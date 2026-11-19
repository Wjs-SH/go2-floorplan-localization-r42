#!/usr/bin/env python3
"""SCI front end: drives the UNCHANGED legacy E5 candidate pipeline (proposals -> tracking -> wall regions -> SigLIP
rerank -> legacy decide) and exposes, per scene, every region with the values the SCI Observation Policy needs.

The legacy decision (`legacy_ev`) is kept as the A0 baseline output. The SCI policy never changes legacy state; it
only reads it. Legacy algorithm defaults are taken from live_event_runner.add_algo_args (no facility/VLM branch).
"""
from __future__ import annotations

import argparse
import base64
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from siglip_transport import PipeTransport, TransportError  # noqa: E402

LEGACY = Path(__file__).resolve().parents[1] / "sci_legacy_e5"
WORKER = Path(__file__).resolve().parent / "siglip_worker.py"
SCI_PY = os.environ.get("SCI_PY", str(Path.home() / "sci_ws" / ".venv" / "bin" / "python"))

_cwd = os.getcwd()
os.chdir(LEGACY)                                  # legacy modules read reports/... grid.json relative to cwd at import
sys.path.insert(0, str(LEGACY))
import live_event_runner as LER  # noqa: E402
import region_memory as M  # noqa: E402
os.chdir(_cwd)

_SPOT_LOG = []                                    # spot ids assigned by legacy decide(), in call order
_legacy_match = LER.match


def _recording_match(spots, xyz, radius, vtol, taken):
    a = _legacy_match(spots, xyz, radius, vtol, taken)
    _SPOT_LOG.append(a)
    return a


LER.match = _recording_match                      # same result; only records the returned id


class WorkerDown(RuntimeError):
    pass


class SciSiglipClient:
    """Drop-in for legacy SiglipClient (same attributes), talking to the SCI worker. Restartable.
    R1: every request has a real read/write deadline (`timeout`, previously stored but unused). After a deadline or an
    invalid reply the transport is poisoned and WorkerDown is raised, so a late reply is never taken as the next answer."""

    def __init__(self, py=SCI_PY, timeout=30.0, ready_timeout=180.0):
        self.py, self.timeout, self.ready_timeout = py, timeout, ready_timeout
        self.want_board, self.p, self.tr = True, None, None
        self.restarts, self.last_error = 0, None
        self.start()

    def start(self):
        env = dict(os.environ)
        env.pop("PYTHONNOUSERSITE", None)         # handoff trap: would hide ~/.local transformers
        self.p = subprocess.Popen([self.py, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(os.environ.get("SCI_SIGLIP_LOG", os.devnull), "ab"), cwd=str(LEGACY), env=env)
        self.tr = PipeTransport(self.p.stdout.fileno(), self.p.stdin.fileno())
        r = self._read(self.ready_timeout)        # model load time is bounded too
        if not r.get("ready"):
            raise WorkerDown("SigLIP worker not ready")
        self.pid = r.get("pid")
        self.prompts, self.model = r.get("prompts"), r.get("model")

    def alive(self):
        return self.p is not None and self.p.poll() is None and (self.tr is None or not self.tr.poisoned)

    def _read(self, deadline_s=None):
        try:
            return self.tr.recv(self.timeout if deadline_s is None else deadline_s)
        except TransportError as e:
            self.last_error = str(e)
            raise WorkerDown(f"SigLIP transport: {e}")

    def score(self, bgr, regs):
        if not self.alive():
            raise WorkerDown("SigLIP worker not running")
        try:
            r = self.tr.request({"bgr": bgr, "regions": regs, "board": self.want_board, "sci": True}, self.timeout)
        except TransportError as e:
            self.last_error = str(e)
            raise WorkerDown(f"SigLIP transport: {e}")
        if "sim" not in r or "sec" not in r:
            self.tr.poisoned = "invalid_reply:missing_fields"
            raise WorkerDown("SigLIP transport: invalid reply (missing sim/sec)")
        self.last_adv = np.array(r.get("adv", []))
        self.last_board = np.array(r.get("board", []))
        self.last_ceil = np.array(r.get("ceil", []))
        self.last_tiles = r.get("tiles", [])
        self.last_neg, self.last_ctx, self.last_ctx_plate = r.get("neg", {}), r.get("ctx", {}), r.get("ctx_plate", [])
        self.last_infer = r.get("t_infer", r["sec"])
        self.last_emb = r.get("emb", [])
        self.last_gpu_ms, self.last_n_tiles, self.last_n_ctx = r.get("gpu_ms"), r.get("n_tiles"), r.get("n_ctx")   # E5-RT telemetry
        return np.array(r["sim"]), r["sec"]

    def close(self):
        if self.p is not None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=5)
            except Exception:
                self.p.kill()


def legacy_args(extra=()):
    p = argparse.ArgumentParser()
    LER.add_algo_args(p)
    return p.parse_args(["--no-facility", *extra])


class Frontend:
    """One legacy LiveEvents instance + SCI per-region export. Scene cadence and allow_stop follow the legacy live()."""

    def __init__(self, siglip=None, workers=4, extra_args=()):
        self.args = legacy_args(extra_args)
        self.args.workers = workers
        LER.SiglipClient = lambda *a, **k: siglip or SciSiglipClient()   # build_events constructs the client
        cwd = os.getcwd()
        os.chdir(LEGACY)
        try:
            self.L, self.vlm, self.pool = LER.build_events(self.args)
        finally:
            os.chdir(cwd)
        self.buf = LER.Frames()
        # E5RT2: the scene may run in a worker thread while the owner keeps adding frames / prefetching. Proposal
        # bookkeeping (L.pending / L.props / L.sent_g) is shared by prefetch() and accumulate*(); this lock serialises
        # them. The owner never waits for it: a busy lock skips that prefetch (the scene recomputes what it needs).
        self._props_lock = threading.Lock()
        self.prefetch_skipped = 0
        self.prompt_set = os.environ.get("KCI_SIGN_PROMPTS")
        if self.prompt_set != self.args.sign_prompts:
            raise RuntimeError(f"SigLIP prompt env mismatch: {self.prompt_set} != {self.args.sign_prompts}")

    def add_frame(self, fid, t, bgr):
        self.buf.add(fid, t, t, bgr)

    def prefetch(self, t_next):
        if not self._props_lock.acquire(blocking=False):
            self.prefetch_skipped += 1
            return
        try:
            self.L.prefetch(self.buf, t_next)
        finally:
            self._props_lock.release()

    def scene(self, t, dimg, pose, allow_stop=True, buf=None):
        """Replica of LiveEvents.scene (same calls, same order) that also returns the vanishing point and regions.
        E5RT2: `buf` = frozen snapshot of the frame buffer (e5rt.snapshot.FrozenFrames) when the scene runs in the
        scene worker; default = the live buffer (synchronous mode, unchanged behaviour)."""
        ts_ = getattr(type(self), "travel_source", None)      # R34: real robot travel for the spot-memory expiry (None = legacy behaviour)
        self.L.cur_travel = ts_() if ts_ else None
        self.L.spot_expire_m = float(getattr(type(self), "spot_expire_m", 0.0))
        L, tm = self.L, {}
        buf = self.buf if buf is None else buf
        t0 = time.perf_counter()
        L.update_floor(dimg)
        if L.prev is not None and t - L.prev["t"] > 3.5:
            L.prev = None                          # legacy live(): long gaps cut the track
        with self._props_lock:
            base, recs = L.accumulate_sparse(buf, t) if L.sparse else L.accumulate(buf, t)
        tm["proposals_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        regs, vp = L.wall_regions(base, recs, dimg)
        tm["regions_ms"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        t_sig_req = time.time()
        self.last_pre_regs = [list(r) for r in regs]
        if hasattr(self.vlm, "set_context"):               # SEM1: value-ordered crop budget needs the scene geometry
            self.vlm.set_context(dimg=dimg, vp=vp, pose=pose, region_world=M.region_world, match=_legacy_match, P=LER.P)
        regs, _ = L.rerank(base["bgr"], regs)
        tm["siglip_roundtrip_ms"] = (time.perf_counter() - t2) * 1e3
        tm["siglip_infer_ms"] = getattr(self.vlm, "last_infer", 0.0) * 1e3
        # E5-RT: device time per forward (CUDA events) and real crop counts; cache stats when the cache wrapper is on
        tm["siglip_gpu_ms"] = getattr(self.vlm, "last_gpu_ms", None)
        tm["siglip_n_tiles"], tm["siglip_n_ctx"] = getattr(self.vlm, "last_n_tiles", None), getattr(self.vlm, "last_n_ctx", None)
        if hasattr(self.vlm, "last_cached"):
            tm["siglip_cached"] = int(sum(bool(c) for c in (self.vlm.last_cached or [])))
            tm["siglip_fresh"] = len(regs) - tm["siglip_cached"]
        bud = getattr(self.vlm, "last_budget", None) if hasattr(self.vlm, "max_crops") else None
        if bud is not None:                        # E5RT2: real neural-crop budget of this request
            tm["crop_budget"] = {k: bud.get(k) for k in ("max_crops", "binding", "crops_requested_upper", "crops_planned_upper",
                                                          "tiles_planned", "worker_tiles", "worker_ctx", "deferred", "tiles_mismatch")
                                 if k in bud}
        t3 = time.perf_counter()
        _SPOT_LOG.clear()
        legacy_ev, thr = L.decide(buf, t, regs, dimg, vp, pose, allow_stop)
        tm["decide_ms"] = (time.perf_counter() - t3) * 1e3
        tm["scene_ms"] = (time.perf_counter() - t0) * 1e3
        ids = list(_SPOT_LOG)
        return dict(t=t, base=base, regs=regs, vp=vp, thr=thr, legacy_ev=legacy_ev, timing=tm, t_siglip_req=t_sig_req,
                    floor_up=None if L.floor_up is None else [float(x) for x in L.floor_up],
                    regions=self.export(regs, dimg, vp, pose, thr, ids))

    def export(self, regs, dimg, vp, pose, thr, ids):
        """Per region: legacy score/filters + SCI hard-negative evidence + world position + legacy spot memory."""
        L, out = self.L, []
        neg = getattr(self.vlm, "last_neg", {}) or {}
        ctx = getattr(self.vlm, "last_ctx", {}) or {}
        ctxp = getattr(self.vlm, "last_ctx_plate", []) or []
        sst = list(getattr(self.vlm, "last_score_status", None) or [])   # E5RT2: FRESH | CACHED | DEFERRED_BUDGET (typed)
        it = iter(ids)                             # decide() calls match() once per region with a world position, in order
        for sim, b in regs:
            key = tuple(float(v) for v in b)
            xyz, z = M.region_world(b, dimg, vp, pose)
            spot, nobs, nscene, msim = None, 0, 0, None
            if xyz is not None:
                spot = next(it, None)
                if spot is not None:
                    sp = L.spots[spot]
                    nobs, nscene = len(sp["sims"]), len(set(sp.get("ts", [])))
                    msim = L.spot_mean(sp)
            hgt = L.box_floor_height(dimg, b) if (L.floor_up is not None and dimg is not None) else None
            out.append(dict(box=[float(v) for v in b], sim=float(sim), xyz=None if xyz is None else [float(v) for v in xyz],
                            z=None if z is None else float(z), spot=spot, nobs=nobs, nscene=nscene, msim=msim,
                            board=float(L.board_of.get(key, -1.0)), adv=float(L.adv_of.get(key, 0.0)), hgt=hgt,
                            fired=spot is not None and spot in L.fired, key=key, like=bool(msim is not None and msim >= thr)))
        idx = {tuple(float(v) for v in b): i for i, b in enumerate(getattr(self, "last_pre_regs", []))}
        for r in out:
            i = idx.get(r["key"])
            r["neg"] = {c: (v[i] if i is not None and i < len(v) else None) for c, v in neg.items()}
            r["ctx"] = {c: (v[i] if i is not None and i < len(v) else None) for c, v in ctx.items()}
            r["ctx_plate"] = ctxp[i] if i is not None and i < len(ctxp) else None
            em = getattr(self.vlm, "last_emb", []) or []
            r["emb"] = base64.b64encode(em[i]).decode() if i is not None and i < len(em) and em[i] is not None else None
            tl = getattr(self.vlm, "last_tiles", []) or []
            r["tile"] = [float(v) for v in tl[i]] if i is not None and i < len(tl) and tl[i] else r["box"]
            r["score_status"] = sst[i] if i is not None and i < len(sst) and sst[i] else "FRESH"
            if r["score_status"] != "FRESH":       # not new evidence: never a normal negative, never a 'like'
                r["like"] = False
            r.pop("key")
        return out
