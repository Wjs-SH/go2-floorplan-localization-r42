#!/usr/bin/env python3
"""e2_worker.py — LIVE2 E2: building-wide Top-K floor/pose hypotheses from the student's metric_fit_v2 internal candidates.

    python e2_worker.py --building kci --store /dev/shm/live2_store --out <LROOT>/runtime/state/e2 --cache-dir <LROOT>/runtime/e2_cache

Pass = every floor of the building fitted on the SAME window end (t_now = newest store sample, W = 60 s, dt = 3 s; the
student floor_id.py / loc_worker floor-role settings). The correct floor is never used to choose maps: all floors are run,
in fixed order 1..N, split over --procs child processes (spawn, like loc_worker).

Per floor: e2_core.fit_window_v2_topk (student fit_window_v2, all de-duplicated candidates kept), the student twoway score
and the student wall_passes check for every candidate, a light map export for E3 (npy, mmap'd by e3), then the floor's
localizer is released. Results are written after EVERY floor (e2_state.json) so E3/E4 see a growing, honestly incomplete
search (SEARCH_INCOMPLETE) instead of waiting for the whole pass.

Resource rules (FAST-LIO, sensor reception and SigLIP keep priority):
  * CPU only by default: CUDA_VISIBLE_DEVICES="" and live2.wallfit_cpu_fast (bit-identical to the student CPU wall fit)
    is installed as the student's 'wallfit_gpu' hook. --device gpu is refused unless approval A lists module e2_gpu.
  * runs under nice / taskset (ctl), one floor resident per child (~0.85-1.2 GB), floor work queue = one pass at a time,
    re-search requests are coalesced (depth 1) and rate limited (--min-pass-gap-s).
  * before opening a floor: FAST-LIO /Odometry >= 90 % of the A1 baseline (8 Hz without baseline), MemAvailable >=
    --mem-start-mb and max temp < --temp-hold-c, otherwise the floor is HELD (HELD_FASTLIO / HELD_MEMORY / HELD_THERMAL, retried) — accumulation, FAST-LIO and the feed are not touched.
  * watchdog: if MemAvailable falls below --mem-kill-mb during a fit, the coordinator stops that child (its floor is
    re-queued and the event is recorded); nothing else is stopped.
"""
import argparse
import gc
import csv
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
SCI_LIVE = CAND / "external" / "sci_live"
sys.path.insert(0, str(HERE))
import e2_region_ledger as RL  # noqa: E402


def atomic_json(path, obj, durable=False):
    path = Path(path)
    if durable:
        return RL.atomic_durable(path, obj, default=_js)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=_js))
    os.replace(tmp, path)


def _js(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def max_temp():
    import glob
    v = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            t = int(open(z).read()) / 1000.0
            if t > -40:
                v.append(t)
        except (OSError, ValueError):
            pass
    return max(v) if v else None


def hold_reason_for(a, state_dir, paused):
    """New heavy work waits while FAST-LIO is below its protection line, memory is short, the SoC is hot or the operator paused.
    Shared by the coordinator (before a floor) and the fit child (before every scale of an exact fit)."""
    if paused():
        return "HELD_PAUSED by operator (no new floor fit until resume)"
    lio = read_json(Path(state_dir) / "lio_state.json") or {}
    base = read_json(Path(state_dir) / "baseline_A1.json") or {}
    hz = (lio.get("odom") or {}).get("hz_10s")
    if hz is not None:
        line = 0.9 * float(base["hz_p50"]) if base.get("hz_p50") else 8.0
        if hz < line:
            return f"HELD_FASTLIO odom {hz:.1f} Hz < {line:.1f} Hz (FAST-LIO keeps priority)"
    mem, t = mem_available_mb(), max_temp()
    if mem < a.mem_start_mb:
        return f"HELD_MEMORY MemAvailable {mem:.0f} < {a.mem_start_mb} MB"
    if t is not None and t >= a.temp_hold_c:
        return f"HELD_THERMAL {t:.1f} >= {a.temp_hold_c} C"
    return None


def rss_mb():
    try:
        return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
    except (OSError, IndexError, ValueError):
        return None


def _paths():
    for p in (SCI_LIVE / "floor_id", SCI_LIVE / "realtime", HERE):
        sys.path.insert(0, str(p))


def install_cpu_hook():
    """Student rt_localizer imports 'wallfit_gpu' at import time; give it the bit-identical CPU kernel instead."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import wallfit_cpu_fast
    sys.modules["wallfit_gpu"] = wallfit_cpu_fast
    return wallfit_cpu_fast


# ------------------------------------------------------------------ child: one floor at a time
class FloorRunner:
    def __init__(self, a):
        _paths()
        self.a = a
        self.FW = install_cpu_hook() if a.device == "cpu" else None
        import floor_id as F
        self.F = F
        self.cfg = F.load_cfg()
        self.b = F.setup_env(self.cfg, a.building)
        import numpy as np
        import rt_localizer as RT, run_floor_id as R, twoway as TW, metric_fit_v2 as V
        import e2_core as E
        self.np, self.RT, self.R, self.TW, self.V, self.E = np, RT, R, TW, V, E
        self.W = float(self.cfg["consts"]["window_s"])
        self.DT = float(self.cfg["consts"]["dt_s"])
        self.DT_STUDENT = self.DT
        if getattr(a, "exact_dt", None):          # 2026-10-28 replay (X20261027_234844 / _235345, 9 floors each, same window end):
            self.DT = float(a.exact_dt)           # 3 -> 6 s frame spacing = 1.7x faster (54 -> 32 s/floor, 1 proc), best pose per
        self.SC = F.map_scales(self.b, R, int(self.b["n_maps"]))   # floor unchanged (<= 3 px, <= 0.3 deg); window W unchanged
        self.store = str(Path(a.store).resolve())
        self.maps_root = os.environ["SEMLOC_MAPS_ROOT"]
        self.beta_memo = {}

    def share_beta(self, Z):
        """Student fit_beta reads only the store (scans + odometry), never the floor map, so its result for one window is
        the same on every floor (live 2026-10-19: 230.0 deg / mirror 1 on all 6 floors). Memoise it per (odom session,
        interval, step) in this child: same inputs -> the same value, only the repeated 3.3 s KD-tree search is skipped."""
        orig, memo = Z.fit_beta, self.beta_memo
        sess = (read_json(Path(self.store) / "session.json") or {}).get("odom_session_id")

        def fit_beta(t_start=None, t_end=None, stride=1.5, fine=2.0):
            key = (sess, None if t_start is None else float(t_start), None if t_end is None else float(t_end), stride, fine)
            if sess is not None and key in memo:
                Z.BETA, q = memo[key]
                Z.beta_quality = dict(q); Z.MIRROR = q['mirror']
                return float(self.np.degrees(Z.BETA)), q['error_m']
            r = orig(t_start, t_end, stride=stride, fine=fine)
            if sess is not None:
                if len(memo) > 16:
                    memo.clear()
                memo[key] = (Z.BETA, dict(Z.beta_quality))
            return r
        Z.fit_beta = fit_beta

    def cache_key(self, m):
        import hashlib, glob
        h = hashlib.md5()
        h.update(repr((self.a.building, m, float(self.R.MID), sys.version, self.np.__version__,
                       sorted((k, v) for k, v in os.environ.items() if k.startswith("SEMLOC_")))).encode())
        files = []
        for k in ("SEMLOC_MAPS_ROOT", "SEMLOC_SEMMAPS_ROOT"):
            root = os.environ.get(k)
            if root:
                files += sorted(glob.glob(f"{root}/{m}F/**/*", recursive=True))
        files += sorted(glob.glob(str(SCI_LIVE / "floor_id" / "localizer" / "*.py")))
        for fp in files:
            if os.path.isfile(fp):
                h.update(fp.encode()); h.update(open(fp, "rb").read())
        return h.hexdigest()[:16]

    def open_floor(self, m):
        """Same as loc_worker.Loc.open_floor (pickle cache keyed by maps + localizer code), cache kept in the LIVE2 tree."""
        import pickle, gzip
        t1 = time.time(); Z = None; src = "built"
        cdir = Path(self.a.cache_dir); cdir.mkdir(parents=True, exist_ok=True)
        f = cdir / f"{self.a.building}_{m}F_{self.cache_key(m)}.pkl.gz"
        if f.exists():
            try:
                with gzip.open(f, "rb") as fh:
                    Z = pickle.load(fh)
                src = "cache"
            except Exception:
                Z = None
        if Z is None:
            Z = self.RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=self.R.MID, data_dir=self.store, bag=None)
            self.V._maps(Z, m)
            try:
                far, Z.L.FAR = Z.L.FAR, None
                tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
                with gzip.open(tmp, "wb", compresslevel=1) as fh:
                    pickle.dump(Z, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, f); Z.L.FAR = far
                for old in cdir.glob(f"{self.a.building}_{m}F_*.pkl.gz"):
                    if old != f:
                        old.unlink()
            except Exception:
                pass
        Z.L.dir = self.store
        self.refresh(Z)
        return Z, src, time.time() - t1

    def refresh(self, Z):
        """loc_worker.Loc.refresh: re-read the store with the localizer's own reader (algorithm unchanged)."""
        L = Z.L
        L._load_odom(self.store)
        with self.np.load(f"{self.store}/far_points.npz") as fz:
            from loc_worker import _Far
            L.FAR = _Far({k: fz[k] for k in fz.files})
        c = getattr(L, "_lscn_cache", None)
        if c:
            c.clear()            # scene labels are rebuilt from the data actually in the store (no stale cache)

    def export_light(self, Z, m, mapver):
        """E3 light map (npy, read with mmap): wall distance field F and building-outside distance (px * 4 as uint16),
        through-wall mask, traversable mask, real wall (twoway). Written once per floor + map version."""
        np = self.np
        d = Path(self.a.cache_dir) / "light" / f"{self.a.building}_{m}F_{mapver}"
        if (d / "meta.json").exists():
            return str(d)
        d.mkdir(parents=True, exist_ok=True)
        M = self.V._maps(Z, m)
        L = Z.L
        q = lambda a: np.minimum(np.rint(np.asarray(a, float) * 4.0), 65535).astype(np.uint16)  # noqa: E731
        np.save(d / "F_q4.npy", q(M["F"]))
        np.save(d / "outd_q4.npy", q(M["out_d"]))
        np.save(d / "wall_tr.npy", np.asarray(M["wall_tr"], bool))
        np.save(d / "ok.npy", np.asarray(Z.ok_px, bool))
        np.save(d / "wall_real.npy", np.asarray(getattr(L, "wall_real", L.wall), bool))
        atomic_json(d / "meta.json", dict(floor=m, H=int(L.H), W=int(L.W), map_version=mapver, quant="px*4 uint16",
                                          created_wall=time.time(), source="metric_fit_v2._maps + RTLocalizer.ok_px"))
        return str(d)

    def scale_gate(self, m, progress=None, sleep=time.sleep):
        """between() hook of an exact fit: report scale progress and WAIT (same thread, no result change) while held."""
        state_dir = Path(self.a.out).parent

        def paused():
            return bool((read_json(state_dir / "control" / "pause.json") or {}).get("paused"))

        def gate(i, n):
            t0, why0 = time.time(), None
            while True:
                why = hold_reason_for(self.a, state_dir, paused)
                if why is None:
                    break
                if why != why0 and progress:
                    progress(dict(scale=i, n_scales=n, held=why))
                why0 = why
                sleep(1.0)
            self.held_s += time.time() - t0
            if progress:
                progress(dict(scale=i, n_scales=n, held=None, held_s=round(self.held_s, 1)))
        return gate

    def run_floor(self, m, t_now, W, stage="exact", progress=None, region=None):
        np = self.np
        out = dict(floor=m, stage=stage)
        t0 = time.time()
        Z, src, open_s = self.open_floor(m)
        self.share_beta(Z)
        out.update(open_s=round(open_s, 1), open_src=src, rss_open_mb=rss_mb())
        mapver = self.E.map_version(self.maps_root, m)
        t1 = time.time()
        if stage in ("coarse", "refresh"):
            import e2_coarse as EC
            r = EC.coarse_floor(Z, t_now, self.SC[m], floor=m, W=W, **EC.COARSE_DEFAULT)
        else:
            self.held_s = 0.0
            r = self.E.fit_window_v2_topk(Z, t_now, self.SC[m], floor=m, W=W, dt=self.DT, between=self.scale_gate(m, progress),
                                          **(dict(region=region, topk=REGION_TOPK) if region else {}))
        out["fit_s"] = round(time.time() - t1, 1)
        out["held_in_fit_s"] = round(getattr(self, "held_s", 0.0), 1) if stage not in ("coarse", "refresh") else 0.0
        L = Z.L
        pts, yaw = L._scan(t_now)
        wall_real = getattr(L, "wall_real", L.wall)
        import csv as _csv
        import pose_tools as PT
        from loc_worker import wall_passes
        rows = [q for q in _csv.DictReader(open(f"{self.store}/odom_link.csv"))]
        recs = []
        t2 = time.time()
        for i, c in enumerate(r["candidates"]):
            d_obs = self.TW.sector_obs(np.asarray(pts, float) * [1, c["mirror"]], np.deg2rad(c["phi"]) + yaw + c["beta"])
            tw = float(self.TW.twoway_score(wall_real, *c["pos"], c["ppm"], d_obs)[0])
            wp = None
            if i < self.a.wall_pass_top:
                fit = dict(phi=c["phi"], ppm=c["ppm"], anchor=c["anchor"], pos=c["pos"], beta=c["beta"])
                tr = [PT.map_pose_px(fit, float(q["fastlio_x"]), float(q["fastlio_y"]), float(q["fastlio_yaw_rad"]))[:2]
                      for q in rows if c["t"] - W <= float(q["t_sec"]) <= c["t"]]
                try:
                    wp = len(wall_passes(self.V._maps(Z, m)["wall_tr"], tr, float(c["ppm"])))
                except Exception:
                    wp = None
            recs.append(dict(self.E.candidate_record(c, m, mapver, None, L.H, twoway=tw, wall_pass=wp, now_wall=time.time()),
                             stage=stage))
        out["score_s"] = round(time.time() - t2, 1)
        out["light"] = self.export_light(Z, m, mapver)          # cached per floor + map version (E3 needs it for coarse too)
        out.update(n_raw=r["n_raw"], n_feasible=r["n_feasible"], n_picked=r["n_picked"], candidates=recs,
                   student_best=dict(pos=[float(v) for v in r["best"]["pos"]], phi=r["best"]["phi"], ppm=r["best"]["ppm"],
                                     merit=r["best"]["cost_m"]),
                   beta_deg=float(np.degrees(r["best"]["beta"]) % 360), mirror=int(r["best"]["mirror"]),
                   beta_source=r["best"].get("beta_source"), map_version=mapver, H=int(L.H), W=int(L.W),
                   t=float(r["best"]["t"]), sec=round(time.time() - t0, 1), rss_peak_mb=rss_mb(),
                   wallfit=dict(self.FW.STATS) if self.FW else None)
        del Z, r
        gc.collect()
        return out


def child_main(a, part, conn):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)          # die with the coordinator
    except Exception:
        pass
    try:
        FR = FloorRunner(a)
        conn.send(("ready", part, rss_mb()))
        while True:
            msg = conn.recv()
            if msg is None:
                return
            m, t_now, W, stage = msg[:4]
            region = msg[4] if len(msg) > 4 else None
            try:
                prog = (lambda d, m=m: conn.send(("progress", m, d))) if stage not in ("coarse", "refresh") else None
                conn.send(("floor_done", m, FR.run_floor(m, t_now, W, stage, progress=prog, region=region)))
            except Exception as e:
                conn.send(("floor_failed", m, dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        return


# ------------------------------------------------------------------ coordinator
class Coordinator:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        _paths()
        import floor_id as F
        cfg = F.load_cfg()
        self.cfg_for_hash = cfg
        self.b = F.setup_env(cfg, a.building)
        self.W = float(cfg["consts"]["window_s"])
        self.floors = a.floors or list(range(1, int(self.b["n_maps"]) + 1))
        self.store = str(Path(a.store).resolve())
        self.pass_id = 0
        self.region_added = []
        self._resume = None          # R34: a pass interrupted by a crash continues from its remaining floors
        self.state = dict(kind="LIVE2_E2_STATE", building=a.building, building_name=self.b.get("name"), status="WAITING_DATA",
                          floors_total=len(self.floors), floors=self.floors, device=a.device, procs=a.procs, exact_dt=getattr(a, "exact_dt", None), pass_id=0,
                          algorithm="student metric_fit_v2 internal candidates (all de-duplicated, T4-scored) + twoway + wall_passes",
                          candidates=[], per_floor={}, floors_done=[], floors_failed={}, floors_pending=list(self.floors),
                          held=None, events=[], started_wall=time.time())
        self.stop = False
        self.ledger = None
        self.restore()

    def log(self, event_kind, /, **k):
        """Append to e2_passes.jsonl. A field that collides with a reserved key is kept as field_<key>, and a logging failure
        is counted, never raised: at 11-01 a duplicate pass_id here killed the worker AFTER a successful regional fit."""
        reserved = dict(wall=time.time(), kind=event_kind, pass_id=getattr(self, "pass_id", None))
        rec = dict(reserved)
        for kk, v in k.items():
            rec[f"field_{kk}" if kk in reserved else kk] = v
        try:
            line = json.dumps(rec, ensure_ascii=False, default=_js)
            with open(self.out / "e2_passes.jsonl", "a") as f:
                f.write(line + "\n")
        except Exception as ex:      # logging must never invalidate saved results
            self.state["log_errors"] = int(self.state.get("log_errors") or 0) + 1
            self.state["last_log_error"] = f"{type(ex).__name__}: {ex}"[:200]
        self.state["events"] = ((self.state.get("events") or []) + [{kk: rec[kk] for kk in rec if kk not in ("candidates", "tb")}])[-20:]

    def write(self, durable=False):
        self.state["updated_wall"] = time.time()
        atomic_json(self.out / "e2_state.json", self.state, durable=durable)

    # ------------------------------------------------------------------ R34: identity + restore after a crash
    def identity(self):
        """What must be the same for saved candidates to stay valid. Each part is a content hash or an id, never a time."""
        import hashlib
        ex = read_json(Path(self.a.out).resolve().parent.parent / "ctl" / "experiment.json") or {}
        h = hashlib.sha256()
        maps_root = os.environ.get("SEMLOC_MAPS_ROOT")
        for m in self.floors:
            d = Path(str(maps_root)) / f"{m}F" / "maps"
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.yaml")):
                h.update(p.name.encode()); h.update(p.read_bytes())
        return dict(run_id=ex.get("exp_id"), odom_session=self.session().get("odom_session_id"), maps=h.hexdigest()[:16],
                    calib=hashlib.sha256(json.dumps([self.cfg_for_hash, self.session().get("fastlio_cfg_sha256")], sort_keys=True, default=str).encode()).hexdigest()[:16])

    def restore(self):
        """Same run, odometry session, maps and calibration -> floors, candidates and requests continue from e2_state.json. Any
        difference, or an unreadable file -> a fresh search, and the reason is shown. Hypotheses are never carried across sessions."""
        idn = self.identity()
        for _ in range(RESTORE_SESSION_WAIT_S):          # a restart may meet a store that is only just readable
            if idn.get("odom_session") or not (self.out / "e2_state.json").exists():
                break
            time.sleep(1.0)
            idn = self.identity()
        self.state["identity"] = idn
        self.ledger = RL.Ledger(self.out / "e2_region_ledger.json")
        p = self.out / "e2_state.json"
        led = self.ledger.load(idn)
        try:
            old = json.loads(p.read_text())
        except OSError:
            old = None
        except ValueError:
            old, led = None, led + "+STATE_CORRUPT"
            self.state["restore"] = dict(status="FRESH", why="e2_state.json unreadable", wall=time.time())
        if not old:
            self.state.setdefault("restore", dict(status="FRESH", why="no earlier state of this run", ledger=led, wall=time.time()))
            return
        o_id = old.get("identity") or {}
        diff = [k for k in sorted(idn) if not idn.get(k) or o_id.get(k) != idn.get(k)]
        if diff:
            self.log("restore_refused", why=f"identity differs: {diff}")
            self.state["restore"] = dict(status="FRESH", why=f"identity differs: {diff} (earlier candidates not used)", ledger=led, wall=time.time())
            return
        keep = ("pass_id", "pass_why", "t_now", "window_s", "odom_session", "pass_started_wall", "finished_wall", "pass_sec", "floors_done", "floors_failed",
                "per_floor", "candidates", "candidates_pass", "coarse", "candidates_coarse", "regional", "events", "started_wall", "wait_note", "restarts")
        for k in keep:
            if k in old:
                self.state[k] = old[k]
        self.pass_id = int(old.get("pass_id") or 0)
        self.state["restarts"] = int(old.get("restarts") or 0) + 1
        self.state["floors_failed"] = {k: v for k, v in (old.get("floors_failed") or {}).items() if v != "CHILD_DIED"}
        self.state["per_floor"] = {k: v for k, v in (old.get("per_floor") or {}).items() if v.get("status") != "RUNNING" or v.get("light")}
        for k, v in self.state["per_floor"].items():
            if v.get("status") == "RUNNING":
                v["status"] = "INTERRUPTED"
        self.state["floors_pending"] = [f for f in self.floors if f not in self.state.get("floors_done", []) and str(f) not in self.state["floors_failed"]]
        self.region_added = [c for c in old.get("candidates_pass") or [] if c.get("regional_request")]
        for r in self.state.get("regional") or []:
            if r.get("status") == "RUNNING":
                r.update(status="INTERRUPTED", why_failed="worker ended during this fit", finished_wall=time.time())
        rec = self.ledger.recover_after_restart(self.state.get("regional"))
        unfinished = old.get("status") == "RUNNING" and self.pass_id >= 1
        cs = old.get("coarse") or {}
        mode = "PASS_COMPLETE"
        if unfinished and (self.state["floors_done"] or cs.get("status") in ("DONE", "OFF")):
            self._resume = "RESUME_AFTER_RESTART"
            mode = "RESUME_REMAINING_FLOORS"
        elif unfinished:                       # died in the coarse stage: no exact floor was lost, the pass simply starts again
            self.pass_id -= 1
            mode = "COARSE_STAGE_RESTART"
        self.state["pass_id"] = self.pass_id
        self.state["status"] = "RUNNING" if self._resume else "IDLE" if self.pass_id else "WAITING_DATA"
        self.state["restore"] = dict(status="RESTORED", mode=mode, wall=time.time(), floors_done=list(self.state.get("floors_done") or []), ledger=led,
                                     n_candidates=len(self.state.get("candidates") or []), running_requests=rec)
        self.state["regional_ledger"] = self.ledger.summary()
        self.write(durable=True)
        self.log("restored", **{k: v for k, v in self.state["restore"].items() if k != "wall"})

    def span(self):
        sys.path.insert(0, str(SCI_LIVE / "realtime"))
        from live_store import read_span
        return read_span(self.store)

    def session(self):
        return read_json(Path(self.store) / "session.json") or {}

    def paused(self):
        """Operator pause (web console): no new pass and no new floor fit; a floor already running finishes."""
        p = read_json(self.out.parent / "control" / "pause.json") or {}
        return bool(p.get("paused"))

    def window_path_m(self, t1, W=None):
        """Odometry path inside the E2 window: the student calibration needs translation (field 2026-10-18: stationary
        windows failed on every floor with CalibrationUnavailable)."""
        W = self.W if W is None else W
        import csv
        try:
            rows = [(float(r["t_sec"]), float(r["fastlio_x"]), float(r["fastlio_y"]))
                    for r in csv.DictReader(open(Path(self.store) / "odom_link.csv"))]
        except (OSError, ValueError, KeyError):
            return 0.0
        xy = [(x, y) for t, x, y in rows if t1 - W <= t <= t1]
        return float(sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(xy, xy[1:])))

    def early_on(self):
        return self.pass_id == 0 and bool(self.a.coarse) and self.a.early_window_s > 0

    def want_pass(self, last_pass_wall):
        if self.paused():
            return None, "PAUSED_BY_OPERATOR"
        s = self.span()
        need, W = (self.a.early_window_s + 1.5, self.a.early_window_s) if self.early_on() else (self.a.min_span, self.W)
        if s is None or s[1] - s[0] < need:
            return None, None if s is None else round(s[1] - s[0], 1)
        path = self.window_path_m(s[1], W)
        if path < self.a.min_window_path_m:
            return None, f"WAITING_MOTION path {path:.1f} m < {self.a.min_window_path_m:.1f} m in the E2 window"
        req = read_json(self.out / "e2_request.json")
        if self.pass_id == 0:
            return "INITIAL", None
        if req and req.get("wall", 0) > (self.state.get("finished_wall") or 0):
            if time.time() - last_pass_wall >= self.a.min_pass_gap_s:
                return "RESEARCH:" + str(req.get("why")), None
            return None, "REQUEST_RATE_LIMITED"
        return None, None

    def run(self):
        import multiprocessing as mp
        from multiprocessing.connection import wait as mp_wait
        sys.path.insert(0, str(SCI_LIVE / "realtime"))
        import live_store as LS
        ctx = mp.get_context("spawn")
        K = max(1, min(self.a.procs, len(self.floors)))
        kids = {}

        def spawn(j):
            pc, cc = ctx.Pipe()
            pr = ctx.Process(target=child_main, args=(self.a, None, cc), daemon=True)
            pr.start()
            msg = pc.recv()
            kids[j] = dict(pr=pr, pc=pc, busy=None, since=None, rss0=msg[2])
        self._spawn = spawn
        for j in range(K):
            spawn(j)
        self.log("children_ready", n=K)
        last_pass_wall = float(self.state.get("finished_wall") or 0.0)
        self.write()
        while not self.stop:
            self.apply_budget(kids, spawn)
            rq = self.region_pending() if self.budget() > 0 else None
            if rq and not self._resume:
                self.region_pass(rq, kids, LS)
                continue
            if self._resume:
                sess = self.session()
                sp = self.span()
                t_now = self.state.get("t_now")
                if sp is None or t_now is None or sp[0] > t_now - self.W:      # the saved window left the store (900 s): newest window
                    if sp is None or sp[1] - sp[0] < self.a.min_span or self.window_path_m(sp[1]) < self.a.min_window_path_m:
                        self.state["exact_wait"] = "restored pass: remaining floors wait for the student window"
                        self.write(); time.sleep(1.0)
                        continue
                    t_now = sp[1]
                self._resume = None
                self.state.update(status="RUNNING", t_now=t_now, held=None, exact_wait=None)
                CS = self.state.get("coarse") or dict(enabled=False, status="OFF", floors_done=[], floors_failed={}, per_floor={}, ranking=[])
                self.state["coarse"] = CS
                Wc, early = self.W, False
                queue = [] if KCI_ACC else [("exact", f, t_now, self.W) for f in exact_order(CS.get("ranking") or [], self.state["floors_pending"])
                         if f in self.state["floors_pending"]]
                exact_queued = True
                cands = [c for c in self.state.get("candidates_pass") or [] if not c.get("regional_request")]
                ccands = []
                self.log("pass_resume", t_now=t_now, floors_done=self.state.get("floors_done"), remaining=[q[1] for q in queue])
            else:
                why, note = self.want_pass(last_pass_wall)
                if not why and self.pass_id >= 1 and getattr(self.a, "refresh_s", 0) and (KCI_ACC or not self.state.get("candidates_pass") == []) and self.refresh_idle(kids, mp_wait, spawn):
                    continue
                if not why:
                    self.state["status"] = wait_status(note, self.pass_id)
                    self.state["wait_note"] = note
                    self.write()
                    time.sleep(1.0)
                    continue
                self.pass_id += 1
                t_now = self.span()[1]
                sess = self.session()
                LS.pin(self.store, "e2_pass", t_now - self.W - 10.0, t_now, ttl_s=3600.0)
                self.state.update(status="RUNNING", pass_id=self.pass_id, pass_why=why, t_now=t_now, window_s=self.W,
                                  odom_session=sess.get("odom_session_id"), pass_started_wall=time.time(), finished_wall=None,
                                  floors_done=[], floors_failed={}, floors_pending=list(self.floors), per_floor={},
                                  candidates_pass=[], held=None)
                self.log("pass_start", why=why, t_now=t_now, odom_session=sess.get("odom_session_id"))
                coarse_on = bool(self.a.coarse)
                early = self.pass_id == 1 and coarse_on and self.a.early_window_s > 0
                Wc = self.a.early_window_s if early else self.W            # EARLY coarse: short window, first Top-K < 60 s
                queue = [("coarse" if coarse_on else "exact", m, t_now, Wc if coarse_on else self.W) for m in self.floors]
                exact_queued = not coarse_on
                cands, ccands = [], []
                CS = self.state["coarse"] = dict(enabled=coarse_on, status="RUNNING" if coarse_on else "OFF", floors_done=[],
                                                 floors_failed={}, per_floor={}, ranking=[], started_wall=time.time(),
                                                 all_floors_wall=None, sec=None, params=coarse_params() if coarse_on else None,
                                                 window_s=Wc if coarse_on else None, t_now=t_now, early=early,
                                                 note="COARSE = approximate pre-ranking (thinner window); orders the exact pass, "
                                                      "never counts as a searched floor")
                self.state["candidates_coarse"] = []
            while (queue or any(k["busy"] for k in kids.values()) or not exact_queued) and not self.stop:
                self.refresh_due(queue, CS)
                self.apply_budget(kids, spawn)
                if exact_queued and CS.get("status") != "RUNNING":
                    t_now = self.new_structure_window(queue, t_now, LS)
                if not exact_queued and not queue and not any(k["busy"] for k in kids.values()):
                    if CS["status"] == "RUNNING":
                        CS.update(status="DONE", sec=round(time.time() - CS["started_wall"], 1))
                        self.log("coarse_done", sec=CS["sec"], ranking=CS["ranking"], floors_failed=CS["floors_failed"], window_s=Wc)
                    if early and not KCI_ACC:      # the exact pass needs the student window (W + 10 s of data, W s of motion)
                        sp = self.span()
                        if sp is None or sp[1] - sp[0] < self.a.min_span or self.window_path_m(sp[1]) < self.a.min_window_path_m \
                                or self.paused():
                            self.state["exact_wait"] = f"exact pass waits for the {self.W:.0f} s window ({self.a.min_span:.0f} s of data, " \
                                                       f">= {self.a.min_window_path_m:.1f} m path)"
                            self.write(); time.sleep(1.0)
                            continue
                        t_now = sp[1]
                        LS.pin(self.store, "e2_pass", t_now - self.W - 10.0, t_now, ttl_s=3600.0)
                        self.state.update(t_now=t_now, window_s=self.W, exact_wait=None)
                        self.log("exact_window", t_now=t_now)
                    order = [] if KCI_ACC else exact_order(CS["ranking"], self.floors)   # 2026-11-13 esy v2: with e2_acc.on the 4 children go to the 20 s coarse rounds
                    queue += [("exact", f, t_now, self.W) for f in order]
                    exact_queued = True
                for j, k in kids.items():
                    if k["busy"] is None and self.pass_id >= 1 and self.budget() > 0 and not any(kk["busy"] and kk["busy"][0] == "region" for kk in kids.values()):
                        rq = self.region_pending()              # 2026-11-01: a semantic region request does not wait for the pass end
                        if rq:
                            rrec, msg = self.region_start(rq)
                            if msg is not None:
                                k["pc"].send(msg); k["busy"], k["since"] = ("region", msg[0], msg[1], msg[2], rrec), time.time()
                                continue
                    if k["busy"] is None and queue:
                        hold = self.hold_reason()
                        if hold:
                            if self.state.get("held") != hold:
                                self.log("held", why=hold)
                            self.state["held"] = hold
                            break
                        self.state["held"] = None
                        stage, m, t_i, W_i = queue.pop(0)
                        k["pc"].send((m, t_i, W_i, stage)); k["busy"], k["since"] = (stage, m, t_i, W_i), time.time()
                        if stage != "refresh":
                            (CS["per_floor"] if stage == "coarse" else self.state["per_floor"])[str(m)] = dict(
                                status="RUNNING", started_wall=time.time(), child=j)
                self.write()
                ready = mp_wait([k["pc"] for k in kids.values() if k["busy"] is not None], timeout=1.0)
                low = mem_available_mb() < self.a.mem_kill_mb
                tmp = max_temp()
                hot = tmp is not None and tmp >= self.a.temp_stop_c      # in-flight fits stop too (a fit lasts 60-110 s live)
                stopped_low = False                  # R39 (X20261102_210840 +277 / +280 s: VLM start stopped both fits, about 100 s of work each, exact stayed 1 of 9): low memory stops one child per check; heat stops all
                for j, k in sorted(kids.items(), reverse=True):
                    if k["busy"] is not None and ((low and not stopped_low) or hot):
                        stopped_low = stopped_low or low
                        stage, m, t_i, W_i = k["busy"][:4]
                        k["pr"].terminate(); k["pr"].join(10)
                        why = "MEMORY" if low else "THERMAL"
                        if stage == "region":                 # a region fit is retried later, never queued as a floor
                            self.region_guarded(self.region_transient, k["busy"][4], f"child stopped ({why})", True)
                            del kids[j]
                            spawn(j)
                            continue
                        self.log("child_stopped_" + why.lower(), floor=m, stage=stage, mem_mb=round(mem_available_mb()), temp_c=tmp)
                        if stage != "refresh":                # a repeat search is simply skipped under memory / heat
                            (CS["per_floor"] if stage == "coarse" else self.state["per_floor"])[str(m)] = dict(status="REQUEUED_" + why, wall=time.time())
                            queue.insert(0, (stage, m, t_i, W_i))
                        del kids[j]
                        spawn(j)
                        continue
                    if k["pc"] not in ready:
                        continue
                    stage, m, t_i, W_i = k["busy"][:4]
                    try:
                        kind, m, payload = k["pc"].recv()
                    except EOFError:
                        self.log("child_died", floor=m, stage=stage)
                        if stage == "region":                 # a dead region fit is a retry of that request, not a failed floor
                            self.region_guarded(self.region_transient, k["busy"][4], "child died during the regional fit", True)
                            del kids[j]; spawn(j)
                            continue
                        if stage == "refresh":
                            del kids[j]; spawn(j)
                            continue
                        (CS["floors_failed"] if stage == "coarse" else self.state["floors_failed"])[str(m)] = "CHILD_DIED"
                        del kids[j]; spawn(j)
                        continue
                    if kind == "progress":                    # exact fit: scale i/n reached, or waiting on a hold
                        if stage not in ("region", "refresh"):
                            self.on_progress(m, payload)      # the child is still busy with this floor
                            self.write()
                        continue
                    if stage == "region":                     # 2026-11-01 in-pass semantic region fit finished
                        rrec = k["busy"][4]
                        k["busy"] = None
                        self.region_guarded(self.region_finish, rrec, kind, m, payload)
                        continue
                    k["busy"] = None
                    if stage == "refresh":
                        self.refresh_result(kind, m, payload)
                        self.write()
                        continue
                    if stage == "coarse":
                        self.coarse_result(CS, ccands, kind, m, payload, sess, t_i)
                        self.write()
                        continue
                    if kind == "floor_done":
                        for c in payload["candidates"]:
                            c["pass_id"], c["odom_session"] = self.pass_id, sess.get("odom_session_id")
                        cands += payload["candidates"]
                        if m not in self.state["floors_done"]:     # R40: a floor can be searched again on a newer window
                            self.state["floors_done"].append(m)
                        self.state["per_floor"][str(m)] = {kk: payload[kk] for kk in payload if kk != "candidates"}
                        self.state["per_floor"][str(m)]["status"] = "DONE"
                        # R34: the floor result is written durably BEFORE the log (the restart continues from here)
                        self.state["floors_pending"] = [f for f in self.floors if f not in self.state["floors_done"]
                                                        and str(f) not in self.state["floors_failed"]]
                        self.state["candidates_pass"] = rank(cands + self.region_added)
                        if self.pass_id == 1 or not self.state["candidates"]:
                            self.state["candidates"] = self.state["candidates_pass"]
                        self.write(durable=True)
                        self.log("floor_done", floor=m, sec=payload["sec"], fit_s=payload["fit_s"], open_s=payload["open_s"],
                                 score_s=payload["score_s"], held_in_fit_s=payload.get("held_in_fit_s"), n=len(payload["candidates"]),
                                 rss_peak_mb=payload["rss_peak_mb"],
                                 t_now=t_now, candidates=payload["candidates"])
                    else:
                        self.state["floors_failed"][str(m)] = payload["error"]
                        self.state["per_floor"][str(m)] = dict(status="FAILED", error=payload["error"])
                        self.log("floor_failed", floor=m, error=payload["error"], tb=payload.get("tb"))
                    self.state["floors_pending"] = [f for f in self.floors if f not in self.state["floors_done"]
                                                    and str(f) not in self.state["floors_failed"]]
                    self.state["candidates_pass"] = rank(cands + self.region_added)
                    if self.pass_id == 1 or not self.state["candidates"]:
                        self.state["candidates"] = self.state["candidates_pass"]
                    self.write()
            if CS["status"] == "RUNNING":
                CS["status"] = "INTERRUPTED"
            LS.unpin(self.store, "e2_pass")
            self.state.update(status=pass_status(self.state["floors_done"], self.state["floors_failed"], self.stop), finished_wall=time.time(),
                              candidates=rank(cands + self.region_added), pass_sec=round(time.time() - self.state["pass_started_wall"], 1))
            last_pass_wall = time.time()
            self.write(durable=True)
            self.log("pass_done", sec=self.state["pass_sec"], n=len(cands), floors_done=self.state["floors_done"],
                     floors_failed=self.state["floors_failed"], status=self.state["status"])
            self.write()
        for k in kids.values():
            try:
                k["pc"].send(None)
            except (OSError, BrokenPipeError):
                pass

    # ------------------------------------------------------------------ regional E2 (R34: persistent, idempotent, bounded)
    def _ledger_sync(self):
        """requests -> ledger; E3 acknowledgements -> ledger; a short copy into e2_state for the tablet."""
        led = self.ledger
        cur = self.session().get("odom_session_id")
        known = (led.d.get("identity") or {}).get("odom_session")
        if cur and cur != known:              # FAST-LIO restarted (new frame): requests and candidates of the old session stop here
            idn = dict(led.d.get("identity") or {}, odom_session=cur)
            why = led.load(idn) if known else "SESSION_BECAME_KNOWN"
            if not known:
                led.d["identity"] = idn; led.save()
            self.state["identity"] = idn
            self.log("ledger_session", why=why, odom_session=cur)
        for rq in RL.read_requests(self.out):
            if rq.get("odom_session") == self.session().get("odom_session_id") or led.get(rq["request_id"]) is None:
                led.ingest(rq)
        led.merge_ack(read_json(self.out / "e2_region_ack.json"))
        self.state["regional_ledger"] = led.summary()

    def region_pending(self):
        """The newest runnable request of THIS odometry session, or None. Finished, final and not-yet-due entries are not returned."""
        if self.pass_id < 1:
            return None
        self._ledger_sync()
        e = self.ledger.ready()
        return e["request"] if e else None

    def region_radius(self, R, t_now):
        """-> (radius_px or None, info). R34: request radius + (NET displacement since the capture + odometry uncertainty) x generous
        px/m. A walk there and back no longer grows the region: 11-01 O003 went 3554.8 px on a 110 m path, the whole map. Over the
        limit the robot's place is no longer tied to the reading: None -> a new observation is needed."""
        radius = float(R["radius_px"])
        info = {}
        if R.get("capture_wall") and R.get("px_per_m_hi") and R.get("radius_base_px"):
            net, path, n = self.motion_between(float(R["capture_wall"]), t_now)
            if n < 2 and t_now - float(R["capture_wall"]) > REGION_MOTION_GAP_S:
                return None, dict(walked_since_capture_at_fit_m=None, why=f"no odometry between the capture and now ({t_now - float(R['capture_wall']):.0f} s)")
            unc = max(REGION_ODOM_ERR_FLOOR_M, REGION_ODOM_REL_ERR * path)
            ppm = float(R["px_per_m_hi"])
            radius = float(R["radius_base_px"]) + (net + unc) * ppm
            info.update(walked_since_capture_at_fit_m=round(path, 1), net_since_capture_m=round(net, 1), odom_uncertainty_m=round(unc, 1))
            if radius / ppm > REGION_RADIUS_MAX_M:
                info["why"] = f"region {radius / ppm:.0f} m > {REGION_RADIUS_MAX_M:.0f} m (net {net:.0f} m, path {path:.0f} m since the capture)"
                return None, info
        return radius, info

    def region_rec(self, rid, e, R, rq):
        """The tablet / E8 record of this request: one per request, updated by each attempt."""
        for r in reversed(self.state.setdefault("regional", [])):
            if r.get("request_id") == rid:
                r.update(status="RUNNING", attempt=e["attempts"], started_wall=time.time(), why_failed=None, seq=e["seq"])
                return r
        rec = dict(request_id=rid, floor=R.get("floor"), center_px=R.get("center_px"), radius_px=R.get("radius_px"), rooms=R.get("rooms"),
                   why=rq.get("why"), started_wall=time.time(), status="RUNNING", attempt=e["attempts"], basis=R.get("basis"), seq=e["seq"])
        self.state["regional"].append(rec)
        return rec

    def region_start(self, rq):
        """-> (rec, child message) or (rec, None) when the request cannot run now (rec already final / RETRY_WAIT)."""
        rid = rq["request_id"]
        e = self.ledger.ingest(rq)
        R = rq.get("region") or {}
        if R.get("floor") is None:
            self.ledger.start(e)
            rec = self.region_rec(rid, e, R, rq)
            self.ledger.fail(e, "bad request (no floor)")
            rec.update(status="FAILED", why_failed="bad request (no floor)", finished_wall=time.time())
            self.write(durable=True)
            self.log("region_done", **rec)
            return rec, None
        sp = self.span()
        hold = self.hold_reason()
        if hold or sp is None or self.window_path_m(sp[1]) < self.a.min_window_path_m:
            why = hold or f"E2 window has < {self.a.min_window_path_m} m motion (robot at rest)"
            self.ledger.wait(e, why)
            rec = self.region_rec(rid, e, R, rq)
            return rec, self._region_settle(rec, e, why)
        t_now = sp[1]
        radius, info = self.region_radius(R, t_now)
        self.ledger.start(e)
        rec = self.region_rec(rid, e, R, rq)
        rec.update(info)
        if radius is None:
            why = "NEEDS_NEW_OBSERVATION: " + str(info.get("why"))
            self.ledger.fail(e, why)
            rec.update(status="FAILED", why_failed=why, finished_wall=time.time())
            self.write(durable=True)
            self.log("region_done", **rec)
            return rec, None
        rec.update(t_now=t_now, radius_px_at_fit=round(radius, 1))
        self.state["regional_ledger"] = self.ledger.summary()
        self.write(durable=True)
        return rec, (int(R["floor"]), t_now, self.W, "exact", dict(center_px=R["center_px"], radius_px=radius))

    def _region_settle(self, rec, e, why):
        """Mirror the ledger state into the tablet record, save, then log."""
        final = e["state"] == RL.FAILED_FINAL
        rec.update(status="FAILED" if final else "RETRY_WAIT", why_failed=e.get("why") or why, finished_wall=time.time(), attempt=e.get("attempts"))
        self.state["regional_ledger"] = self.ledger.summary()
        self.write(durable=True)
        self.log("region_done", **{kk: v for kk, v in rec.items() if kk != "candidates"})
        return None

    def motion_between(self, t0, t1):
        """(net displacement, path length, samples) of FAST-LIO between t0 and t1."""
        try:
            import csv as _csv
            xy = [(float(q["fastlio_x"]), float(q["fastlio_y"])) for q in _csv.DictReader(open(f"{self.store}/odom_link.csv"))
                  if t0 <= float(q["t_sec"]) <= t1]
        except (OSError, ValueError, KeyError):
            return 0.0, 0.0, 0
        if len(xy) < 2:
            return 0.0, 0.0, len(xy)
        path = float(sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(xy, xy[1:])))
        return float(((xy[-1][0] - xy[0][0]) ** 2 + (xy[-1][1] - xy[0][1]) ** 2) ** 0.5), path, len(xy)

    def path_between(self, t0, t1):
        return self.motion_between(t0, t1)[1]

    def region_transient(self, rec, why, fit_started=True):
        """The fit was refused or stopped (calibration on a resting robot, memory / thermal stop): retried, bounded (RL.MAX_ATTEMPTS)."""
        e = self.ledger.get(rec["request_id"])
        self.ledger.retry(e, why)
        self._region_settle(rec, e, why)

    def region_guarded(self, fn, *args):
        """An error while handling one regional result must not end the worker or the running floor search."""
        rec = args[0]
        try:
            return fn(*args)
        except Exception as ex:
            why = f"{type(ex).__name__}: {ex}"[:200]
            sys.stderr.write(traceback.format_exc())
            e = self.ledger.get(rec["request_id"])
            if e is not None and e["state"] == RL.RUNNING:
                self.ledger.interrupted(e, why)
                rec.update(status="FAILED" if e["state"] == RL.FAILED_FINAL else "RETRY_WAIT", why_failed=e.get("why"), finished_wall=time.time())
            self.state["regional_ledger"] = self.ledger.summary()
            self.write(durable=True)
            self.log("region_error", request_id=rec.get("request_id"), error=why)

    def region_finish(self, rec, kind, m, payload):
        """Order (R34): candidates + state saved atomically -> ledger COMPLETED -> log. A logging failure cannot lose candidates."""
        sess = self.session()
        e = self.ledger.get(rec["request_id"])
        if kind == "floor_done":
            tag = f"{self.pass_id}R{e['seq'] if e else len(self.state.get('regional') or [])}"
            for c in payload["candidates"]:
                c.update(pass_id=tag, odom_session=sess.get("odom_session_id"), regional_request=rec["request_id"], stage="exact")
            self.state["candidates_pass"] = rank(list(self.state.get("candidates_pass") or []) + payload["candidates"])
            self.state["candidates"] = rank(list(self.state.get("candidates") or []) + payload["candidates"])
            self.region_added = self.region_added + payload["candidates"]   # kept across the pass reset of candidates
            pf = self.state["per_floor"].setdefault(str(m), {})
            for kk in ("light", "H", "W", "map_version"):      # R41: W was missing: e34 read pf["W"] whenever H was there and died (integrated replay X20261103_001810 +325 s; reachable in the field since an early reading asks a regional fit before the floor's exact result)
                pf.setdefault(kk, payload.get(kk))
            rec.update(status="DONE", n=len(payload["candidates"]), sec=payload["sec"], region_pass_id=tag,
                       candidates_hash=RL.candidates_hash(payload["candidates"]), finished_wall=time.time(),
                       candidates=[dict(px=c.get("px"), py=c.get("py"), phi_deg=c.get("phi_deg"), ppm=c.get("ppm"), merit=c.get("merit"))
                                   for c in payload["candidates"]])
            self.write(durable=True)
            if e:
                self.ledger.complete(e, payload["candidates"])
        elif "Calibration" in str(payload.get("error")) or "Too few scan points" in str(payload.get("error")):
            return self.region_transient(rec, str(payload.get("error"))[:200])
        else:
            rec.update(status="FAILED", why_failed=payload.get("error"), finished_wall=time.time())
            if e:
                self.ledger.fail(e, str(payload.get("error"))[:300])
        self.state["regional_ledger"] = self.ledger.summary()
        self.write(durable=True)
        self.log("region_done", **{kk: v for kk, v in rec.items() if kk != "candidates"})

    def region_pass(self, rq, kids, LS):
        """2026-10-30 SEMANTIC RECOVERY between passes (idle child, blocking): ONE floor, student seed grid limited to the region;
        candidates ADDED (stage exact, pass_id '<pass>R<n>', regional_request); floors_done unchanged. 2026-11-01: also run
        INSIDE a pass on the next idle child (see run()), radius recomputed at fit time."""
        rec, msg = self.region_start(rq)
        if msg is None:
            return
        j = next(iter(kids))
        k = kids[j]
        k["pc"].send(msg)
        try:
            while True:
                kind, m, payload = k["pc"].recv()
                if kind != "progress":
                    break
        except EOFError:
            self.region_guarded(self.region_transient, rec, "child died during the regional fit")
            del kids[j]
            self._spawn(j)
            return
        self.region_guarded(self.region_finish, rec, kind, m, payload)

    def new_structure_window(self, queue, t_now, LS):
        """R40 (X20261103_003729: the exact window was pinned at +102 s, a straight corridor. 4F there: every candidate 22-44 m off. The same fit on a window with the robot's turn: +200 s 2.3 m, +300 s 2.2 m, +385 s 3.3 m). When the odometry has turned since the pinned window,
        the floors still queued use the newest window and the floors already searched go again behind them. Same fit, later data: nothing about the search itself changes. -> the window end now in use."""
        try:
            rows = [(float(r["t_sec"]), float(r["fastlio_yaw_rad"])) for r in csv.DictReader(open(Path(self.store) / "odom_link.csv"))]
        except (OSError, ValueError, KeyError):
            return t_now
        now = time.time()
        if now - getattr(self, "_nsw_check", 0.0) < 5.0:
            return t_now
        self._nsw_check = now
        sp = self.span()
        n = int(self.state.get("exact_new_windows") or 0)
        if sp is None or self.paused() or n >= NEW_WINDOW_MAX or not new_window_due(rows, t_now, sp[1], self.W) or self.window_path_m(sp[1]) < self.a.min_window_path_m:
            return t_now
        t2 = sp[1]
        LS.pin(self.store, "e2_pass", t2 - self.W - 10.0, t2, ttl_s=3600.0)
        waiting = {q[1] for q in queue if q[0] == "exact"}
        queue[:] = [(q[0], q[1], t2, q[3]) if q[0] == "exact" else q for q in queue]
        again = [f for f in self.floors if f not in waiting]
        queue += [("exact", f, t2, self.W) for f in again]
        self.state.update(t_now=t2, exact_new_windows=n + 1)
        self.log("exact_new_window", t_now=t2, previous=t_now, requeued=sorted(waiting), again=again, n=n + 1)
        return t2

    def budget(self):
        """R39: how many children may work now (<out>/e2_budget.json from auto_fsm). Default: all. 4 where signs are not in use, 2 where SigLIP, OCR and the VLM need the cores and the memory."""
        try:
            mb = int(json.loads((Path(self.a.out) / "e2_budget.json").read_text())["max_busy"])
        except (OSError, ValueError, KeyError, TypeError):
            return max(1, int(self.a.procs))
        return max(0, min(mb, int(self.a.procs)))

    def apply_budget(self, kids, spawn):
        """An idle child above the budget ends (its floor leaves memory for the VLM); a busy one finishes its floor first. Missing ones start again when the budget rises."""
        mb = self.budget()
        for j in sorted(kids):
            if kids[j]["busy"] is None and ((mb == 0 and not kids[j].get("fresh")) or (mb > 0 and j >= mb)):      # 0 (VLM needs the memory): every idle child gives its floor back; one fresh child stays so that the loops keep a pipe to wait on
                kids[j]["pr"].terminate(); kids[j]["pr"].join(10)
                del kids[j]
        if mb == 0 and not kids:
            spawn(0)
            kids[0]["fresh"] = True
        for j in range(mb):
            if j not in kids:
                spawn(j)
            else:
                kids[j].pop("fresh", None)
        return mb

    def refresh_idle(self, kids, mp_wait, spawn):
        """Repeat searches go on after the pass is done. -> True when one ran."""
        q = []
        if not self.refresh_due(q):
            return False
        while (q or any(k["busy"] for k in kids.values())) and not self.stop:
            self.apply_budget(kids, spawn)
            for k in kids.values():
                if k["busy"] is None and q and not self.hold_reason():
                    stage, m, t_i, W_i = q.pop(0)
                    k["pc"].send((m, t_i, W_i, stage)); k["busy"], k["since"] = (stage, m, t_i, W_i), time.time()
            ready = mp_wait([k["pc"] for k in kids.values() if k["busy"] is not None], timeout=1.0)
            for j, k in list(kids.items()):
                if k["busy"] is None or k["pc"] not in ready:
                    continue
                try:
                    kind, m, payload = k["pc"].recv()
                except EOFError:
                    del kids[j]; spawn(j)
                    continue
                if kind == "progress":
                    continue
                k["busy"] = None
                self.refresh_result(kind, m, payload)
            self.write()
        return True

    def refresh_due(self, queue, CS=None):
        """R37 (loc1, --refresh-s > 0): every floor searched again, coarsely, on the newest window, in front of the exact queue. The exact pass looks at one early window, so a floor can look far worse than it is
        (11-02: F7 long-window cost 0.145 -> 0.024 from a repeat search). 5 s per floor. Candidates accumulate per floor; nothing is removed. This is what makes a floor comparison fair; it is not a shortcut around the exact pass."""
        rs = float(getattr(self.a, "refresh_s", 0) or 0) * int(self.a.procs) / max(1, self.budget())    # R39: half the children, half as often: a repeat pass must not swallow the exact queue
        if rs <= 0 or self.pass_id < 1 or any(q[0] == "refresh" for q in queue) or ((CS or {}).get("status") == "RUNNING"):
            return False
        R = self.state.setdefault("refresh", dict(n_passes=0, last_wall=0.0))
        now = time.time()
        forced = False
        try:                                  # R40: final_verify asks for one whole search on the newest window before its deciding comparison
            forced = float(json.loads((Path(self.a.out) / "e2_refresh_now.json").read_text()).get("wall") or 0) > R["last_wall"]
        except (OSError, ValueError, TypeError):
            pass
        if now - R["last_wall"] < rs and not forced:
            return False
        sp = self.span()
        if sp is None or self.paused():
            return False
        R.update(last_wall=now, n_passes=R["n_passes"] + 1, t_now=sp[1])
        queue[0:0] = [("refresh", f, sp[1], REFRESH_W_S) for f in self.floors]
        self.log("refresh_start", n=R["n_passes"], t_now=sp[1])
        return True

    def kci_acc_add(self, m, payload):
        """2026-11-13 esy: per-floor best merit of every coarse/refresh round, averaged (lower = better) -> e2_state kci_floor_acc.rank.
        Offline on 17 robot recordings (1-4F): accumulated coarse rounds put the true floor first in 71-79 % at 80-100 s vs 20 % for one early round."""
        if not KCI_ACC or not payload.get("candidates"):
            return
        sess = self.state.get("odom_session")
        A = self.state.get("kci_floor_acc")
        if not A or A.get("session") != sess:
            A = self.state["kci_floor_acc"] = dict(session=sess, merits={}, rank=[], mean={}, n={},
                                                   how="esy 2026-11-13: mean of each round's best coarse merit per floor, lower = better")
        cs = payload["candidates"]
        ok = [c["merit"] for c in cs if c.get("wall_pass") == 0]           # 2026-11-13 esy v2: physical model (10-13 rule): a pose that passes a wall to the other side is not this floor's score
        checked = [c for c in cs if c.get("wall_pass") is not None]
        sc = min(ok) if ok else (min(c["merit"] for c in cs) + (0.5 if checked else 0.0))
        A["merits"].setdefault(str(m), []).append(round(float(sc), 4))
        mean = {int(f): sum(v) / len(v) for f, v in A["merits"].items()}
        A["rank"] = sorted(mean, key=mean.get)
        A["mean"] = {str(f): round(v, 4) for f, v in mean.items()}
        A["n"] = {f: len(v) for f, v in A["merits"].items()}
        A["updated_wall"] = time.time()

    def refresh_result(self, kind, m, payload):
        if kind != "floor_done":
            self.log("refresh_floor_failed", floor=m, error=payload.get("error"))
            return
        for c in payload["candidates"]:
            c.update(pass_id=self.pass_id, odom_session=self.session().get("odom_session_id"), stage="refresh", refresh_n=self.state["refresh"]["n_passes"])
        self.kci_acc_add(m, payload)
        keep = [c for c in self.state.get("candidates_refresh") or [] if not (c["floor"] == m and c.get("refresh_n", 0) < self.state["refresh"]["n_passes"] - 3)]
        self.state["candidates_refresh"] = keep + payload["candidates"][:3]
        self.log("refresh_floor_done", floor=m, sec=payload["sec"], fit_s=payload["fit_s"], n=len(payload["candidates"]))

    def coarse_result(self, CS, ccands, kind, m, payload, sess, t_now):
        """Coarse floor finished: rank all coarse candidates so far; the exact pass will follow this floor order."""
        if kind != "floor_done":
            CS["floors_failed"][str(m)] = payload["error"]
            CS["per_floor"][str(m)] = dict(status="FAILED", error=payload["error"])
            self.log("coarse_floor_failed", floor=m, error=payload["error"])
            return
        for c in payload["candidates"]:
            c.update(pass_id=self.pass_id, odom_session=sess.get("odom_session_id"), stage="coarse")
        self.kci_acc_add(m, payload)
        ccands += payload["candidates"]
        CS["floors_done"].append(m)
        CS["per_floor"][str(m)] = dict({kk: payload[kk] for kk in ("sec", "fit_s", "open_s", "n_raw", "n_picked", "light", "H", "W")},
                                       status="DONE")
        ranked = rank(ccands)
        self.state["candidates_coarse"] = ranked[:24]
        CS["ranking"] = list(dict.fromkeys(c["floor"] for c in ranked))
        if len(CS["floors_done"]) + len(CS["floors_failed"]) >= len(self.floors) and CS["all_floors_wall"] is None:
            CS["all_floors_wall"] = time.time()
            CS["all_floors_s_after_pass_start"] = round(CS["all_floors_wall"] - self.state["pass_started_wall"], 1)
        self.log("coarse_floor_done", floor=m, sec=payload["sec"], fit_s=payload["fit_s"], n=len(payload["candidates"]), t_now=t_now,
                 ranking=CS["ranking"], candidates=payload["candidates"])

    def on_progress(self, m, payload):
        """Exact-fit progress from a child (scale i of n, or waiting on a hold inside the fit) -> per-floor state + log."""
        pf = self.state["per_floor"].setdefault(str(m), dict(status="RUNNING"))
        pf.update(payload, progress_wall=time.time())
        if payload.get("held"):
            self.log("held_in_fit", floor=m, why=payload["held"], scale=payload.get("scale"))

    def hold_reason(self):
        """New heavy fits wait while FAST-LIO is below its protection line, memory is short or the SoC is hot."""
        if self.budget() == 0:
            return "HELD_VLM the VLM needs the memory for one photo (no new floor; a running fit finishes)"
        return hold_reason_for(self.a, Path(self.a.out).parent, self.paused)


REGION_RADIUS_MAX_M = 45.0     # base ~19 m (15 m camera range + room + 15 % scale margin) + about 25 m of net motion; beyond it: new observation
REGION_ODOM_REL_ERR, REGION_ODOM_ERR_FLOOR_M = 0.05, 0.5   # same FAST-LIO relative-drift assumption as e8_pair
RESTORE_SESSION_WAIT_S = 30
REGION_MOTION_GAP_S = 30.0     # no odometry for this long after the capture: the motion is unknown
REGION_TOPK = 6          # semantic recovery: more student seeds per scale / direction inside the (small) region


def coarse_params():
    import e2_coarse as EC
    return dict(EC.COARSE_DEFAULT)


def pass_status(done, failed, stopped):
    """End-of-pass status. A pass cut short by a stop (experiment end / approval expiry) is INTERRUPTED, not FAILED:
    2026-10-19 21:35 a pass stopped after 68 s with 0 floors done and 0 floors failed was shown as 'all floors failed'."""
    if done:
        return "PARTIAL_INTERRUPTED" if stopped else "DONE"
    if stopped:
        return "INTERRUPTED"
    return "FAILED" if failed else "NO_RESULT"


NEW_WINDOW_MAX, NEW_TURN_DEG, NEW_MIN_AGE_S = 3, 45.0, 30.0


def new_window_due(rows, t_old, t_new, W):
    """rows = [(t, yaw rad)]. Pure. True when the window ending at t_new holds a turn of >= NEW_TURN_DEG that the window ending at t_old does not, and the two are at least NEW_MIN_AGE_S apart. A straight corridor never asks; time alone never asks."""
    if t_new - t_old < NEW_MIN_AGE_S:
        return False

    def turn(t1):
        y = [a for t, a in rows if t1 - W <= t <= t1]
        if len(y) < 2:
            return 0.0
        un, lo, hi = 0.0, 0.0, 0.0
        for a, b in zip(y, y[1:]):
            un += (b - a + math.pi) % (2 * math.pi) - math.pi
            lo, hi = min(lo, un), max(hi, un)
        return math.degrees(hi - lo)
    return turn(t_new) >= NEW_TURN_DEG and turn(t_old) < NEW_TURN_DEG


def wait_status(note, pass_id):
    """Idle status from want_pass()'s note: str reason, float store span (s) while the window fills, or None.
    (2026-10-19 field crash: a float note reached str.startswith -> E2 exited rc=1 at its first idle tick.)"""
    note = "" if note is None else str(note)
    if note == "PAUSED_BY_OPERATOR":
        return "PAUSED"
    if note.startswith("WAITING_MOTION"):
        return "WAITING_MOTION"
    return "WAITING_DATA" if pass_id == 0 else "IDLE"


REFRESH_W_S = 60.0
KCI_ACC = (Path.home() / "kci_loadtest_20261113" / "e2_acc.on").exists()   # 2026-11-13 esy: file present -> coarse every 20 s on a 20 s window + per-floor score accumulation (e2_state kci_floor_acc)


def exact_order(ranking, floors):
    """Exact-pass queue: coarse ranking first, then every floor the coarse stage did not rank. NO floor is ever dropped
    (2026-10-21 EXP2: the true floor 4F was 5th of 9 in the coarse ranking)."""
    return list(ranking) + [f for f in floors if f not in ranking]


def rank(cands):
    """Building-wide order = the student's floor criterion: wall_pass > 0 last, then twoway (higher better), then merit."""
    return sorted(cands, key=lambda c: ((c.get("wall_pass") or 0) > 0, -(c["twoway"] if c["twoway"] is not None else -1e9),
                                        c["merit"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--floors", type=int, nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--refresh-s", type=float, default=0.0, help="R37: repeat coarse search of every floor every N s (loc1); 0 = off")
    ap.add_argument("--exact-dt", type=float, default=6.0,
                    help="exact-pass frame spacing in the W window (student config 3 s; 6 s replay-checked, see FloorRunner)")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--min-span", type=float, default=70.0, help="store span before the first pass (student W + 10)")
    ap.add_argument("--min-pass-gap-s", type=float, default=120.0)
    ap.add_argument("--min-window-path-m", type=float, default=2.0, help="odometry path needed inside the E2 window")
    ap.add_argument("--wall-pass-top", type=int, default=4, help="student wall_passes check for the best N candidates per floor")
    ap.add_argument("--mem-start-mb", type=float, default=3500.0)
    ap.add_argument("--mem-kill-mb", type=float, default=2800.0)
    ap.add_argument("--temp-stop-c", type=float, default=78.0,
                    help="stop the floor fits already running at/above this temperature (requeued; FAST-LIO / lio2 keep priority)")
    ap.add_argument("--temp-hold-c", type=float, default=76.0,
                    help="no new floor fit at or above this temperature: E2 (CPU-heavy, non-essential) yields before the SigLIP 80 C pause")
    ap.add_argument("--early-window-s", type=float, default=20.0,
                    help="first pass: COARSE stage on this short window as soon as it exists (0 = off); exact stage waits for the "
                         "student window. 2026-10-19 real 4F data: W=20 first all-floor Top-K 58.5 s, 9/9 floors in 6/6 windows")
    ap.add_argument("--coarse", type=int, choices=(0, 1), default=1,
                    help="1: approximate all-floor pre-ranking (e2_coarse) before the exact pass, exact pass in coarse order")
    a = ap.parse_args()
    if KCI_ACC:                                  # 2026-11-13 esy (switch file ~/kci_loadtest_20261113/e2_acc.on): repeat coarse search on the newest 20 s every 20 s
        global REFRESH_W_S
        REFRESH_W_S = 20.0
        a.refresh_s = 20.0
    if a.device == "gpu" and os.environ.get("LIVE2_E2_GPU_APPROVED") != "1":
        sys.exit("REFUSED --device gpu: approval A must list module e2_gpu (ctl sets LIVE2_E2_GPU_APPROVED)")
    co = Coordinator(a)
    signal.signal(signal.SIGTERM, lambda *_: setattr(co, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(co, "stop", True))
    co.run()


if __name__ == "__main__":
    main()
