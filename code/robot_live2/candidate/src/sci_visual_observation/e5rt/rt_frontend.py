"""E5-RT glue between the unchanged legacy front end and the SigLIP worker (CPU side, no torch import).

RtSiglipCache wraps the SciSiglipClient (same .score(bgr, regs) API and the same last_* attributes the legacy code reads):
  1. associates each region of the current scene with the previous scene's regions (track id) by propagating the
     previous boxes with sparse LK through the buffered frames between the two scene frames (same as legacy
     _propagate, but on boxes) and IoU >= assoc_iou,
  2. asks VerifyScheduler which regions need a FRESH forward (NEW / IMPROVED_SCALE / IMPROVED_SHARP / APPEARANCE /
     AGED / INVALIDATED) and sends only those to the worker,
  3. fills cache hits from the last fresh result of the same track, marks them in `last_cached`, and tells the legacy
     decide() not to append them to the spot memory (LiveEvents.no_evidence_keys) -> a cache hit is never new evidence.
Telemetry per call: n_regions, n_fresh, n_cached, reasons, assoc stats, worker time, host time.
mode 'off' = pass-through (bit-identical to the baseline request).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .scheduler import VerifyScheduler, crop_signature
from .tracker import iou


def _roi_stats(gray, b):
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in b[:4])
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None
    r = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(r, cv2.CV_32F).var()), float(r.mean())


class BoxAssociator:
    """Track ids for scene regions via LK propagation of the previous scene's boxes (geometry only)."""

    def __init__(self, assoc_iou=0.5, step_s=0.2, win=21, levels=3, fb_max=1.5):
        self.assoc_iou, self.step_s, self.win, self.levels, self.fb_max = assoc_iou, step_s, (win, win), levels, fb_max
        self.prev = None                     # dict(th, gray, boxes, ids)
        self.next_id = 0
        self.stats = dict(calls=0, matched=0, new=0, lk_fail=0)

    def _frames_between(self, buf, t_a, t_b):
        items = list(buf.items)
        ts = np.array([f["th"] for f in items]) if items else np.zeros(0)
        out = []
        for t in list(np.arange(t_a + self.step_s, t_b, self.step_s)) + [t_b]:
            if len(ts):
                out.append(items[int(np.argmin(np.abs(ts - t)))])
        return out

    def assign(self, buf, base, boxes):
        """E5RT2: LK None / non-finite / shape-mismatch results, time going backwards and a base frame newer than the
        buffer are explicit failures (no association, new ids) instead of exceptions or silent re-use."""
        self.stats["calls"] += 1
        self.stats.setdefault("time_backwards", 0)
        self.stats.setdefault("future_base", 0)
        ids = [None] * len(boxes)
        if self.prev is not None and base is not None and base["th"] < self.prev["th"]:
            self.stats["time_backwards"] += 1             # replay restart / clock reset: old ids are not continued
            self.prev = None
        newest = buf.items[-1]["th"] if getattr(buf, "items", None) else None
        if base is not None and newest is not None and base["th"] > newest + 1e-6:
            self.stats["future_base"] += 1                # base not in the buffer timeline: refuse propagation
            self.prev = None
        if self.prev is not None and self.prev["boxes"] and base is not None and base["th"] - self.prev["th"] <= 3.5:
            cen = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in self.prev["boxes"]], np.float32).reshape(-1, 1, 2)
            ok = np.ones(len(cen), bool)
            g0 = self.prev["gray"]
            cur = cen.copy()
            for f in self._frames_between(buf, self.prev["th"], base["th"]):
                g1 = f["gray"]
                if g1 is g0:
                    continue
                try:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, cur, None, winSize=self.win, maxLevel=self.levels)
                    if nxt is None or st is None or nxt.shape != cur.shape or not np.isfinite(nxt).all():
                        raise ValueError("missing_or_nonfinite_forward_flow")
                    back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, nxt, None, winSize=self.win, maxLevel=self.levels)
                    if back is None or st2 is None or back.shape != cur.shape or not np.isfinite(back).all():
                        raise ValueError("missing_or_nonfinite_backward_flow")
                except (cv2.error, ValueError, TypeError):
                    ok[:] = False                         # explicit LK failure: nothing is associated
                    break
                fb = np.linalg.norm((cur - back).reshape(-1, 2), axis=1)
                ok &= (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb <= self.fb_max)
                cur, g0 = nxt, g1
            moved = []
            for k, b in enumerate(self.prev["boxes"]):
                if not ok[k]:
                    self.stats["lk_fail"] += 1
                    moved.append(None)
                    continue
                d = cur.reshape(-1, 2)[k] - cen.reshape(-1, 2)[k]
                moved.append([b[0] + d[0], b[1] + d[1], b[2] + d[0], b[3] + d[1]])
            used = set()
            for j, b in enumerate(boxes):
                best, bv = None, self.assoc_iou
                for k, mb in enumerate(moved):
                    if mb is None or k in used:
                        continue
                    v = iou(mb, b)
                    if v >= bv:
                        best, bv = k, v
                if best is not None:
                    used.add(best)
                    ids[j] = self.prev["ids"][best]
                    self.stats["matched"] += 1
        for j in range(len(ids)):
            if ids[j] is None:
                ids[j] = self.next_id
                self.next_id += 1
                self.stats["new"] += 1
        if base is not None:
            self.prev = dict(th=base["th"], gray=base["gray"], boxes=[list(b) for b in boxes], ids=list(ids))
        return ids


class RtSiglipCache:
    LEGACY_ATTRS = ("last_adv", "last_board", "last_ceil", "last_tiles", "last_neg", "last_ctx", "last_ctx_plate", "last_emb")

    def __init__(self, client, frontend_buf, legacy_events=None, revision=None, mode="cache", scheduler=None, clock=time.monotonic):
        self.c, self.buf, self.L, self.mode, self.clock = client, frontend_buf, legacy_events, mode, clock
        self.sched = scheduler or VerifyScheduler(revision or dict(model=getattr(client, "model", None),
                                                                   prompts=str(getattr(client, "prompts", None)), preprocess="legacy_prep_v1"),
                                                  clock=clock)
        self.assoc = BoxAssociator()
        self.last_cached, self.last_track_ids, self.last_reasons = [], [], {}
        self.tele = []
        self.want_board = getattr(client, "want_board", True)

    def __getattr__(self, k):                       # pid, prompts, model, alive(), close(), timeout ...
        return getattr(self.c, k)

    def _base_for(self, bgr):
        for f in reversed(list(self.buf.items)):
            if f["bgr"] is bgr:
                return f
        return None

    def score(self, bgr, regs):
        t0 = time.perf_counter()
        if self.mode == "off" or not regs:
            r = self.c.score(bgr, regs)
            self._copy_from_client(len(regs))
            self.last_cached = [False] * len(regs)
            if self.L is not None:
                self.L.no_evidence_keys = set()
            return r
        base = self._base_for(bgr)
        gray = base["gray"] if base is not None else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ids = self.assoc.assign(self.buf, base, regs)
        now = self.clock()
        items = []
        for j, b in enumerate(regs):
            sh, mg = _roi_stats(gray, b)
            items.append(dict(key=j, track_id=ids[j], sig=crop_signature(b, sh, mg)))
        fresh, cached, dec = self.sched.plan(items, now)
        # a cache hit is only usable if we really hold a result for that track
        cached = [j for j in cached if self.sched.cached_result(ids[j]) is not None]
        # E5RT2: never undo the scheduler's DEFERRED_BUDGET decision.
        # Missing cached result is deferred/invalidated for next scheduling, not an unbudgeted forward.
        fresh = sorted(set(fresh))
        deferred = sorted(set(range(len(regs))) - set(fresh) - set(cached))
        self.last_deferred = deferred
        self.last_score_status = ["FRESH" if j in fresh else "CACHED" if j in cached else "DEFERRED_BUDGET" for j in range(len(regs))]
        n = len(regs)
        sim = np.full(n, -1.0)
        adv, board, ceil = np.zeros(n), np.full(n, -1.0), np.full(n, -1.0)
        tiles, emb = [None] * n, [None] * n
        neg, ctx, ctxp = {}, {}, [None] * n
        sec = 0.0
        if fresh:
            s_f, sec = self.c.score(bgr, [regs[j] for j in fresh])
            fa = lambda name, default: list(getattr(self.c, name, default))  # noqa: E731
            a_adv, a_board, a_ceil = fa("last_adv", []), fa("last_board", []), fa("last_ceil", [])
            a_tiles, a_emb, a_ctxp = fa("last_tiles", []), fa("last_emb", []), fa("last_ctx_plate", [])
            a_neg, a_ctx = getattr(self.c, "last_neg", {}) or {}, getattr(self.c, "last_ctx", {}) or {}
            for k, j in enumerate(fresh):
                sim[j] = float(s_f[k])
                adv[j] = a_adv[k] if k < len(a_adv) else 0.0
                board[j] = a_board[k] if k < len(a_board) else -1.0
                ceil[j] = a_ceil[k] if k < len(a_ceil) else -1.0
                tiles[j] = a_tiles[k] if k < len(a_tiles) else None
                emb[j] = a_emb[k] if k < len(a_emb) else None
                ctxp[j] = a_ctxp[k] if k < len(a_ctxp) else None
                for c_, v in a_neg.items():
                    neg.setdefault(c_, [None] * n)[j] = v[k] if k < len(v) else None
                for c_, v in a_ctx.items():
                    ctx.setdefault(c_, [None] * n)[j] = v[k] if k < len(v) else None
                res = dict(sim=sim[j], adv=adv[j], board=board[j], ceil=ceil[j],
                           tile_rel=None if tiles[j] is None else [tiles[j][0] - regs[j][0], tiles[j][1] - regs[j][1],
                                                                  tiles[j][2] - regs[j][0], tiles[j][3] - regs[j][1]],
                           neg={c_: neg[c_][j] for c_ in neg}, ctx={c_: ctx[c_][j] for c_ in ctx}, ctx_plate=ctxp[j], emb=emb[j])
                self.sched.store(ids[j], items[j]["sig"], res, now)
        for j in cached:
            r = self.sched.cached_result(ids[j])
            sim[j], adv[j], board[j], ceil[j] = r["sim"], r["adv"], r["board"], r["ceil"]
            tr = r.get("tile_rel")
            tiles[j] = None if tr is None else [regs[j][0] + tr[0], regs[j][1] + tr[1], regs[j][0] + tr[2], regs[j][1] + tr[3]]
            emb[j], ctxp[j] = r.get("emb"), r.get("ctx_plate")
            for c_, v in (r.get("neg") or {}).items():
                neg.setdefault(c_, [None] * n)[j] = v
            for c_, v in (r.get("ctx") or {}).items():
                ctx.setdefault(c_, [None] * n)[j] = v
        self.last_adv, self.last_board, self.last_ceil = adv, board, ceil
        self.last_tiles, self.last_emb, self.last_neg, self.last_ctx, self.last_ctx_plate = tiles, emb, neg, ctx, ctxp
        self.last_infer = sec
        self.last_gpu_ms = getattr(self.c, "last_gpu_ms", None) if fresh else None
        self.last_n_tiles = getattr(self.c, "last_n_tiles", None) if fresh else 0
        self.last_n_ctx = getattr(self.c, "last_n_ctx", None) if fresh else 0
        self.last_cached = [j in cached for j in range(n)]
        self.last_track_ids, self.last_reasons = ids, dec
        if self.L is not None:
            self.L.no_evidence_keys = {tuple(float(v) for v in regs[j]) for j in cached + deferred}
        self.tele.append(dict(n=n, fresh=len(fresh), cached=len(cached), deferred=len(deferred), worker_s=sec, host_ms=(time.perf_counter() - t0) * 1e3,
                              reasons={str(k): v for k, v in dec.items()}))
        self.tele = self.tele[-300:]
        return sim, sec

    def _copy_from_client(self, n):
        for a in self.LEGACY_ATTRS:
            setattr(self, a, getattr(self.c, a, None))
        self.last_infer = getattr(self.c, "last_infer", 0.0)
        self.last_track_ids, self.last_reasons = [None] * n, {}
