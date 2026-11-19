"""E5RT2: real neural-crop budget in front of the SigLIP worker.

A region is not one crop: the worker tiles every region into ceil(w/100) x ceil(h/100) crops (legacy RegionSiglip /
SciSiglip.score_sci) and adds at most one 2.5x context crop per region (lazy: only regions with sim >= ctx_min).
The budget therefore counts tiles + one context upper bound per region BEFORE sending, and only whole regions are
sent (a partial region would change its max-over-tiles score).

Regions that do not fit are DEFERRED_BUDGET: typed status, score = the legacy "no score" value -1 for array shape
compatibility, never counted as evidence (legacy decide skips them via no_evidence_keys; SCI export/P2/stage policies
see score_status != FRESH).  When everything fits, the request is byte-identical to the unbudgeted one
(decision-preserving); a budget that binds is a POLICY change and must be evaluated as such.

Fair order when the budget binds: regions deferred in the previous scene (IoU match) first, then cheaper regions
(small candidates), alternating image halves (left/right), then the original wall-region rank.
"""
from __future__ import annotations

import math
import time

import numpy as np

FRESH, CACHED, DEFERRED = "FRESH", "CACHED", "DEFERRED_BUDGET"


def n_tiles(b, tile=100):
    return max(1, int(math.ceil((b[2] - b[0]) / tile))) * max(1, int(math.ceil((b[3] - b[1]) / tile)))


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def plan_budget(regs, max_crops, prev_deferred=(), width=None, tile=100, ctx_per_region=1):
    """Pure. Returns (send_idx, deferred_idx, info). send_idx keeps the original order of regs."""
    cost = [n_tiles(b, tile) + ctx_per_region for b in regs]
    total = sum(cost)
    info = dict(n_regions=len(regs), crops_requested_upper=total, max_crops=max_crops, binding=False)
    if max_crops is None or total <= max_crops:
        info.update(crops_planned_upper=total, tiles_planned=total - ctx_per_region * len(regs))
        return list(range(len(regs))), [], info
    w = width or max([b[2] for b in regs] + [1.0])

    def waited(b):
        return any(_iou(b, d) >= 0.3 for d in prev_deferred)

    left = [i for i, b in enumerate(regs) if (b[0] + b[2]) / 2 < w / 2]
    right = [i for i, b in enumerate(regs) if (b[0] + b[2]) / 2 >= w / 2]
    key = lambda i: (0 if waited(regs[i]) else 1, cost[i], i)  # noqa: E731
    left.sort(key=key)
    right.sort(key=key)
    order, li, ri, turn_left = [], 0, 0, True
    # alternate image halves; a region that already waited (deferred last scene) goes first in either half
    while li < len(left) or ri < len(right):
        lc = left[li] if li < len(left) else None
        rc = right[ri] if ri < len(right) else None
        if lc is None or rc is None:
            pick_left = rc is None
        else:
            kl, kr = key(lc)[0], key(rc)[0]
            pick_left = kl < kr or (kl == kr and turn_left)
        if pick_left:
            order.append(lc)
            li += 1
            turn_left = False
        else:
            order.append(rc)
            ri += 1
            turn_left = True
    send, used = [], 0
    for i in order:
        if used + cost[i] <= max_crops:
            send.append(i)
            used += cost[i]
    send.sort()
    deferred = [i for i in range(len(regs)) if i not in send]
    info.update(binding=True, crops_planned_upper=used, tiles_planned=sum(n_tiles(regs[i], tile) for i in send),
                deferred_waited=[i for i in deferred if waited(regs[i])])
    return send, deferred, info


SEM_FLOOR = 0.078


def semantic_confirmed_rule(sims, n_scenes, thr, floor=SEM_FLOOR, window=3, min_scenes=2):
    """Shared causal rule (simulator == runtime): mean of the last `window` FRESH sims >= thr, last sim >= floor,
    seen in >= min_scenes scenes. This is SEMANTIC confirmation only; it says nothing about a STOP having fired."""
    if not sims or not math.isfinite(thr):
        return False
    return float(np.mean(sims[-window:])) >= thr and sims[-1] >= floor and n_scenes >= min_scenes


def select_value(cands, max_crops, thr, reserve=1):
    """FOLLOW-UP: the ONE value-selection function used by the stored-score simulator and by the runtime wrapper.
    cands: list of dict(idx, cost, new, sims, waited_s, semantic_confirmed, observed). Causal features only.
    Order: (0) existing spots that still need semantic confirmation (not semantic_confirmed, not observed), nearest to
    the threshold first, then longest waiting, then cheaper; (1) the rest. Exploration reserve: the first `reserve`
    NEW spots (in cost order) are served before that list. Returns (send_idx_sorted, deferred={idx: reason}, used)."""
    def near(c):
        return abs(float(np.mean(c["sims"][-3:])) - thr) if (c["sims"] and math.isfinite(thr)) else math.inf

    def key(c):
        need = 0 if (not c["new"] and not c["semantic_confirmed"] and not c["observed"] and math.isfinite(thr)) else 1
        return (need, near(c), -float(c["waited_s"] or 0.0), c["cost"], c["idx"])
    order = sorted(cands, key=key)
    send, used = [], 0
    for c in sorted((c for c in cands if c["new"]), key=lambda c: (c["cost"], c["idx"]))[:reserve]:
        if used + c["cost"] <= max_crops:
            send.append(c["idx"])
            used += c["cost"]
    for c in order:
        if c["idx"] not in send and used + c["cost"] <= max_crops:
            send.append(c["idx"])
            used += c["cost"]
    deferred = {c["idx"]: "DEFERRED_BUDGET:value" for c in cands if c["idx"] not in send}
    return sorted(send), deferred, used


def value_plan(regs, max_crops, spot_info, thr, reserve=1, tile=100):
    """Runtime adapter: builds the shared candidate list from spot_info[i] (None | dict(new, sims, waited_s,
    semantic_confirmed, observed)) and calls select_value. Non-binding budgets send everything (decision-preserving)."""
    cost = [n_tiles(b, tile) + 1 for b in regs]
    total = sum(cost)
    info = dict(n_regions=len(regs), crops_requested_upper=total, max_crops=max_crops, binding=False, policy="value")
    if max_crops is None or total <= max_crops:
        info.update(crops_planned_upper=total, tiles_planned=total - len(regs))
        return list(range(len(regs))), [], info
    cands = [dict(idx=i, cost=cost[i], new=bool((s or {}).get("new", True)), sims=list((s or {}).get("sims") or []),
                  waited_s=(s or {}).get("waited_s") or 0.0, semantic_confirmed=bool((s or {}).get("semantic_confirmed")),
                  observed=bool((s or {}).get("observed"))) for i, s in enumerate(spot_info)]
    send, dmap, used = select_value(cands, max_crops, thr, reserve=reserve)
    deferred = sorted(dmap)
    info.update(binding=True, crops_planned_upper=used, tiles_planned=sum(n_tiles(regs[i], tile) for i in send),
                deferred_reasons={str(k): v for k, v in dmap.items()})
    return send, deferred, info


class BudgetedSiglip:
    """Wraps the SigLIP client (or the opt-in cache wrapper). Same score(bgr, regs) API + typed per-region status."""

    def __init__(self, client, legacy_events=None, max_crops=None, ms_per_crop=15.0, tile=100, policy="fair"):
        self.c, self.L = client, legacy_events
        self.max_crops = max_crops
        self.policy = policy                    # SEM1: "fair" (E5RT2 default) | "value" (needs set_context each scene)
        self._ctx = None
        self.spot_last_fresh = {}
        self.semantic_confirmed = {}            # spot -> mono of first causal semantic confirmation (not a STOP)
        self.ms_per_crop, self.tile = float(ms_per_crop), tile
        self.prev_deferred = []
        self.last_score_status, self.last_budget = [], None
        self.counters = dict(requests=0, binding=0, deferred=0, tiles_mismatch=0)
        self.tele = []

    def __getattr__(self, k):
        return getattr(self.c, k)

    def set_context(self, **ctx):
        """Frontend.scene passes depth / vanishing point / pose / legacy state right before rerank (value policy)."""
        self._ctx = ctx

    def _spot_info(self, regs):
        """Map each region to the legacy world spot it would join (same rule as legacy decide: region_world + match)."""
        ctx, L = self._ctx or {}, self.L
        if L is None or not ctx.get("region_world") or not ctx.get("match"):
            return [None] * len(regs), math.inf
        seen = list(getattr(L, "seen", []) or [])
        P = ctx.get("P") or {}
        thr = float(np.quantile(seen, P.get("quant", 0.9))) if len(seen) >= P.get("warm", 50) else math.inf
        out, now = [], time.monotonic()
        for b in regs:
            try:
                xyz, _ = ctx["region_world"](b, ctx.get("dimg"), ctx.get("vp"), ctx.get("pose"))
            except Exception:
                xyz = None
            if xyz is None:
                out.append(None)
                continue
            a = ctx["match"](L.spots, xyz, P.get("radius", 0.7), None, set()) if L.spots else None
            if a is None or a >= len(L.spots):
                out.append(dict(new=True))
                continue
            sp = L.spots[a]
            sims = list(sp.get("sims") or [])
            if self.semantic_confirmed.get(a) is None and semantic_confirmed_rule(sims, len(set(sp.get("ts") or [])), thr):
                self.semantic_confirmed[a] = now                    # latched on causal (already scored) evidence
            out.append(dict(new=False, sims=sims, waited_s=now - self.spot_last_fresh.get(a, now),
                            semantic_confirmed=a in self.semantic_confirmed,
                            observed=a in getattr(L, "fired", set()), spot=a))   # fired = STOP side, kept separate
        return out, thr

    def score(self, bgr, regs):
        t0 = time.perf_counter()
        regs = [list(map(float, b)) for b in regs]
        if self.L is not None:
            self.L.no_evidence_keys = set()           # reset every scene (the inner cache may set its own keys)
        sinfo = None
        if self.policy == "value":
            sinfo, thr = self._spot_info(regs)
            send, deferred, info = value_plan(regs, self.max_crops, sinfo, thr, tile=self.tile)
        else:
            send, deferred, info = plan_budget(regs, self.max_crops, self.prev_deferred, width=bgr.shape[1], tile=self.tile)
        self.counters["requests"] += 1
        if not deferred:
            sim, sec = self.c.score(bgr, regs)
            inner = list(getattr(self.c, "last_score_status", None) or [FRESH] * len(regs))
            self.last_score_status = inner
        else:
            self.counters["binding"] += 1
            self.counters["deferred"] += len(deferred)
            sub = [regs[i] for i in send]
            s_sub, sec = self.c.score(bgr, sub) if sub else (np.zeros(0), 0.0)
            inner = list(getattr(self.c, "last_score_status", None) or [FRESH] * len(sub))
            n = len(regs)
            sim = np.full(n, -1.0)
            st = [DEFERRED] * n

            def scatter(name, fill):
                v = getattr(self.c, name, None)
                if v is None:
                    return None
                out = [fill] * n
                for k, i in enumerate(send):
                    if k < len(v):
                        out[i] = v[k]
                return out

            for k, i in enumerate(send):
                sim[i] = float(s_sub[k])
                st[i] = inner[k] if k < len(inner) else FRESH
            # the legacy rerank/Frontend.export read these attributes by region index
            self._last = dict(last_adv=np.array(scatter("last_adv", 0.0) or [0.0] * n),
                              last_board=np.array(scatter("last_board", -1.0) or [-1.0] * n),
                              last_ceil=np.array(scatter("last_ceil", -1.0) or [-1.0] * n),
                              last_tiles=scatter("last_tiles", None) or [None] * n,
                              last_emb=scatter("last_emb", None) or [None] * n,
                              last_ctx_plate=scatter("last_ctx_plate", None) or [None] * n,
                              last_neg={c: scatter_list(v, send, n) for c, v in (getattr(self.c, "last_neg", {}) or {}).items()},
                              last_ctx={c: scatter_list(v, send, n) for c, v in (getattr(self.c, "last_ctx", {}) or {}).items()})
            self.last_score_status = st
        if deferred and self.L is not None:
            keys = set(getattr(self.L, "no_evidence_keys", set()) or set())
            keys |= {tuple(float(v) for v in regs[i]) for i in deferred}
            self.L.no_evidence_keys = keys
        elif self.L is not None and not hasattr(self.L, "no_evidence_keys"):
            self.L.no_evidence_keys = set()
        if not deferred:
            self._last = None
        nt = getattr(self.c, "last_n_tiles", None)
        if nt is not None and int(nt) != int(info.get("tiles_planned", -1)) and not any(s == CACHED for s in self.last_score_status):
            self.counters["tiles_mismatch"] += 1
            info["tiles_mismatch"] = dict(planned=info.get("tiles_planned"), worker=int(nt))
        info.update(worker_tiles=nt, worker_ctx=getattr(self.c, "last_n_ctx", None), deferred=deferred,
                    est_gpu_ms_upper=info.get("crops_planned_upper", 0) * self.ms_per_crop, host_ms=(time.perf_counter() - t0) * 1e3)
        if sinfo is not None:                                       # FOLLOW-UP: last_fresh only on a FRESH result
            t_now = time.monotonic()
            for i, stt in enumerate(self.last_score_status):
                if stt == FRESH and sinfo[i] and sinfo[i].get("spot") is not None:
                    self.spot_last_fresh[sinfo[i]["spot"]] = t_now
        self.last_budget = info
        self.prev_deferred = [regs[i] for i in deferred]
        self.tele = (self.tele + [info])[-300:]
        return sim, sec

    def __getattribute__(self, k):
        if k.startswith("last_") and k not in ("last_score_status", "last_budget"):
            last = object.__getattribute__(self, "__dict__").get("_last")
            if last is not None and k in last:
                return last[k]
        return object.__getattribute__(self, k)


def scatter_list(v, send, n):
    out = [None] * n
    for k, i in enumerate(send):
        if k < len(v):
            out[i] = v[k]
    return out
