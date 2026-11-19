"""e5_e4_link.py — LIVE2 E4 -> E5 link (installed by observe_node when LIVE2_E4_FEED is set). SigLIP / FAST-LIO untouched.

What it changes, and only this:
  * e5rt.budget.plan_budget: when the SigLIP crop budget BINDS and E4 asks for semantic observation (priority HIGH or
    MEDIUM, fresh <= 3 s), regions whose horizontal bearing is within BEAR_TOL_DEG of an E4 map target that is
    OBSERVABLE_NOW are served first (then the original fair order). When the budget does not bind every region is sent,
    exactly as before (decision preserving). The SigLIP scene interval / GPU guards (1 s, 2 s degraded) are NOT changed.
  * every scene records what happened in LINK.last (exposed in E5 status as 'live2_e4_link'): E4 state / priority,
    targets, per-region bearing, which regions matched, whether the budget bound and how many were promoted.
Bearing of a region = atan((cx - u) / fx) (left positive) from the hub camera_info K; without K it is not computed
(BEARING_UNAVAILABLE) — nothing is guessed. Pan-tilt is never commanded.
"""
import json
import math
import time
from pathlib import Path

BEAR_TOL_DEG, STALE_S = 15.0, 3.0


class Link:
    def __init__(self, e4_path, hub_dir=None):
        self.e4_path = Path(e4_path)
        self.hub_dir = Path(hub_dir) if hub_dir else None
        self.K = None
        self.last = dict(installed=True, scenes=0, promoted_total=0, binding_scenes=0)

    def e4(self):
        try:
            d = json.loads(self.e4_path.read_text())
        except (OSError, ValueError):
            return None
        if time.time() - float(d.get("stamp", 0)) > STALE_S:
            return dict(d, stale=True)
        return d

    def fx_cx(self):
        if self.K is None and self.hub_dir is not None:
            try:
                ci = json.loads((self.hub_dir / "index.json").read_text()).get("caminfo") or {}
                K = ci.get("K") or ci.get("k")
                if K and len(K) >= 3:
                    self.K = (float(K[0]), float(K[2]))
            except (OSError, ValueError, TypeError):
                pass
        return self.K

    def bearing(self, b):
        k = self.fx_cx()
        if not k:
            return None
        u = 0.5 * (b[0] + b[2])
        return math.degrees(math.atan2(k[1] - u, k[0]))

    def priority_regions(self, regs):
        d = self.e4()
        info = dict(e4_state=None if d is None else d.get("level"), priority=None if d is None else d.get("priority"),
                    stale=None if d is None else d.get("stale", False), targets=[], matched=[], bearings=[])
        if d is None or d.get("stale") or d.get("priority") not in ("HIGH", "MEDIUM"):
            info["why"] = "E4_UNAVAILABLE_OR_STALE" if (d is None or d.get("stale")) else "E4_PRIORITY_" + str(d.get("priority"))
            return set(), info
        tg = [t for t in d.get("targets") or [] if t.get("observability") == "OBSERVABLE_NOW"]
        info["targets"] = [dict(label=t["label"], floor=t["floor"], bearing_deg=t["bearing_deg"], dist_m=t["dist_m"]) for t in tg]
        out = set()
        for i, b in enumerate(regs):
            br = self.bearing(b)
            info["bearings"].append(None if br is None else round(br, 1))
            if br is None:
                continue
            if any(abs(br - t["bearing_deg"]) <= BEAR_TOL_DEG for t in tg):
                out.add(i)
        info["matched"] = sorted(out)
        if self.fx_cx() is None:
            info["why"] = "BEARING_UNAVAILABLE (no camera_info K in hub index)"
        return out, info


LINK = None


def install(budget_module, e4_path, hub_dir=None):
    """Wrap plan_budget in the given e5rt.budget module (in-process only; files on disk unchanged)."""
    global LINK
    LINK = Link(e4_path, hub_dir)
    orig = budget_module.plan_budget

    def plan_budget(regs, max_crops, prev_deferred=(), width=None, tile=100, ctx_per_region=1):
        prio, info = LINK.priority_regions(regs)
        send, deferred, binfo = orig(regs, max_crops, prev_deferred, width=width, tile=tile, ctx_per_region=ctx_per_region)
        LINK.last["scenes"] += 1
        promoted = []
        if binfo.get("binding") and prio:
            LINK.last["binding_scenes"] += 1
            cost = [budget_module.n_tiles(b, tile) + ctx_per_region for b in regs]
            first = [i for i in sorted(prio, key=lambda i: cost[i])]
            rest = [i for i in send if i not in prio] + [i for i in deferred if i not in prio]   # original fair order kept
            new_send, used = [], 0
            for i in first + rest:
                if used + cost[i] <= max_crops:
                    new_send.append(i); used += cost[i]
            promoted = [i for i in new_send if i in prio and i not in send]
            send = sorted(new_send)
            deferred = [i for i in range(len(regs)) if i not in send]
            binfo = dict(binfo, crops_planned_upper=used, tiles_planned=sum(budget_module.n_tiles(regs[i], tile) for i in send),
                         e4_promoted=promoted)
        LINK.last["promoted_total"] += len(promoted)
        LINK.last.update(last_scene=dict(info, binding=bool(binfo.get("binding")), promoted=promoted, n_regions=len(regs),
                                         wall=time.time()))
        return send, deferred, binfo

    budget_module.plan_budget = plan_budget
    return LINK
