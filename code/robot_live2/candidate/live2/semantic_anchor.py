#!/usr/bin/env python3
"""semantic_anchor.py (R34, 2026-11-01): the SEMANTIC_ANCHORED candidate channel.

An added source of E3 hypotheses, next to the student's E2 search and never in place of it. A reliable room reading, matched exactly to a room on a floor, gives a
place where the plate must be: somewhere on that room's corridor wall. Its position in odometry is the E6 target. What is unknown is the rotation (phi), and only
LiDAR can tell: phi is scanned, each transform is scored with E7's own wall cost (e7_rematch.cost), the best 1-3 separated transforms become hypotheses.

Rules:
  * PROVISIONAL ONLY. One reading never gives a FIX. A fix needs the E8 pair rule + E7 with its rivals, exactly as for E2 candidates.
  * The room centre is not assumed to be the plate: the position is scanned along the wall. Depth is not assumed true: the scale is scanned too.
  * beta (LiDAR yaw) and px/m come from E2 and the floor scale file as PRIORS (estimates, not ground truth) and are scanned around their values.
  * Floors are never removed: the reading only adds candidates. Other floors' hypotheses stay in E3.
  * Model status UNVERIFIED_ANCHOR_MODEL until it has been reproduced in the field: such a candidate is tagged and can only be displayed (shadow) unless
    LIVE2_ANCHOR_E3=1 lets it enter E3. See SEMANTIC_ANCHOR_408_TOILETS.md for the offline numbers on 11-01 and 10-31.
2026-11-18 (user decision "경로까지 보고 만드는걸로"):
  * WALL CHECK while choosing (accept): a transform whose path goes through walls is not chosen — the same tests as the floor rule (fused_floor):
    the last PATH_LAST_M of path does not pass through a wall (e3_tracker.wall_pass_any) and the last 90 s of path crosses walls by <= XMAX_M.
    Field 11-12~17: 46 of 110 anchored candidates (2nd/3rd: 45 of 85) were suspended by E3 within 5 s for wall violations. With the check those
    slots get the best transform that does not cross walls. Without accept the scan is unchanged.
  * PATH ANCHOR (e6e7_link, second decisive plate on the same floor): the EARLIER plate is pinned on its room wall and the rotation is found with the LiDAR
    of the whole path between the plates (window from the earlier plate's stop); the newer plate only filters (its room within PLATE2_TOL_M). Rotation from
    the corridor walls, position along the corridor from the plates. Replay 21 runs: 18 FIX, all right floor, start error median 0.98 m (two-point 1.02,
    current 1.11), never worse than the current method on the same runs.
GT is never an input."""
import json
import math
import os
from pathlib import Path

import numpy as np

KIND = "SEMANTIC_ANCHORED"
MODEL_STATUS = "UNVERIFIED_ANCHOR_MODEL"
TOP_K = int(os.environ.get("LIVE2_ANCHOR_TOPK", "3"))
PHI_COARSE_DEG, PHI_FINE_DEG, PHI_SEP_DEG = 3.0, 1.0, 8.0
SCALE_FACTORS = (0.95, 1.0, 1.05)
BETA_OFFSETS_DEG = (-6.0, 0.0, 6.0)
WALL_STEP_M = 0.5
WINDOW_S = 240.0
XMAX_M, PATH_LAST_M, WIN90_S = 0.3, 30.0, 90.0       # wall check = fused_floor XMAX / PASS_PATH_M / E7 window
PLATE2_TOL_M = 3.0                                  # = e6_semantic.PLACE_TOL_M
LAST_STATS = {}


def _edge_points(poly):
    p = np.asarray(poly, float)
    return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]


def plate_wall_points(room, corridors, px_per_m, step_m=WALL_STEP_M):
    """Where this room's plate can be: points along the room edge that faces a corridor, on the wall line midway to the corridor polygon.
    -> list of [x, y] px. Without a corridor polygon: the room's whole boundary (wider, never narrower)."""
    best = None
    for (a, b) in _edge_points(room):
        mid = (a + b) / 2
        for c in corridors:
            for (u, v) in _edge_points(c):
                t = float(np.clip(np.dot(mid - u, v - u) / max(np.dot(v - u, v - u), 1e-9), 0, 1))
                q = u + t * (v - u)
                d = float(np.hypot(*(mid - q)))
                L = float(np.hypot(*(b - a)))
                if L > 1 and (best is None or d < best[0]):
                    best = (d, a, b, (q - mid) / 2)
    edges = [(best[1] + best[3], best[2] + best[3])] if best else _edge_points(room)
    pts = []
    step = step_m * px_per_m
    for a, b in edges:
        n = max(1, int(round(np.hypot(*(b - a)) / step)))
        pts += [a + (b - a) * i / n for i in range(n + 1)]
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def scan(cost, frames, light, target_xy, wall_pts, ppm0, beta0, mirror, floor, top_k=TOP_K, accept=None):
    """Two stages: phi over the circle at the wall centre -> around the best separated phis, position along the wall x scale x beta x fine phi.
    cost(light, rec, frames, d) is e7_rematch.cost. -> ranked transforms, best first, each with its separation from the next distinct phi.
    accept(rec) -> (ok, why) (2026-11-18): when given, every chosen transform must pass it (best passing one per direction; more directions are tried)."""
    mid = wall_pts[len(wall_pts) // 2]

    def rec(phi, pos=mid, ppm=ppm0, beta=beta0):
        return dict(pos=list(pos), anchor=list(target_xy), phi=float(phi), ppm=float(ppm), beta=float(beta), mirror=mirror, floor=floor, light=None)
    coarse = sorted((float(cost(light, rec(ph), frames, (0, 0, 0, 1))[0]), float(ph)) for ph in np.arange(0.0, 360.0, PHI_COARSE_DEG))
    n_dirs = max(top_k + 2, 5) if accept is None else max(2 * top_k + 2, 8)
    picks = []
    for c, ph in coarse:
        if all(abs((ph - q + 180) % 360 - 180) > PHI_SEP_DEG for q in picks):
            picks.append(ph)
        if len(picks) >= n_dirs:
            break
    out, stats = [], dict(directions=len(picks), rejected=0, why=[])
    for ph0 in picks:
        tried = []
        for ph in np.arange(ph0 - 3.0, ph0 + 3.01, PHI_FINE_DEG):
            for pos in wall_pts:
                r = rec(ph % 360, pos)
                tried.append((float(cost(light, r, frames, (0, 0, 0, 1))[0]), r))
        tried.sort(key=lambda x: x[0])
        best = None
        for k_, (c0, r0) in enumerate(tried):
            if accept is None:
                best = (c0, r0); break
            if k_ >= 12:                                    # at most 12 wall checks per direction (best 12 by LiDAR cost)
                break
            ok, why = accept(r0)
            if ok:
                best = (c0, r0); break
            stats["rejected"] += 1
            if len(stats["why"]) < 6:
                stats["why"].append(why)
        if best is None:
            continue
        c0, r0 = best
        for sf in SCALE_FACTORS:
            for db in BETA_OFFSETS_DEG:
                r = dict(r0, ppm=ppm0 * sf, beta=beta0 + math.radians(db))
                c = float(cost(light, r, frames, (0, 0, 0, 1))[0])
                if c < best[0] and (accept is None or accept(r)[0]):
                    best = (c, r)
        out.append(dict(cost=best[0], rec=best[1], wall_checked=accept is not None))
    out.sort(key=lambda x: x["cost"])
    for i, o in enumerate(out):
        nxt = out[i + 1]["cost"] if i + 1 < len(out) else None
        o["next_cost"] = None if nxt is None else round(nxt, 4)
        o["separation_ratio"] = None if nxt is None else round(nxt / max(o["cost"], 1e-3), 2)
    LAST_STATS.clear(); LAST_STATS.update(stats, kept=len(out))
    return out[:top_k], len(coarse)


def project(rec, xy):
    """odom_link point -> floor-plan px (same transform as e3_tracker.pose_in_map, point only)."""
    phi = math.radians(rec["phi"]); c, s_ = math.cos(phi), math.sin(phi)
    dx, dy = float(xy[0]) - rec["anchor"][0], float(xy[1]) - rec["anchor"][1]
    return [rec["pos"][0] + rec["ppm"] * (c * dx - s_ * dy), rec["pos"][1] + rec["ppm"] * (-s_ * dx - c * dy)]


def path_parts(rows, t_end, last_m=PATH_LAST_M, win_s=WIN90_S):
    """store rows (sid, t, x, y, ...) -> (last last_m metres of path, path of the last win_s seconds), odom_link xy, both ending at t_end."""
    rr = [(float(r[2]), float(r[3])) for r in rows if float(r[1]) <= t_end]
    d, i = 0.0, len(rr) - 1
    while i > 0 and d < last_m:
        d += math.hypot(rr[i][0] - rr[i - 1][0], rr[i][1] - rr[i - 1][1]); i -= 1
    win = [(float(r[2]), float(r[3])) for r in rows if t_end - win_s <= float(r[1]) <= t_end]
    return rr[i:], win


def wall_checker(light, last_path, win_path, wall_pass_any, xmax=XMAX_M, extra=None):
    """-> accept(rec) for scan: the last PATH_LAST_M of path must not pass through a wall and the last 90 s of path must cross walls by <= xmax
    (fused_floor's tests). extra(rec) -> (ok, why) is ANDed (path anchor: the newer plate within its room)."""
    lp, wp = np.asarray(last_path, float), np.asarray(win_path, float)

    def accept(rec):
        ppm = float(rec["ppm"])
        if len(lp) > 1 and bool(wall_pass_any(light.wall_tr, np.array([project(rec, q) for q in lp]), ppm)):
            return False, "last path passes through a wall"
        if len(wp) > 1:
            x = float(light.crossing_m(np.array([project(rec, q) for q in wp]), ppm))
            if x > xmax:
                return False, f"90 s path {x:.2f} m through walls"
        if extra is not None:
            return extra(rec)
        return True, None
    return accept


def priors(e2, floor, res_px_per_m):
    """(beta rad, mirror, merit prior, source) from E2's candidates on this floor; None when E2 has not reached it (beta unknown -> not anchored yet)."""
    cs = [c for c in (e2.get("candidates") or []) + (e2.get("candidates_coarse") or []) if int(c.get("floor", -1)) == int(floor) and c.get("beta") is not None]
    if not cs:
        allc = [c for c in (e2.get("candidates") or []) + (e2.get("candidates_coarse") or []) if c.get("beta") is not None]
        if not allc:
            return None
        cs = allc
    ang = np.array([float(c["beta"]) for c in cs])
    beta = float(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang))))
    mir = [int(c.get("mirror", 1)) for c in cs]
    mer = [float(c["merit"]) for c in cs if c.get("merit") is not None]
    return dict(beta=beta, mirror=max(set(mir), key=mir.count), merit=float(np.median(mer)) if mer else 0.3, n=len(cs),
                source="E2_MEDIAN (estimate, not ground truth)", floor_specific=any(int(c.get("floor", -1)) == int(floor) for c in cs))


def candidate(t, room, floor, o, H, light_dir, map_version, session, merit, t_fit, why_rank):
    """A record E3 can take like an E2 candidate (same fields, stage 'anchored'), tagged with its evidence."""
    r = t["rec"]
    return dict(floor=int(floor), px=round(r["pos"][0], 1), py=round(r["pos"][1], 1), phi=float(r["phi"]), phi_deg=round(float(r["phi"]), 2), ppm=round(float(r["ppm"]), 3),
                beta=float(r["beta"]), beta_deg=round(math.degrees(r["beta"]) % 360, 2), mirror=r["mirror"], anchor=[float(v) for v in r["anchor"]], pos=[float(v) for v in r["pos"]],
                merit=float(merit), twoway=None, wall_pass=None, fit_t=float(t_fit), H=H, map_version=map_version, odom_session=session, stage="anchored", pass_id=f"A_{o['observation_id']}",
                provenance=KIND, model_status=MODEL_STATUS, provisional=True, verified=False, status="UNVERIFIED_HYPOTHESIS",
                evidence=dict(observation_id=o["observation_id"], image_sha256=o.get("image_sha256"), room=room, label=o.get("label"), lidar_cost=round(t["cost"], 4),
                              next_cost=t.get("next_cost"), separation_ratio=t.get("separation_ratio"), reprojection_t=float(t_fit), map_version=map_version, rank=why_rank),
                light_dir=light_dir)


def write(e2_dir, doc):
    p = Path(e2_dir) / "semantic_anchor.json"
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, default=str))
    os.replace(tmp, p)
