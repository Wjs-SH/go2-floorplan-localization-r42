"""e8_pair.py — LIVE2 E8 dual semantic observation (runs inside e34; owned by e6e7_link when policy E8_PAIR).

Operator order (2026-10-22 directive, reading changed 2026-10-26): OBS1 = E5 STOP -> stop -> E6 aim / (manual focus by the
operator, no DC motor) / IMX477 capture -> HOME, image STORED -> two-tone "drive on" -> its OCR (VLM only where OCR is
insufficient and the GPU gate admits it) runs in the background at once -> PROVISIONAL place (never a fix) -> joystick ->
E5/SigLIP finds the next plate -> STOP -> OBS2 captured + read -> semantic map + odometry consistency of the pair -> region
candidate -> E7 (independent LiDAR) -> FIX / NO_FIX / MORE_OBSERVATION_REQUIRED / CANDIDATE_MISSING. A single observation never
gives a global fix (read_each=False restores the old "read both after the second" order). Nothing here commands the robot or the pan-tilt; the bridge (e6_precision_bridge --defer-reading) only
stores captures.

States (pair group; n = observation number in the group):
  PAIR_IDLE, OBS{n}_STOP_CONFIRMED, OBS{n}_E6_RUNNING, OBS{n}_CAPTURE_STORED, OBS{n}_WAITING_RESUME, OBS{n+1}_SEARCHING,
  PAIR_READING, PAIR_CONSISTENCY_CHECK, REGION_CANDIDATE_READY, E7_VERIFYING, GLOBAL_FIX,
  MORE_OBSERVATION_REQUIRED (group stays open: next observation re-evaluates EVERY observation, no majority vote),
  CANDIDATE_MISSING, NO_FIX (MAX_OBS reached without a fix), HOLD_ERROR (session change / device block: no drive-on).
Records: <state>/e8_pair_state.json (snapshot), e8_observations.jsonl (one line per observation event, never overwritten),
e8_pairs.jsonl (one line per group evaluation / E7 result). GT is never read here.

Consistency (per candidate, OFFLINE-testable, numbers below are stated assumptions, not measured on this robot):
  * reading: student fuse (OCR first; VLM only when OCR is insufficient; VLM-only number = no label), decisive p >= E3 0.6
  * same plate twice: targets closer than INDEP_MIN_M with the same label -> not independent evidence
  * per E3 hypothesis: every labelled observation SEMANTIC_CONSISTENT under the SAME odom->map transform (e6_semantic; 2026-11-19 half-corridor place rule, LIVE2_PLATE_HALF_CORRIDOR=0 -> room polygon 3 m,
    PLACE_TOL_M 3 m each) — one rigid transform = the odometry-relative displacement and heading between the plates
  * hypothesis-free: |target_j - target_i| (FAST-LIO odom_link, plate to plate — NOT room centre to room centre, NOT robot
    displacement) inside [min, max] distance between the two labelled room polygons +- (2 x PLACE_TOL_M + odometry error)
  * corridor topology: both rooms within ADJ_M of corridor polygons of one connected corridor set (map 'corridors');
    missing corridor data -> TOPOLOGY_UNKNOWN (reported, not invented)
  * odometry: same FAST-LIO session, no sample < 8 Hz between the captures (same line as the E7 FIX block)
  * yaw: the map has no plate facing -> no independent yaw residual (DATA_LIMIT, reported); the heading between the
    observations is constrained only through the single rigid transform per hypothesis
"""
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

INDEP_MIN_M = 2.0          # E6 target error: re-detection offset 29 px (max 67 px) at 3.5-5 m ~ 0.2-0.5 m, plate ~0.3 m wide
ODOM_REL_ERR = 0.05        # FAST-LIO relative drift per metre of path (assumption; N1 drifted 0.84 -> 5.2 m over a long run)
ODOM_ERR_FLOOR_M = 0.5
ADJ_M = 2.0                # room polygon to corridor polygon gap still counted as adjacent (door / wall thickness)
CORRIDOR_LINK_M = 1.0
REGION_CLUSTER_M = 5.0     # 2026-11-18: no longer gates the verdict when PATH_RULE is on (kept only as a recorded diagnostic)
# 2026-11-18 (user decision, R42 replay E8_fig11): a both-plate-consistent hypothesis is dropped when its transform puts (a) the path between the
# first and last plate photo or (b) the last PATH_RULE_LAST_M of path through a wall (e3_tracker.wall_pass_any, the E3 rule). If the remaining
# hypotheses are all on one floor -> REGION_CANDIDATE_READY (E7 picks the place). Replay of 41 recorded pair checks: best candidate never dropped,
# 8/8 ambiguous checks reach E7; 45 runs: same floors. The whole path from the start is NOT used (it dropped right candidates 5 times).
# Off: LIVE2_E8_PATH_RULE=0 restores the 5 m region rule exactly.
PATH_RULE = os.environ.get("LIVE2_E8_PATH_RULE", "1") == "1"
PATH_RULE_LAST_M = 30.0
# 2026-11-18 B1 (user decision): a path-infeasible candidate is ranked LAST (E8 candidate order and E7 choice), not removed. The two-plate transform is the one
# before E7's correction, so a right place can look like it crosses a wall (replay 11-15 20:20: the right 4F candidate left for 13 s). The verdict uses the
# feasible ones (all on one floor -> READY); with none feasible and every candidate on one floor E7 still looks, and fused_floor's check after E7's correction decides.
PATH_RANK = os.environ.get("LIVE2_E8_PATH_RANK", "1") == "1"
# 2026-11-18 B6 (user decision): one misread plate (another floor's number) no longer blocks the whole group until the budget ends. When NO floor explains every
# plate-to-plate pair, each plate is left out in turn; if exactly one leave-one-out set is explained by a floor, that plate is set aside (recorded, not deleted)
# and the rest are judged by the same rules. Needs >= DROP_MISREAD_MIN valid plates: with two, which one is wrong cannot be told.
DROP_MISREAD = os.environ.get("LIVE2_E8_DROP_MISREAD", "1") == "1"
DROP_MISREAD_MIN = 3


def path_feasible(rec, wall, rows, t_a, t_b, last_m=PATH_RULE_LAST_M):
    """(ok, why) for ONE hypothesis transform. rows = store rows (sid, t, x, y, yaw) in the odom_link frame; t_a/t_b = first/last plate stop time."""
    import e3_tracker as E3
    import e6_core as G
    if wall is None or not rows:
        return True, None
    proj = lambda rr: np.array([G.map_projection(rec, r[2], r[3]) for r in rr]) if rr else np.zeros((0, 2))
    seg = [r for r in rows if t_a <= r[1] <= t_b]
    if len(seg) > 1 and E3.wall_pass_any(wall, proj(seg), float(rec["ppm"])):
        return False, "path between the plates crosses a wall"
    d, i = 0.0, len(rows) - 1
    while i > 0 and d < last_m:
        d += math.hypot(rows[i][2] - rows[i - 1][2], rows[i][3] - rows[i - 1][3]); i -= 1
    last = rows[i:]
    if len(last) > 1 and E3.wall_pass_any(wall, proj(last), float(rec["ppm"])):
        return False, f"last {last_m:.0f} m of path crosses a wall"
    return True, None
LOW_RATE_HZ = 8.0
MAX_OBS = 4                # R34: valid independent observations (a failed read does not use one up)
MAX_ATTEMPTS = 8           # stored captures in one group, readable or not
GROUP_BUDGET_S = 900.0     # one group's time budget: after this a NO_FIX ends it even with a free attempt
READ_TIMEOUT_S = 150.0
WAIT_STATES = ("WAITING_RESUME", "SEARCHING")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _atomic(path, obj):
    tmp = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.{threading.get_ident()}.tmp")   # reader thread writes too
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str))
    os.replace(tmp, path)


# ------------------------------------------------------------------ map helpers (read-only semantic maps)
SCALE_FILE = Path(__file__).resolve().with_name("semantic_scale_v1.json")
try:
    SCALE_OVERRIDE = (json.loads(SCALE_FILE.read_text()).get("floors") or {})
except (OSError, ValueError):
    SCALE_OVERRIDE = {}


def floor_res(floor, floor_map):
    """-> (m per px, override or None). 2026-10-30: the 2F-5F semantic map_meta scale is wrong (4F -28 %): use the separate,
    versioned per-floor estimate (semantic_scale_v1.json, ESTIMATED_CORE_STACK) instead; the map file itself is untouched."""
    res = float((floor_map.get("map_meta") or {}).get("resolution_m_per_px") or 0.0) or None
    ov = SCALE_OVERRIDE.get(str(floor))
    # bound to the exact map it corrects: only when this map's own px/m equals the recorded wrong value (+-0.5 %)
    if ov and ov.get("px_per_m") and res and abs(1.0 / res - float(ov["semantic_map_px_per_m"])) <= 0.005 * float(ov["semantic_map_px_per_m"]):
        return 1.0 / float(ov["px_per_m"]), ov
    return res, None


def _poly_m(space, res, force_px=False):
    if space.get("polygon_map_m") and not force_px:
        return np.asarray(space["polygon_map_m"], float)
    return np.asarray(space["polygon_px"], float) * res


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return float(np.hypot(*(a + t * ab - p)))


def poly_interval(P, Q):
    """[min, max] distance between two room polygons (m). min = 0 when they touch/overlap."""
    dmax = max(float(np.hypot(*(p - q))) for p in P for q in Q)
    dmin = min(min(_seg_dist(p, a, b) for a, b in zip(Q, np.roll(Q, -1, 0))) for p in P)
    dmin = min(dmin, min(_seg_dist(q, a, b) for q in Q for a, b in zip(P, np.roll(P, -1, 0))))
    return dmin, dmax


def corridor_components(corridors, res, force_px=False):
    polys = [_poly_m(c, res, force_px) for c in corridors if (c.get("polygon_px") if force_px else (c.get("polygon_px") or c.get("polygon_map_m")))]
    comp = list(range(len(polys)))

    def find(i):
        while comp[i] != i:
            i = comp[i]
        return i
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if poly_interval(polys[i], polys[j])[0] <= CORRIDOR_LINK_M:
                comp[find(i)] = find(j)
    return polys, [find(i) for i in range(len(polys))]


def topology(floor_map, room_i, room_j, floor=None):
    """-> (status, detail). CONNECTED / NOT_CONNECTED / TOPOLOGY_UNKNOWN (no corridor polygons in the map)."""
    res, ov = floor_res(floor, floor_map)
    cors = floor_map.get("corridors") or []
    if not cors or res is None:
        return "TOPOLOGY_UNKNOWN", "semantic map has no corridor polygons / resolution"
    polys, comp = corridor_components(cors, res, force_px=bool(ov))
    Pi, Pj = _poly_m(room_i, res, bool(ov)), _poly_m(room_j, res, bool(ov))
    ci = {comp[k] for k, c in enumerate(polys) if poly_interval(Pi, c)[0] <= ADJ_M}
    cj = {comp[k] for k, c in enumerate(polys) if poly_interval(Pj, c)[0] <= ADJ_M}
    if not ci or not cj:
        return "TOPOLOGY_UNKNOWN", f"room not adjacent to any corridor polygon ({room_i.get('id')}: {bool(ci)}, {room_j.get('id')}: {bool(cj)})"
    return ("CONNECTED", f"corridor set {sorted(ci & cj)}") if ci & cj else ("NOT_CONNECTED", "rooms on separate corridor sets")


# ------------------------------------------------------------------ evaluation (pure)
def reading_of(o):
    """Observation reading -> (alternatives, fusion info, decisive, label_source)."""
    import e3_tracker as E3
    import e6_semantic as S
    alts, info = S.label_alternatives([tuple(x) for x in o.get("ocr_lines") or []], o.get("vlm"), with_category=True)
    fu = dict(info.get("fusion") or {})
    p_max = max((p for _, p in alts), default=0.0)
    import plate_text as PT
    room = None if p_max >= E3.SEM_DECISIVE_P else PT.vlm_only_room(o)
    if room:                                       # R40: not a digit alone: number + department of one sharp crop
        return [(room, max(float(E3.SEM_DECISIVE_P), 0.8))], dict(fu, status="VLM_TEXT_CONFIRMED"), True, "VLM_TEXT_CONFIRMED"
    if info.get("category"):
        fu["category"] = info["category"]
        return alts, fu, p_max >= E3.SEM_DECISIVE_P, "CATEGORY_" + info["category"]["category"].upper()
    return alts, fu, bool(alts) and p_max >= E3.SEM_DECISIVE_P, fu.get("status")


def is_category_reading(alts):
    return bool(alts) and all(str(a[0]).startswith("cat:") for a in alts)


def path_segments(xy, turn_deg=45.0, min_len_m=2.0):
    """Straight legs of an odometry path (a new leg when the direction over >= min_len_m turns by > turn_deg)."""
    legs, cur, anchor, last_dir = [], 0.0, None, None
    for p, q in zip(xy, xy[1:]):
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        cur += d
        if anchor is None:
            anchor = p
        v = math.hypot(q[0] - anchor[0], q[1] - anchor[1])
        if v >= min_len_m:
            ang = math.atan2(q[1] - anchor[1], q[0] - anchor[0])
            if last_dir is not None and abs((ang - last_dir + math.pi) % (2 * math.pi) - math.pi) > math.radians(turn_deg):
                legs.append(cur)
                cur = 0.0
            last_dir, anchor = ang, q
    if cur > 0:
        legs.append(cur)
    return legs


def segment_odom_err(xy, net):
    """R34: odometry error between two stops = floor + rel x sqrt(sum of squared leg lengths). The errors of separate straight legs do not simply add up, so a 79 m walk
    there and back no longer gives a 4.5 m tolerance. Still a stated assumption, not a measured drift model."""
    legs = path_segments(xy) if len(xy) >= 2 else []
    eff = math.sqrt(sum(L * L for L in legs)) if legs else net
    return ODOM_ERR_FLOOR_M + ODOM_REL_ERR * max(eff, net), legs


def odom_between(o1, o2, rows, lio_series):
    """Robot displacement between two stops (FAST-LIO raw frame of the stop records) + odometry health in between."""
    a, b = o1.get("odom_pose_at_stop") or {}, o2.get("odom_pose_at_stop") or {}
    out = dict(session_same=o1.get("session") == o2.get("session"), session_1=o1.get("session"), session_2=o2.get("session"))
    if a.get("x") is None or b.get("x") is None:
        return dict(out, ok=False, why="stop pose missing")
    dx, dy = float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])
    dyaw = math.degrees((float(b["yaw"]) - float(a["yaw"]) + math.pi) % (2 * math.pi) - math.pi)
    t1, t2 = float(a.get("t") or 0), float(b.get("t") or 0)
    xy = [(r[2], r[3]) for r in rows or [] if t1 <= r[1] <= t2]
    path = float(sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(xy, xy[1:]))) if len(xy) >= 2 else None
    w1, w2 = float(o1.get("stopped_wall") or 0), float(o2.get("stopped_wall") or 0)
    low = [s for s in lio_series or [] if w1 <= float(s.get("wall", 0)) <= w2 and (s.get("hz") or 99) < LOW_RATE_HZ]
    net = math.hypot(dx, dy)
    err, legs = segment_odom_err(xy, net)
    ok, why = True, None
    if not out["session_same"]:
        ok, why = False, f"ODOM_SESSION_CHANGED ({o1.get('session')} -> {o2.get('session')})"
    elif low:
        ok, why = False, f"FAST-LIO below {LOW_RATE_HZ:.0f} Hz in {len(low)} samples between the observations"
    return dict(out, ok=ok, why=why, delta_x_m=round(dx, 3), delta_y_m=round(dy, 3), delta_yaw_deg=round(dyaw, 1),
                net_displacement_m=round(net, 3), path_length_m=None if path is None else round(path, 3),
                path_samples=len(xy), elapsed_s=round(w2 - w1, 1), low_rate_samples=len(low),
                odom_rel_err_m=round(err, 2), odom_legs_m=[round(x, 1) for x in legs],
                odom_err_model=f"{ODOM_ERR_FLOOR_M} m + {ODOM_REL_ERR:.0%} of sqrt(sum leg^2) (ASSUMPTION)")


def _target(o):
    t = o.get("target_odom_link_xy")
    return None if t is None else np.asarray(t, float)


def low_quality_override(o):
    q = o.get("image_quality") or {}
    return bool(q.get("operator_override_low_sharpness")) or bool({"LOW_SHARPNESS", "TARGET_ROI_NOT_DETECTED"} & set(q.get("flags") or []))


def operator_roi(o):
    q = o.get("image_quality") or {}
    return q.get("roi") if q.get("roi_source") == "OPERATOR_SELECTED" and q.get("roi") else None


def _pairs(ev, sf, rows=None, lio_series=None):
    """Consecutive plate pairs of an evidence list -> pair records (moved unchanged out of evaluate_group, 2026-11-18, so B6 can judge a subset)."""
    import e6_semantic as S
    import e8_map_data as MD
    pairs = []
    for (o1, r1, a1, s1), (o2, r2, a2, s2) in zip(ev, ev[1:]):
        od = odom_between(o1, o2, rows, lio_series)
        t1, t2 = _target(o1), _target(o2)
        sep = float(np.hypot(*(t2 - t1)))
        pr = dict(observation_ids=[o1["observation_id"], o2["observation_id"]], odometry=od, target_separation_m=round(sep, 3),
                  labels=[[a[0] for a in a1], [a[0] for a in a2]], independent=True, floors={})
        for f, m in (getattr(sf, "maps", None) or {}).items():
            res, ov = floor_res(f, m)
            best = None
            for l1 in {a[0] for a in a1}:
                for l2 in {a[0] for a in a2}:
                    for sp1 in sf.spaces(f, l1):
                        for sp2 in sf.spaces(f, l2):
                            if not res:
                                continue
                            dmin, dmax = poly_interval(_poly_m(sp1, res, bool(ov)), _poly_m(sp2, res, bool(ov)))
                            tol = 2 * S.PLACE_TOL_M + od.get("odom_rel_err_m", ODOM_ERR_FLOOR_M) \
                                + (float(ov.get("rel_unc") or 0.0) * dmax if ov else 0.0)   # estimated (unsurveyed) floor scale
                            resid = max(0.0, dmin - sep - tol, sep - dmax - tol)
                            topo, tdet = topology(m, sp1, sp2, floor=f)
                            c = dict(rooms=[sp1.get("id"), sp2.get("id")], map_interval_m=[round(dmin, 2), round(dmax, 2)],
                                     tolerance_m=round(tol, 2), translation_residual_m=round(resid, 2),
                                     metric_ok=resid == 0.0, topology=topo, topology_detail=tdet,
                                     scale=dict(px_per_m=round(1.0 / res, 2), source=(ov or {}).get("status") or "SEMANTIC_MAP_META"))
                            wi = MD.wall_interval(m, sp1, sp2)          # derived corridor-wall interval: diagnostic only
                            if wi:
                                w0, w1 = wi["interval_m"]
                                wt = 2 * wi["uncertainty_m"] + od.get("odom_rel_err_m", ODOM_ERR_FLOOR_M)
                                c.update(derived_wall_interval_m=wi["interval_m"], derived_provenance=wi["provenance"],
                                         derived_tolerance_m=round(wt, 2), derived_verified=False,
                                         derived_residual_m=round(max(0.0, w0 - sep - wt, sep - w1 - wt), 2))
                            if best is None or (c["metric_ok"], -resid) > (best["metric_ok"], -best["translation_residual_m"]):
                                best = c
            if best:
                pr["floors"][f] = best
        r2_ok = od["ok"]
        pr["explains"] = sorted(f for f, c in pr["floors"].items() if c["metric_ok"] and c["topology"] != "NOT_CONNECTED")
        pr["usable"] = r2_ok
        pairs.append(pr)
    return pairs


def evaluate_group(obs, hyps, sf, rows=None, lio_series=None, light_of=None, excluded=None):
    """obs: observation dicts of ONE group, oldest first, each read (ocr_lines / vlm) — invalid ones are kept and explained.
    hyps: E3 alive [{'id','floor','rec'}]. sf: e6_semantic.SemanticFloors. -> evaluation dict (verdict + every number)."""
    import e6_semantic as S
    import e8_map_data as MD
    per_obs, usable = [], []
    for o in obs:
        alts, fu, decisive, src = reading_of(o)
        sem = S.evaluate(alts, hyps, sf, list(o["target_odom_link_xy"]), half_corridor=S.HALF_CORRIDOR) if alts and o.get("target_odom_link_xy") else \
            dict(alternatives=alts, per_hypothesis={})
        rec = dict(observation_id=o["observation_id"], capture_id=o.get("capture_id"), labels=alts, fusion_status=src, category=fu.get("category"),
                   reading_kind="CATEGORY" if is_category_reading(alts) else "ROOM" if alts else "NONE",
                   decisive=decisive, floor_likelihood=sem.get("floor_likelihood"),
                   per_hypothesis={k: dict({kk: v.get(kk) for kk in ("verdict", "dist_m", "map_room_id", "label", "target_px", "floor")},
                                           **{kk: v[kk] for kk in ("place_rule", "wall_seg_m", "wall_perp_m", "half_corridor_m") if kk in v})
                                   for k, v in (sem.get("per_hypothesis") or {}).items()},
                   provisional=provisional_pose(o, hyps, sem), ocr_lines=o.get("ocr_lines"), vlm=o.get("vlm"))
        why = None
        if not (o.get("image_quality") or {}).get("ok"):
            why = "IMAGE_NOT_USABLE"
        elif not decisive:
            why = "NO_DECISIVE_LABEL" + (" (VLM-only / conflict)" if alts else "")
        elif src != "VLM_TEXT_CONFIRMED" and low_quality_override(o) and max((float(c) for _, c in (o.get("ocr_lines") or [])), default=0.0) < OVERRIDE_MIN_OCR_CONF:
            why = f"LOW_QUALITY_OVERRIDE_WEAK_READ (photo taken over a quality warning, OCR conf < {OVERRIDE_MIN_OCR_CONF})"
        elif _target(o) is None:
            why = "NO_TARGET_POSITION"
        elif excluded and o["observation_id"] in excluded:                        # 2026-11-18 B3: set aside by the single-plate E7 misread check
            why = f"EXCLUDED_AS_MISREAD ({excluded[o['observation_id']]})"
        rec["usable"], rec["why_not"] = why is None, why
        per_obs.append(rec)
        if why is None:
            usable.append((o, rec, alts, sem))
    out = dict(kind="LIVE2_E8_EVALUATION", observation_ids=[o["observation_id"] for o in obs], per_observation=per_obs,
               thresholds=dict(PLACE_TOL_M=S.PLACE_TOL_M, INDEP_MIN_M=INDEP_MIN_M, ODOM_REL_ERR=ODOM_REL_ERR,
                               ODOM_ERR_FLOOR_M=ODOM_ERR_FLOOR_M, ADJ_M=ADJ_M, REGION_CLUSTER_M=REGION_CLUSTER_M,
                               basis="PLACE_TOL from e6_semantic; others stated assumptions, NOT measured on this robot"),
               data_limits=["semantic map has no physical plate position / facing: room polygon used, yaw residual not computable"],
               map_data=dict(plates={f: MD.validate_plates(m)["status"] for f, m in (getattr(sf, "maps", None) or {}).items()},
                             yaw_check="BLOCKED_BY_MAP_DATA", gating_interval="ROOM_POLYGON (conservative)",
                             derived_interval="DERIVED_ROOM_CORRIDOR_ADJACENCY (recorded, UNVERIFIED, never gates a fix)"),
               pairs=[], candidates=[], verdict=None, why=None)
    # same-session, independent, odometry-healthy evidence set (newest session wins; same plate twice counts once)
    ev = []
    for o, rec, alts, sem in usable:
        if ev and o.get("session") != ev[-1][0].get("session"):
            for e in ev:
                e[1].update(usable=False, why_not="ODOM_SESSION_CHANGED (older session)")
            ev = []
        dup = None
        for e in ev:
            same_label = {a[0] for a in alts} & {a[0] for a in e[2]}
            if same_label and float(np.hypot(*(_target(o) - _target(e[0])))) < INDEP_MIN_M:
                dup = e[0]["observation_id"]
        if dup:
            rec.update(usable=False, why_not=f"SAME_PLATE_AS {dup} (< {INDEP_MIN_M} m, same label): not independent")
            continue
        ev.append((o, rec, alts, sem))
    out["pairs"] = _pairs(ev, sf, rows, lio_series)
    out["n_valid_independent"] = len(ev)
    out["n_room_readings"] = sum(1 for e in ev if not is_category_reading(e[2]))
    if len(ev) < 2:
        rejected = "; ".join(f"{r['observation_id']}: {r['why_not']}" for r in per_obs if not r["usable"])
        out.update(verdict="MORE_OBSERVATION_REQUIRED",
                   why=f"{len(ev)} usable independent observation(s) of {len(obs)} — a second independent plate is needed"
                       + (f" ({rejected})" if rejected else ""))
        return out
    bad = [p for p in out["pairs"] if not p["usable"]]
    if bad:
        out.update(verdict="MORE_OBSERVATION_REQUIRED", why="odometry between observations not usable: " +
                   "; ".join(p["odometry"]["why"] for p in bad))
        return out
    if out["n_room_readings"] == 0:              # a category exists on every floor: categories alone never choose a floor
        out.update(verdict="MORE_OBSERVATION_REQUIRED", why=f"{len(ev)} category reading(s) and no room label: a room plate is needed to choose a floor")
        return out
    floors_ok = set.intersection(*[set(p["explains"]) for p in out["pairs"]])
    if DROP_MISREAD and not floors_ok and len(ev) >= DROP_MISREAD_MIN:
        fits = []
        for i in range(len(ev)):
            sub = ev[:i] + ev[i + 1:]
            if all(is_category_reading(e[2]) for e in sub):
                continue
            prs = _pairs(sub, sf, rows, lio_series)
            if not prs or any(not p["usable"] for p in prs):
                continue
            fo = set.intersection(*[set(p["explains"]) for p in prs])
            if fo:
                fits.append((i, sub, prs, fo))
        out["misread_check"] = dict(on=True, n_plates=len(ev), fits_without=[ev[i][0]["observation_id"] for i, _, _, _ in fits], left_out=None)
        if len(fits) == 1:
            i, sub, prs, fo = fits[0]
            ev[i][1].update(usable=False, why_not=f"LEFT_OUT_AS_MISREAD: the other {len(sub)} plates fit {sorted(fo)}F together, this one fits no floor with them")
            out["misread_check"].update(left_out=ev[i][0]["observation_id"], label=[a[0] for a in ev[i][2]][:3], floors=sorted(fo))
            out["pairs_with_all_plates"] = out["pairs"]
            ev, out["pairs"], floors_ok = sub, prs, fo
        else:
            out["misread_check"]["why"] = "no plate can be left out to make the rest consistent" if not fits else "more than one plate could be the wrong one: none is left out"
    cands = []
    for h in hyps:
        vs = [s["per_hypothesis"].get(h["id"]) or {} for _, _, _, s in ev]
        if all(v.get("verdict") == "SEMANTIC_CONSISTENT" for v in vs) and h["floor"] in floors_ok:
            cands.append(dict(id=h["id"], floor=h["floor"], residuals_m=[v.get("dist_m") for v in vs],
                              mean_residual_m=round(float(np.mean([v.get("dist_m") or 0 for v in vs])), 2),
                              target_px=vs[-1].get("target_px"), rooms=[v.get("map_room_id") for v in vs]))
    if PATH_RULE and cands and rows and light_of is not None:
        stops = [float((o.get("odom_pose_at_stop") or {}).get("t") or o.get("stopped_wall") or 0.0) for o, _, _, _ in ev]
        t_a, t_b = min(stops), max(stops)
        hy = {h["id"]: h for h in hyps}
        keep, dropped = [], []
        for c in cands:
            h = hy.get(c["id"]) or {}
            try:
                L = light_of(h.get("light")) if h.get("light") else None
                ok, why = path_feasible(h["rec"], getattr(L, "wall_tr", None), rows, t_a, t_b)
            except Exception as ex:                                                   # noqa: BLE001  a failed check never drops a candidate
                ok, why = True, f"path check error {type(ex).__name__}: {ex}"
            c["path_ok"] = bool(ok)
            if not ok:
                c["path_why"] = why
            (keep if ok else dropped).append(c if ok else dict(id=c["id"], floor=c["floor"], why=why))
        out["path_rule"] = dict(on=True, last_m=PATH_RULE_LAST_M, rank_only=PATH_RANK, dropped=[] if PATH_RANK else dropped, ranked_down=dropped if PATH_RANK else [],
                                n_before=len(cands), n_after=len(keep), plate_stops=[round(t_a, 3), round(t_b, 3)])
        cands = keep + ([c for c in cands if not c["path_ok"]] if PATH_RANK else [])
    elif PATH_RULE:
        out["path_rule"] = dict(on=True, skipped="no candidates / rows / map loader")
    out["floors_explaining"] = sorted(floors_ok)
    out["candidates"] = cands
    out["n_evidence"] = len(ev)
    if cands:
        regions = []
        for c in cands:
            for g in regions:
                h0 = g[0]
                ppm = next((h["rec"]["ppm"] for h in hyps if h["id"] == h0["id"]), 1.0)
                if h0["floor"] == c["floor"] and h0["target_px"] and c["target_px"] and \
                        math.hypot(h0["target_px"][0] - c["target_px"][0], h0["target_px"][1] - c["target_px"][1]) / ppm <= REGION_CLUSTER_M:
                    g.append(c)
                    break
            else:
                regions.append([c])
        out["regions"] = [dict(floor=g[0]["floor"], hypotheses=[c["id"] for c in g]) for g in regions]
        other_floors = floors_ok - {regions[0][0]["floor"]}
        cand_floors = {c["floor"] for c in cands}
        feas = {c["floor"] for c in cands if c.get("path_ok", True)}
        n_down = sum(1 for c in cands if c.get("path_ok") is False)
        if PATH_RULE and (len(feas) == 1 or (not feas and len(cand_floors) == 1)):   # 2026-11-18: no 5 m grouping; E7 picks the place
            fl_ = next(iter(feas or cand_floors))
            out.update(verdict="REGION_CANDIDATE_READY",
                       why=f"{len(ev)} independent plates consistent under {len(cands)} hypothesis(es), {len(cands) - n_down} without a wall crossing on {fl_}F "
                           f"(semantic + plate-to-plate odometry + path between the plates / last {PATH_RULE_LAST_M:.0f} m)"
                           + (f"; {n_down} crossing a wall ranked last, judged again after E7's correction" if n_down else "") + " — not a fix until E7")
        elif PATH_RULE:
            out.update(verdict="MORE_OBSERVATION_REQUIRED", why=f"AMBIGUOUS: feasible hypotheses on floors {sorted(feas or cand_floors)}")
        elif len(regions) == 1 and not other_floors:
            out.update(verdict="REGION_CANDIDATE_READY",
                       why=f"{len(ev)} independent plates consistent under {len(cands)} hypothesis(es) on {regions[0][0]['floor']}F "
                           f"(semantic + plate-to-plate odometry within tolerance) — not a fix until E7")
        else:
            out.update(verdict="MORE_OBSERVATION_REQUIRED",
                       why=f"AMBIGUOUS: {len(regions)} separated regions / floors {sorted(floors_ok)} explain the observations")
    elif PATH_RULE and (out.get("path_rule") or {}).get("dropped"):
        out.update(verdict="MORE_OBSERVATION_REQUIRED", why="PATH_RULE: every plate-consistent hypothesis puts the path through a wall: " +
                   "; ".join(f"{d['id']} ({d['why']})" for d in out["path_rule"]["dropped"]))
    elif floors_ok:
        out.update(verdict="CANDIDATE_MISSING",
                   why=f"the readings fit the map on {sorted(floors_ok)}F (rooms " +
                       ", ".join(str(p["floors"][f]["rooms"]) for p in out["pairs"] for f in sorted(floors_ok)) +
                       ") but no live E3 hypothesis places both plates there — no hypothesis is invented; re-search")
    else:
        out.update(verdict="MORE_OBSERVATION_REQUIRED",
                   why="INCONSISTENT: no floor / room pair explains the plate-to-plate odometry (" +
                       "; ".join(f"{f}F resid {c['translation_residual_m']} m {c['topology']}" for p in out["pairs"]
                                 for f, c in p["floors"].items()) + ")")
    return out


def _crop_for_vlm(img, roi, pad=0.5):
    """The VLM prompt reads the sign in the CENTRE of the image: give it the operator's sign box (1280x720 page coordinates) with a margin, the whole photo otherwise."""
    if not roi:
        return img
    h, w = img.shape[:2]
    sx, sy = w / 1280.0, h / 720.0
    x0, y0, x1, y1 = [float(v) for v in roi]
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    a, b, c, d = int(max(0, (x0 - px) * sx)), int(max(0, (y0 - py) * sy)), int(min(w, (x1 + px) * sx)), int(min(h, (y1 + py) * sy))
    return img[b:d, a:c] if c > a and d > b else img


def provisional_pose(o, hyps, alts_sem):
    """Single-observation PROVISIONAL place: the best hypothesis under which this plate alone is consistent (never a fix)."""
    best = None
    for h in hyps:
        v = (alts_sem.get("per_hypothesis") or {}).get(h["id"]) or {}
        if v.get("verdict") == "SEMANTIC_CONSISTENT" and (best is None or v["dist_m"] < best["dist_m"]):
            best = dict(hypothesis=h["id"], floor=h["floor"], target_px=v.get("target_px"), dist_m=v["dist_m"], room=v.get("map_room_id"))
    return best


E7_RETRY_S, E7_RETRY_M, E7_RETRY_MAX = 60.0, 3.0, 12   # R41: the next window shares no frame with the last and is long enough to converge (49 s: not converged, replay load3)
VLM_TRIES, VLM_RETRY_S, VLM_YIELD_S = 60, 5.0, 120.0   # R41: E2 gives way for at most VLM_YIELD_S per photo (a running fit lasts 40-60 s); after that the VLM keeps trying beside E2 and the state says why   # R40: 4 x 20 s gave up on 5 of 6 photos of X20261103_001810 (MemAvailable 5300-5800 < 6200); e2_budget now frees memory while a photo waits
REGION_RADIUS_M = 15.0        # semantic recovery: robot within ~15 m of the plate room (camera range + margin)
SINGLE_FLOOR_P = 0.8          # 2026-11-01 single-reading regional search: one floor must explain the reading this strongly
# 2026-11-01: a photo the operator took ANYWAY over a low-sharpness / no-sign-ROI warning counts only when its OCR line was read
# with conf >= OVERRIDE_MIN_OCR_CONF (PROVISIONAL, set conservatively from the one field case: O002 '402' 0.975 with the auto ROI
# on the wrong object). Stricter only — it never makes an observation usable that the old rule refused.
OVERRIDE_MIN_OCR_CONF = 0.9
REGION_PPM_MARGIN = 1.15      # x the floor px/m (estimated scales are +-5 %): a generous region, never a narrow one


# ------------------------------------------------------------------ manager (stateful, inside e34)
class PairManager:
    def __init__(self, state_dir, tracker, sem_floors, request_e7, clock=time.time, reader=None, vlm=None, vlm_admit=None,
                 lio_series=None, read_each=True):
        self.state, self.tr, self.sf, self.request_e7, self.clock = Path(state_dir), tracker, sem_floors, request_e7, clock
        self.reader, self.vlm, self.vlm_admit = reader, vlm, vlm_admit
        self.lio_series = lio_series
        self.group, self.gid, self.n_groups = [], None, 0
        self.status, self.why, self.since = "PAIR_IDLE", None, clock()
        self.last_eval, self.e7, self.job, self.lock = None, None, None, threading.Lock()
        self.read_each = read_each   # 2026-10-26: OCR (+VLM gate) of EVERY stored capture at once -> 1st = PROVISIONAL place
        self._running, self._again = False, False
        self.provisional = None
        self.history = []
        self.processed = []          # observation ids already handled (the tablet flow waits for this before any guidance)

    # -- inputs
    def _session(self):
        sess = getattr(getattr(self.tr, "store", None), "session", None)
        return (sess() or {}).get("odom_session_id") if callable(sess) else None

    def _set(self, status, why=None):
        if status != self.status:
            self.since = self.clock()
        self.status, self.why = status, why

    def on_precision(self, r):
        """One E6 precision record (bridge). Stored captures join the group; failures are recorded with their reason."""
        oid = r.get("observation_id")
        self.processed = (self.processed + [oid])[-30:]
        n = len(self.group) + 1
        if r.get("status") in ("FOCUS_FAULT_BLOCKED", "DEVICE_ERROR_BLOCKED", "BLOCKED", "REFUSED_NO_VALID_APPROVAL", "USER_ABORTED") \
                or r.get("home_confirmed") is False:
            self._set("HOLD_ERROR", f"{oid}: {r.get('status')} {r.get('why') or ''}")
            self._log_obs(r, accepted=False, why=r.get("status"))
            return self.snapshot()
        stored = r.get("status") in ("CAPTURE_STORED", "CAPTURED") and (r.get("image_quality") or {}).get("ok")
        if not stored:
            why = f"{r.get('status')}: {r.get('why') or ''} — not counted as an observation (no automatic retry)"
            self._log_obs(r, accepted=False, why=why)
            if self.status != "HOLD_ERROR":
                self._set(f"OBS{len(self.group)}_WAITING_RESUME" if self.group else "PAIR_IDLE", why)
            return self.snapshot()
        sess = self._session()
        if self.group and sess != self.group[-1].get("session"):
            self._log_obs(r, accepted=False, why="ODOM_SESSION_CHANGED")
            self._close("HOLD_ERROR", f"FAST-LIO session changed ({self.group[-1].get('session')} -> {sess}): the group is void")
            return self.snapshot()
        if not self.group and getattr(self, "closed", None) and self.history and self.history[-1]["status"] == "CANDIDATE_MISSING" and (self.closed["observations"] or [{}])[-1].get("session") == sess:
            # R41 (full-load replay X20261103_001810): 407 + 405 closed as CANDIDATE_MISSING while their fit was running; the next photo started a new group and the pair was gone for good. A missing hypothesis says nothing against the readings: they stay.
            self.group, self.gid, self.e7 = list(self.closed["observations"]), self.closed["group_id"], None
            n = len(self.group) + 1
        if not self.group:
            self.n_groups += 1
            self.gid = f"PAIR_{time.strftime('%H%M%S', time.localtime(self.clock()))}_{self.n_groups}"
        o = dict(observation_id=oid, capture_id=r.get("capture_id"), capture_file=r.get("capture_file"),
                 image_sha256=r.get("image_sha256") or (sha256_file(r["capture_file"]) if r.get("capture_file") and
                                                         Path(r["capture_file"]).exists() else None),
                 resolution=r.get("resolution"), image_quality=r.get("image_quality"), focus=r.get("focus"),
                 target_odom_link_xy=r.get("target_odom_link_xy"), odom_pose_at_stop=r.get("odom_pose_at_stop"),
                 stopped_wall=r.get("stopped_wall") or r.get("wall"), capture_wall=r.get("capture_wall") or r.get("wall"), session=sess,
                 odometry_at_e6=r.get("odometry_at_e6"),
                 experiment_id=r.get("experiment_id"), source=r.get("source"), n_in_group=n, group_id=self.gid,
                 ocr_lines=r.get("ocr_lines") if r.get("status") == "CAPTURED" else None, vlm=r.get("vlm"),
                 read=r.get("status") == "CAPTURED", reading=None, joined_clock=self.clock())
        self.group.append(o)
        self._log_obs(o, accepted=True)
        if len(self.group) >= 2:
            self._set("PAIR_READING", f"{len(self.group)} observations stored: reading {sum(not g['read'] for g in self.group)} image(s)")
            self._start_reading()
        else:
            self._set(f"OBS{n}_WAITING_RESUME", f"{oid} stored (sha256 {str(o['image_sha256'])[:12]}): "
                      + ("OCR running in the background (PROVISIONAL place, never a fix)" if self.read_each
                         else "reading after the next observation"))
            if self.read_each:
                self._start_reading()
        return self.snapshot()

    def on_e6_progress(self, e6_recent, odom_now=None):
        """Display states between captures: OBS{n}_STOP_CONFIRMED / _E6_RUNNING / SEARCHING (from the E6 tracker)."""
        if self.status in ("PAIR_READING", "PAIR_CONSISTENCY_CHECK", "REGION_CANDIDATE_READY", "E7_VERIFYING", "HOLD_ERROR"):
            return
        n = len(self.group) + 1
        known = {g["observation_id"] for g in self.group}
        new = [o for o in e6_recent or [] if o.get("state") == "STOPPED_TARGET_READY" and o.get("observation_id") not in known
               and float(o.get("stopped_wall") or 0) > self.since - 1]
        if new:
            self._set(f"OBS{n}_E6_RUNNING", f"{new[-1]['observation_id']}: E6 aim / AF / capture")
        elif self.status.endswith("_WAITING_RESUME") and self.group and odom_now is not None:
            a = self.group[-1].get("odom_pose_at_stop") or {}
            if a.get("x") is not None and math.hypot(odom_now[1] - a["x"], odom_now[2] - a["y"]) > 0.3:
                self._set(f"OBS{n}_SEARCHING", "moving: E5/SigLIP searches the next plate")

    # -- reading (background, CPU OCR; VLM only through the GPU gate)
    def _start_reading(self):
        with self.lock:
            if self._running:                    # the running reader picks the new capture up before it ends
                self._again = True
                return
            self._running, self._again = True, False
            self.job = threading.Thread(target=self._read_and_evaluate, daemon=True)
            self.job.start()

    def _read_and_evaluate(self):
        try:                                    # R37: OCR on core 5 at nice 19 under E3 + E7 took 31-63 s (idle 8 s). The search cores, same priority as e34, ahead of E2; it inherits to the OCR child
            os.sched_setaffinity(threading.get_native_id(), {6, 7})
        except (AttributeError, OSError):
            pass
        t0 = self.clock()
        lat = []
        try:
            while True:
                self._read_pending(lat)
                with self.lock:
                    if self._again:
                        self._again = False
                        continue
                    break
            group = list(self.group)
            if len(group) >= 2:
                self._set("PAIR_CONSISTENCY_CHECK", "semantic map + odometry consistency")
                self.evaluate(read_latency=lat, read_s=round(self.clock() - t0, 2))
            elif group:
                self.provisional_eval(read_latency=lat, read_s=round(self.clock() - t0, 2))
        finally:
            with self.lock:                      # a capture stored meanwhile is read by a follow-up job (no gap)
                if self._again:
                    self._again = False
                    self.job = threading.Thread(target=self._read_and_evaluate, daemon=True)
                    self.job.start()
                else:
                    self._running = False

    def provisional_eval(self, read_latency=None, read_s=None):
        """One stored + read observation: its PROVISIONAL place under the live E3 hypotheses (never a fix, no E7)."""
        hyps = [dict(id=h.id, floor=h.floor, rec=h.rec) for h in self.tr.alive()]
        ev = evaluate_group(list(self.group), hyps, self.sf)
        ev.update(kind="LIVE2_E8_PROVISIONAL", verdict="PROVISIONAL", group_id=self.gid, wall=self.clock(),
                  read_latency=read_latency, read_s=read_s, resources=_resources(),
                  why="1 observation read: PROVISIONAL place only — a second independent plate + E7 are needed for a fix")
        self.last_eval = self.provisional = ev
        _jsonl(self.state / "e8_pairs.jsonl", ev)
        self.snapshot()
        return ev

    def _read_pending(self, lat):
        import cv2
        for o in [g for g in list(self.group) if not g["read"]]:
            r0 = time.time()
            img = cv2.imread(o["capture_file"]) if o.get("capture_file") else None
            if img is None or (o.get("image_sha256") and sha256_file(o["capture_file"]) != o["image_sha256"]):
                o.update(read=True, ocr_lines=[], reading=dict(error="IMAGE_MISSING_OR_CHANGED"))
                continue
            roi = operator_roi(o)
            if self.reader is None:                    # 2026-11-01: the operator's sign ROI (manual focus) is the OCR ROI too
                lines, meta = __import__("e6_ocr").sign_read(img, roi=roi)
            else:
                lines, meta = self.reader(img)
            ocr_s = round(time.time() - r0, 2)
            # R37: OCR is evaluated at once; the VLM reviews the same stored photo in its own thread whatever OCR said (11-02: OCR success meant NOT_NEEDED, and a VLM load held the next reading 65 s). Nothing waits for it.
            vmeta = dict(state="QUEUED") if self.vlm is not None else dict(state="UNAVAILABLE", reason="VLM not enabled for this experiment")
            o.update(read=True, ocr_lines=lines, vlm=None, reading=dict(ocr_s=ocr_s, engine=meta.get("engine"), vlm=vmeta,
                                                                      ocr_load_s=meta.get("load_s"), ocr_roi_s=meta.get("roi_s"),
                                                                      ocr_full_s=meta.get("full_s"), ocr_roi_source=meta.get("roi_source"),
                                                                          observation_id=o["observation_id"], capture_id=o.get("capture_id")))
            lat.append(dict(observation_id=o["observation_id"], ocr_s=ocr_s, vlm=vmeta))
            self._log_obs(o, accepted=True, event="READ")
            self.snapshot()
            if self.vlm is not None:
                self._vlm_enqueue(o, img, roi)

    # -- VLM review (R37): own thread, one at a time, same photo, result kept apart from OCR
    def _vlm_enqueue(self, o, img, roi):
        with self.lock:
            self._vlm_q = getattr(self, "_vlm_q", [])
            self._vlm_q.append(dict(o=o, crop=_crop_for_vlm(img, roi), tries=0, cropped=bool(roi)))
            o["vlm_wait_since"] = self.clock()
            th = getattr(self, "_vlm_th", None)
            if th is None or not th.is_alive():
                self._vlm_th = threading.Thread(target=self._vlm_loop, daemon=True)
                self._vlm_th.start()

    def _vlm_loop(self):
        while True:
            with self.lock:
                job = self._vlm_q.pop() if getattr(self, "_vlm_q", []) else None    # R41: the newest photo first: a refused older one must not hold the plate that can decide (O005 kept 405 waiting for minutes)
            if job is None:
                stop = getattr(getattr(self.vlm, "__self__", None), "stop", None)    # R41: resident it starves E2 and SigLIP (4290 MB); a reload takes 8 s
                if stop is not None:
                    stop("BATCH_DONE")
                return
            o = job["o"]
            adm, why = (self.vlm_admit or (lambda: (True, "no admission rule")))()
            if not adm:
                job["tries"] += 1
                o["reading"]["vlm"] = dict(state="NOT_ADMITTED_RESOURCE", why=why, tries=job["tries"])
                self.snapshot()
                if job["tries"] < VLM_TRIES:                  # memory / rate may recover; the robot does not wait for it
                    time.sleep(VLM_RETRY_S)
                    with self.lock:
                        self._vlm_q.insert(0, job)                # behind the newer ones
                continue
            o["reading"]["vlm"] = dict(state="ASKING", why=why)
            self.snapshot()
            t0 = time.time()
            try:
                v = self.vlm(job["crop"])
                o["vlm"] = (v or {}).get("parsed")
                o["reading"]["vlm"] = dict(state="ASKED", reason="OCR_CROSS_CHECK", crop=job.get("cropped"), sec=round(time.time() - t0, 2), model=(v or {}).get("model"), raw=str((v or {}).get("raw"))[:200])
            except Exception as e:                            # noqa: BLE001
                o["reading"]["vlm"] = dict(state="ERROR", error=f"{type(e).__name__}: {e}", sec=round(time.time() - t0, 2))
            self._log_obs(o, accepted=True, event="VLM_READ")
            self.snapshot()
            if o in self.group:
                self._start_reading()                         # a new evaluation with the VLM result: agreement, conflict or nothing new

    def evaluate(self, read_latency=None, read_s=None):
        hyps = [dict(id=h.id, floor=h.floor, rec=h.rec, light=getattr(h, "light_dir", None)) for h in self.tr.alive()]
        rows = getattr(getattr(self.tr, "store", None), "rows", None)
        lio = self.lio_series() if callable(self.lio_series) else self.lio_series
        ev = evaluate_group(self.group, hyps, self.sf, rows, lio, light_of=getattr(self.tr, "light", None), excluded=getattr(self, "excluded", None))
        ev.update(group_id=self.gid, wall=self.clock(), read_latency=read_latency, read_s=read_s, resources=_resources())
        self.last_eval = ev
        _jsonl(self.state / "e8_pairs.jsonl", ev)
        v = ev["verdict"]
        if v != "REGION_CANDIDATE_READY" and len(self.group) >= 2:       # R35 record only: E7 does not run, the shadow says why
            try:
                import e7_semantic_shadow as SH
                SH.write(self.state, SH.evaluate(ev, list(self.group), None, None, None, 0.05, 0.10, None))
            except Exception:                                            # noqa: BLE001
                pass
        if v == "REGION_CANDIDATE_READY":
            self._set("REGION_CANDIDATE_READY", ev["why"])
            self._apply_and_request_e7(ev, hyps)
        elif v == "CANDIDATE_MISSING":
            self._close("CANDIDATE_MISSING", ev["why"])
        elif self.budget_used(ev):
            self._close("NO_FIX", f"{self.budget_used(ev)} without a consistent unique region: {ev['why']}")
        else:
            self._set("MORE_OBSERVATION_REQUIRED", ev["why"])
        return ev

    def budget_used(self, ev):
        """R34 (11-01: 4 captures, 2 unreadable -> group closed with one room reading): valid independent observations, capture attempts and time are separate limits.
        -> text when the group must end, else None. Thresholds of the pair rule and E7 are not touched."""
        nv, na = int(ev.get("n_valid_independent") or 0), len(self.group)
        first = float(self.group[0].get("joined_clock") or self.clock()) if self.group else self.clock()
        if nv >= MAX_OBS:
            return f"{nv} valid independent observations"
        if na >= MAX_ATTEMPTS:
            return f"{na} capture attempts ({nv} valid)"
        if self.clock() - first > GROUP_BUDGET_S:
            return f"{self.clock() - first:.0f} s in this group ({nv} valid)"
        return None

    def _apply_and_request_e7(self, ev, hyps):
        ids = {c["id"] for c in ev["candidates"]}
        down = {c["id"] for c in ev["candidates"] if c.get("path_ok") is False}
        for h in self.tr.alive():                         # B1: E7 takes the path-feasible ones first (e6e7_link.request_e7); every flag is set anew
            h.e8_path_down = h.id in down
        last = self.group[-1]
        po = next((p for p in reversed(ev["per_observation"]) if p.get("usable")), ev["per_observation"][-1])     # B6: never the labels of a plate left out
        res = dict(alternatives=po["labels"], floor_likelihood=po["floor_likelihood"],
                   per_hypothesis={h["id"]: dict(verdict="SEMANTIC_CONSISTENT" if h["id"] in ids else "PAIR_INCONSISTENT")
                                   for h in hyps})
        self.tr.apply_semantic(res, self.gid, "E8_PAIR")
        cons = [h for h in self.tr.alive() if h.id in ids]
        self._set("E7_VERIFYING", f"E7 LiDAR check of {sorted(ids)} for {self.gid}")
        self.request_e7(cons, dict(observation_id=self.gid, source="E8_PAIR", pair_observation_ids=[g["observation_id"] for g in self.group],
                                   target_odom_link_xy=last.get("target_odom_link_xy")))

    def reevaluate_on_new_hypotheses(self):
        """R41: a fit that places these plates (anchored, regional) came in while the group waits: judge now, not at the next photo. Same rules. -> True when started."""
        ev = self.last_eval or {}
        if len(self.group) < 2 or self._running or self.status != "MORE_OBSERVATION_REQUIRED":
            return False
        alive = list(self.tr.alive())
        sig = tuple(sorted(h.id for h in alive if (getattr(h, "rec", {}) or {}).get("provenance") == "SEMANTIC_ANCHORED" or (getattr(h, "rec", {}) or {}).get("regional_request")))
        if ev.get("verdict") == "REGION_CANDIDATE_READY":
            # 2026-11-18: after a READY look, only a NEW path-anchored candidate (second plate, earlier plate pinned) asks for a new judgement, so the next E7 includes it
            old = set(getattr(self, "_hyp_sig", ()) or ())
            new_path = os.environ.get("LIVE2_ANCHOR_PATH", "1") == "1" and any(
                h.id not in old and ((getattr(h, "rec", {}) or {}).get("evidence") or {}).get("kind") == "PATH_ANCHORED" for h in alive)
            if not new_path:
                return False
        if sig == getattr(self, "_hyp_sig", ()):
            return False
        self._hyp_sig = sig
        self._start_reading()
        return True

    def retry_e7(self):
        """R40: readings ready, E7 not convinced yet: more path may be all it needs, so ask again on new ground, without waiting for another plate. Never after a rejection (a contradiction stays one)."""
        ev, e7 = self.last_eval or {}, self.e7 or {}
        if self.status != "MORE_OBSERVATION_REQUIRED" or ev.get("verdict") != "REGION_CANDIDATE_READY" or not e7 or e7.get("observation_id") != self.gid:
            return False
        try:
            if (json.loads((self.state / "e7_semantic_shadow.json").read_text()).get("rejected_by") or []):
                return False
        except (OSError, ValueError):
            return False
        rows = getattr(getattr(self.tr, "store", None), "rows", None) or []
        p = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(rows, rows[1:]))
        now = self.clock()
        n = getattr(self, "_e7_retry", dict(gid=None))
        if n["gid"] != self.gid:
            n = self._e7_retry = dict(gid=self.gid, wall=now, path=p, n=0)
        t_last = float(((e7.get("e7") or {}).get("t_end")) or n["wall"])        # the frames of the next window start where the last one ended; the time E7 itself took already counts
        if now - t_last < E7_RETRY_S or p - n["path"] < E7_RETRY_M or n["n"] >= E7_RETRY_MAX:
            return False
        ids = {c["id"] for c in ev.get("candidates") or []}
        cons = [h for h in self.tr.alive() if h.id in ids]
        if not cons:
            return False
        n.update(wall=now, path=p, n=n["n"] + 1)
        self._set("E7_VERIFYING", f"E7 again on {E7_RETRY_M:.0f}+ m of new path (try {n['n']})")
        self.request_e7(cons, dict(observation_id=self.gid, source="E8_PAIR", pair_observation_ids=[g["observation_id"] for g in self.group], target_odom_link_xy=self.group[-1].get("target_odom_link_xy")))
        return True

    def on_e7(self, out):
        """e6e7_link -> E7 result for this group (already FAST-LIO / session gated there)."""
        if out.get("observation_id") != self.gid:
            return
        self.e7 = dict(out, used_observation_ids=[g["observation_id"] for g in self.group], best_result=e7_best(out))
        _jsonl(self.state / "e8_pairs.jsonl", dict(kind="LIVE2_E8_E7_RESULT", group_id=self.gid, wall=self.clock(),
                                                   global_decision=out.get("global_decision"), why=out.get("why"),
                                                   e7_s=out.get("e7_s"), e7=out.get("e7"), used_observation_ids=self.e7["used_observation_ids"]))
        if out.get("global_decision") == "GLOBAL_CONFIRMED":
            self._close("GLOBAL_FIX", out.get("why"))
        elif self.budget_used(self.last_eval or {}):
            self._close("NO_FIX", f"E7 {out.get('global_decision')}: {out.get('why')}")
        else:
            self._set("MORE_OBSERVATION_REQUIRED", f"E7 {out.get('global_decision')}: {out.get('why')} — another observation")

    # -- 2026-10-30 semantic recovery (CANDIDATE_MISSING -> regional E2 -> re-evaluate the SAME closed group)
    def recovery_region(self, rows=None):
        """-> (region dict, None) or (None, why). Only for a closed CANDIDATE_MISSING group whose readings fit exactly ONE
        floor; region = disk around the last plate's room (semantic px = structure-map px) with radius REGION_RADIUS_M (+ the
        odometry path since that capture) at a generous px/m, so the robot's current place is inside it."""
        if not self.history or self.history[-1]["status"] != "CANDIDATE_MISSING" or not self.closed:
            return None, "no CANDIDATE_MISSING group"
        ev = getattr(self, "closed_eval", None) or {}
        floors = ev.get("floors_explaining") or []
        if len(floors) != 1:
            return None, f"SEMANTIC_RECOVERY_SKIPPED: readings fit {len(floors)} floors {floors} (needs exactly one)"
        f = floors[0]
        rooms = None
        for pr in reversed(ev.get("pairs") or []):
            c = (pr.get("floors") or {}).get(f) or (pr.get("floors") or {}).get(str(f))
            if c and c.get("rooms"):
                rooms = c["rooms"]; break
        m = (getattr(self.sf, "maps", None) or {}).get(f) or (getattr(self.sf, "maps", None) or {}).get(str(f))
        sp = next((s_ for s_ in (m or {}).get("spaces", []) if rooms and s_.get("id") == rooms[-1]), None)
        if not sp or not sp.get("polygon_px"):
            return None, f"SEMANTIC_RECOVERY_SKIPPED: room {rooms} has no polygon on {f}F"
        P = np.asarray(sp["polygon_px"], float)
        res, ov = floor_res(f, m)
        if not res:
            return None, f"SEMANTIC_RECOVERY_SKIPPED: no scale for {f}F"
        last = self.closed["observations"][-1]
        walked = 0.0
        t_cap = last.get("capture_wall") or last.get("stopped_wall")
        if rows and t_cap:
            xy = [(r[2], r[3]) for r in rows if r[1] >= float(t_cap)]
            walked = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])))
        ppm_hi = REGION_PPM_MARGIN / res
        half = float(np.max(np.hypot(*(P - P.mean(0)).T)))
        region = dict(floor=f, center_px=[round(float(v), 1) for v in P.mean(0)], rooms=rooms, group_id=self.closed["group_id"],
                      radius_px=round(half + (REGION_RADIUS_M + walked) * ppm_hi, 1), walked_since_capture_m=round(walked, 1),
                      radius_base_px=round(half + REGION_RADIUS_M * ppm_hi, 1), px_per_m_hi=round(ppm_hi, 3),
                      capture_wall=None if not t_cap else float(t_cap), basis="PAIR_CANDIDATE_MISSING",
                      scale_source=(ov or {}).get("status") or "SEMANTIC_MAP_META")
        return region, None

    def single_obs_region(self, r, rows=None):
        """2026-11-01: (region, None) or (None, why) for ONE decisive, usable reading (per_observation record r of the current
        group) whose floor likelihood is >= SINGLE_FLOOR_P on one floor and whose label names exactly ONE space on that floor.
        Used only to GENERATE candidates (regional E2) early — it is never evidence for a fix (the pair rule + E7 still decide)."""
        fl = r.get("floor_likelihood") or {}
        if not fl:
            return None, "no floor likelihood"
        f_s, p = max(fl.items(), key=lambda kv: kv[1])
        if p < SINGLE_FLOOR_P:
            return None, f"floor likelihood {p:.2f} < {SINGLE_FLOOR_P} (several floors possible)"
        f = int(f_s)
        maps = getattr(self.sf, "maps", None) or {}
        m = maps.get(f) or maps.get(str(f)) or {}
        spaces = {}
        for lab, _ in r.get("labels") or []:
            for sp in self.sf.spaces(f, lab) if hasattr(self.sf, "spaces") else []:
                if sp.get("polygon_px"):
                    spaces[sp.get("id")] = sp
        if len(spaces) != 1:
            return None, f"label {[l for l, _ in r.get('labels') or []]} names {len(spaces)} spaces on {f}F (needs exactly one)"
        sp = next(iter(spaces.values()))
        P = np.asarray(sp["polygon_px"], float)
        res, ov = floor_res(f, m)
        if not res:
            return None, f"no scale for {f}F"
        o = next((g for g in self.group if g.get("observation_id") == r.get("observation_id")), {})
        walked = 0.0
        t_cap = o.get("capture_wall") or o.get("stopped_wall")
        if rows and t_cap:
            xy = [(q[2], q[3]) for q in rows if q[1] >= float(t_cap)]
            walked = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])))
        half = float(np.max(np.hypot(*(P - P.mean(0)).T)))
        return dict(floor=f, center_px=[round(float(v), 1) for v in P.mean(0)], rooms=[sp.get("id")], observation_id=r.get("observation_id"),
                    radius_px=round(half + (REGION_RADIUS_M + walked) * REGION_PPM_MARGIN / res, 1), walked_since_capture_m=round(walked, 1),
                    radius_base_px=round(half + REGION_RADIUS_M * REGION_PPM_MARGIN / res, 1), px_per_m_hi=round(REGION_PPM_MARGIN / res, 3),
                    capture_wall=None if not t_cap else float(t_cap),
                    scale_source=(ov or {}).get("status") or "SEMANTIC_MAP_META", basis="SINGLE_DECISIVE_READING"), None

    def recover_closed(self):
        """Re-evaluate the last closed CANDIDATE_MISSING group with the CURRENT hypotheses (after the regional E2 search).
        Same observations, same rules (evaluate_group): a new hypothesis is NEVER a fix by itself; REGION_CANDIDATE_READY
        goes to E7 exactly as before."""
        if not self.history or self.history[-1]["status"] != "CANDIDATE_MISSING" or not self.closed or self.group:
            return None
        self.group, self.gid, self.e7 = list(self.closed["observations"]), self.closed["group_id"], None
        ev = self.evaluate()
        ev["recovery"] = True
        return ev

    def _close(self, status, why):
        self.history.append(dict(group_id=self.gid, status=status, why=why, observation_ids=[g["observation_id"] for g in self.group],
                                 wall=self.clock()))
        self.closed = dict(group_id=self.gid, observations=list(self.group), evaluation=self.last_eval, e7=self.e7)
        self.closed_eval = self.last_eval            # semantic recovery reads the verdict / floors / rooms of this group
        self._set(status, why)
        self.group, self.e7 = [], None

    def _log_obs(self, o, accepted, why=None, event="STORED"):
        _jsonl(self.state / "e8_observations.jsonl", dict(
            kind="LIVE2_E8_OBSERVATION", event=event if accepted else "REJECTED", group_id=self.gid if accepted else None,
            wall=self.clock(), why=why, **{k: o.get(k) for k in (
                "observation_id", "capture_id", "experiment_id", "session", "capture_file", "image_sha256", "resolution",
                "image_quality", "target_odom_link_xy", "odom_pose_at_stop", "odometry_at_e6", "stopped_wall", "capture_wall", "n_in_group",
                "ocr_lines", "vlm", "reading", "status", "source")}))

    # -- outputs
    def resume_ok(self):
        """Drive-on is allowed only between observations (stored or not counted) and after a terminal / more-obs verdict."""
        s = self.status
        return not s.endswith("_E6_RUNNING") and s != "HOLD_ERROR"   # R36: reading / E7 run in the background; only an E6 in progress or a fault holds the robot

    def siglip_override(self, sreq):
        """E5/SigLIP must look for the next plate while the group waits for one; it yields the GPU while the pair is read."""
        s = self.status
        if s.endswith("_WAITING_RESUME") or s.endswith("_SEARCHING") or s in ("MORE_OBSERVATION_REQUIRED", "PAIR_READING", "PAIR_CONSISTENCY_CHECK", "REGION_CANDIDATE_READY", "E7_VERIFYING"):
            return dict(sreq, mode="ACTIVE", why=f"E8_PAIR_NEXT_OBSERVATION ({s})", e8_override=True)   # R36: the robot may already be driving while the last photo is read: SigLIP never pauses for it
        return sreq

    def vlm_pending(self):
        """R40: a photo waits for the VLM (asks auto_fsm to lower the E2 budget so that the load is admitted)."""
        now = self.clock()
        for o in list(self.group):
            v = (o.get("reading") or {}).get("vlm") or {}
            if v.get("state") == "ASKING":
                return True
            if v.get("state") in ("QUEUED", "NOT_ADMITTED_RESOURCE") and now - float(o.get("vlm_wait_since") or now) < VLM_YIELD_S:
                return True
        return False

    def snapshot(self):
        def short(o):
            return {k: o.get(k) for k in ("observation_id", "capture_id", "capture_file", "image_sha256", "resolution", "image_quality",
                                          "ocr_lines", "vlm", "reading", "target_odom_link_xy", "odom_pose_at_stop", "n_in_group",
                                          "session", "read")}
        closed = getattr(self, "closed", None)
        snap = dict(kind="LIVE2_E8_PAIR_STATE", policy="E8_PAIR", state=self.status, why=self.why, since=self.since,
                    group_id=self.gid, observations=[short(o) for o in self.group], evaluation=self.last_eval,
                    e7=None if not self.e7 else {k: self.e7.get(k) for k in ("global_decision", "why", "e7_s", "used_observation_ids",
                                                                             "semantic_hypothesis", "best_result", "fix_blocked")},
                    resume_ok=self.resume_ok(), history=self.history[-6:], processed=self.processed,
                    last_closed=None if not closed else dict(group_id=closed["group_id"],
                                                             observations=[short(o) for o in closed["observations"]],
                                                             e7=None if not closed["e7"] else {k: closed["e7"].get(k) for k in (
                                                                 "global_decision", "why", "e7_s", "used_observation_ids",
                                                                 "semantic_hypothesis", "best_result", "fix_blocked")}),
                    written_wall=self.clock())
        _atomic(self.state / "e8_pair_state.json", snap)
        return snap


def e7_best(out):
    """E7 best candidate summary for the tablet: floor, map pose (px, phi), alignment score / residual / inlier ratio, gap."""
    e7 = out.get("e7") or {}
    b = next((r for r in e7.get("results") or [] if isinstance(r, dict) and r.get("candidate_id") == e7.get("best")), None)
    if not b:
        return None
    g = b.get("candidate_global_pose") or {}
    return dict(candidate_id=b.get("candidate_id"), floor=b.get("floor_id"), pos_px=g.get("pos_px"), phi_deg=g.get("phi_deg"),
                ppm=g.get("ppm"), alignment_score=b.get("alignment_score"), residual_m=b.get("residual"),
                inlier_ratio=b.get("inlier_ratio"), convergence=b.get("convergence"), gap_m=e7.get("gap_m"),
                runner_up=e7.get("runner_up"), n_frames=e7.get("n_frames"))


def _resources():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live1"))
        import live1_common as LC
        r = LC.resources()
        return dict(temp_c=r.get("max_temp_c"), mem_avail_mb=round(r.get("mem_available_mb") or 0), cpu_pct=r.get("cpu_pct"),
                    gpu_pct=r.get("gpu_load_pct"))
    except Exception as e:                                                   # noqa: BLE001
        return dict(error=f"{type(e).__name__}: {e}")
