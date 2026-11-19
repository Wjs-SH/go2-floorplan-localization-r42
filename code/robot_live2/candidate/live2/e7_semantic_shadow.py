#!/usr/bin/env python3
"""e7_semantic_shadow.py (R35, 2026-11-01): the structure + semantic judgement, RECORD ONLY. It has no path to e7_state.json or to a FIX.

Why it exists: in a straight corridor LiDAR cannot separate floors (loc2 at +930 s: true 4F 0.0021, 2F 0.0000, misread 5F 0.0009), so "structure first place" can almost never confirm, while
"semantic wins when structure allows" also confirms a misread floor (offline: 708+TOILETS, 508+TOILETS). The floor then rests on OCR digits alone. This module records what a rule would decide, for field evaluation.

Verdicts (primary = strict rule):
  WOULD_REJECT     real structural contradiction (a rival better by >= GAP_M), a floor marker that disagrees, or E8 found the readings inconsistent
  WOULD_AMBIGUOUS  nothing contradicts, but a listed condition is missing (see blocked_by)
  WOULD_FIX        every condition met. Never acted on.
Conditions: >= 2 different room numbers on different plates, read from different images, one floor, one E8 region, the semantic candidate converged, no contradiction, scale within SCALE_REL of the floor scale file, crossing margin below the E7 veto,
and a THIRD floor-bearing evidence, because two readings can be wrong together (505+508, 705+708 pass everything else): a third room number, a floor marker (4F, 4층) or a VLM room number that agrees.
would_fix_2room says what the rule gives without the third evidence, would_fix_no_crossing without the crossing condition, both for the field comparison only."""
import json
import math
import os
import re
import time
from pathlib import Path

SCALE_REL = 0.10
INDEP_MIN_M = 2.0
MIN_ROOM_CONF = 0.85
XW_ABS = 0.35
_FLOOR_RE = re.compile(r"^(?:([1-9])F|([1-9])층|F([1-9]))$")


def floor_markers(obs):
    """Text like 4F / 4층 in the raw OCR of ANY observation of the group, usable or not: [(floor, observation_id, text, conf)]."""
    out = []
    for o in obs or []:
        for text, conf in o.get("ocr_lines") or []:
            m = _FLOOR_RE.match("".join(str(text).split()).upper())
            if m and float(conf) >= MIN_ROOM_CONF:
                out.append((int(next(g for g in m.groups() if g)), o.get("observation_id"), str(text), float(conf)))
    return out


def _vlm_room(o):
    v = o.get("vlm")
    return str(v.get("room_number")) if isinstance(v, dict) and v.get("room_number") else None


def evaluate(ev, obs, res, sem_id, e7_global, gap_m, xw_veto, floor_px_per_m, maps=None, long_state=None, experiment_id=None):
    """ev: E8 evaluation, obs: observation dicts of the group, res: E7 result or None, floor_px_per_m: {floor: px/m}. -> shadow record."""
    per = {r.get("observation_id"): r for r in (ev or {}).get("per_observation") or []}
    rooms, cats = [], []
    for o in obs or []:
        r = per.get(o.get("observation_id")) or {}
        if not r.get("usable"):
            continue
        if r.get("reading_kind") == "ROOM":
            top = max(r.get("labels") or [["", 0]], key=lambda a: a[1])
            conf = max((float(c) for t, c in o.get("ocr_lines") or [] if top[0] in "".join(str(t).split())), default=0.0)
            if r.get("fusion_status") == "VLM_TEXT_CONFIRMED":     # R40: no OCR line; the crop's number + department stand for it. Floor evidence is still judged below
                conf = max(conf, MIN_ROOM_CONF)
            rooms.append(dict(label=top[0], conf=round(conf, 3), observation_id=o.get("observation_id"), image=o.get("image_sha256"), capture_id=o.get("capture_id"), target=o.get("target_odom_link_xy"), wall=o.get("capture_wall"), odom=o.get("odom_pose_at_stop")))
        elif r.get("reading_kind") == "CATEGORY":
            cats.append(dict(category=(r.get("category") or {}).get("category"), observation_id=o.get("observation_id"), text=(r.get("category") or {}).get("original_text")))
    distinct, seen = [], []
    for x in rooms:
        same_image = x["image"] and any(x["image"] == y["image"] for y in seen)
        close = x["target"] and any(y["target"] and math.hypot(x["target"][0] - y["target"][0], x["target"][1] - y["target"][1]) < INDEP_MIN_M for y in seen)
        same_label = any(x["label"] == y["label"] for y in seen)
        if not (same_image or close or same_label) and x["conf"] >= MIN_ROOM_CONF:
            distinct.append(x)
        seen.append(x)
    floors = (ev or {}).get("floors_explaining") or []
    cands = (ev or {}).get("candidates") or []
    regions = (ev or {}).get("regions") or []
    fm = floor_markers(obs)
    fl = floors[0] if len(floors) == 1 else None
    fm_agree = [x for x in fm if fl is not None and x[0] == fl]
    fm_conflict = [x for x in fm if fl is not None and x[0] != fl]
    labels = {x["label"] for x in distinct}
    ids = {x["observation_id"]: x["label"] for x in distinct}
    vlm_agree = [o.get("observation_id") for o in obs or [] if o.get("observation_id") in ids and _vlm_room(o) == ids[o.get("observation_id")]]   # R37: the VLM must read THAT plate's number
    # R38: evidence of the floor that does not come from the room digits. Any one is enough; two readings agreeing on digits (OCR, VLM, or a third room) are not.
    import plate_text as PT
    by_id = {o.get("observation_id"): o for o in obs or []}
    plates = [(by_id[x["observation_id"]], fl) for x in rooms if fl is not None and x["observation_id"] in by_id]
    pj = PT.judge(plates, maps)
    long_ok = bool(long_state) and long_state.get("status") == "WOULD_CONFIRM" and long_state.get("leader_floor") == fl and len(distinct) >= 3
    fe = []
    if fm_agree:
        fe.append(dict(kind="FLOOR_MARKER", detail=fm_agree[0][2], observation_id=fm_agree[0][1]))
    if pj["name_plates"]:
        fe.append(dict(kind="PERSON_NAME_ON_PLATE", observation_ids=pj["name_plates"]))
    if len(set(pj["dept_plates"])) >= 2:
        fe.append(dict(kind="DEPARTMENT_ON_2_PLATES", observation_ids=pj["dept_plates"]))
    if long_ok:
        fe.append(dict(kind="LIDAR_FAIR_RIVALS", gap_m=long_state.get("next_gap_m")))
    third = bool(fe)
    s, rivals, klass = None, [], None
    if res and res.get("results"):
        s = next((q for q in res["results"] if q.get("candidate_id") == sem_id), None)
        rivals = [dict(candidate_id=q.get("candidate_id"), floor=q.get("floor_id"), score=q.get("alignment_score"), convergence=q.get("convergence"), scale=(q.get("correction") or {}).get("scale"),
                       crossing=(q.get("wall_crossing") or {}).get("violation")) for q in res["results"] if q.get("candidate_id") != sem_id]
        klass = res.get("structural_class")
    block, reject = [], []
    if (ev or {}).get("verdict") not in ("REGION_CANDIDATE_READY",):
        (reject if str((ev or {}).get("why", "")).startswith("INCONSISTENT") else block).append(f"E8_{(ev or {}).get('verdict')}")
    if len(distinct) < 2:
        block.append(f"ROOM_READINGS_{len(distinct)}_OF_2")
    if len(floors) != 1:
        block.append(f"FLOORS_EXPLAINING_{floors}")
    if len(regions) > 1:
        block.append(f"E8_REGIONS_{len(regions)}")
    if fm_conflict:
        reject.append(f"FLOOR_MARKER_CONFLICT {fm_conflict[0][2]}")
    if pj["conflicts"]:
        c0 = pj["conflicts"][0]
        reject.append(f"PLATE_TEXT_CONFLICT '{c0['token']}' ({c0['source']}) is {c0['floors']}F, its number says {c0['digit_floor']}F")
    foreign = [o.get("observation_id") for o in obs or [] if experiment_id and o.get("experiment_id") and o.get("experiment_id") != experiment_id]
    if foreign:
        reject.append(f"OTHER_EXPERIMENT {foreign}")
    if s is None:
        block.append("NO_E7_RESULT")
    else:
        if klass == "STRUCTURAL_CONTRADICTION":
            reject.append("STRUCTURAL_CONTRADICTION")
        if not s.get("convergence"):
            block.append("SEMANTIC_CANDIDATE_NOT_CONVERGED")
        if (s.get("local_ambiguity") or {}).get("ambiguous"):
            block.append("LOCAL_AMBIGUITY")
    scale_rel, xm = None, None
    if s is not None:
        pose = s.get("candidate_global_pose") or {}
        base = (floor_px_per_m or {}).get(s.get("floor_id")) or (floor_px_per_m or {}).get(str(s.get("floor_id")))
        if base and pose.get("ppm"):
            scale_rel = round(float(pose["ppm"]) * float((s.get("correction") or {}).get("scale") or 1.0) / float(base) - 1.0, 3)
            if abs(scale_rel) > SCALE_REL:
                block.append(f"SCALE_{scale_rel:+.3f}")
        # R41 wall crossing, measured on stored runs (reports/r41): a right place reads 0.05-0.3 and swings from window to window, a wrong place reads far more. Two questions, kept apart:
        #   absolute: does this place cross walls too much to be the place?  XW_ABS
        #   rival: does another floor explain the same LiDAR at least as well with clearly fewer crossings?  The R20 veto, threshold unchanged, but a rival that fits worse no longer vetoes (X20261103_001810 +598 s: 3F fits worse, 0.048 was the only low value of 8 floors)
        own = (s.get("wall_crossing") or {}).get("violation")
        own_c = s.get("alignment_score")
        xs = [(q.get("wall_crossing") or {}).get("violation") for q in res["results"] if q.get("floor_id") != s.get("floor_id") and (own_c is None or (q.get("alignment_score") is not None and q["alignment_score"] <= own_c))]
        xs = [x for x in xs if x is not None]
        if xs and own is not None:
            xm = round(own - min(xs), 4)
        if own is not None and own > XW_ABS:
            block.append(f"CROSSING_ABS {own:.2f} > {XW_ABS}")
    cross_block = xm is not None and xm >= xw_veto
    guard_block = not third
    base_ok = not reject and not block
    verdict = "WOULD_REJECT" if reject else "WOULD_FIX" if (base_ok and not cross_block and not guard_block) else "WOULD_AMBIGUOUS"
    all_block = block + (["CROSSING_MARGIN"] if cross_block else []) + (["NO_FLOOR_EVIDENCE"] if guard_block else [])
    return dict(kind="LIVE2_E7_SEMANTIC_SHADOW", verdict=verdict, authority="input of fix_allowed; this function writes nothing", wall=time.time(), semantic_hypothesis=sem_id, e7_global_decision=e7_global, structural_class=klass,
                blocked_by=all_block, rejected_by=reject, would_fix_2room=base_ok and not cross_block, would_fix_no_crossing=base_ok and not guard_block,
                room_readings=distinct, floor_evidence=fe, plate_text=pj, category_readings=cats, floor_markers=fm, vlm_agree=vlm_agree, floors_explaining=floors, e8_verdict=(ev or {}).get("verdict"), e8_candidates=[dict(id=c.get("id"), residuals_m=c.get("residuals_m"), rooms=c.get("rooms")) for c in cands],
                pairs=[dict(observation_ids=p.get("observation_ids"), separation_m=p.get("target_separation_m"), odom_err_m=(p.get("odometry") or {}).get("odom_rel_err_m"), explains=p.get("explains")) for p in (ev or {}).get("pairs") or []],
                semantic_score=None if s is None else s.get("alignment_score"), semantic_convergence=None if s is None else s.get("convergence"), scale_rel=scale_rel, crossing_margin=xm, e7_best=(res or {}).get("best"), e7_gap_m=(res or {}).get("gap_m"), rivals=rivals, gap_m=gap_m, xw_veto=xw_veto)


def fix_allowed(rec):
    """R38: may this record become a FIX? WOULD_FIX now requires evidence that does not depend on the room digits (see evaluate). R37's rule (OCR and VLM agree on both plates) is gone: it confirmed 505 + 502. -> (bool, why)"""
    if not rec or rec.get("verdict") != "WOULD_FIX":
        return False, f"verdict {None if not rec else rec.get('verdict')}, blocked by {None if not rec else rec.get('blocked_by')}, rejected by {None if not rec else rec.get('rejected_by')}"
    return True, f"rooms {[x['label'] for x in rec.get('room_readings') or []]} on {rec.get('floors_explaining')}F, floor evidence {rec.get('floor_evidence')}, odometry consistent, LiDAR converged, no contradiction"


def write(state, rec):
    state = Path(state)
    tmp = state / f".e7_semantic_shadow.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, state / "e7_semantic_shadow.json")
    with open(state / "e7_semantic_shadow.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
