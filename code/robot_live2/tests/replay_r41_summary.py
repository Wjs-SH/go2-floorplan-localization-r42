#!/usr/bin/env python3
"""R41: reads one replay's records and the run's GT (scoring only). Current-position reference: GT start pose + FAST-LIO delta, used only while the robot is within REF_M of the start, where a heading error of the reference cannot matter much. Elsewhere: no accuracy claim. usage: replay_r41_summary.py RUN TAG"""
import json, math, sys
from pathlib import Path
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
run, tag = sys.argv[1:3]; ROOT = Path.home() / "sci_ws_alt_imu_20261029_204740/runs/r41" / tag; S = ROOT / "state"
GT = json.loads((OPR / "evaluation" / run / "gt.jsonl").read_text().splitlines()[0]); PPM = 24.9; REF_M = 15.0
def jl(p):
    try:
        return [json.loads(l) for l in open(p)]
    except OSError:
        return []
TL = jl(ROOT / "replay_timeline.jsonl"); sh = TL[0]["shift"]; T0 = json.loads((OPR / "experiments" / run / "experiment.json").read_text())["created_wall"] + sh
T = lambda w: round(w - T0, 1)
print("##", run, tag, "code", TL[0]["code"][-40:], "vlm", TL[0]["vlm"])
for r in jl(S / "auto_transitions.jsonl"):
    if r["kind"] == "CLASS": print(" class", T(r["wall"]), r["frm"], "->", r["to"], "lead", r.get("leader_floor"), "riv", r.get("rival_floor"), "gap", r.get("gap"), (r.get("why") or "")[:70])
for r in jl(S / "e2/e2_passes.jsonl"):
    if r["kind"] in ("coarse_done", "exact_window", "exact_new_window", "pass_done", "child_stopped_memory", "child_stopped_thermal", "floor_failed", "child_died"): print(" e2", T(r["wall"]), r["kind"], {k: r.get(k) for k in ("floor", "requeued", "again", "n", "sec", "mem_mb") if r.get(k) is not None}, "" if r["kind"] != "exact_new_window" else ("window_end", T(r["t_now"])))
fd = [(T(r["wall"]), r["floor"], r.get("sec"), T(r["t_now"])) for r in jl(S / "e2/e2_passes.jsonl") if r["kind"] == "floor_done"]
print(" exact done (t, floor, sec, window_end)", fd)
for r in jl(S / "e7_long.jsonl"):
    print(" long", T(r["wall"]), r["status"], r.get("leader_floor"), "gap", r.get("next_gap_m"), "sec", r.get("sec"), [(x["floor"], x["stage"][0], round(x["cost"], 3), x.get("crossing")) for x in r["floors"][:4]], (r.get("why") or "")[:80])
for r in jl(S / "final_verify.jsonl"):
    print(" final", T(r["wall"]), r["status"], str(r.get("why"))[:120])
for r in TL:
    if r["kind"] in ("e6_inject", "e5_stop_request", "end_of_record"): print(" ", r)
for r in jl(S / "e8_observations.jsonl"):
    rd = r.get("reading") or {}
    print(" obs", T(r["wall"]), r["event"], (r.get("observation_id") or "")[-4:], "ocr", r.get("ocr_lines"), "vlm", (r.get("vlm") or {}).get("room_number"), (r.get("vlm") or {}).get("text"), (rd.get("vlm") or {}).get("state"), (rd.get("vlm") or {}).get("sec"), rd.get("ocr_s"), str((rd.get("vlm") or {}).get("why"))[:60])
for r in jl(S / "e8_pairs.jsonl"):
    print(" e8", T(r["wall"]) if r.get("wall") else "", r.get("kind")[9:], r.get("verdict"), r.get("global_decision"), (r.get("why") or "")[:130], "e7_s", r.get("e7_s"))
for r in jl(S / "e7_history.jsonl"):
    print(" e7", T(r["written_wall"]), r.get("global_decision"), r.get("path"), r.get("semantic_hypothesis"), str(r.get("why"))[:140], r.get("e7_s"))
for f in ("e7_state.json", "e7_semantic_shadow.json", "final_presentation.json", "fix_overlay.json"):
    try:
        d = json.loads((S / f).read_text())
    except (OSError, ValueError):
        print(" ", f, "none"); continue
    if f == "e7_state.json": print(" E7", d.get("global_decision"), d.get("path"), d.get("semantic_hypothesis"), T(d["written_wall"]), str(d.get("why"))[:160])
    if f == "e7_semantic_shadow.json": print(" shadow", d.get("verdict"), d.get("blocked_by"), d.get("rejected_by"), d.get("crossing_margin"), [x.get("kind") for x in d.get("floor_evidence") or []])
    if f == "fix_overlay.json": print(" fix_overlay", d.get("floor"), d.get("pose_m"), d.get("pose_px"), d.get("heading_deg"), d.get("correction"), d.get("shift_m"), d.get("why"))
# reference
tr = jl(OPR / "experiments" / run / "archive/field_eval/odom_trace.jsonl")
x0, y0, y0a = tr[0]["x"], tr[0]["y"], tr[0]["yaw_deg"]
move = next(r for r in tr if math.hypot(r["x"] - x0, r["y"] - y0) > 1.5)
# raw FAST-LIO (x, y) maps to image pixels (y down) by a pure rotation: odom_link negates x and yaw, and the student's _A is a reflection, so raw -> map (y up) is a reflection. So: pixel = GT + ppm * Rot(a) * delta_raw, heading(y up) = GT - delta_yaw_raw. a from the first steps: the robot starts walking along its GT heading.
a_px = math.atan2(-math.sin(math.radians(GT["yaw_deg"])), math.cos(math.radians(GT["yaw_deg"]))) - math.atan2(move["y"] - y0, move["x"] - x0)
near, lead_changes, last = [], [], None
E3 = [r for r in TL if r["kind"] == "e3" and r.get("top")]
for r in E3:
    if not r["odom"]: continue
    dx, dy = r["odom"][0] - x0, r["odom"][1] - y0; d = math.hypot(dx, dy)
    tx = GT["px"] + PPM * (dx * math.cos(a_px) - dy * math.sin(a_px)); ty = GT["py"] + PPM * (dx * math.sin(a_px) + dy * math.cos(a_px)); th = GT["yaw_deg"] - (r["odom"][2] - y0a)
    top = r["top"][0]; gp = top.get("global_pose") or {}
    best4 = None
    for h in r["top"]:
        g = h.get("global_pose") or {}
        if h["floor"] == GT["floor"] and g.get("px") is not None:
            e = math.hypot(g["px"] - tx, g["py"] - ty) / PPM
            if best4 is None: best4 = (h["id"], round(e, 1), round((g["heading_deg"] - th + 180) % 360 - 180, 1), h["score"])
    e_lead = None if gp.get("px") is None or top["floor"] != GT["floor"] else (round(math.hypot(gp["px"] - tx, gp["py"] - ty) / PPM, 1), round((gp["heading_deg"] - th + 180) % 360 - 180, 1))
    if top["id"] != last:
        last = top["id"]; lead_changes.append((r["t"], top["id"], round(top["score"], 3), top.get("ppm") and round(top["ppm"], 1)))
    if d <= REF_M:
        near.append(dict(t=r["t"], d=round(d, 1), lead=top["id"], lead_floor=top["floor"], lead_err=e_lead, best4F=best4, ref_px=[round(tx), round(ty)], ref_th=round(th)))
print(" leaders", lead_changes[-25:])
print(" CURRENT-pose vs reference, within", REF_M, "m of the start"); [print("   ", x) for x in near if x["t"] > 80][:40]
res = jl(S / "res_history.jsonl")
if res:
    print(" res temp max", max(x["temp"] for x in res), "mem min", min(x["mem_avail_mb"] for x in res), "kf_p95 max", max(x.get("kf_ms_p95") or 0 for x in res), "e3tick p", sorted(x.get("e3_tick_ms") or 0 for x in res)[int(.95*len(res))])
