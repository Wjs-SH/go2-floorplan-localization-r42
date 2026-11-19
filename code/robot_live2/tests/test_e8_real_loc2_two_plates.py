"""STAGING (2026-11-01, r32): E8 recovery + 2nd observation on the REAL loc2 record X20261031_231940 (read-only).
REAL: LiveStore odometry, the O002 photo read by the REAL e6_ocr ('402'), the r32 single-reading region (room_402), the REGIONAL
E2 fit of the student e2_core in that region at the time the reading was done (+287.9 s), the real 4F semantic map, the real
O002 / O003 stop records (target, stop pose, session), the field E3 final hypotheses as rivals.
SIMULATED (never field evidence): the 2nd plate READING. Its label is the room that really lies at the real O003 target under
the GT-anchored transform (evaluation side: GT start pose + odometry; validated first: the same transform must put the REAL
O002 target on room_402). The real O003 photo was unreadable — this case asks "what would E8 do with a readable 2nd plate".
GT is used only for scoring and for choosing the simulated label; the E8 / E2 / E7 inputs never contain GT.
Run: python -s -B test_e8_real_loc2_two_plates.py OUT"""
import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(STG))
import e3_tracker as E3  # noqa: E402
import e6_core as G  # noqa: E402
import e6_semantic as S  # noqa: E402
import e8_pair as EP  # noqa: E402
import e6_ocr  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
X = OPR / "experiments/X20261031_231940"
STORE = X / "archive/live2_store"
GT = json.loads((OPR / "evaluation/X20261031_231940/gt.jsonl").read_text().splitlines()[0])
RES = dict(cases=[], ok=True)


def check(name, ok, **kw):
    RES["cases"].append(dict(name=name, ok=bool(ok), **kw)); RES["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, default=str)[:400], flush=True)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    st = E3.Store(str(STORE)); st.refresh(); rows = st.rows
    obs = {json.loads(l)["observation_id"][-4:]: json.loads(l) for l in open(X / "state/e8_observations.jsonl")
           if json.loads(l).get("event") == "STORED"}
    t_exp0 = json.loads((X / "experiment.json").read_text())["created_wall"]
    sf = S.SemanticFloors()
    room402 = next(s for s in sf.maps[4]["spaces"] if s.get("id") == "room_402")

    # ---- truth-anchored transform (EVALUATION ONLY)
    x0, y0, yaw0 = rows[0][2], rows[0][3], rows[0][4]
    # The odom yaw is NOT the travel direction in this frame (body-IMU odom_link: travel at -136 deg while yaw ~ 0; E3 absorbs it in
    # beta). So the rotation is taken from the MOTION: the robot left the GT start along the GT heading and drove straight to the
    # O002 stop (planar lateral -0.24 m over 4.13 m, i.e. within ~3.3 deg) -> phi such that that displacement points along GT yaw.
    row_at = lambda t: min(rows, key=lambda q: abs(q[1] - t))                    # noqa: E731
    r2 = row_at(obs["O002"]["odom_pose_at_stop"]["t"]); dv = np.array([r2[2] - x0, r2[3] - y0]); dv /= np.linalg.norm(dv)
    g = math.radians(GT["yaw_deg"]); want = np.array([math.cos(g), -math.sin(g)])
    phi_t = float(min(np.arange(0, 360, 0.1), key=lambda ph: np.linalg.norm(E3._A(math.radians(ph)) @ dv - want)))
    truth = dict(floor=4, pos=[GT["px"], GT["py"]], anchor=[x0, y0], phi=phi_t, ppm=23.96)
    d402 = S._poly_dist_px(room402["polygon_px"], np.array(G.map_projection(truth, *obs["O002"]["target_odom_link_xy"]))) / 23.96
    check("truth transform validated by the REAL reading: real O002 target lies on room_402", d402 <= S.PLACE_TOL_M, dist_m=round(d402, 2),
          place_tol_m=S.PLACE_TOL_M)

    def room_at(xy, exclude=()):
        p = np.array(G.map_projection(truth, *xy))
        best = min(((S._poly_dist_px(s["polygon_px"], p) / 23.96, s) for s in sf.maps[4]["spaces"]
                    if s.get("polygon_px") and str(s.get("id", "")).startswith("room_") and s.get("id") not in exclude), key=lambda x: x[0])
        return best[1], best[0]
    sp2, d2 = room_at(obs["O003"]["target_odom_link_xy"], exclude=("room_402",))
    lab2 = str(sp2.get("name") or sp2["id"]).split()[0]
    far = max((s for s in sf.maps[4]["spaces"] if str(s.get("id", "")).startswith("room_") and s.get("polygon_px")),
              key=lambda s: float(np.hypot(*(np.asarray(s["polygon_px"], float).mean(0) - np.asarray(room402["polygon_px"], float).mean(0)))))
    lab_far = str(far.get("name") or far["id"]).split()[0]
    print(json.dumps(dict(simulated_second_label=lab2, room=sp2["id"], room_dist_m=round(d2, 2), wrong_far_label=lab_far)), flush=True)

    # ---- REAL regional E2 (student fit) in the r32 room_402 region at the time the reading was done
    reg = json.loads((Path.home() / "sci_ws/reports/field_20261031_analysis/data/r32_single_region_real_loc2.json").read_text())[0]["region"]
    t_now = t_exp0 + 287.9
    spec = importlib.util.spec_from_file_location("e2w", STG / "e2_worker.py")
    W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
    W.SCI_LIVE = OPR / "candidate" / "external" / "sci_live"
    a = argparse.Namespace(building="kci", store=str(STORE), out=str(out / "e2"), cache_dir=str(OPR / "runtime/e2_cache"), floors=None,
                           procs=1, exact_dt=6.0, device="cpu", min_span=70.0, min_pass_gap_s=120.0, min_window_path_m=2.0,
                           wall_pass_top=4, mem_start_mb=3500.0, mem_kill_mb=2800.0, temp_stop_c=78.0, temp_hold_c=76.0,
                           early_window_s=20.0, coarse=1)
    (out / "e2").mkdir(exist_ok=True)
    FR = W.FloorRunner(a)
    t1 = time.time()
    r = FR.run_floor(4, float([q[1] for q in rows if q[1] <= t_now][-1]), FR.W, "exact",
                     region=dict(center_px=reg["center_px"], radius_px=reg["radius_px"]))
    cands = r["candidates"]

    def start_err(rec):
        px, py, hd = E3.pose_in_map(rec, x0, y0, yaw0)
        return round(math.hypot(px - GT["px"], py - GT["py"]) / 23.96, 2)
    errs = [start_err(c) for c in cands]
    check("REAL regional E2 around room_402 gives candidates; one within 1 m of the true start (scoring only)", cands and min(errs) <= 1.0,
          n=len(cands), fit_s=round(time.time() - t1, 1), start_err_m_by_rank=errs, best=min(errs) if errs else None)
    regional = [dict(id=f"R{i:02d}_F4", floor=4, rec=dict(c, floor=4)) for i, c in enumerate(cands)]
    b = cands[int(np.argmin(errs))]
    dd = []
    for k in ("O002", "O003", "O005"):
        rr = row_at(obs[k]["odom_pose_at_stop"]["t"])
        dd.append(round(float(np.hypot(*(np.array(G.map_projection(b, rr[2], rr[3])) - np.array(G.map_projection(truth, rr[2], rr[3]))))) / 23.96, 2))
    check("cross-check: the best REAL regional candidate and the evaluation transform agree at the real stops O002/O003/O005",
          max(dd) <= 3.0, stop_diff_m=dd, cand_phi=round(b["phi"], 1), truth_phi=round(truth["phi"], 1))

    # ---- rivals: the field E3 hypotheses at the end of the run (9F / 7F / 3F ...), rebuilt as test_loc_overlay does
    S3 = json.loads((X / "state/e3_state.json").read_text())
    rivals = []
    for h in S3["hypotheses"]:
        gp = h.get("global_pose")
        if not gp or h.get("status") not in ("ACTIVE", "SUSPENDED"):
            continue
        row = min(rows, key=lambda q: abs(q[1] - gp["odom_t"]))
        rivals.append(dict(id=h["id"], floor=h["floor"], rec=dict(pos=[gp["px"], gp["py"]], anchor=[row[2], row[3]], phi=h["phi"],
                                                                 ppm=h["ppm"], beta=math.radians(h["beta_deg"]), mirror=1, floor=h["floor"])))
    print(json.dumps(dict(rivals=len(rivals), rival_floors=sorted({h["floor"] for h in rivals}))), flush=True)

    # ---- observations
    for k in obs:                                         # archived with the experiment
        obs[k]["capture_file"] = str(X / "state/e6_captures" / Path(obs[k]["capture_file"]).name)
    lines, meta = e6_ocr.sign_read(__import__("cv2").imread(obs["O002"]["capture_file"]))
    check("REAL OCR of the field O002 photo reads '402' (one model load)", any(t == "402" for t, _ in lines), lines=lines, ocr_s=meta["ocr_s"])

    def O(key, label_lines, target=None, session=None, oid=None):
        o = dict(obs[key], read=True, ocr_lines=label_lines, vlm=None)
        o["observation_id"] = oid or o["observation_id"]
        if target is not None:
            o["target_odom_link_xy"] = list(target)
        if session is not None:
            o["session"] = session
        return o
    o1 = O("O002", lines)
    sim = lambda lab, **k: O("O003", [[lab, 0.99]], **k)                     # noqa: E731  SIMULATED 2nd reading
    H_all = regional + rivals

    def verdict(o2, hyps=H_all, lio=None):
        return EP.evaluate_group([o1, o2], hyps, sf, rows=rows, lio_series=lio or [])

    ev = verdict(sim(lab2))
    # diagnostics (evaluation side): raw stop poses are a DIFFERENT frame from the store rows — only targets are compared here
    (out / "regional_candidates.json").write_text(json.dumps(dict(region=reg, t_now=t_now, candidates=cands, start_err_m=errs), default=float))
    diag = []
    for key in ("O002", "O003"):
        o_ = obs[key]; sp_ = o_["odom_pose_at_stop"]
        rr = row_at(sp_["t"]); ptru = np.array(G.map_projection(truth, rr[2], rr[3])); pc = np.array(G.map_projection(b, rr[2], rr[3]))
        ttru = np.array(G.map_projection(truth, *o_["target_odom_link_xy"])); tc = np.array(G.map_projection(b, *o_["target_odom_link_xy"]))
        diag.append(dict(obs=key, stop_truth_px=ptru.round(1).tolist(), stop_cand_px=pc.round(1).tolist(), stop_diff_m=round(float(np.hypot(*(pc - ptru))) / 23.96, 2),
                         target_truth_px=ttru.round(1).tolist(), target_cand_px=tc.round(1).tolist(), target_diff_m=round(float(np.hypot(*(tc - ttru))) / 23.96, 2)))
    bid = f"R{int(np.argmin(errs)):02d}_F4"
    print(json.dumps(dict(best_candidate=bid, best_rec={k: b[k] for k in ("pos", "anchor", "phi", "ppm", "beta")}, diag=diag,
                          per_obs=[(r_["observation_id"][-4:], r_["per_hypothesis"].get(bid)) for r_ in ev["per_observation"]]), default=float), flush=True)
    cons = [c["id"] for c in ev.get("candidates") or []]
    cons_err = sorted(start_err(next(h["rec"] for h in H_all if h["id"] == i)) for i in cons)
    check(f"two different plates (402 real + {lab2} simulated) consistent -> REGION_CANDIDATE_READY on 4F (E7 next, not a fix)",
          ev["verdict"] == "REGION_CANDIDATE_READY" and ev["floors_explaining"] == [4] and all(i.startswith("R") for i in cons),
          verdict=ev["verdict"], why=ev["why"], consistent=cons, consistent_start_err_m=cons_err,
          pair=dict((k, ev["pairs"][0].get(k)) for k in ("target_separation_m", "explains")),
          odometry={k: ev["pairs"][0]["odometry"].get(k) for k in ("net_displacement_m", "path_length_m", "odom_rel_err_m", "ok")})
    # ---- PairManager plumbing: E7 is requested; E7 not passing -> no FIX
    tmp = out / "pm"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    req = []
    tr = types.SimpleNamespace(alive=lambda: [types.SimpleNamespace(id=h["id"], floor=h["floor"], rec=h["rec"], status="ACTIVE",
                                                                   light_dir=f"/L{h['floor']}", rank_key=lambda: 0) for h in H_all],
                               store=types.SimpleNamespace(refresh=lambda: None, rows=rows, session=lambda: {"odom_session_id": o1["session"]}),
                               apply_semantic=lambda *a_, **k_: 1, light=lambda d: d)
    pm = EP.PairManager(tmp, tr, sf, lambda c_, r_: req.append([h.id for h in c_]), reader=lambda img: ([], {}))
    pm.group, pm.gid = [o1, sim(lab2)], "PAIR_TEST"
    ev2 = pm.evaluate()
    check("PairManager on the same input requests E7 for the consistent regional candidate(s)", bool(req) and set(req[-1]) == set(cons),
          status=pm.status if hasattr(pm, "status") else None, e7_request=req[-1] if req else None)
    for dec in ("REJECTED", "AMBIGUOUS"):
        pm.on_e7(dict(observation_id=pm.gid, global_decision=dec, why=f"test: E7 {dec}"))
        check(f"E7 {dec} -> no FIX (group stays open for another observation)", pm.status == "MORE_OBSERVATION_REQUIRED", status=pm.status, why=pm.why)

    # ---- failure matrix (each must end in NO FIX / more observation)
    sp_near402, _ = room_at(obs["O002"]["target_odom_link_xy"])
    other_floor = next(lab for f in (7, 9, 5) for s in sf.maps[f]["spaces"] if str(s.get("id", "")).startswith("room_")
                       for lab in [str(s["id"]).split("_", 1)[1]] if lab.isdigit() and sf.spaces(f, lab) and not sf.spaces(4, lab))
    lio_low = [dict(wall=float(obs["O003"]["stopped_wall"]) - 30, hz=5.0)]
    t2 = np.asarray(obs["O002"]["target_odom_link_xy"]) + 0.5
    cases = [("2nd OCR read nothing", dict(sim(lab2), ocr_lines=[]), H_all, None),
             (f"2nd plate is a WRONG room ({lab_far}, far end of 4F)", sim(lab_far), H_all, None),
             ("same plate twice ('402' again 0.7 m away)", O("O002", lines, target=t2, oid="O002_again"), H_all, None),
             (f"2nd plate only exists on another floor ({other_floor})", sim(other_floor), H_all, None),
             ("regional E2 failed (no regional candidate, field rivals only)", sim(lab2), rivals, None),
             ("FAST-LIO session changed between the plates (jump / restart)", sim(lab2, session="fastlio_other_session"), H_all, None),
             ("FAST-LIO below rate between the plates", sim(lab2), H_all, lio_low)]
    pm2 = EP.PairManager(tmp / "b", tr, sf, lambda c_, r_: None, reader=lambda img: ([], {})) if (tmp / "b").mkdir() is None else None
    pm2.group, pm2.gid = [o1, sim(lab2)], "PAIR_TEST2"
    pm2.evaluate(); pm2.on_e7(dict(observation_id=pm2.gid, global_decision="GLOBAL_CONFIRMED", why="test: E7 confirmed"))
    check("plumbing reference: only an E7 GLOBAL_CONFIRMED closes the group as GLOBAL_FIX", pm2.status == "GLOBAL_FIX", status=pm2.status)
    for name, o2, hyps, lio in cases:
        e = verdict(o2, hyps, lio)
        check(f"{name} -> {e['verdict']} (no E7 request, no FIX)", e["verdict"] != "REGION_CANDIDATE_READY", verdict=e["verdict"], why=(e.get("why") or "")[:200])
    (out / "result.json").write_text(json.dumps(RES, indent=1, default=str))
    print("ALL_OK" if RES["ok"] else "SOME_FAILED", flush=True)
    return RES["ok"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
