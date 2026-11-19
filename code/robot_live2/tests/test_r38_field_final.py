"""R38: the REAL loc2 run X20261102_052047 (405 at O004, 402 at O006, field OCR; field VLM values measured on the same photos, VLM_MEASUREMENT) through the REAL evaluate_group and REAL E6E7Link._run.
Hypotheses: real E2 + regional E2 + SEMANTIC_ANCHORED candidates of that run; wrong readings get their own anchored fit on the wrong floor. Semantic policy forced ENABLED here (env-free file) to see what it WOULD write. GT only scores. OFFLINE_COUNTERFACTUAL. usage: test_r37_field_final.py OUT"""
import importlib.util, json, math, shutil, sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
L, E3 = RP.T.load(RP.STG); sys.path.insert(0, str(RP.STG))
import e6_semantic as ES, e8_pair as EP, semantic_anchor as SA, e7_policy as P7  # noqa
E7 = L.E7
RUN = "X20261102_052047"; X = RP.OPR / "experiments" / RUN; T0 = json.loads((X / "experiment.json").read_text())["created_wall"]
GT = json.loads((RP.OPR / "evaluation" / RUN / "gt.jsonl").read_text().splitlines()[0])
OBS = {json.loads(l)["observation_id"][-4:]: json.loads(l) for l in open(X / "state/e8_observations.jsonl") if json.loads(l).get("event") == "STORED"}
SF = ES.SemanticFloors(); ST = E3.Store(str(X / "archive/live2_store")); ST.refresh(); ROWS = list(ST.rows)
TE = float(sys.argv[2]) if len(sys.argv) > 2 else 660.0
PASS = [json.loads(l) for l in open(X / "state/e2/e2_passes.jsonl")]
def lf(f): return str(next(RP.LROOT.glob(f"kci_{f}F_*")))
CANDS = [dict(c, light=lf(p["floor"]), stage="exact", floor=p["floor"]) for p in PASS if p.get("kind") == "floor_done" and p["wall"] <= T0 + TE for c in p["candidates"]]
REG = [dict(c, light=lf(4), stage="exact", floor=4, provenance="E2_REGIONAL") for p in PASS if p.get("kind") == "region_done" for c in p.get("candidates") or []]
BETA = math.radians(float(np.median([math.degrees(c["beta"]) % 360 for c in CANDS])))
RES = {f: 1.0 / EP.floor_res(f, m)[0] for f, m in SF.maps.items()}
def anchored(floor, label, oid="O004"):
    m = SF.maps[floor]; sp = SF.spaces(floor, label)
    cor = [c["polygon_px"] for c in m.get("corridors") or [] if c.get("polygon_px")]
    fr = E7.frames(ST, float(OBS[oid]["capture_wall"]), SA.WINDOW_S, structural=True)
    top, _ = SA.scan(E7.cost, fr, E3.LightMap(lf(floor)), OBS[oid]["target_odom_link_xy"], SA.plate_wall_points(sp[0]["polygon_px"], cor, RES[floor]), RES[floor], BETA, 1, floor)
    return [dict(t["rec"], floor=floor, light=lf(floor), merit=0.26, id=f"A{i + 1}_F{floor}") for i, t in enumerate(top[:2])]
# what the VLM really read on the crops of these photos (VLM_MEASUREMENT.md, same prompt as the field): number and text
REAL_TEXT = dict(O004="로봇공학과 가나다", O006="로봇공학과", O008="로봇공학과 가나다")
def obs(oid, text, conf=.98, vlm=None, vtext=None, crop=True):
    v = None if vlm is None else dict(sign_type="room_number_plate", room_number=vlm, text=vtext if vtext is not None else "")
    return dict(OBS[oid], ocr_lines=[[text, conf]], vlm=v, reading=dict(vlm=dict(state="ASKED", crop=crop)), read=True, image_quality=dict(ok=True, flags=[]), image_sha256=f"sha_{oid}", experiment_id=RUN)
RESULTS = dict(run=RUN, t_end_rel=TE, kind="OFFLINE_COUNTERFACTUAL", cases=[], ok=True)
best = {}
for c in sorted(CANDS, key=lambda c: c["merit"]):
    best.setdefault(c["floor"], c)
def start_err(rec):
    px, py, h = E3.pose_in_map(dict(rec, light=None), ROWS[0][2], ROWS[0][3], ROWS[0][4])
    return round(math.hypot(px - GT["px"], py - GT["py"]) / 24.5, 2), round((math.degrees(h) - GT["yaw_deg"] + 180) % 360 - 180, 1)
CAT = {"TRUE": "TRUE_FIX", "WRONG": "FALSE_FIX"}
def category(r):
    if r["decision"] == "GLOBAL_CONFIRMED":
        return "TRUE_FIX" if r["truth"].startswith("TRUE") else "FALSE_FIX"
    if r["decision"] == "REJECTED" or (r.get("rejected") or []) or r["e8"] in ("MORE_OBSERVATION_REQUIRED",) and "INCONSISTENT" in r["why_e8"]:
        return "REJECTED"
    if r["decision"] == "AMBIGUOUS":
        return "AMBIGUOUS"
    return "INSUFFICIENT_EVIDENCE"
def case(name, group, floor, truth, out, label=None, extra=(), want=None, long_state=None):
    A = (anchored(floor, label) if label else []) + list(extra)
    hyps = [RP.T.Hyp(c.get("id") or f"R{i}_F4", dict(c), 1000 + i, light=c["light"]) for i, c in enumerate(A)] + [RP.T.Hyp(f"E2_{f}F", dict(c, floor=f), 2000 + f, light=c["light"]) for f, c in best.items() if f != floor]
    rows = [r for r in ROWS if r[1] <= T0 + TE]
    hy = [dict(id=h.id, floor=h.floor, rec=h.rec) for h in hyps]
    ev = EP.evaluate_group(group, hy, SF, rows, None)
    r = dict(name=name, truth=truth, e8=ev["verdict"], why_e8=str(ev.get("why"))[:120], candidates=[c["id"] for c in ev.get("candidates") or []], decision=None, rule=None)
    sd = out / name; shutil.rmtree(sd, ignore_errors=True); (sd / "e2").mkdir(parents=True)
    if ev["verdict"] == "REGION_CANDIDATE_READY":
        ids = {c["id"] for c in ev["candidates"]}
        (sd / "e2/e2_state.json").write_text(json.dumps(dict(odom_session=ST.session().get("odom_session_id"), candidates=CANDS, per_floor={str(c["floor"]): dict(light=c["light"], status="DONE") for c in CANDS}, floors_done=sorted({c["floor"] for c in CANDS})), default=float))
        cs = RP.CutStore(ST); cs.cut(T0 + TE)
        tr = RP.Tracker(cs, hyps, sd / "e2", ST.session().get("odom_session_id")); tr.E3 = E3; tr.experiment_id = RUN
        link = L.E6E7Link(sd, tr, sem_floors=SF, policy="SINGLE_DIAGNOSTIC"); link.max_odom_age_s = None; link.min_windows = 1   # the rule table of one window; the two-window rule has its own test
        if long_state:
            (sd / "e7_long.json").write_text(json.dumps(long_state))
        link.pair = types.SimpleNamespace(last_eval=ev, group=group, on_e7=lambda o: None, snapshot=lambda: None)
        link.fastlio_degraded = lambda *a, **k: (False, None)
        cons = [h for h in hyps if h.id in ids]; riv = [h for h in hyps if h.id not in ids][:L.E7_RIVALS]
        link._run([dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:3] + riv], dict(observation_id=f"T_{name}", source="E8_PAIR", pair_observation_ids=[o["observation_id"] for o in group], consistent_ids=sorted(ids)), cons[0].id)
        d = json.loads((sd / "e7_state.json").read_text()); sh = json.loads((sd / "e7_semantic_shadow.json").read_text())
        win = next((h for h in hyps if h.id == d["semantic_hypothesis"]), None)
        r.update(decision=d["global_decision"], path=d.get("path"), hyp=d["semantic_hypothesis"], floor=win.floor if win else None, start_err=start_err(win.rec) if win and win.floor == 4 else None, best=d["e7"].get("best"), gap=d["e7"].get("gap_m"), shadow=sh["verdict"], blocked=sh["blocked_by"], rejected=sh["rejected_by"], evidence=sh.get("floor_evidence"), rule=d.get("semantic_rule"), sec=d.get("e7_s"))
    shutil.rmtree(sd, ignore_errors=True)
    r["category"] = category(r)
    r["ok"] = r["category"] != "FALSE_FIX" and (want is None or r["category"] == want)
    RESULTS["ok"] &= r["ok"]; RESULTS["cases"].append(r)
    print(("PASS " if r["ok"] else "FAIL ") + json.dumps({k: r.get(k) for k in ("name", "category", "e8", "decision", "path", "floor", "start_err", "gap", "shadow", "blocked", "rejected", "evidence", "sec")}, ensure_ascii=False, default=str)[:520], flush=True)
    if r["e8"] != "REGION_CANDIDATE_READY":
        print("     e8:", r["why_e8"], flush=True)
def main(out):
    out = Path(out); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    P7.semantic_policy = lambda *a, **k: dict(mode="ENABLED", why="same as the enable file")
    A4 = anchored(4, "405"); E = A4 + REG[:6]
    T4, T6 = REAL_TEXT["O004"], REAL_TEXT["O006"]
    # real photos, real readings
    case("REAL_405_402_with_plate_text", [obs("O004", "405", .988, "405", T4), obs("O006", "402", .997, "402", T6)], 4, "TRUE", out, extra=E, want="TRUE_FIX")
    case("REAL_405_402_OCR_only", [obs("O004", "405", .988), obs("O006", "402", .997)], 4, "TRUE", out, extra=E, want="AMBIGUOUS")
    case("REAL_405_402_numbers_only_no_text", [obs("O004", "405", .988, "405"), obs("O006", "402", .997, "402")], 4, "TRUE", out, extra=E, want="AMBIGUOUS")
    case("REAL_405_402_text_from_whole_frame_only", [obs("O004", "405", .988, "405", T4, crop=False), obs("O006", "402", .997, "402", T6, crop=False)], 4, "TRUE", out, extra=E, want="AMBIGUOUS")
    case("SAME_PLATE_405_405", [obs("O004", "405", .988, "405", T4), obs("O008", "405", .955, "405", REAL_TEXT["O008"])], 4, "TRUE", out, extra=E)
    # the R37 failure: OCR and VLM misread the digit on both plates; the rest of each plate is what was really printed
    for f in (5, 7, 8, 3):
        case(f"MISREAD_{f}05_{f}02_OCR_VLM_both_wrong", [obs("O004", f"{f}05", .98, f"{f}05", T4), obs("O006", f"{f}02", .98, f"{f}02", T6)], f, "WRONG", out, label=f"{f}05")
        case(f"MISREAD_{f}05_{f}02_OCR_VLM_both_wrong_no_text", [obs("O004", f"{f}05", .98, f"{f}05"), obs("O006", f"{f}02", .98, f"{f}02")], f, "WRONG", out, label=f"{f}05")
        case(f"MISREAD_{f}05_{f}02_VLM_disagrees", [obs("O004", f"{f}05", .98, "405", T4), obs("O006", f"{f}02", .98, "402", T6)], f, "WRONG", out, label=f"{f}05")
    case("MIXED_FLOORS_505_402", [obs("O004", "505", .98, "505", T4), obs("O006", "402", .997, "402", T6)], 4, "WRONG", out, extra=E)
    wm = dict(OBS["O006"]); 
    case("WRONG_FLOOR_SIGN_5F_with_405_402", [obs("O004", "405", .988, "405", T4), obs("O006", "402", .997, "402", T6), dict(obs("O007", "5F", .95), observation_id="SIGN")], 4, "WRONG", out, extra=E)
    far = dict(obs("O006", "402", .997, "402", T6)); far["target_odom_link_xy"] = [far["target_odom_link_xy"][0] + 40.0, far["target_odom_link_xy"][1]]
    case("ODOMETRY_MISMATCH_plate_40m_off", [obs("O004", "405", .988, "405", T4), far], 4, "WRONG", out, extra=E)
    other = dict(obs("O006", "402", .997, "402", T6), experiment_id="X_OTHER_RUN")
    case("OTHER_EXPERIMENT_observation", [obs("O004", "405", .988, "405", T4), other], 4, "WRONG", out, extra=E)
    same = dict(obs("O006", "402", .997, "402", T6), image_sha256="sha_O004")
    case("SAME_IMAGE_used_twice", [obs("O004", "405", .988, "405", T4), same], 4, "WRONG", out, extra=E)
    wrong = sorted((c for c in CANDS if c["floor"] == 4), key=lambda c: c["merit"])[-1]
    case("WRONG_POSITION_4F", [obs("O004", "405", .988, "405", T4), obs("O006", "402", .997, "402", T6)], 4, "WRONG", out, extra=[dict(wrong, id="WRONGPOS")])
    # SYNTHETIC inputs on the real run's geometry: paths the real record does not contain
    case("SYNTH_405_402_separate_floor_sign_4F", [obs("O004", "405", .988, "405"), obs("O006", "402", .997, "402"), dict(obs("O007", "4F", .95), observation_id="SIGN")], 4, "TRUE", out, extra=E, want="TRUE_FIX")
    case("SYNTH_3_rooms_with_fair_lidar", [obs("O004", "405", .988, "405"), obs("O006", "402", .997, "402"), obs("O007", "403", .95, "403")], 4, "TRUE", out, extra=E, want="TRUE_FIX", long_state=dict(status="WOULD_CONFIRM", leader_floor=4, next_gap_m=0.07))
    case("SYNTH_3_rooms_lidar_not_separating", [obs("O004", "405", .988, "405"), obs("O006", "402", .997, "402"), obs("O007", "403", .95, "403")], 4, "TRUE", out, extra=E, want="AMBIGUOUS", long_state=dict(status="AMBIGUOUS_FLOORS", leader_floor=4))
    # known limit, reported: the digits AND the department / name all read as another floor. The record cannot tell this from really being on 5F: it is 5F in every field. Not counted in the pass rule
    r_ok = RESULTS["ok"]
    case("RESIDUAL_ALL_TEXT_SAYS_5F", [obs("O004", "505", .98, "505", "전기공학과 라마바"), obs("O006", "502", .98, "502", "전기공학과")], 5, "RESIDUAL", out, label="505")
    RESULTS["ok"] = r_ok
    (out / "result.json").write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False, default=str))
    import collections
    print(collections.Counter((c["truth"], c["category"]) for c in RESULTS["cases"]), flush=True)
    print("ALL_OK" if RESULTS["ok"] else "SOME_FAILED", flush=True)
    return RESULTS["ok"]
if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
