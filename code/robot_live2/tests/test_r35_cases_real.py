"""R35 §2.3, §3, §10: the real loc2 record X20261101_193139 through the REAL evaluate_group and the REAL E6E7Link._run (E7, FIX gates, new structural class, new shadow). The reading is what the case sets (field OCR, the true plate, or a misread). OFFLINE_COUNTERFACTUAL.
Pass rule: no GLOBAL_CONFIRMED on any wrong case; the shadow never writes e7_state; every wrong case is not WOULD_FIX unless it is listed as a known residual risk. Run: test_r35_cases_real.py OUT"""
import importlib.util, json, math, shutil, sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
L, E3 = RP.T.load(RP.STG); sys.path.insert(0, str(RP.STG))
import e6_semantic as ES, e8_pair as EP, semantic_anchor as SA  # noqa: E402,E401
E7 = L.E7
RUN = "X20261101_193139"; X = RP.OPR / "experiments" / RUN; T0 = json.loads((X / "experiment.json").read_text())["created_wall"]
OBS = {json.loads(l)["observation_id"][-4:]: json.loads(l) for l in open(X / "state/e8_observations.jsonl") if json.loads(l).get("event") == "STORED"}
SF = ES.SemanticFloors(); ST = E3.Store(str(X / "archive/live2_store")); ST.refresh()
FD = [json.loads(l) for l in open(X / "state/e2/e2_passes.jsonl") if '"floor_done"' in l][:9]
CANDS = [dict(c, light=f"{RP.LROOT}/kci_{p['floor']}F_{c['map_version']}", stage="exact") for p in FD for c in p["candidates"]]
BETA = math.radians(float(np.median([math.degrees(c["beta"]) % 360 for c in CANDS])))
RES = {f: 1.0 / EP.floor_res(f, m)[0] for f, m in SF.maps.items()}
def lf(f): return str(next(RP.LROOT.glob(f"kci_{f}F_*")))
def anchored(floor, label):
    m = SF.maps[floor]; sp = SF.spaces(floor, label)
    cor = [c["polygon_px"] for c in m.get("corridors") or [] if c.get("polygon_px")]
    fr = E7.frames(ST, T0 + 521.0, SA.WINDOW_S, structural=True)
    top, _ = SA.scan(E7.cost, fr, E3.LightMap(lf(floor)), OBS["O003"]["target_odom_link_xy"], SA.plate_wall_points(sp[0]["polygon_px"], cor, RES[floor]), RES[floor], BETA, 1, floor)
    return [dict(t["rec"], floor=floor, light=lf(floor), merit=0.26, id=f"A{i + 1}_F{floor}") for i, t in enumerate(top[:2])]
def obs(oid, text, conf=.95, vlm=None, extra=None):
    return dict(OBS[oid], ocr_lines=[[text, conf]] + (extra or []), vlm=vlm, read=True, image_quality=dict(ok=True, flags=[]), image_sha256=f"sha_{oid}")
T = lambda: obs("O006", "TOILETS", .935)
best = {}
for c in sorted(CANDS, key=lambda c: c["merit"]):
    best.setdefault(c["floor"], c)
RESULTS = dict(run=RUN, kind="OFFLINE_COUNTERFACTUAL", cases=[], ok=True)
# the decision R34 would have given (operational rollback copy of the old rule)
def old_decision(res, sem_id):
    if not res.get("results"): return "MORE_OBSERVATION_REQUIRED"
    if res.get("best") != sem_id: return "REJECTED"
    if res.get("final_decision") == "GLOBAL_CONFIRMED": return "GLOBAL_CONFIRMED"
    return "AMBIGUOUS" if res.get("ambiguity") else "MORE_OBSERVATION_REQUIRED"
def case(name, group, floor, label, truth, out, sem_rec=None, residual_ok=False):
    A = [sem_rec] if sem_rec else anchored(floor, label)
    hyps = [RP.T.Hyp(c["id"], dict(c), 1000 + i, light=c["light"]) for i, c in enumerate(A)] + [RP.T.Hyp(f"E2_{f}F", dict(c, floor=f), 2000 + f, light=c["light"]) for f, c in best.items() if f != floor]
    rows = [r for r in ST.rows if r[1] <= T0 + 930.0]
    hy = [dict(id=h.id, floor=h.floor, rec=h.rec) for h in hyps]
    ev = EP.evaluate_group(group, hy, SF, rows, None)
    res = dict(name=name, truth=truth, readings=[o["ocr_lines"][0][0] for o in group], e8=ev["verdict"], candidates=[c["id"] for c in ev.get("candidates") or []], old=None, decision=None, cls=None, shadow=None, blocked=None, residual_ok=residual_ok)
    sd = out / name; shutil.rmtree(sd, ignore_errors=True); sd.mkdir(parents=True); (sd / "e2").mkdir()
    if ev["verdict"] == "REGION_CANDIDATE_READY" or sem_rec:
        sem = (ev.get("candidates") or [dict(id=A[0]["id"])])[0]["id"]
        (sd / "e2/e2_state.json").write_text(json.dumps(dict(odom_session=ST.session().get("odom_session_id"), candidates=CANDS, per_floor={str(c["floor"]): dict(light=c["light"], status="DONE") for c in CANDS}, floors_done=list(range(1, 10))), default=float))
        cs = RP.CutStore(ST); cs.cut(T0 + 930.0)
        tr = RP.Tracker(cs, hyps, sd / "e2", ST.session().get("odom_session_id")); tr.E3 = E3
        link = L.E6E7Link(sd, tr, sem_floors=SF, policy="SINGLE_DIAGNOSTIC")
        link.pair = types.SimpleNamespace(last_eval=ev, group=group, on_e7=lambda o: None, snapshot=lambda: None)
        ids = {c["id"] for c in ev.get("candidates") or []} or {sem}
        cons = [h for h in hyps if h.id in ids]; riv = [h for h in hyps if h.id not in ids][:L.E7_RIVALS]
        link._run([dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:3] + riv], dict(observation_id=f"T_{name}", source="E8_PAIR", pair_observation_ids=[o["observation_id"] for o in group]), sem)
        d = json.loads((sd / "e7_state.json").read_text()); sh = json.loads((sd / "e7_semantic_shadow.json").read_text())
        res.update(old=old_decision(d["e7"], sem), decision=d["global_decision"], cls=d.get("structural_class"), why=str(d.get("why"))[:170], best=d["e7"].get("best"), gap=d["e7"].get("gap_m"), shadow=sh["verdict"], blocked=sh["blocked_by"] + sh["rejected_by"], would2=sh["would_fix_2room"], wouldnx=sh["would_fix_no_crossing"], scale=sh["scale_rel"], cross=sh["crossing_margin"], conv=sh["semantic_convergence"])
    else:
        import e7_semantic_shadow as SH
        sh = SH.evaluate(ev, group, None, None, None, .05, .10, None)
        res.update(shadow=sh["verdict"], blocked=sh["blocked_by"] + sh["rejected_by"], why=str(ev.get("why"))[:170])
    shutil.rmtree(sd, ignore_errors=True)
    bad = truth == "WRONG" and (res["decision"] == "GLOBAL_CONFIRMED" or (res["shadow"] == "WOULD_FIX" and not residual_ok))
    res["ok"] = not bad
    RESULTS["ok"] &= res["ok"]; RESULTS["cases"].append(res)
    print(("PASS " if res["ok"] else "FAIL ") + json.dumps({k: res.get(k) for k in ("name", "e8", "old", "decision", "cls", "best", "gap", "shadow", "would2", "wouldnx", "conv", "scale", "cross", "blocked")}, default=str)[:520], flush=True)
    return res
def main(out):
    out = Path(out); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    r408 = lambda: obs("O003", "408", .997); r405 = lambda: obs("O002", "405", .982)
    case("TRUE_408_ONLY", [r408()], 4, "408", "TRUE", out)
    case("TRUE_408_T", [r408(), T()], 4, "408", "TRUE", out)
    case("TRUE_405_408", [r405(), r408()], 4, "408", "TRUE", out)
    case("TRUE_405_408_T", [r405(), r408(), T()], 4, "408", "TRUE", out)
    case("TRUE_405_407_408_T", [r405(), obs("O005", "407"), r408(), T()], 4, "408", "TRUE", out)
    case("TRUE_405_408_T_marker4F", [r405(), r408(), obs("O006", "TOILETS", .935, extra=[["4F", .9]])], 4, "408", "TRUE", out)
    case("TRUE_405_408_T_vlm", [obs("O002", "405", .982, vlm=dict(room_number="405", sign_type="room")), r408(), T()], 4, "408", "TRUE", out)
    for f in (5, 7):
        case(f"MISREAD_{f}08_T", [obs("O003", f"{f}08"), T()], f, f"{f}08", "WRONG", out)
        case(f"MISREAD_{f}05_{f}08_T", [obs("O002", f"{f}05"), obs("O003", f"{f}08"), T()], f, f"{f}08", "WRONG", out)
    case("MISREAD_505_507_508_T_triple", [obs("O002", "505"), obs("O005", "507"), obs("O003", "508"), T()], 5, "508", "WRONG", out, residual_ok=True)
    case("MISREAD_505_508_T_marker4F_conflict", [obs("O002", "505"), obs("O003", "508"), obs("O006", "TOILETS", .935, extra=[["4F", .9]])], 5, "508", "WRONG", out)
    case("MIX_505_408_T", [obs("O002", "505"), r408(), T()], 4, "408", "WRONG", out)
    wrong = sorted((c for c in CANDS if c["floor"] == 4), key=lambda c: c["merit"])[1]
    case("WRONG_POSITION_4F", [r405(), r408(), T()], 4, "408", "WRONG", out, sem_rec=dict(wrong, id="WRONGPOS", light=wrong["light"]))
    case("SESSION_CHANGED", [r408(), dict(T(), session="OTHER")], 4, "408", "TRUE", out)
    (out / "result.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    print("ALL_OK" if RESULTS["ok"] else "SOME_FAILED", flush=True)
    return RESULTS["ok"]
if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
