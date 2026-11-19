"""R34 §11-5,6: 408 + TOILETS on the REAL loc2 record X20261101_193139 with the REAL evaluate_group and the REAL E6E7Link._run (E7 + all FIX gates).
Hypotheses: the run's real E2 candidates (e2_passes.jsonl) + SEMANTIC_ANCHORED candidates fitted with frames up to the reading only. The reading itself is what the case sets
(field OCR, or a wrong one for the counterexamples). GT only scores. OFFLINE_COUNTERFACTUAL; run: test_e8_anchor_real.py OUT"""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
L, E3 = RP.T.load(RP.STG)
sys.path.insert(0, str(RP.STG))
import e6_semantic as ES  # noqa: E402
import e8_pair as EP  # noqa: E402
import semantic_anchor as SA  # noqa: E402
_s = importlib.util.spec_from_file_location("live2_e7_rematch", RP.STG / "e7_rematch.py"); E7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(E7)
RUN = "X20261101_193139"
X = RP.OPR / "experiments" / RUN
T0 = json.loads((X / "experiment.json").read_text())["created_wall"]
OBS = {json.loads(l)["observation_id"][-4:]: json.loads(l) for l in open(X / "state/e8_observations.jsonl") if json.loads(l).get("event") == "STORED"}
GT = json.loads((RP.OPR / "evaluation" / RUN / "gt.jsonl").read_text().splitlines()[0])
SF = ES.SemanticFloors()
ST = E3.Store(str(X / "archive/live2_store")); ST.refresh()
PASS = [json.loads(l) for l in open(X / "state/e2/e2_passes.jsonl")]
FD = [p for p in PASS if p.get("kind") == "floor_done"]
CANDS = [dict(c, light=f"{RP.LROOT}/kci_{p['floor']}F_{c['map_version']}", H=None) for p in FD[:9] for c in p["candidates"]]
BETA = math.radians(float(np.median([math.degrees(c["beta"]) % 360 for c in CANDS])))
RES = dict(run=RUN, kind="OFFLINE_COUNTERFACTUAL", cases=[], ok=True)


def light_for(floor):
    return str(next(RP.LROOT.glob(f"kci_{floor}F_*")))


def anchored(oid, floor, room, t_rel):
    m = SF.maps[floor]
    sp = next(s for s in m["spaces"] if s["id"] == room)
    cor = [c["polygon_px"] for c in m.get("corridors") or [] if c.get("polygon_px")]
    res, _ = EP.floor_res(floor, m)
    fr = E7.frames(ST, T0 + t_rel, SA.WINDOW_S, structural=True)
    lm = E3.LightMap(light_for(floor))
    top, _n = SA.scan(E7.cost, fr, lm, OBS[oid]["target_odom_link_xy"], SA.plate_wall_points(sp["polygon_px"], cor, 1.0 / res), 1.0 / res, BETA, 1, floor)
    return [dict(t["rec"], floor=floor, merit=0.26, twoway=None, light=light_for(floor), provenance=SA.KIND, sep=t["separation_ratio"], cost=t["cost"], id=f"A{i + 1}_{room}") for i, t in enumerate(top)]


def obs(oid, text, conf=0.95):
    o = dict(OBS[oid], ocr_lines=[[text, conf]], vlm=None, read=True, image_quality=dict(ok=True, flags=["MANUAL_ACCEPTED"]))
    return o


def run_case(name, readings, anch, t_rel, out, rival_n=9):
    hyps = []
    for i, c in enumerate(anch):
        hyps.append(RP.T.Hyp(c["id"], dict(c, floor=c["floor"]), 1000 + i, light=c["light"]))
    best = {}
    for c in sorted(CANDS, key=lambda c: c["merit"]):
        best.setdefault(c["floor"], c)
    for f, c in best.items():
        hyps.append(RP.T.Hyp(f"E2_{f}F", dict(c, floor=f), 2000 + f, light=c["light"]))
    rows = [r for r in ST.rows if r[1] <= T0 + t_rel]
    hy = [dict(id=h.id, floor=h.floor, rec=h.rec) for h in hyps]
    ev = EP.evaluate_group(readings, hy, SF, rows, None)
    res = dict(name=name, readings=[(o["observation_id"][-4:], o["ocr_lines"]) for o in readings], verdict=ev["verdict"], why=str(ev.get("why"))[:300], n_valid=ev.get("n_valid_independent"),
               n_room=ev.get("n_room_readings"), floors=ev.get("floors_explaining"), candidates=[(c["id"], c["residuals_m"], c["rooms"]) for c in ev.get("candidates") or []],
               pairs=[dict(sep=p["target_separation_m"], odom_err=p["odometry"].get("odom_rel_err_m"), legs=p["odometry"].get("odom_legs_m"), path=p["odometry"].get("path_length_m"),
                           explains=p.get("explains")) for p in ev.get("pairs") or []], decision=None)
    if ev["verdict"] == "REGION_CANDIDATE_READY":
        sd = out / name; shutil.rmtree(sd, ignore_errors=True); sd.mkdir(parents=True)
        e2d = sd / "e2"; e2d.mkdir()
        (e2d / "e2_state.json").write_text(json.dumps(dict(odom_session=ST.session().get("odom_session_id"), candidates=CANDS, per_floor={str(c["floor"]): dict(light=c["light"], status="DONE") for c in CANDS},
                                                           floors_done=list(range(1, 10))), default=float))
        cs = RP.CutStore(ST); cs.cut(T0 + t_rel)
        tr = RP.Tracker(cs, hyps, e2d, ST.session().get("odom_session_id")); tr.E3 = E3
        link = L.E6E7Link(sd, tr, policy="SINGLE_DIAGNOSTIC")
        ids = {c["id"] for c in ev["candidates"]}
        cons = [h for h in hyps if h.id in ids]
        rivals = [h for h in hyps if h.id not in ids][:L.E7_RIVALS]
        link._run([dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:3] + rivals],
                  dict(observation_id=f"T_{name}", source="E8_PAIR", pair_observation_ids=[o["observation_id"] for o in readings]), cons[0].id)
        d = json.loads((sd / "e7_state.json").read_text())
        res.update(decision=d.get("global_decision"), decision_why=str(d.get("why"))[:300], e7=[(r["candidate_id"], r.get("level") or r.get("verdict"), r.get("cost"), r.get("gap")) for r in (d.get("e7") or {}).get("results", [])][:8])
        shutil.rmtree(sd, ignore_errors=True)
    RES["cases"].append(res)
    print(json.dumps(res, default=float)[:900], flush=True)
    return res


def start_err(rec):
    px, py, h = E3.pose_in_map(dict(rec, light=None), ST.rows[0][2], ST.rows[0][3], ST.rows[0][4])
    return round(math.hypot(px - GT["px"], py - GT["py"]) / 23.96, 2), round((math.degrees(h) - GT["yaw_deg"] + 180) % 360 - 180, 1)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    a408, a405, a508 = anchored("O003", 4, "room_408", 521.0), anchored("O003", 4, "room_405", 521.0), anchored("O003", 5, "room_508", 521.0)
    RES["anchor_408"] = [dict(id=c["id"], phi=round(c["phi"], 1), cost=round(c["cost"], 4), sep=c["sep"], start_err=start_err(c)) for c in a408]
    print(RES["anchor_408"], flush=True)
    t = 930.0
    TRUE = run_case("TRUE_408_TOILETS", [obs("O003", "408", 0.997), obs("O006", "TOILETS", 0.935)], a408, t, out)
    A = run_case("MISREAD_405_TOILETS", [obs("O003", "405"), obs("O006", "TOILETS", 0.935)], a405, t, out)
    B = run_case("MISREAD_508_TOILETS_OTHER_FLOOR", [obs("O003", "508"), obs("O006", "TOILETS", 0.935)], a508, t, out)
    C = run_case("TOILETS_ONLY_NO_ROOM", [obs("O006", "TOILETS", 0.935)], [], t, out)
    D = run_case("TWO_CATEGORIES_NO_ROOM", [obs("O003", "TOILETS", 0.935), obs("O006", "WC", 0.9)], [], t, out)
    E = run_case("408_THEN_FAR_ROOM_WRONG", [obs("O003", "408", 0.997), obs("O006", "401")], a408, t, out)
    F = run_case("SINGLE_408_ONLY", [obs("O003", "408", 0.997)], a408, 521.0, out)
    o2 = obs("O006", "TOILETS", 0.935); o2["session"] = "OTHER"
    G = run_case("SESSION_CHANGED", [obs("O003", "408", 0.997), o2], a408, t, out)
    bad = [x["name"] for x in (A, B, C, D, E, F, G) if x["decision"] == "GLOBAL_CONFIRMED" or x["verdict"] == "GLOBAL_FIX"]
    RES["false_fix"] = bad
    RES["ok"] = not bad
    RES["true_case"] = dict(verdict=TRUE["verdict"], decision=TRUE["decision"])
    (out / "result.json").write_text(json.dumps(RES, indent=1, default=float))
    print("FALSE_FIX", bad, "TRUE", TRUE["verdict"], TRUE["decision"], flush=True)
    return RES["ok"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
