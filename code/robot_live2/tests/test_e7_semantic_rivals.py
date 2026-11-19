"""STAGING ONLY (2026-10-30): PATH B (semantic -> E8 pair -> E7) rival completion on the REAL loc1 record, deterministic.
Calls E6E7Link._run (the E7 job of an E8 pair request) directly with the real LiveStore cut at a fixed record time; the
semantic reading itself is NOT simulated — the test only fixes which hypothesis the reading points to (sem_id) and checks what
E7 + the FIX gates decide. Operational code vs staging code on the same inputs.
  SEM_OK   reading -> true 4F leader, junction window (+120 s)                 staging: GLOBAL_CONFIRMED expected (still possible)
  MISREAD  reading -> 7F hypothesis, the 4F geometry merged out of E3          operational: recorded (the defect if it confirms)
                                                                              staging: never GLOBAL_CONFIRMED (E2 4F rival restored)
  STRAIGHT reading -> true 4F leader, final straight/rest window (+204.6 s)    recorded (structure alone cannot separate it)
  NO_E2    reading -> true 4F leader, junction window, no E2 list              staging: never GLOBAL_CONFIRMED (fail closed)
Run: python3 -s -B test_e7_semantic_rivals.py OUT_DIR"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
OPC = RP.OPR / "candidate/live2"


def case(code_dir, name, sem_id, alive_ids, t_rel, out, e2_mode="real"):
    L, E3 = RP.T.load(code_dir)
    st, e2 = RP.record("loc1")
    D = json.loads(Path(RP.REC["loc1"]["recon"]).read_text())
    hyps = [RP.T.Hyp(c["candidate_id"], dict(c["rec"], floor=c["floor"]), i, light=c["rec"]["light"])
            for i, c in enumerate(D["cands"]) if c["candidate_id"] in alive_ids]
    sd = out / name; sd.mkdir(parents=True)
    e2dir = sd / "e2"; e2dir.mkdir()
    if e2_mode == "real":
        (e2dir / "e2_state.json").write_text(json.dumps(e2, default=float))
    cs = RP.CutStore(st)
    tr = RP.Tracker(cs, hyps, e2dir, e2["odom_session"]); tr.E3 = E3
    cs.cut(st.rows[0][1] + t_rel)
    link = L.E6E7Link(sd, tr, policy="SINGLE_DIAGNOSTIC")
    cons = [h for h in hyps if h.id == sem_id]
    rivals = [h for h in sorted(hyps, key=lambda h: h.rank_key()) if h not in cons][:L.E7_RIVALS]
    cands = [dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:3] + rivals]
    link._run(cands, dict(observation_id=f"T_{name}", source="E8_PAIR", pair_observation_ids=["T_a", "T_b"]), sem_id)
    o = link.last or {}
    e7 = o.get("e7") or {}
    return dict(case=name, code="operational" if code_dir == OPC else "staging", sem_id=sem_id, t_rel=t_rel, e2_mode=e2_mode,
                global_decision=o.get("global_decision"), why=o.get("why"), best=e7.get("best"), runner_up=e7.get("runner_up"),
                gap=e7.get("gap_m"), e2_rivals=e7.get("e2_rivals"))


def main(out):
    out = Path(out); out.mkdir(parents=True)
    ALL = {"H025_F4", "H047_F7", "H003_F2", "H122_F3"}
    NO4 = ALL - {"H025_F4"}
    C = [(RP.STG, "SEM_OK_staging", "H025_F4", ALL, 120.0, "real", lambda r: r["global_decision"] == "GLOBAL_CONFIRMED"),
         (OPC, "MISREAD_operational", "H047_F7", NO4, 120.0, "real", lambda r: True),
         (RP.STG, "MISREAD_staging", "H047_F7", NO4, 120.0, "real", lambda r: r["global_decision"] != "GLOBAL_CONFIRMED"),
         (RP.STG, "STRAIGHT_staging", "H025_F4", ALL, 204.6, "real", lambda r: True),
         (RP.STG, "NO_E2_staging", "H025_F4", ALL, 120.0, "missing", lambda r: r["global_decision"] != "GLOBAL_CONFIRMED")]
    R = []
    for code, name, sem, alive, t, e2m, want in C:
        r = case(code, name, sem, alive, t, out, e2m)
        r["expected"] = bool(want(r)); R.append(r)
        print(json.dumps(r, default=str), flush=True)
    ok = all(r["expected"] for r in R)
    (out / "e7_semantic_rivals_result.json").write_text(json.dumps(dict(results=R, all_ok=ok), indent=1, default=str))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
