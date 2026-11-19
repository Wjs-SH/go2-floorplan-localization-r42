"""STAGING (2026-10-30 21:xx): E7 PATH A with RETRO windows on the REAL loc1 / loc2 LiveStores, deterministic replay clock.
E4 CONFIDENT times are the MEASURED times of the operational replays (store time), never moved earlier:
  loc1 GPU replay (A_gpu_loc1)   +154.6 s, leader H025_F4
  loc1 CPU-3 replay (A_p3_loc1)  +202.6 s (leader H039_F4 = the same 4F place; the recon transform H025_F4 is used)
  loc1 CPU-2 replay (baseline)   +250.7 s (the store ends at +204.6 s: no data after CONFIDENT)
loc2 never reached CONFIDENT: there E4 is FORCED for each leader as a NEGATIVE control (it can only add false-FIX chances).
The retro job's run time is modelled as RETRO_LAT_S of record time (the robot's data keeps arriving meanwhile); live windows
that ended meanwhile are caught up from the store, as in the field. Candidate transforms: final E3 hypotheses (data/recon_*.json).
Run: python3 -s -B test_e7_final_replay.py OUT_DIR"""
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
assert RP.STG == HERE.parents[0] / "candidate/live2" or RP.STG.resolve() == (HERE.parent / "candidate/live2").resolve()
L, E3, P7 = RP.L, RP.E3, RP.P7
sys.path.insert(0, str(RP.STG))          # spawn workers (e7_retro) import e6e7_link / e3_tracker from here
BENCH = RP.BENCH
PASSES = dict(loc1=BENCH / "baseline_loc1/state/e2/e2_passes.jsonl", loc2=BENCH / "baseline_loc2/state/e2/e2_passes.jsonl")
RETRO_LAT_S = 60.0


def run(name, rec_name, leader_id, t_conf, out, mode="ENABLED", e2_mode="real", hyp_ids=None, jump_at=None, retro_lat=RETRO_LAT_S,
        live_lat=10.0):
    st, e2 = RP.record(rec_name)
    D = json.loads(Path(RP.REC[rec_name]["recon"]).read_text())
    cands = [c for c in D["cands"] if hyp_ids is None or c["candidate_id"] in hyp_ids]
    hyps = [RP.T.Hyp(c["candidate_id"], dict(c["rec"], floor=c["floor"]), i, light=c["rec"]["light"]) for i, c in enumerate(cands)]
    sd = out / name; sd.mkdir(parents=True)
    e2dir = sd / "e2"; e2dir.mkdir()
    if e2_mode == "real":
        (e2dir / "e2_state.json").write_text(json.dumps(e2, default=float))
        shutil.copy(PASSES[rec_name], e2dir / "e2_passes.jsonl")
    elif e2_mode == "twin":
        lead = next(c for c in D["cands"] if c["candidate_id"] == leader_id)
        twin = dict({k: v for k, v in lead["rec"].items() if k != "light"}, floor=7, stage="exact", merit=1.0)
        (e2dir / "e2_state.json").write_text(json.dumps(dict(odom_session=e2["odom_session"], candidates=[twin], t_now=e2.get("t_now"),
                                                             window_s=e2.get("window_s"),
                                                             per_floor={"7": dict(light=lead["rec"]["light"], H=1825)}), default=float))
        shutil.copy(PASSES[rec_name], e2dir / "e2_passes.jsonl")
    cs = RP.CutStore(st)
    if jump_at is not None:                                # a 3 m odometry jump at record time t0 + jump_at (copy of the rows)
        t0 = st.rows[0][1]
        rows = [list(r) for r in st.rows]
        for r in rows:
            if r[1] >= t0 + jump_at:
                r[2] += 3.0
        cs.st = type("S", (), {})()
        cs.st.rows = [tuple(r) for r in rows]
        for k in ("root", "points", "session"):
            setattr(cs.st, k, getattr(st, k))
        cs.st._structural = getattr(st, "_structural", None)
    tr = RP.Tracker(cs, hyps, e2dir, e2["odom_session"])
    t0, tN = st.rows[0][1], st.rows[-1][1]
    cs.cut(t0 + t_conf)
    link = L.E6E7Link(sd, tr, policy="SINGLE_DIAGNOSTIC")
    if mode == "ENABLED":
        pf = sd / "e7_fix_policy.json"
        pf.write_text(json.dumps(dict(structure_fix="ENABLED", verification_report="test_e7_final_replay (harness only)",
                                      code_sha256=P7.code_sha256())))
        link.struct_policy = P7.structure_policy(policy_file=pf, env={})
    elif mode == "SHADOW":
        link.struct_policy = P7.structure_policy(policy_file=sd / "absent.json", env={})
    else:
        link.struct_policy = P7.structure_policy(policy_file=sd / "absent.json", env={"LIVE2_STRUCTURE_FIX": "0"})
    assert link.struct_policy["mode"] == mode, link.struct_policy
    leader = next(h for h in hyps if h.id == leader_id)
    e4 = dict(state="CONFIDENT_GEOMETRY", metrics=dict(best_id=leader.id, best_floor=leader.floor, gap=0.2))
    t, sa, wall0 = t0 + t_conf, None, time.time()
    t_stop = max(tN, t0 + t_conf) + 120.0                  # after the record: the clock runs on, no new data
    while t <= t_stop:
        cs.cut(min(t, tN))
        sa = link.structure_tick(e4, t, stop=None)
        if link.job and link.job.is_alive():
            retro = sa is not None and sa.get("phase") == "RETRO_VERIFY"
            link.job.join()
            t += retro_lat if retro else live_lat
        if sa is None or sa.get("status") not in (None, "RUNNING"):
            break
        t += 1.0
    if (sd / "structure_path.json").exists():
        sa = json.loads((sd / "structure_path.json").read_text())
    e7 = json.loads((sd / "e7_state.json").read_text()) if (sd / "e7_state.json").exists() else {}
    rel = (lambda x: None if x is None else round(x - t0, 1))
    return dict(scenario=name, record=rec_name, leader=leader_id, t_conf=t_conf, mode=mode, e2_mode=e2_mode,
                status=None if sa is None else sa.get("status"), why=None if sa is None else sa.get("why"),
                global_decision=e7.get("global_decision"), evidence=None if sa is None else sa.get("evidence"),
                retro=None if sa is None else {k: v for k, v in (sa.get("retro") or {}).items() if k != "e2_input_windows"},
                passes=[] if sa is None else [dict(t_rel=rel(p.get("t_end")), ok=p.get("ok"), best=p.get("best"), gap=p.get("gap_m"),
                                                   skipped=p.get("skipped"), retro=p.get("retro"), live=p.get("live"), path_m=p.get("path_m"))
                                              for p in sa.get("passes", [])],
                bench_s=round(time.time() - wall0, 1))


def main(out):
    out = Path(out); out.mkdir(parents=True)
    pol = dict(default=P7.structure_policy(env={}), env0=P7.structure_policy(env={"LIVE2_STRUCTURE_FIX": "0"}))
    R = dict(policy={k: {kk: v.get(kk) for kk in ("mode", "enabled", "why")} for k, v in pol.items()}, runs=[])
    pol_ok = pol["default"]["mode"] == "SHADOW" and not pol["default"]["enabled"] and pol["env0"]["mode"] == "OFF"
    nofix = lambda r: r["status"] not in ("CONFIRMED", "SHADOW_WOULD_CONFIRM") and r["global_decision"] != "GLOBAL_CONFIRMED"   # noqa: E731
    S = [("REAL_GPU_loc1_true_ENABLED_rep0", "loc1", "H025_F4", 154.6, dict(), lambda r: True),
         ("REAL_GPU_loc1_true_ENABLED_rep1", "loc1", "H025_F4", 154.6, dict(), lambda r: True),
         ("REAL_GPU_loc1_true_SHADOW", "loc1", "H025_F4", 154.6, dict(mode="SHADOW"), lambda r: r["global_decision"] != "GLOBAL_CONFIRMED"),
         ("REAL_CPU3_loc1_true_ENABLED", "loc1", "H025_F4", 202.6, dict(), lambda r: True),
         ("REAL_CPU2_loc1_true_ENABLED", "loc1", "H025_F4", 250.7, dict(), lambda r: True),
         ("OFF_env0_loc1", "loc1", "H025_F4", 154.6, dict(mode="OFF"), lambda r: r["status"] is None),
         ("FC_no_e2_loc1", "loc1", "H025_F4", 154.6, dict(e2_mode="missing"), nofix),
         ("TWIN_hidden_loc1", "loc1", "H025_F4", 154.6, dict(e2_mode="twin"), nofix),
         ("JUMP_at95_loc1", "loc1", "H025_F4", 154.6, dict(jump_at=95.0), lambda r: True)]
    S += [(f"W_REAL_GPU_loc1_{h}", "loc1", h, 154.6, dict(), nofix) for h in ("H047_F7", "H003_F2", "H122_F3")]
    S += [("W_REAL_GPU_loc1_7F_e3_rivals_removed", "loc1", "H047_F7", 154.6, dict(hyp_ids={"H047_F7"}), nofix)]
    S += [(f"W_FORCED_loc2_{h}_t{tc:.0f}", "loc2", h, tc, dict(), nofix)
          for h in ("H146_F4", "H136_F3", "H147_F4", "H141_F3") for tc in (150.0, 235.0)]
    for name, rec, lead, tc, kw, want in S:
        r = run(name, rec, lead, tc, out, **kw)
        r["expected"] = bool(want(r))
        R["runs"].append(r)
        print(json.dumps(dict(scenario=name, status=r["status"], gd=r["global_decision"], expected=r["expected"], evidence=r["evidence"],
                              retro=r["retro"], why=(r["why"] or "")[:160], bench_s=r["bench_s"]), default=str), flush=True)
        (out / "e7_final_replay_result.json").write_text(json.dumps(R, indent=1, default=str))
    reps = [json.dumps(r["passes"], sort_keys=True, default=str) for r in R["runs"] if "_rep" in r["scenario"]]
    R["deterministic"] = len(set(reps)) == 1
    R["wrong_fix"] = [r["scenario"] for r in R["runs"] if (r["status"] in ("CONFIRMED", "SHADOW_WOULD_CONFIRM") or
                                                           r["global_decision"] == "GLOBAL_CONFIRMED") and (r["leader"] != "H025_F4" or r["record"] == "loc2")]
    R["all_ok"] = pol_ok and R["deterministic"] and not R["wrong_fix"] and all(r["expected"] for r in R["runs"])
    (out / "e7_final_replay_result.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(dict(policy_ok=pol_ok, deterministic=R["deterministic"], wrong_fix=R["wrong_fix"], all_ok=R["all_ok"])), flush=True)
    return R["all_ok"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
