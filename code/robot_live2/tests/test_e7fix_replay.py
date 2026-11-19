"""STAGING ONLY (2026-10-30): E7 structure-path safety on the REAL loc1 / loc2 LiDAR records with a deterministic replay clock.
The real archived LiveStore is cut at a virtual time t (rows <= t); every structure_tick gets now = t; every E7 job is joined
before the clock advances -> the same inputs give the same outputs on every run (no wall-clock growth, no load dependence).
Candidate transforms: final E3 hypotheses of the operational replay (data/recon_*.json, rebuilt without GT) + the replay's
E2 exact list for the rival pre-screen. 'E4 CONFIDENT from t_conf' is FORCED by the harness (a counterfactual timing for the
code path, NOT a field result); the robot-stop signal is derived from the record (<= 0.10 m in the last 2 s).
Pass/fail rules of E7 (GAP_M, segment overlap, wall crossing) are the unchanged operational ones.
Run: python3 -s -B test_e7fix_replay.py OUT_DIR [quick]"""
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HOME = Path.home()
OPR = HOME / "sci_ws/repairs/livefield2_20261018_151606"
STG = Path(__file__).resolve().parents[1] / "candidate/live2"
REP = HOME / "sci_ws/reports/e7_fix_safety_20261030_1840/data"
BENCH = HOME / "sci_ws_alt_imu_20261029_204740/runs/bench_20261030"
LROOT = OPR / "runtime/e2_cache/light"
REC = dict(loc1=dict(store=OPR / "experiments/X20261030_102108/archive/live2_store", recon=REP / "recon_base.json",
                     e2=BENCH / "baseline_loc1/state/e2/e2_state.json"),
           loc2=dict(store=OPR / "experiments/X20261030_102650/archive/live2_store", recon=REP / "recon_loc2.json",
                     e2=BENCH / "baseline_loc2/state/e2/e2_state.json"))
spec = importlib.util.spec_from_file_location("t7", OPR / "tests/test_e7_structure_fix.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
L, E3 = T.load(STG)
import e7_policy as P7  # noqa: E402  (loaded from STG by T.load's sys.path order)
assert Path(P7.__file__).resolve().parent == STG.resolve() or Path(P7.__file__).parent == STG, P7.__file__


class CutStore:
    """the real LiveStore, visible only up to the virtual clock."""
    def __init__(self, st):
        self.st, self.rows = st, []

    def cut(self, t):
        self.rows = [r for r in self.st.rows if r[1] <= t]

    def refresh(self):
        pass

    def __getattr__(self, k):
        return getattr(self.st, k)


class Tracker(T.Tracker):
    def __init__(self, store, hyps, e2_dir, session):
        super().__init__(store, hyps, E3)
        self.e2_dir, self.session = str(e2_dir), session


_STORES, _E2 = {}, {}


def record(name):
    if name not in _STORES:
        st = E3.Store(str(REC[name]["store"])); st.refresh()
        _STORES[name] = st
        e2 = json.loads(Path(REC[name]["e2"]).read_text())
        for f, m in e2["per_floor"].items():
            m["light"] = f"{LROOT}/kci_{f}F_{m.get('map_version') or next(c['map_version'] for c in e2['candidates'] if str(c['floor']) == f)}"
        _E2[name] = e2
    return _STORES[name], _E2[name]


def stopped_at(rows, t):
    xy = [(r[2], r[3]) for r in rows if t - 2.0 <= r[1] <= t]
    return len(xy) >= 2 and sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])) <= 0.10


def run(name, rec_name, leader_id, t_conf, out, hyp_ids=None, policy="good", e2_mode="real", t_stop_at=None, step=1.0, e7_lat=0.0):
    """e7_lat: modeled E7 job duration in record seconds (the virtual clock advances by it after each job: the field E7 took
    7.6-30 s while FAST-LIO keeps writing); deterministic, unlike measuring the bench time."""
    st, e2 = record(rec_name)
    D = json.loads(Path(REC[rec_name]["recon"]).read_text())
    cands = [c for c in D["cands"] if hyp_ids is None or c["candidate_id"] in hyp_ids]
    hyps = [T.Hyp(c["candidate_id"], dict(c["rec"], floor=c["floor"]), i, light=c["rec"]["light"]) for i, c in enumerate(cands)]
    sd = out / name; sd.mkdir(parents=True)
    e2dir = sd / "e2"; e2dir.mkdir()
    if e2_mode == "real":
        (e2dir / "e2_state.json").write_text(json.dumps(e2, default=float))
    elif e2_mode == "twin":                               # the leader's exact geometry ONLY in the E2 list, labelled floor 7
        lead = next(c for c in D["cands"] if c["candidate_id"] == leader_id)
        twin = dict({k: v for k, v in lead["rec"].items() if k != "light"}, floor=7, stage="exact", merit=1.0)
        (e2dir / "e2_state.json").write_text(json.dumps(dict(odom_session=e2["odom_session"], candidates=[twin],
                                                             per_floor={"7": dict(light=lead["rec"]["light"], H=1825)}), default=float))
    elif e2_mode == "other_session":
        (e2dir / "e2_state.json").write_text(json.dumps(dict(e2, odom_session="SOME_OTHER_SESSION"), default=float))
    cs = CutStore(st)
    tr = Tracker(cs, hyps, e2dir, e2["odom_session"])
    t0, tN = st.rows[0][1], st.rows[-1][1]
    cs.cut(t0 + t_conf)
    link = L.E6E7Link(sd, tr, policy="SINGLE_DIAGNOSTIC")
    if policy == "good":                                  # a verified enable file for THIS staged code (exercises the real check)
        pf = sd / "e7_fix_policy.json"
        pf.write_text(json.dumps(dict(structure_fix="ENABLED", verification_report="test_e7fix_replay (harness only)",
                                      code_sha256=P7.code_sha256())))
        link.struct_policy = P7.structure_policy(policy_file=pf, env={})
    leader = next(h for h in hyps if h.id == leader_id)
    e4 = dict(state="CONFIDENT_GEOMETRY", metrics=dict(best_id=leader.id, best_floor=leader.floor, gap=0.2))
    t, sa, ticks = t0 + t_conf, None, 0
    while t <= tN + 1e-6:
        cs.cut(t)
        stop = dict(stopped=(t - t0 >= t_stop_at) if t_stop_at is not None else stopped_at(cs.rows, t))
        sa = link.structure_tick(e4, t, stop=stop)
        if link.job and link.job.is_alive():
            link.job.join()
            t += e7_lat
        ticks += 1
        if sa is None or sa.get("status") in ("CONFIRMED", "NOT_CONFIRMED"):
            break
        t += step
    if (sd / "structure_path.json").exists():
        sa = json.loads((sd / "structure_path.json").read_text())
    e7 = json.loads((sd / "e7_state.json").read_text()) if (sd / "e7_state.json").exists() else {}
    pol = json.loads((sd / "e7_policy.json").read_text()) if (sd / "e7_policy.json").exists() else None
    return dict(scenario=name, record=rec_name, leader=leader_id, t_conf=t_conf, policy=policy, e2_mode=e2_mode, e7_lat=e7_lat,
                status=None if sa is None else sa.get("status"), why=None if sa is None else sa.get("why"),
                global_decision=e7.get("global_decision"), policy_report=None if pol is None else pol.get("structure"),
                passes=[] if sa is None else [dict(t_rel=None if p.get("t_end") is None else round(p["t_end"] - t0, 1),
                                                   **{k: p.get(k) for k in ("ok", "best", "final_decision", "gap_m", "window_s",
                                                                            "path_m", "streak", "independence", "e2_rivals")})
                                              for p in sa.get("passes", [])])


def policy_unit():
    """e7_policy without any E7 run."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="e7pol_"))
    good = dict(structure_fix="ENABLED", verification_report="r", code_sha256=P7.code_sha256())
    cases = []

    def case(name, cfg, env, want):
        pf = d / f"{name}.json"
        if cfg is not None:
            pf.write_text(cfg if isinstance(cfg, str) else json.dumps(cfg))
        r = P7.structure_policy(policy_file=pf, env=env)
        cases.append(dict(case=name, enabled=r["enabled"], why=r["why"], ok=r["enabled"] == want))
    case("no_file", None, {}, False)
    case("no_file_env1", None, {"LIVE2_STRUCTURE_FIX": "1"}, False)
    case("good", good, {}, True)
    case("good_env0", good, {"LIVE2_STRUCTURE_FIX": "0"}, False)
    case("good_env1", good, {"LIVE2_STRUCTURE_FIX": "1"}, True)
    case("disabled", dict(good, structure_fix="DISABLED"), {}, False)
    case("no_report", dict(good, verification_report=""), {}, False)
    case("wrong_sha", dict(good, code_sha256="0" * 64), {}, False)
    case("garbage", "{not json", {}, False)
    case("list", "[1,2]", {}, False)
    default = P7.structure_policy(env={})                 # the staging tree's own <root>/config: must not exist -> OFF
    cases.append(dict(case="staging_default_root", enabled=default["enabled"], why=default["why"], ok=not default["enabled"]))
    return cases


def main(out, quick=False):
    out = Path(out); out.mkdir(parents=True)
    R = dict(policy_unit=policy_unit(), runs=[])
    S = []
    # P: real loc1, true leader, E4 forced CONFIDENT at +45 s (counterfactual timing) — repeated: must give identical windows
    for k in range(1 if quick else 3):
        S.append((f"P_loc1_true_tconf45_rep{k}", "loc1", "H025_F4", 45.0, {}, lambda r: r["status"] == "CONFIRMED"))
    S += [
        ("REAL_loc1_true_tconf155_gpu_timing", "loc1", "H025_F4", 155.0, {}, lambda r: r["status"] != "CONFIRMED"),
        ("W_loc1_7F_leader_tconf45", "loc1", "H047_F7", 45.0, {}, lambda r: r["status"] != "CONFIRMED"),
        ("W_loc1_2F_leader_tconf45", "loc1", "H003_F2", 45.0, {}, lambda r: r["status"] != "CONFIRMED"),
        ("OFF_loc1_true_no_enable_file", "loc1", "H025_F4", 45.0, dict(policy="default"), lambda r: r["status"] is None and r["passes"] == []),
        ("FC_loc1_true_no_e2_list", "loc1", "H025_F4", 45.0, dict(e2_mode="missing"), lambda r: r["status"] == "NOT_CONFIRMED"),
        ("FC_loc1_true_e2_other_session", "loc1", "H025_F4", 45.0, dict(e2_mode="other_session"), lambda r: r["status"] == "NOT_CONFIRMED"),
        ("H_loc1_7F_leader_e3_rivals_removed", "loc1", "H047_F7", 45.0, dict(hyp_ids={"H047_F7"}), lambda r: r["status"] != "CONFIRMED"),
    ]
    if not quick:
        S += [(f"S_loc1_true_tconf{tc:.0f}", "loc1", "H025_F4", tc, {}, lambda r: True) for tc in (30.0, 60.0, 75.0, 90.0, 120.0)]
        S += [(f"L_loc1_true_tconf45_e7lat{la:.0f}", "loc1", "H025_F4", 45.0, dict(e7_lat=la), lambda r: True) for la in (15.0, 25.0)]
        S += [(f"WL_loc1_{h}_tconf45_e7lat15", "loc1", h, 45.0, dict(e7_lat=15.0), lambda r: r["status"] != "CONFIRMED")
              for h in ("H047_F7", "H003_F2")]
        S += [(f"W_loc1_{h}_tconf{tc:.0f}", "loc1", h, tc, {}, lambda r: r["status"] != "CONFIRMED")
              for h in ("H047_F7", "H003_F2", "H122_F3") for tc in (60.0, 90.0, 120.0)]
        S += [(f"W_loc2_{h}_tconf{tc:.0f}", "loc2", h, tc, {}, lambda r: r["status"] != "CONFIRMED")
              for h in ("H146_F4", "H136_F3", "H147_F4", "H141_F3") for tc in (30.0, 90.0, 150.0)]
    for name, rec, lead, tc, kw, want in S:
        t1 = time.time()
        r = run(name, rec, lead, tc, out, **kw)
        r["expected"] = bool(want(r)); r["run_s"] = round(time.time() - t1, 1)
        R["runs"].append(r)
        print(json.dumps(dict(scenario=name, status=r["status"], expected=r["expected"],
                              passes=[(p["t_rel"], p["ok"], p["best"], p["gap_m"], p.get("path_m")) for p in r["passes"]],
                              why=(r["why"] or "")[:120])), flush=True)
        (out / "e7fix_replay_result.json").write_text(json.dumps(R, indent=1, default=str))
    reps = [json.dumps(r["passes"], sort_keys=True, default=str) for r in R["runs"] if r["scenario"].startswith("P_loc1_true_tconf45_rep")]
    R["deterministic_P"] = len(set(reps)) == 1
    wrong_fix = [r["scenario"] for r in R["runs"] if r["status"] == "CONFIRMED" and r["leader"] not in ("H025_F4",)]
    wrong_fix += [r["scenario"] for r in R["runs"] if r["status"] == "CONFIRMED" and r["record"] == "loc2"]
    R["wrong_fix"] = sorted(set(wrong_fix))
    R["all_ok"] = all(c["ok"] for c in R["policy_unit"]) and all(r["expected"] for r in R["runs"]) and R["deterministic_P"] and not R["wrong_fix"]
    (out / "e7fix_replay_result.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(dict(policy_unit_ok=all(c["ok"] for c in R["policy_unit"]), deterministic_P=R["deterministic_P"],
                          wrong_fix=R["wrong_fix"], all_ok=R["all_ok"])))
    return R["all_ok"]


if __name__ == "__main__":
    # 2026-10-30 22:xx: the 20:53 rule tested here (moving windows only, E4 time forced earlier) is SUPERSEDED by the retro-window
    # rule; this file is kept as the helper module of test_e7_final_replay.py. Run that test instead.
    if "--legacy" not in sys.argv:
        print("SUPERSEDED: run test_e7_final_replay.py (this file = helpers; '--legacy' runs the old 20:53 scenarios)")
        sys.exit(0)
    sys.argv.remove("--legacy")
    sys.exit(0 if main(sys.argv[1], quick=len(sys.argv) > 2 and sys.argv[2] == "quick") else 1)
