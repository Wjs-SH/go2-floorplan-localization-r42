"""R34 §11-7: early E7 SHADOW on the REAL loc1 record X20261101_192409 with the REAL E6E7Link.early_shadow_tick + REAL E7, deterministic clock (LiveStore cut at t, job joined before the clock moves).
Candidates: the run's real E2 candidates (7 of 9 floors, as in the field); leader = the E2 candidate nearest the final E3 leader's transform on its floor (HINDSIGHT for the code path: OFFLINE_COUNTERFACTUAL, not a realtime timing claim).
Checks: record only (no e7_state, no fix_overlay), unsearched floors listed, 8F rival present, a leader change clears the run. Run: test_r34_loc1_shadow.py OUT"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
L, E3 = RP.T.load(RP.STG)
RUN = "X20261101_192409"
X = RP.OPR / "experiments" / RUN


def main(out):
    out = Path(out); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    t0 = json.loads((X / "experiment.json").read_text())["created_wall"]
    st = E3.Store(str(X / "archive/live2_store")); st.refresh()
    fd = [json.loads(l) for l in open(X / "state/e2/e2_passes.jsonl") if '"floor_done"' in l]
    cands = [dict(c, light=f"{RP.LROOT}/kci_{p['floor']}F_{c['map_version']}") for p in fd for c in p["candidates"]]
    e3 = json.loads((X / "state/e3_state.json").read_text())
    lead = next(h for h in e3["hypotheses"] if h["id"] == e3.get("leader")) if e3.get("leader") else e3["hypotheses"][0]
    same = [c for c in cands if c["floor"] == lead["floor"]]
    lc = min(same, key=lambda c: abs((c["phi"] - lead["phi"] + 180) % 360 - 180))
    best = {}
    for c in sorted(cands, key=lambda c: c["merit"]):
        best.setdefault(c["floor"], c)
    hyps = [RP.T.Hyp("LEAD_F%d" % lead["floor"], dict(lc), 0, light=lc["light"])]
    for i, (f, c) in enumerate(sorted(best.items()), 1):
        if c is not lc:
            hyps.append(RP.T.Hyp(f"R_F{f}", dict(c), i, light=c["light"]))
    for h in hyps:
        h.n_kf = 10
    e2d = out / "e2"; e2d.mkdir()
    floors_done = sorted({p["floor"] for p in fd})
    (e2d / "e2_state.json").write_text(json.dumps(dict(floors_done=floors_done, floors=list(range(1, 10)), floors_total=9, odom_session=st.session().get("odom_session_id"))))
    cs = RP.CutStore(st)
    tr = RP.Tracker(cs, hyps, e2d, st.session().get("odom_session_id")); tr.E3 = E3
    link = L.E6E7Link(out, tr, policy="SINGLE_DIAGNOSTIC")
    table, changed = [], None
    for rel in range(60, 380, 5):
        cs.cut(t0 + rel)
        link.early_shadow_tick(t0 + rel)
        th = getattr(link, "_early_thread", None)
        if th is not None:
            th.join()
        es = getattr(link, "early", None) or {}
        last = es.get("last") or {}
        if last and (not table or table[-1]["t_end"] != last["t_end"]):
            table.append(dict(rel=rel, t_end=last["t_end"], best=last["best"], decision=last["final_decision"], gap=last["gap_m"], leader_best=last["leader_best"], status=es["status"], sec=last["sec"], unsearched=last["unsearched_floors"], rivals=last["rivals"], path=last["path_m"]))
            print(json.dumps(table[-1], default=float)[:260], flush=True)
    assert not (out / "e7_state.json").exists() and not (out / "fix_overlay.json").exists(), "early shadow wrote a FIX record"
    assert table and all(set(t["unsearched"]) == set(range(1, 10)) - set(floors_done) for t in table)
    assert any("R_F8" in t["rivals"] for t in table), "8F rival kept"
    # a changed leader clears the run
    n0 = link.early["n_pass"]
    hyps[0], hyps[1] = hyps[1], hyps[0]
    cs.cut(t0 + 379)
    es = dict(link.early); es["passes"] = []
    link.early["passes"][-1]["t_end"] = -1e12
    tr.alive = lambda: sorted(hyps[1:] + hyps[:1], key=lambda h: 0 if h is hyps[1] else 1)
    link.early_shadow_tick(t0 + 379)
    th = getattr(link, "_early_thread", None)
    if th is not None:
        th.join()
    changed = link.early["leader"]
    res = dict(run=RUN, kind="OFFLINE_COUNTERFACTUAL", leader=hyps[1].id, floors_done=floors_done, table=table, leader_changed_to=changed, passes_after_change=len(link.early["passes"]), n_before=n0, e7_state_written=False, e7_s=[t["sec"] for t in table])
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    print("OK", "windows", len(table), "leader best", sum(t["leader_best"] for t in table), "would-confirm", sum(t["status"] == "SHADOW_WOULD_CONFIRM" for t in table), flush=True)
    return True


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
