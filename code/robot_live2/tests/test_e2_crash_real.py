"""R34 §11-1,2: E2 regional success path and restart, through the REAL Coordinator constructor, REAL log(), REAL files. Nothing under test is stubbed (only span/hold where the test sets the scene).
Also the REAL loc2 request OBS_..._O003 and the REAL loc2 odometry (X20261101_193139): OFFLINE_COUNTERFACTUAL. tools/e2e_region_replay.py runs the same with real child fits (data/e2e_*.json)."""
import argparse
import ast
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(STG))
spec = importlib.util.spec_from_file_location("e2w_real", STG / "e2_worker.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
RL = W.RL
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments"
SESS = "fastlio_T:1"
RID = "OBS_live1_observe_20261101_193218_O003"
RQ = dict(request_id=RID, odom_session=SESS, wall=1.0, why="SEMANTIC_RECOVERY_SINGLE",
          region=dict(floor=4, center_px=[1738.5, 559.0], rooms=["room_408"], observation_id="O003", radius_px=519.5, radius_base_px=518.8, px_per_m_hi=27.554, capture_wall=100.0))
CAND = [dict(floor=4, px=1700.0, py=650.0, phi_deg=135.0, ppm=24.0, merit=0.25, twoway=0.6)]
DONE = dict(candidates=CAND, sec=70.0, light="/L4", H=1825, map_version="mv")


def rig(tmp, rows=None, exp="X_T", sess=SESS):
    store, out, ctl = tmp / "store", tmp / "runtime/state/e2", tmp / "runtime/ctl"
    for d in (store, out, ctl):
        d.mkdir(parents=True, exist_ok=True)
    (ctl / "experiment.json").write_text(json.dumps(dict(exp_id=exp)))
    (store / "session.json").write_text(json.dumps(dict(odom_session_id=sess, fastlio_cfg_sha256="abc")))
    with open(store / "odom_link.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "t_sec", "fastlio_x", "fastlio_y", "fastlio_yaw_rad", "delta_s_m", "delta_yaw_deg"])
        for i, (t, x, y) in enumerate(rows or [(100.0 + i, float(i) * 0.5, 0.0) for i in range(40)]):
            w.writerow([f"S{i}", t, x, y, 0, 0, 0])
    (store / "UPDATED").write_text("1")
    return argparse.Namespace(building="kci", store=str(store), out=str(out), cache_dir=str(tmp / "c"), floors=[4], procs=1, exact_dt=6.0, device="cpu", min_span=70.0, min_pass_gap_s=120.0,
                              min_window_path_m=2.0, wall_pass_top=4, mem_start_mb=3500, mem_kill_mb=2800, temp_stop_c=78, temp_hold_c=76, early_window_s=20.0, coarse=1)


def coord(a):
    c = W.Coordinator(a)
    c.hold_reason = lambda: None
    return c


def passes(a):
    return [json.loads(l) for l in open(Path(a.out) / "e2_passes.jsonl")]


def state(a):
    return json.loads((Path(a.out) / "e2_state.json").read_text())


def one_pass(c):
    c.pass_id = 1
    c.state.update(pass_id=1, status="DONE", floors_done=[4], floors_pending=[], t_now=130.0, candidates=[dict(floor=4, px=1.0, py=2.0, phi_deg=3.0, ppm=24.0, merit=0.3, twoway=0.5, pass_id=1)],
                   candidates_pass=[dict(floor=4, px=1.0, py=2.0, phi_deg=3.0, ppm=24.0, merit=0.3, twoway=0.5, pass_id=1)], per_floor={"4": dict(status="DONE", light="/L4", H=1825)}, finished_wall=1.0)
    c.write()


def test_success_path_through_real_log_no_typeerror_and_order_of_saving(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    q = c.region_pending()
    rec, msg = c.region_start(q)
    assert msg[0] == 4 and rec["status"] == "RUNNING"
    c.region_finish(rec, "floor_done", 4, dict(DONE))            # the 11-01 TypeError was here (5 crashes)
    st = state(a)
    r = st["regional"][-1]
    assert r["status"] == "DONE" and r["region_pass_id"] == "1R1" and "pass_id" not in r
    assert st["candidates_pass"][0].get("regional_request") or any(x.get("regional_request") == RID for x in st["candidates_pass"])
    line = [p for p in passes(a) if p["kind"] == "region_done"][-1]
    assert line["pass_id"] == 1 and line["region_pass_id"] == "1R1"     # both survive in the JSON line
    e = c.ledger.get(RID)
    assert e["state"] == "COMPLETED" and e["n_candidates"] == 1 and e["candidates_hash"] and e["delivery"] == "WAITING_E3"


def test_log_cannot_raise_on_any_field_name(tmp_path):
    a = rig(tmp_path); c = coord(a)
    c.log("x", pass_id="1R1", wall=5, kind="k", candidates=[1], tb="t")
    p = passes(a)[-1]
    assert p["pass_id"] == 0 and p["field_pass_id"] == "1R1" and p["field_wall"] == 5 and p["field_kind"] == "k" and p["kind"] == "x"
    c.log("y", obj=object(), bad=float("nan"))                   # not JSON native: still written


def test_no_call_of_log_in_the_worker_can_collide():
    src = (STG / "e2_worker.py").read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "log":
            assert not {k.arg for k in n.keywords if k.arg} & {"wall", "pass_id", "kind", "event_kind"}, n.lineno


def test_log_failure_does_not_lose_candidates(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    rec, _ = c.region_start(c.region_pending())
    (Path(a.out) / "e2_passes.jsonl").mkdir()    # the log cannot be opened
    c.region_finish(rec, "floor_done", 4, dict(DONE))
    st = state(a)
    assert c.state["log_errors"] >= 1 and st["regional"][-1]["status"] == "DONE" and any(x.get("regional_request") == RID for x in st["candidates"])
    assert c.ledger.get(RID)["state"] == "COMPLETED"


def test_restart_keeps_floors_candidates_and_never_reruns_a_completed_request(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    rec, _ = c.region_start(c.region_pending())
    c.region_finish(rec, "floor_done", 4, dict(DONE))
    c2 = coord(a)
    st = c2.state
    assert st["restore"]["status"] == "RESTORED" and st["floors_done"] == [4] and c2.pass_id == 1 and st["restarts"] == 1
    assert any(x.get("regional_request") == RID for x in st["candidates"]) and len(c2.region_added) == 1
    assert c2.region_pending() is None and c2.ledger.get(RID)["attempts"] == 1          # nothing repeated
    RL.submit_request(a.out, RQ)                                                         # the same request again: idempotent
    assert c2.region_pending() is None


def test_restart_during_a_fit_retries_and_a_killer_request_is_bounded(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    n = 0
    while True:
        for e in c.ledger.d["entries"].values():
            e["not_before"] = 0.0
        q = c.region_pending()
        if q is None:
            break
        c.region_start(q)                                        # ... and the process dies here
        n += 1
        c = coord(a)
        assert n <= RL.MAX_INTERRUPTED
    e = c.ledger.get(RID)
    assert n == RL.MAX_INTERRUPTED and e["state"] == "FAILED_FINAL" and "no crash loop" in e["why"]


def test_death_between_saving_candidates_and_ledger_is_reconciled_without_refit(tmp_path, monkeypatch):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    rec, _ = c.region_start(c.region_pending())

    def die(*_a, **_k):
        raise SystemExit("killed")
    monkeypatch.setattr(c.ledger, "complete", die)
    with pytest.raises(SystemExit):
        c.region_finish(rec, "floor_done", 4, dict(DONE))
    c2 = coord(a)
    assert c2.ledger.get(RID)["state"] == "COMPLETED" and c2.region_pending() is None
    assert ("OBS_live1_observe_20261101_193218_O003", "RECONCILED_COMPLETED") in [tuple(x) for x in c2.state["restore"]["running_requests"]]


def test_interrupted_atomic_save_leaves_the_previous_complete_file(tmp_path, monkeypatch):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    before = (Path(a.out) / "e2_state.json").read_text()

    def boom(*_a, **_k):
        raise OSError("disk")
    monkeypatch.setattr(os, "replace", boom)
    c.state["floors_done"] = [4, 5]
    with pytest.raises(OSError):
        c.write(durable=True)
    monkeypatch.undo()
    assert (Path(a.out) / "e2_state.json").read_text() == before
    c2 = coord(a)
    assert c2.state["floors_done"] == [4]


@pytest.mark.parametrize("change", ["session", "run", "corrupt"])
def test_restore_refused_on_changed_identity_or_broken_file(tmp_path, change):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    if change == "session":
        rig(tmp_path, sess="fastlio_OTHER:2")
    elif change == "run":
        rig(tmp_path, exp="X_OTHER")
    else:
        (Path(a.out) / "e2_state.json").write_text("{broken")
    c2 = coord(a)
    assert c2.pass_id == 0 and c2.state["floors_done"] == [] and c2.state["restore"]["status"] == "FRESH" and c2.state["restore"]["why"]
    assert not c2.state["candidates"]


def test_request_of_another_session_never_runs(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, dict(RQ, odom_session="OLD"))
    assert c.region_pending() is None


def test_older_request_survives_a_newer_one_overwriting_the_file(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, dict(RQ, request_id="R1", region=dict(RQ["region"], observation_id="O1")))
    RL.submit_request(a.out, dict(RQ, request_id="R2", region=dict(RQ["region"], observation_id="O2")))
    assert c.region_pending()["request_id"] == "R2"
    c.region_start(c.region_pending())
    assert c.region_pending()["request_id"] == "R1"


def test_child_death_during_a_regional_fit_does_not_mark_the_floor_failed(tmp_path):
    a = rig(tmp_path); c = coord(a); one_pass(c)
    RL.submit_request(a.out, RQ)
    rec, _ = c.region_start(c.region_pending())
    c.region_guarded(c.region_transient, rec, "child died during the regional fit", True)
    assert rec["status"] == "RETRY_WAIT" and c.state["floors_done"] == [4] and "4" not in c.state["floors_failed"]


def real_rows(cut):
    x = OPR / "X20261101_193139"
    t0 = json.loads((x / "experiment.json").read_text())["created_wall"]
    rows = list(csv.DictReader(open(x / "archive/live2_store/odom_link.csv")))
    return t0, [(float(r["t_sec"]), float(r["fastlio_x"]), float(r["fastlio_y"])) for r in rows if float(r["t_sec"]) <= t0 + cut]


def test_real_loc2_o003_radius_uses_net_motion_not_the_path(tmp_path):
    """Field: 3554.8 px (whole map) from a 110 m path. Now: net motion + uncertainty, or a request for a new observation."""
    req = json.loads((OPR / "X20261101_193139/state/e2/e2_region_request.json").read_text())
    res = {}
    for cut in (620.0, 1032.0):
        t0, rows = real_rows(cut)
        sub = tmp_path / str(int(cut)); sub.mkdir()
        a = rig(sub, rows=rows)
        c = coord(a)
        r, info = c.region_radius(req["region"], rows[-1][0])
        res[cut] = (r, info)
    r620, i620 = res[620.0]
    assert r620 is not None and r620 < 1282.7 and i620["walked_since_capture_at_fit_m"] > i620["net_since_capture_m"]      # 1282.7 = R33 at the same time
    r1032, i1032 = res[1032.0]
    assert i1032["walked_since_capture_at_fit_m"] > 100
    assert r1032 is None or r1032 < 0.5 * 3554.8                                                                           # never the whole map again


def test_round_trip_does_not_grow_the_region(tmp_path):
    out_and_back = [(100.0 + i, float(i), 0.0) for i in range(41)] + [(141.0 + i, 40.0 - i, 0.0) for i in range(41)]       # 80 m path, net 0
    a = rig(tmp_path, rows=out_and_back); c = coord(a)
    r, info = c.region_radius(RQ["region"], 181.0)
    assert info["net_since_capture_m"] < 1.0 and r < 518.8 + 5.0 * 27.554
