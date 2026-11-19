"""STAGING (2026-10-30): semantic recovery code path — CANDIDATE_MISSING -> regional E2 request -> new hypothesis -> the SAME
closed pair re-evaluated -> E7 only via REGION_CANDIDATE_READY. SYNTHETIC maps / readings / E2 (helpers of test_e8_pair_fake);
the regional E2 fit itself is tested on the real loc2 store in test_e2_region_loc2.py. Nothing here is field evidence."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_e8_pair_fake as T  # noqa: E402
EP = T.EP


def missing_pair(tmp, hyps, fl=(4, 5)):
    pm, tr, e7 = T.manager(tmp, hyps, T.floors(*fl), T.OCR)
    pm.on_precision(T.stored(tmp, "O1", 10, T.T401))
    pm.on_precision(T.stored(tmp, "O2", 20, T.T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    T.wait(pm)
    return pm, tr, e7


def test_region_is_the_last_plate_room_on_the_single_explaining_floor(tmp_path):
    pm, _, e7 = missing_pair(tmp_path, [T.Hyp("H5", 5)])
    assert pm.status == "CANDIDATE_MISSING" and not e7
    reg, why = pm.recovery_region(rows=[])
    assert why is None and reg["floor"] == 4 and reg["rooms"][-1] == "room_405"
    assert reg["center_px"] == [750.0, 350.0]                     # room 405 centre (synthetic px)
    assert reg["radius_px"] >= EP.REGION_RADIUS_M / T.RES        # generous: >= 15 m at the map scale


def test_no_region_when_nothing_is_missing_or_floors_are_ambiguous(tmp_path):
    pm, _, e7 = missing_pair(tmp_path, [T.Hyp("H4", 4)])         # consistent pair -> E7, not missing
    assert pm.recovery_region(rows=[])[0] is None and e7
    (tmp_path / "b").mkdir()
    pm2, _, _ = missing_pair(tmp_path / "b", [T.Hyp("H9", 9)], fl=(4, 6))   # same rooms exist on 4F and 6F
    if pm2.status == "CANDIDATE_MISSING" and len(pm2.last_eval.get("floors_explaining") or []) > 1:
        reg, why = pm2.recovery_region(rows=[])
        assert reg is None and "needs exactly one" in why


def test_recovered_hypothesis_goes_to_e7_only_through_region_candidate_ready(tmp_path):
    hyps = [T.Hyp("H5", 5)]
    pm, tr, e7 = missing_pair(tmp_path, hyps)
    assert pm.recover_closed() is not None and pm.status == "CANDIDATE_MISSING" and not e7   # nothing new yet: still missing
    hyps.append(T.Hyp("H4R", 4))                                  # what the regional E2 adds (right place)
    ev = pm.recover_closed()
    assert ev["verdict"] == "REGION_CANDIDATE_READY" and ev["recovery"] and e7 and e7[-1][0] == ["H4R"]
    assert e7[-1][1]["pair_observation_ids"] == ["O1", "O2"]       # the same two plates, not new evidence
    assert pm.status == "E7_VERIFYING"                            # not a fix: E7 decides


def test_wrong_place_recovered_hypothesis_is_not_consistent(tmp_path):
    hyps = [T.Hyp("H5", 5)]
    pm, tr, e7 = missing_pair(tmp_path, hyps)
    hyps.append(T.Hyp("H4X", 4, pos=(1450.0, 400.0)))             # regional E2 found a place near room 409, not 401/405
    ev = pm.recover_closed()
    assert ev["verdict"] != "REGION_CANDIDATE_READY" and not e7


def test_link_recovery_tick_writes_the_request_and_reevaluates_after_regional_e2(tmp_path, monkeypatch):
    import e6e7_link as L
    st = tmp_path / "state"; st.mkdir()
    hyps = [T.Hyp("H5", 5)]
    tr = T.Tracker(hyps)
    tr.store.rows = [(0, 10.0)]
    (tmp_path / "e2").mkdir()
    tr.e2_dir, tr.session = str(tmp_path / "e2"), "S1"
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(dict(odom_session="S1", candidates=[], per_floor={})))
    monkeypatch.setattr(L.E7, "evaluate", lambda *a, **k: dict(results=[1], best="H4R", final_decision="GLOBAL_CONFIRMED"))
    link = L.E6E7Link(st, tr, sem_floors=T.floors(4, 5), policy="E8_PAIR", reader=T.Reader(T.OCR))
    for r in (T.stored(st, "O1", 10, T.T401), T.stored(st, "O2", 20, T.T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0)):
        with open(st / "e6_precision_results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        link.tick()
    T.wait(link.pair)
    link.tick()
    assert link.pair.status == "CANDIDATE_MISSING"
    req = json.loads((tmp_path / "e2" / "e2_region_request.json").read_text())
    assert req["region"]["floor"] == 4 and req["odom_session"] == "S1"
    rec = json.loads((st / "semantic_recovery.json").read_text())
    assert rec["status"] == "REGION_REQUESTED"
    # regional E2 waits for motion first (RETRY_WAIT), then is DONE with one candidate that E3 takes in
    e2 = dict(odom_session="S1", candidates=[], per_floor={},
              regional=[dict(request_id=req["request_id"], status="RETRY_WAIT", attempt=1, why_failed="robot at rest")])
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(e2)); link.tick()
    assert json.loads((st / "semantic_recovery.json").read_text())["status"] == "REGION_REQUESTED"
    e2["regional"].append(dict(request_id=req["request_id"], status="DONE", n=1, pass_id="1R2", attempt=2))
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(e2))
    h = T.Hyp("H4R", 4); h.rec["regional_request"] = req["request_id"]; hyps.append(h)
    link.tick()
    T.wait(link.pair)
    if link.job:
        link.job.join(10)
    rec = json.loads((st / "semantic_recovery.json").read_text())
    assert rec["status"] == "RE_EVALUATED" and rec["verdict"] == "REGION_CANDIDATE_READY"
    for _ in range(50):
        if link.pair.status == "GLOBAL_FIX":
            break
        link.tick()
    e7s = json.loads((st / "e7_state.json").read_text())
    # R42: the stub E7 confirms, but 401 + 405 are digits only (no floor evidence) and one window: the R38/R41 rule holds it
    assert e7s["global_decision"] == "MORE_OBSERVATION_REQUIRED" and e7s["fix_blocked"]["e7_decision"] == "GLOBAL_CONFIRMED"
    assert e7s["used_observation_ids"] == ["O1", "O2"] and link.pair.status != "GLOBAL_FIX"


def test_failed_regional_search_never_fixes(tmp_path):
    import e6e7_link as L
    st = tmp_path / "state"; st.mkdir()
    tr = T.Tracker([T.Hyp("H5", 5)])
    (tmp_path / "e2").mkdir()
    tr.e2_dir, tr.session = str(tmp_path / "e2"), "S1"
    link = L.E6E7Link(st, tr, sem_floors=T.floors(4, 5), policy="E8_PAIR", reader=T.Reader(T.OCR))
    for r in (T.stored(st, "O1", 10, T.T401), T.stored(st, "O2", 20, T.T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0)):
        with open(st / "e6_precision_results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        link.tick()
    T.wait(link.pair); link.tick()
    req = json.loads((tmp_path / "e2" / "e2_region_request.json").read_text())
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(dict(odom_session="S1", candidates=[], regional=[
        dict(request_id=req["request_id"], status="FAILED", why_failed="semantic region contains no traversable seed cell")])))
    link.tick()
    rec = json.loads((st / "semantic_recovery.json").read_text())
    assert rec["status"] == "RESEARCH_FAILED" and not (st / "e7_state.json").exists()
