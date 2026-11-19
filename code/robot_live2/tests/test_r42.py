"""R42: E7's own GLOBAL_CONFIRMED passes the same gates as SEMANTIC_2ROOM; an overlapping window is neutral; the structure FIX pose must come from a passing registration."""
import sys, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_r41 as T  # noqa: E402  (fixtures of the R41 window tests)
L = T.L

BAD = dict(T.GOOD, verdict="WOULD_AMBIGUOUS", floor_evidence=[], blocked_by=["NO_FLOOR_EVIDENCE"])
CORE = lambda now: dict(global_decision="GLOBAL_CONFIRMED", semantic_hypothesis="A", why="geometry unique", e7=dict(t_end=now - 5.0), used_observation_ids=["o1", "o2"])


def k_(monkeypatch, tmp_path, mode="ENABLED"):
    k = T.link(monkeypatch, tmp_path); k.pair = object(); k.max_odom_age_s = 60.0
    monkeypatch.setattr(L.P7, "semantic_policy", lambda *a, **kw: dict(mode=mode))
    k.fastlio_degraded = lambda *a, **kw: (False, None)
    k._epoch = lambda: None
    return k


def fix(k, rec, now, t, dx=0.0):
    return k._semantic_fix(CORE(now), rec, T.R_, None, None, cands=[dict(candidate_id="A", rec=T.rec())], res=T.res(t, dx=dx))


def test_core_confirm_without_floor_evidence_is_not_a_fix(monkeypatch, tmp_path):
    k = k_(monkeypatch, tmp_path); now = time.time()
    fix(k, T.GOOD, now, now - 60.0)
    o = fix(k, BAD, now, now - 5.0)
    assert o["global_decision"] == "MORE_OBSERVATION_REQUIRED" and "semantic rule" in o["why"] and o["fix_blocked"]["e7_decision"] == "GLOBAL_CONFIRMED"


def test_core_confirm_obeys_the_policy_switch(monkeypatch, tmp_path):
    k = k_(monkeypatch, tmp_path, mode="SHADOW"); now = time.time()
    fix(k, T.GOOD, now, now - 60.0)
    o = fix(k, T.GOOD, now, now - 5.0)
    assert o["global_decision"] == "MORE_OBSERVATION_REQUIRED" and "policy" in o["why"]


def test_core_confirm_needs_two_windows(monkeypatch, tmp_path):
    k = k_(monkeypatch, tmp_path); now = time.time()
    o1 = fix(k, T.GOOD, now, now - 60.0)
    assert o1["global_decision"] == "MORE_OBSERVATION_REQUIRED" and "1/2" in o1["why"]
    o2 = fix(k, T.GOOD, now, now - 5.0, dx=0.2)
    assert o2["global_decision"] == "GLOBAL_CONFIRMED" and o2["semantic_rule"]["allowed"] is True


def test_core_confirm_with_old_odometry_is_not_a_fix(monkeypatch, tmp_path):
    k = k_(monkeypatch, tmp_path); now = time.time()
    fix(k, T.GOOD, now, now - 400.0)
    o = k._semantic_fix(dict(CORE(now), e7=dict(t_end=now - 300.0)), T.GOOD, T.R_, None, None, cands=[dict(candidate_id="A", rec=T.rec())], res=T.res(now - 300.0))
    assert o["global_decision"] == "MORE_OBSERVATION_REQUIRED" and "old" in o["why"]


def test_core_confirm_without_rule_record_is_not_a_fix(monkeypatch, tmp_path):
    k = k_(monkeypatch, tmp_path); now = time.time()
    o = k._semantic_fix(CORE(now), None, T.R_, None, None, cands=[], res=T.res(now - 5.0))
    assert o["global_decision"] == "MORE_OBSERVATION_REQUIRED"


def test_non_core_paths_unchanged(monkeypatch, tmp_path):
    k = k_(monkeypatch, tmp_path); now = time.time()
    A = dict(global_decision="AMBIGUOUS", semantic_hypothesis="A", e7=dict(t_end=now - 5.0))
    k._semantic_fix(dict(A), T.GOOD, T.R_, None, None, cands=[dict(candidate_id="A", rec=T.rec())], res=T.res(now - 50.0))
    o = k._semantic_fix(dict(A), T.GOOD, T.R_, None, None, cands=[dict(candidate_id="A", rec=T.rec())], res=T.res(now - 5.0, dx=0.2))
    assert o["global_decision"] == "GLOBAL_CONFIRMED" and o["path"] == "SEMANTIC_2ROOM"


def test_overlapping_window_is_neutral(monkeypatch, tmp_path):
    k = T.link(monkeypatch, tmp_path)
    T.W_(k, 100.0)
    w = T.W_(k, 120.0, dx=0.2)                 # window 45 s: [75, 120] shares 25 s with the pass ending at 100
    assert w["n"] == 1 and "shares" in w["why"]
    assert T.W_(k, 146.0, dx=0.2)["n"] == 2    # [101, 146]: disjoint


def _hyp():
    return types.SimpleNamespace(id="H1_F4", floor=4, rec=T.rec(), light_dir="l", rank_key=lambda: 0)


def _structure_link(monkeypatch, tmp_path, decision):
    k = T.link(monkeypatch, tmp_path)
    h = _hyp()
    k.tr = types.SimpleNamespace(alive=lambda: [h], light=lambda d: object(), store=types.SimpleNamespace(rows=[(0, 0.0, 0, 0, 0)]))
    k.fastlio_degraded = lambda *a, **kw: (False, None)
    monkeypatch.setattr(L.E7, "evaluate", lambda *a, **kw: dict(final_decision=decision, why="x", results=[dict(candidate_id="H1_F4", correction=dict(dx_m=0.0, dy_m=0.0, dphi_deg=0.0, scale=1.0))]))
    return k


def test_structure_fix_needs_a_passing_pose_registration(monkeypatch, tmp_path):
    r = dict(floors=[dict(floor=4, hyp="H1_F4")], t_end=100.0, sec=1.0)
    ok, why = _structure_link(monkeypatch, tmp_path, "MORE_OBSERVATION_REQUIRED").write_structure_long_fix(r, "test")
    assert ok is False and not (tmp_path / "e7_state.json").exists()
    ok, why = _structure_link(monkeypatch, tmp_path, "GEOMETRY_CONSISTENT_SEMANTIC_PENDING").write_structure_long_fix(r, "test")
    assert ok is True and (tmp_path / "e7_state.json").exists()
    k = _structure_link(monkeypatch, tmp_path, "GEOMETRY_CONSISTENT_SEMANTIC_PENDING")
    k._e7_ledger = dict(gid="PAIR_OLD", rec_fix=dict(T.rec(px=900.0), floor=7))     # an earlier semantic group's transform must not be the current pose
    assert k.write_structure_long_fix(r, "test")[0] is True
    assert k._e7_ledger["gid"] == "STRUCT_LONG_H1_F4" and k._e7_ledger["rec_fix"]["floor"] == 4 and k._e7_ledger["rec_fix"]["pos"][0] != 900.0


def test_stop_request_numbers_continue_after_a_keeper_restart(tmp_path):
    import json as J
    import visual_stop as VS
    run = tmp_path / "live1_observe_X"; run.mkdir()
    with open(run / "live2_stop_requests.jsonl", "w") as f:
        for n in (1, 2, 3):
            f.write(J.dumps(dict(observation_id=f"live1_observe_X_O{n:03d}", target_track_id="s_O9\"")) + "\n")
    v = VS.VisualStop(run)
    assert v.n == 3                                  # the next request is O004, not O001 again
    assert VS.VisualStop(tmp_path / "fresh").n == 0


def test_e3_tracking_scale_is_bounded_around_the_e2_fit():
    """A map where the trimmed cost is sample/ppm with a constant sample: a bigger scale always looks better. Before R42 tracking followed it +1 % per call for ever."""
    import math
    import numpy as np
    import e3_tracker as E3
    h = types.SimpleNamespace(id="H1_F4", floor=4, light_dir="l", rec=dict(pos=[100.0, 100.0], anchor=[0.0, 0.0], phi=0.0, ppm=24.0, beta=0.0, mirror=1),
                              obs=[(0.0, 0.0, 0.0, np.ones((5, 2))), (1.0, 0.0, 0.0, np.ones((5, 2)))], corr=dict(n=0, xy_m=0.0, deg=0.0, scale=0.0, saturated=0))
    L = types.SimpleNamespace(H=500, W=500, F=None, sample=lambda F, xy, big: np.full(len(xy), 1.0))
    tr = E3.Tracker.__new__(E3.Tracker); tr.light = lambda d: L; tr.log_path = None; tr.clock = lambda: 0.0
    for _ in range(40):
        tr.track(h)
    assert 24.0 * 1.04 < h.rec["ppm"] <= 24.0 * 1.05 + 1e-9      # it moved (the bound is the only limit) and stopped at +5 %
