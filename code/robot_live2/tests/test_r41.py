"""R41: disjoint-window E7 rule, wall-crossing rule, exact queue order. Unit level; the integrated replays are replay_r41.py."""
import json, math, sys, types
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "candidate/live2"), str(HERE.parent / "candidate/external/sci_live/floor_id/localizer")]
import e6e7_link as L  # noqa: E402
import e7_semantic_shadow as SH  # noqa: E402
import e2_worker as W  # noqa: E402


def rec(px=100.0, py=200.0):
    return dict(pos=[px, py], anchor=[0.0, 0.0], phi=0.0, ppm=24.0, beta=0.0, mirror=1, floor=4, light="l")


def link(monkeypatch, tmp_path):
    k = L.E6E7Link.__new__(L.E6E7Link)
    k.tr = types.SimpleNamespace(store=None); k.state = tmp_path
    monkeypatch.setattr(L.E7, "frames", lambda *a, **kw: [(100.0, 1.0, 2.0, 0.3, None)])
    return k


def res(t, sid="A", dx=0.0, xw=0.1):
    return dict(t_end=t, window_s=45.0, results=[dict(candidate_id=sid, alignment_score=0.02, wall_crossing=dict(violation=xw), correction=dict(dx_m=dx, dy_m=0.0, dphi_deg=0.0, scale=1.0))])


GOOD = dict(verdict="WOULD_FIX", floor_evidence=[dict(kind="DEPARTMENT_ON_2_PLATES")], blocked_by=[], rejected_by=[], room_readings=[dict(label="407"), dict(label="405")], floors_explaining=[4], crossing_margin=0.0, semantic_score=0.02)
R_ = dict(observation_id="G")


def W_(k, t, dx=0.0, r=GOOD, sid="A", rc=None):
    return k._window_evidence(R_, [dict(candidate_id=sid, rec=rc or rec())], res(t, sid, dx), r, sid)


def test_one_window_is_not_enough(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path)
    assert W_(k, 100.0)["n"] == 1


def test_two_agreeing_windows_count(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path)
    W_(k, 100.0)
    assert W_(k, 150.0, dx=0.5)["n"] == 2


def test_window_elsewhere_does_not_count(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path)
    W_(k, 100.0)
    assert W_(k, 150.0, dx=9.0)["n"] == 1


def test_passes_of_two_plate_hypotheses_at_one_place_agree(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path)
    W_(k, 100.0, sid="A1_405")
    assert W_(k, 150.0, sid="A1_407", dx=0.3)["n"] == 2


def test_a_failed_window_in_between_is_neutral_but_a_contradiction_resets(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path)
    W_(k, 100.0)
    fail = dict(GOOD, verdict="WOULD_AMBIGUOUS", blocked_by=["LOCAL_AMBIGUITY"])
    assert W_(k, 150.0, r=fail)["n"] == 1
    assert W_(k, 200.0)["n"] == 2
    k = link(monkeypatch, tmp_path)
    W_(k, 100.0)
    hard = dict(GOOD, verdict="WOULD_AMBIGUOUS", blocked_by=["CROSSING_ABS 0.87 > 0.35"])
    assert W_(k, 150.0, r=hard)["n"] == 0
    assert W_(k, 200.0)["n"] == 1
    k = link(monkeypatch, tmp_path)
    W_(k, 100.0)
    rej = dict(GOOD, verdict="WOULD_REJECT", rejected_by=["PLATE_TEXT_CONFLICT"])
    assert W_(k, 150.0, r=rej)["n"] == 0


def test_e34_survives_a_regional_floor_record_without_size():
    src = (HERE.parent / "candidate/live2/e34_node.py").read_text()
    assert 'if pf.get("H") and pf.get("W"):' in src


def test_a_fix_is_not_replaced_by_a_later_look(tmp_path):
    k = L.E6E7Link.__new__(L.E6E7Link)
    k.last = dict(global_decision="GLOBAL_CONFIRMED")
    k.tr = None                     # would raise if the look ran
    assert k._run([], dict(observation_id="G"), "A") is None
    assert k.request_e7([], dict(observation_id="G")) is None


def _lio(tmp_path, hz_list):
    now = __import__("time").time()
    (tmp_path / "lio_series.jsonl").write_text("".join(json.dumps(dict(wall=now - 100 + i, hz=h)) + "\n" for i, h in enumerate(hz_list)))
    k = L.E6E7Link.__new__(L.E6E7Link); k.state = tmp_path
    return k.fastlio_degraded()[0]


def test_fastlio_gate_short_dip_passes_sustained_or_deep_blocks(tmp_path):
    assert _lio(tmp_path, [10.0] * 97 + [7.8] * 3) is False       # X20261103_001810 +583 s
    assert _lio(tmp_path, [10.0] * 80 + [7.5] * 20) is True
    assert _lio(tmp_path, [10.0] * 99 + [5.5]) is True


def test_old_odometry_blocks_a_fix(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path); k.pair = object(); k.max_odom_age_s = 60.0
    monkeypatch.setattr(L.P7, "semantic_policy", lambda *a, **kw: dict(mode="ENABLED"))
    now = __import__("time").time()
    W_(k, now - 300.0)
    r2 = res(now - 200.0)
    out = k._semantic_fix(dict(global_decision="AMBIGUOUS", semantic_hypothesis="A", e7=dict(t_end=now - 200.0)), GOOD, R_, None, None, cands=[dict(candidate_id="A", rec=rec())], res=r2)
    assert out["global_decision"] != "GLOBAL_CONFIRMED" and "old" in out["semantic_rule"]["why"]


def test_one_window_gives_no_fix_two_do(monkeypatch, tmp_path):
    k = link(monkeypatch, tmp_path); k.pair = object(); k.max_odom_age_s = 60.0
    monkeypatch.setattr(L.P7, "semantic_policy", lambda *a, **kw: dict(mode="ENABLED"))
    k.fastlio_degraded = lambda *a, **kw: (False, None)
    k._epoch = lambda: None
    now = __import__("time").time()
    A = dict(global_decision="AMBIGUOUS", semantic_hypothesis="A", e7=dict(t_end=now - 5.0))
    o1 = k._semantic_fix(dict(A), GOOD, R_, None, None, cands=[dict(candidate_id="A", rec=rec())], res=res(now - 50.0))
    assert o1["global_decision"] == "AMBIGUOUS" and "1/2" in o1["semantic_rule"]["why"]
    o2 = k._semantic_fix(dict(A), GOOD, R_, None, None, cands=[dict(candidate_id="A", rec=rec())], res=res(now - 5.0, dx=0.2))
    assert o2["global_decision"] == "GLOBAL_CONFIRMED" and o2["path"] == "SEMANTIC_2ROOM"


def test_vlm_server_outlives_the_thread_that_asked(tmp_path):
    """REAL process: a short-lived thread starts a child with the same PR_SET_PDEATHSIG; before R41 the child died with that thread."""
    import subprocess, threading, time
    import vlm_ondemand as V
    v = V.OnDemandVLM.__new__(V.OnDemandVLM); v.popen = subprocess.Popen
    got = []
    t = threading.Thread(target=lambda: got.append(v._popen_in_owner(["sleep", "30"], subprocess.DEVNULL)))
    t.start(); t.join()
    time.sleep(1.0)
    try:
        assert got[0].poll() is None
    finally:
        got[0].kill(); got[0].wait()


def _pm(tmp_path):
    import e8_pair as EP
    p = EP.PairManager.__new__(EP.PairManager)
    p.state, p.clock, p.status, p.gid, p.e7 = tmp_path, lambda: 1000.0, "CANDIDATE_MISSING", "G1", None
    p.group, p.history, p.processed, p.n_groups, p.closed = [], [dict(group_id="G1", status="CANDIDATE_MISSING")], [], 1, dict(group_id="G1", observations=[dict(observation_id="a", session="S", read=True), dict(observation_id="b", session="S", read=True)])
    p.read_each, p.why, p.since, p._running, p._again, p.last_eval = True, None, 0, False, False, None
    p._session = lambda: "S"; p._start_reading = lambda: None; p.tr = types.SimpleNamespace(alive=lambda: [])
    p.snapshot = lambda: dict(state=p.status, why=p.why)
    return p


def test_a_new_photo_keeps_a_pair_that_waits_for_its_fit(tmp_path):
    p = _pm(tmp_path)
    p.on_precision(dict(observation_id="c", status="CAPTURE_STORED", image_quality=dict(ok=True), image_sha256="x"))
    assert p.gid == "G1" and [g["observation_id"] for g in p.group] == ["a", "b", "c"]


def test_a_new_photo_after_other_outcomes_starts_a_new_group(tmp_path):
    p = _pm(tmp_path)
    p.history = [dict(group_id="G1", status="NO_FIX")]
    p.on_precision(dict(observation_id="c", status="CAPTURE_STORED", image_quality=dict(ok=True), image_sha256="x"))
    assert p.gid != "G1" and [g["observation_id"] for g in p.group] == ["c"]


def test_new_fit_triggers_a_new_judgement_once(tmp_path):
    p = _pm(tmp_path)
    p.group = [dict(observation_id="a"), dict(observation_id="b")]; p.status = "MORE_OBSERVATION_REQUIRED"; p.last_eval = dict(verdict="MORE_OBSERVATION_REQUIRED")
    h = types.SimpleNamespace(id="H1", rec=dict(provenance="SEMANTIC_ANCHORED"))
    p.tr = types.SimpleNamespace(alive=lambda: [h])
    n = []
    p._start_reading = lambda: n.append(1)
    assert p.reevaluate_on_new_hypotheses() is True and p.reevaluate_on_new_hypotheses() is False and len(n) == 1


def test_e2_gives_all_memory_back_for_the_vlm(tmp_path):
    C = next(v for v in vars(W).values() if isinstance(v, type) and hasattr(v, "apply_budget"))
    s = types.SimpleNamespace(a=types.SimpleNamespace(out=str(tmp_path), procs=4))
    s.budget = lambda: C.budget(s)
    (tmp_path / "e2_budget.json").write_text(json.dumps(dict(max_busy=0)))
    assert C.budget(s) == 0
    class P:
        def terminate(self): pass
        def join(self, x): pass
    kids = {0: dict(pr=P(), busy=None), 1: dict(pr=P(), busy=("exact", 4))}
    spawned = []
    def sp(j):
        spawned.append(j); kids[j] = dict(pr=P(), busy=None)
    C.apply_budget(s, kids, sp)
    assert 0 not in kids and 1 in kids and not spawned     # the fit finishes, the idle one goes
    kids[1]["busy"] = None
    C.apply_budget(s, kids, sp)
    assert list(kids) == [0] and kids[0].get("fresh") and spawned == [0]
    n = len(spawned)
    C.apply_budget(s, kids, sp)
    assert len(spawned) == n                                # no respawn loop
    (tmp_path / "e2_budget.json").write_text(json.dumps(dict(max_busy=3)))
    C.apply_budget(s, kids, sp)
    assert sorted(kids) == [0, 1, 2] and not kids[0].get("fresh")


def test_vlm_stops_after_the_batch_and_e2_yield_is_bounded():
    import e8_pair as EP, threading
    p = EP.PairManager.__new__(EP.PairManager)
    p.clock = lambda: 1000.0
    p.group = [dict(vlm_wait_since=1000.0 - EP.VLM_YIELD_S - 1, reading=dict(vlm=dict(state="NOT_ADMITTED_RESOURCE", tries=3)))]
    assert p.vlm_pending() is False
    p.group = [dict(vlm_wait_since=990.0, reading=dict(vlm=dict(state="QUEUED")))]
    assert p.vlm_pending() is True
    class V:
        stopped = None
        def ask(self, img): return {}
        def stop(self, why): V.stopped = why
    p.vlm, p.lock, p._vlm_q = V().ask, threading.Lock(), []
    p._vlm_loop()
    assert V.stopped == "BATCH_DONE"


def test_refresh_due_survives_budget_zero(tmp_path):
    C = next(v for v in vars(W).values() if isinstance(v, type) and hasattr(v, "refresh_due"))
    s = types.SimpleNamespace(a=types.SimpleNamespace(out=str(tmp_path), procs=4, refresh_s=90.0), pass_id=1, state=dict(refresh=dict(n_passes=0, last_wall=__import__("time").time())), budget=lambda: 0)
    assert C.refresh_due(s, []) is False


def test_vlm_newest_photo_first(monkeypatch):
    import e8_pair as EP, threading
    monkeypatch.setattr(EP, "VLM_RETRY_S", 0.0)
    p = EP.PairManager.__new__(EP.PairManager)
    asked = []
    class V:
        def ask(self, img):
            asked.append(img); return {}
        def stop(self, why): pass
    p.vlm, p.lock, p.vlm_admit = V().ask, threading.Lock(), lambda: (True, "")
    p.snapshot = lambda: None; p._log_obs = lambda *a, **k: None; p.group = []
    p._vlm_q = [dict(o=dict(reading={}), crop="old", tries=0), dict(o=dict(reading={}), crop="new", tries=0)]
    p._vlm_loop()
    assert asked == ["new", "old"]
