"""R39: the automatic state machine. REAL = recorded field runs (no device, no robot). SYNTHETIC cases say so in their name."""
import json
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
sys.path.insert(0, str(HERE))
import auto_fsm as A  # noqa: E402
import final_verify as FV  # noqa: E402
import op_clock as OC  # noqa: E402
import status_view as SV  # noqa: E402
import visual_stop as VS  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments"


def hy(floor, score, n_kf=30, status="ACTIVE", i=None):
    return dict(id=i or f"H{floor}", floor=floor, score=score, n_kf=n_kf, status=status)


def e2(coarse=9, exact=0):
    return dict(floors_total=9, floors_done=list(range(exact)), coarse=dict(floors_done=list(range(coarse))))


def run(F, seq):
    """seq = (t, hyps, e2, path, long, extras...) -> class after each."""
    out = []
    for t, h, E, p, L in seq:
        out.append(F.tick(e2=E, e3=dict(hypotheses=h), long_d=L, path_m=p, now=t)["structure_class"])
    return out


# ------------------------------ REAL replays
@pytest.mark.skipif(not (OPR / "X20261102_210254").exists(), reason="field record")
def test_real_latest_structure_rich_run_needs_no_sign_stop():
    import auto_replay as R
    r = R.replay("X20261102_210254")
    tr = r["transitions"]
    assert tr and tr[0]["lead"] == 4    # R40: E3's gap alone no longer settles it: see test_r40


@pytest.mark.skipif(not (OPR / "X20261102_210840").exists(), reason="field record")
def test_real_latest_ambiguous_run_turns_to_signs():
    import auto_replay as R
    r = R.replay("X20261102_210840")
    assert r["final"] == "AMBIGUOUS" and all(o["to"] != "DISCRIMINATIVE" for o in r["transitions"])


@pytest.mark.parametrize("run_id", ["X20261102_053723", "X20261102_023712", "X20261101_192409"])
def test_real_big_e3_gap_does_not_silence_signs_for_ever(run_id):
    """The E3 gap goes up in these recorded runs although fair rivals later tie (4/7/8F): the machine must come back to signs."""
    import auto_replay as R
    if not (OPR / run_id).exists():
        pytest.skip("field record")
    assert R.replay(run_id)["final"] == "AMBIGUOUS"


# ------------------------------ Scenario A (SYNTHETIC inputs): structure sufficient
def test_a_synthetic_clear_leader_withholds_stops_after_hysteresis(tmp_path):
    (tmp_path / "e2").mkdir()
    F = A.AutoFSM(tmp_path)
    H = [hy(4, 0.3), hy(7, 0.8, status="SUSPENDED")]
    c = run(F, [(0, H, e2(), 0, None), (10, H, e2(), 5, None), (31, H, e2(), 9, None)])
    assert c == ["AMBIGUOUS", "AMBIGUOUS", "DISCRIMINATIVE"]   # R40: all floors searched and not yet clear: signs allowed until structure has shown it
    assert F.d["semantic_stop_allowed"] is False and F.d["e2_max_busy"] == 4
    assert A.stop_gate(tmp_path, now=F.d["wall"]) == "SUPPRESSED_STRUCTURE_SUFFICIENT"
    assert (tmp_path / "e2" / "e2_budget.json").exists() and (tmp_path / "auto_transitions.jsonl").read_text().count("DISCRIMINATIVE") >= 1


def test_a_synthetic_one_good_frame_is_not_enough():
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.8)]
    assert run(F, [(0, H, e2(), 0, None), (5, H, e2(), 9, None)])[-1] != "DISCRIMINATIVE"       # no time
    F = A.AutoFSM()
    assert run(F, [(0, H, e2(), 0, None), (31, H, e2(), 2, None)])[-1] != "DISCRIMINATIVE"     # no new path


def test_a_synthetic_no_discriminative_before_all_floors_had_a_search():
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.8)]
    assert run(F, [(0, H, e2(coarse=3), 0, None), (60, H, e2(coarse=3), 20, None)])[-1] == "UNDETERMINED"


def test_a_synthetic_alone_on_evidence_is_not_discriminative():
    """Rivals without keyframes were never compared."""
    assert A.assess([hy(4, 0.3), hy(7, 0.9, n_kf=2)], e2())["raw"] == "MIDDLE"


# ------------------------------ Scenario B (SYNTHETIC inputs): the state follows reading -> E7 -> FIX; the FIX is not made here
def test_b_synthetic_semantic_states():
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.31)]
    run(F, [(0, H, e2(), 0, None), (11, H, e2(), 2, None)])
    assert F.cls == "AMBIGUOUS" and F.d["semantic_stop_allowed"] and F.d["e2_max_busy"] == 3
    assert F.d["state"] == "SEMANTIC_ACQUISITION_ENABLED"
    assert F.tick(e2=e2(), e3=dict(hypotheses=H), semantic_status="PROVISIONAL", now=12)["state"] == "SEMANTIC_CANDIDATE"
    assert F.tick(e2=e2(), e3=dict(hypotheses=H), pair_status="E7_VERIFYING", now=13)["state"] == "E7_SEMANTIC_VERIFY"
    d = F.tick(e2=e2(), e3=dict(hypotheses=H), global_decision="GLOBAL_CONFIRMED", now=14)
    assert d["state"] == "GLOBAL_FIX" and not d["semantic_stop_allowed"]


def test_no_consent_says_so_and_structure_continues():
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.31)]
    F.tick(e2=e2(), e3=dict(hypotheses=H), e6_alive=False, now=0)
    d = F.tick(e2=e2(), e3=dict(hypotheses=H), e6_alive=False, now=11)
    assert d["needs_pantilt_consent"] and d["state"] == "STRUCTURE_AMBIGUOUS"


# ------------------------------ Scenario C (SYNTHETIC inputs): ambiguous corridor, then a distinctive place
def test_c_synthetic_ambiguous_to_structure_without_any_sign():
    F = A.AutoFSM()
    amb = [hy(4, 0.3), hy(7, 0.31)]
    clr = [hy(4, 0.3), hy(7, 0.6)]
    c = run(F, [(0, amb, e2(), 0, None), (11, amb, e2(), 5, None), (20, clr, e2(), 8, None), (40, clr, e2(), 12, None), (51, clr, e2(), 14, None)])
    assert c[1] == "AMBIGUOUS" and c[-1] == "DISCRIMINATIVE"


def test_c_synthetic_would_confirm_starts_the_structure_verification_even_while_ambiguous():
    class Store:
        rows = [(0, 0, 0.0, 0.0), (1, 100, 30.0, 0.0)]
    class Tr:
        store = Store()
    class Link:
        tr = Tr()
        last = {}
    class Long:
        d = dict(status="WOULD_CONFIRM", leader_floor=4, next_gap_m=0.06)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        V = FV.FinalVerify(td, Link(), Long(), lambda: dict(stopped=False), clock=lambda: 1000.0)
        V.tick(1000.0)
        assert V.d["status"] == "FINAL_VERIFICATION_STOP_REQUESTED"
        F = A.AutoFSM()
        F.cls = "AMBIGUOUS"
        d = F.tick(e2=e2(), e3=dict(hypotheses=[hy(4, 0.3), hy(7, 0.31)]), long_d=Long.d, final_status=V.d["status"], now=1000.0)
        assert d["state"] == "E7_STRUCTURE_VERIFY" and d["structure_class"] == "DISCRIMINATIVE"


def test_c_synthetic_escalation_returns_only_with_fair_rivals():
    F = A.AutoFSM()
    clr = [hy(4, 0.3), hy(7, 0.8)]
    run(F, [(0, clr, e2(), 0, None), (31, clr, e2(), 9, None)])
    assert F.cls == "DISCRIMINATIVE"
    assert run(F, [(182, clr, e2(), 60, None)])[-1] == "AMBIGUOUS" and F.escalated
    assert run(F, [(300, clr, e2(), 90, None), (340, clr, e2(), 110, None)])[-1] == "AMBIGUOUS"
    assert run(F, [(350, clr, e2(), 115, dict(status="WOULD_CONFIRM", leader_floor=4))])[-1] == "DISCRIMINATIVE"


def test_long_ambiguous_floors_overrides_a_big_e3_gap():
    F = A.AutoFSM()
    clr = [hy(4, 0.3), hy(7, 0.8)]
    L = dict(status="AMBIGUOUS_FLOORS", leader_floor=4, close_floors=[7, 8])
    assert A.assess(clr, e2(), None, L)["raw"] == "AMB"
    run(F, [(0, clr, e2(), 0, None), (31, clr, e2(), 9, None)])
    assert run(F, [(32, clr, e2(), 9, L), (43, clr, e2(), 9, L)])[-1] == "AMBIGUOUS"


def test_flicker_does_not_switch():
    """SYNTHETIC. One bad tick inside a clear run, and one clear tick inside an ambiguous run."""
    F = A.AutoFSM()
    clr, amb = [hy(4, 0.3), hy(7, 0.8)], [hy(4, 0.3), hy(7, 0.31)]
    run(F, [(0, clr, e2(), 0, None), (31, clr, e2(), 9, None)])
    assert run(F, [(32, amb, e2(), 9, None), (33, clr, e2(), 9, None), (34, amb, e2(), 9, None), (35, clr, e2(), 9, None)])[-1] == "DISCRIMINATIVE"
    F = A.AutoFSM()
    run(F, [(0, amb, e2(), 0, None), (11, amb, e2(), 2, None)])
    assert run(F, [(12, clr, e2(), 2, None), (13, amb, e2(), 2, None), (14, clr, e2(), 3, None)])[-1] == "AMBIGUOUS"


def test_early_ambiguity_before_the_floor_search_ends():
    """E4's early call: no wait for the exact pass."""
    F = A.AutoFSM()
    amb = [hy(4, 0.3), hy(2, 0.31)]
    assert run(F, [(0, amb, e2(coarse=4), 0, None), (21, amb, e2(coarse=4), 5, None)])[-1] == "AMBIGUOUS"


def test_no_decision_is_not_kept_for_ever():
    F = A.AutoFSM()
    mid = [hy(4, 0.3), hy(7, 0.40)]
    assert run(F, [(0, mid, e2(), 0, None), (61, mid, e2(), 20, None)])[-1] == "AMBIGUOUS"


def test_time_does_not_touch_the_machine_at_300_s():
    """Scenario D (SYNTHETIC): E6 work across 300 s: class, stop permission and SigLIP reason stay; no time input exists."""
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.31)]
    run(F, [(0, H, e2(), 0, None), (11, H, e2(), 2, None)])
    for t in (240, 300, 900, 3000):
        d = F.tick(e2=e2(), e3=dict(hypotheses=H), hw="WAIT_MANUAL_FOCUS", now=t)
        assert d["structure_class"] == "AMBIGUOUS" and d["semantic_stop_allowed"] and d["state"] == "SEMANTIC_ACQUISITION_ENABLED" and d["e2_max_busy"] == 2   # hw busy


# ------------------------------ stop gate in the real E5 stop code
def line():
    return dict(odom_raw=dict(x=0.0, y=0.0, yaw=0.0), depth_dt_s=0.01, frame_id=1)


def rec(spot, xyz=(3.0, 1.0, 0.5), sim=0.2, t=1.0):
    return dict(t=t, regions=[dict(sim=sim, msim=sim, z=3.0, spot=spot, box=[10, 10, 50, 50], xyz=list(xyz), dec=dict(P2=dict(state="VERIFIED")))])


@pytest.mark.parametrize("cls,allowed,why", [("DISCRIMINATIVE", False, "SUPPRESSED_STRUCTURE_SUFFICIENT"), ("UNDETERMINED", True, None), ("AMBIGUOUS", True, None)])
def test_stop_gate_follows_the_class(tmp_path, cls, allowed, why):
    now = time.time()
    (tmp_path / "auto_state.json").write_text(json.dumps(dict(state="X", structure_class=cls, semantic_stop_allowed=allowed, wall=now)))
    v = VS.VisualStop(tmp_path, state_dir=tmp_path)
    o = v.feed(rec(1), now, line(), now=now)
    assert (o is not None) == allowed
    if why:
        assert v.n_gate.get(why) == 1
        assert why in (tmp_path / "live2_stop_suppressed.jsonl").read_text()


def test_stop_gate_without_or_with_an_old_file_fails_towards_asking(tmp_path):
    assert A.stop_gate(tmp_path) is None
    (tmp_path / "auto_state.json").write_text(json.dumps(dict(structure_class="DISCRIMINATIVE", semantic_stop_allowed=False, wall=time.time() - 60)))
    assert A.stop_gate(tmp_path) is None


def test_after_fix_no_more_stops(tmp_path):
    (tmp_path / "auto_state.json").write_text(json.dumps(dict(state="GLOBAL_FIX", structure_class="AMBIGUOUS", semantic_stop_allowed=False, wall=time.time())))
    assert A.stop_gate(tmp_path) == "SUPPRESSED_AFTER_GLOBAL_FIX"


# ------------------------------ Scenario E (SYNTHETIC inputs in the real E5 stop code): the next plate after a photo
def test_e_second_plate_after_first_photo_gets_its_own_request(tmp_path):
    now = time.time()
    (tmp_path / "auto_state.json").write_text(json.dumps(dict(state="SEMANTIC_ACQUISITION_ENABLED", structure_class="AMBIGUOUS", semantic_stop_allowed=True, wall=now + 1000)))
    v = VS.VisualStop(tmp_path, state_dir=tmp_path)
    a = v.feed(rec(1), now, line(), now=now)
    assert a and a["observation_id"].endswith("O001")
    (tmp_path / "e6_precision_results.jsonl").write_text(json.dumps(dict(observation_id=a["observation_id"], status="CAPTURE_STORED")) + "\n")
    t = now + 40
    moving = dict(odom_raw=dict(x=10.0, y=0.0, yaw=0.0), depth_dt_s=0.01, frame_id=2)
    v.hist = [(t - 2.5 + i * 0.5, 7.0 + i, 0.0) for i in range(5)]
    b = v.feed(rec(9, xyz=(3.0, 1.0, 0.5)), t, moving, now=t)
    assert b and b["observation_id"].endswith("O002") and b["observation_id"] != a["observation_id"]


def test_e_same_plate_stays_suppressed(tmp_path):
    now = time.time()
    v = VS.VisualStop(tmp_path, state_dir=tmp_path)
    a = v.feed(rec(1), now, line(), now=now)
    (tmp_path / "e6_precision_results.jsonl").write_text(json.dumps(dict(observation_id=a["observation_id"], status="CAPTURE_STORED")) + "\n")
    t = now + 100
    v.hist = [(t - 2.5 + i * 0.5, i, 0.0) for i in range(5)]
    assert v.feed(rec(9), t, line(), now=t) is None and v.n_gate.get("SUPPRESSED_ALREADY_PHOTOGRAPHED") == 1


# ------------------------------ final_verify must not interrupt E6
def test_final_verify_waits_for_e6_and_goes_out_after(tmp_path):
    class Store:
        rows = [(0, 0, 0.0, 0.0), (1, 100, 30.0, 0.0)]
    class Tr:
        store = Store()
    class Link:
        tr = Tr()
        last = {}
    class Long:
        d = dict(status="WOULD_CONFIRM", leader_floor=4)
    busy = [True]
    V = FV.FinalVerify(tmp_path, Link(), Long(), lambda: dict(stopped=False), clock=lambda: 1000.0, busy_fn=lambda: busy[0])
    V.tick(1000.0)
    assert V.d["status"] == "WATCHING"
    busy[0] = False
    V.tick(1001.0)
    assert V.d["status"] == "FINAL_VERIFICATION_STOP_REQUESTED"


def test_final_verify_silent_after_a_semantic_fix(tmp_path):
    class Store:
        rows = [(0, 0, 0.0, 0.0), (1, 100, 30.0, 0.0)]
    class Tr:
        store = Store()
    class Link:
        tr = Tr()
        last = dict(global_decision="GLOBAL_CONFIRMED")
    class Long:
        d = dict(status="WOULD_CONFIRM", leader_floor=4)
    V = FV.FinalVerify(tmp_path, Link(), Long(), lambda: dict(stopped=False), clock=lambda: 1000.0)
    V.tick(1000.0)
    assert V.d["status"] == "WATCHING"


# ------------------------------ hardware names
@pytest.mark.parametrize("ph,st,hw", [("AIMING", None, "PANTILT_AIMING"), ("WAIT_MANUAL_FOCUS", None, "WAIT_MANUAL_FOCUS"), ("CAPTURE_REVIEW", None, "CAPTURE_REVIEW"), ("REPOSITION_NEEDED", None, "REPOSITION_NEEDED"),
                                      ("RETURNING_HOME", None, "HOME_RETURNING"), ("HOME_CONFIRMED", None, "MOVE_ALLOWED"), (None, "WAITING_FOR_SETTLE_APPROVAL", "WAIT_BODY_SETTLE"), (None, None, "E6_IDLE")])
def test_hw_names(ph, st, hw):
    assert A.phase_to_hw(ph, st) == hw


# ------------------------------ the operating timer (REAL events of the latest run)
@pytest.mark.skipif(not (OPR / "X20261102_210840").exists(), reason="field record")
def test_ops_timer_real_latest_run():
    X = OPR / "X20261102_210840"
    t0 = json.loads((X / "experiment.json").read_text())["created_wall"]
    d = OC.ops_timer(X / "state", "X20261102_210840", t0, t0 + 410)
    assert d["ops_paused"] is False and 230 < d["ops_s"] < 260      # about 165 s of E6 work removed from 410 s
    busy = OC.ops_timer(X / "state", "X20261102_210840", t0, t0 + 150)
    assert busy["ops_paused"] is True and busy["ops_s"] < 80


def test_ops_timer_does_not_change_the_other_clocks(tmp_path):
    (tmp_path / "e6_manual_events.jsonl").write_text(json.dumps(dict(kind="PHASE", phase="AIMING", wall=110.0)) + "\n")
    d = OC.ops_timer(tmp_path, "X", 100.0, 200.0)
    assert d == dict(ops_s=10.0, ops_paused=True)
    c = OC.clock(OC.intervals(tmp_path, "X"), 100.0, 200.0)
    assert c["raw_s"] == 100.0


# ------------------------------ VLM status words
@pytest.mark.parametrize("rd,srv,want", [(None, None, "NOT_REQUESTED"), (dict(vlm=dict(state="QUEUED")), dict(state="LOADING"), "LOADING"), (dict(vlm=dict(state="QUEUED")), dict(state="READY"), "QUEUED"), (dict(vlm=dict(state="ASKING")), None, "RUNNING"),
                                          (dict(vlm=dict(state="ASKED")), None, "DONE"), (dict(vlm=dict(state="NOT_ADMITTED_RESOURCE")), None, "SKIPPED_RESOURCE"), (dict(vlm=dict(state="ERROR", error="Timeout: x")), None, "TIMEOUT"), (dict(vlm=dict(state="ERROR", error="boom")), None, "FAILED"),
                                          (dict(vlm=dict(state="UNAVAILABLE")), None, "NOT_REQUESTED")])
def test_vlm_status(rd, srv, want):
    assert SV.vlm_state(rd, srv) == want


# ------------------------------ E2 budget
def test_e2_budget_children(tmp_path):
    import types
    import e2_worker as W
    self = types.SimpleNamespace(a=types.SimpleNamespace(out=str(tmp_path), procs=4))
    self.budget = lambda: W.Coordinator.budget(self) if hasattr(W, "Coordinator") else None
    C = next(v for v in vars(W).values() if isinstance(v, type) and hasattr(v, "apply_budget"))
    self.budget = lambda: C.budget(self)
    assert C.budget(self) == 4
    (tmp_path / "e2_budget.json").write_text(json.dumps(dict(max_busy=2)))
    assert C.budget(self) == 2
    (tmp_path / "e2_budget.json").write_text(json.dumps(dict(max_busy=9)))
    assert C.budget(self) == 4
    class P:
        def terminate(self): self.t = True
        def join(self, x): pass
    kids = {j: dict(pr=P(), busy=None) for j in range(4)}
    kids[3]["busy"] = ("exact", 1)
    spawned = []
    (tmp_path / "e2_budget.json").write_text(json.dumps(dict(max_busy=2)))
    C.apply_budget(self, kids, lambda j: spawned.append(j))
    assert sorted(kids) == [0, 1, 3] and not spawned      # the idle one above the budget ends; the busy one finishes its floor
    kids[3]["busy"] = None
    (tmp_path / "e2_budget.json").write_text(json.dumps(dict(max_busy=4)))
    C.apply_budget(self, kids, lambda j: spawned.append(j))
    assert spawned == [2]


# ------------------------------ no input carries the place
def test_no_zone_name_or_gt_in_the_machine():
    src = (HERE.parent / "candidate/live2/auto_fsm.py").read_text()
    code = src.split('"""', 2)[2]
    for w in ("loc1", "loc2", "LOC1", "LOC2", "gt.jsonl", "start_gt", "evaluation"):
        assert w not in code
    e34 = (HERE.parent / "candidate/live2/e34_node.py").read_text()
    assert "LOC1" not in e34 and "LOC2" not in e34


def test_vlm_resident_leaves_one_e2_child():
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.31)]
    assert F.tick(e2=e2(), e3=dict(hypotheses=H), vlm_resident=True, now=1)["e2_max_busy"] == 0


# ================ R40
@pytest.mark.parametrize("run_id", ["X20261103_000434", "X20261103_001810", "X20261103_003729"])
def test_r40_real_runs_allow_signs_without_long_silence(run_id):
    import auto_replay as R
    if not (OPR / run_id).exists():
        pytest.skip("field record")
    r = R.replay(run_id)
    first_amb = next((o["t"] for o in r["transitions"] if o["to"] == "AMBIGUOUS"), None)
    assert first_amb is not None and first_amb < 130     # signs available soon after the first coarse search of all floors
    # X20261103_003729 is 'discriminative' on 9F for about 30 s in this replay too (the history has no px/m to refuse it); E7's comparison ends it. A wrong call may not silence signs for long
    tr = r["transitions"]
    for a_, b_ in zip(tr, tr[1:] + [dict(t=r["t_end"])]):
        if a_["to"] == "DISCRIMINATIVE" and a_["lead"] != 4:
            assert b_["t"] - a_["t"] < 60


def test_r40_wrong_scale_leader_is_not_believable():
    q = A.pose_quality(dict(id="H106_F4", floor=4, ppm=33.1, recent=0.96, n_kf=40))
    assert q["low"] and any("축척" in x for x in q["reasons"])
    assert not A.pose_quality(dict(id="H154_F4", floor=4, ppm=23.4, recent=0.2, n_kf=92))["low"]


def test_r40_big_gap_with_poor_fit_is_not_structure():
    H = [dict(hy(4, 0.6, n_kf=40), ppm=33.1, recent=0.96), dict(hy(3, 0.9, n_kf=30, status="SUSPENDED"), ppm=28.0, recent=0.5)]
    assert A.assess(H, e2())["raw"] == "MIDDLE"


def test_r40_vlm_waiting_frees_e2_memory():
    F = A.AutoFSM()
    H = [hy(4, 0.3), hy(7, 0.31)]
    assert F.tick(e2=e2(), e3=dict(hypotheses=H), vlm_pending=True, now=1)["e2_max_busy"] == 0


def test_r40_vlm_only_needs_number_department_and_sharpness():
    import plate_text as PT
    base = dict(vlm=dict(sign_type="room_number_plate", room_number="407", text="로봇공학과 차카타"), reading=dict(vlm=dict(crop=True)), image_quality=dict(roi_laplacian_var=32.6), ocr_lines=[])
    assert PT.vlm_only_room(base) == "407"
    assert PT.vlm_only_room(dict(base, vlm=dict(base["vlm"], room_number="507"))) is None
    assert PT.vlm_only_room(dict(base, vlm=dict(base["vlm"], text=""))) is None
    assert PT.vlm_only_room(dict(base, image_quality=dict(roi_laplacian_var=3.2))) is None
    assert PT.vlm_only_room(dict(base, reading=dict(vlm=dict(crop=False)))) is None
    assert PT.vlm_only_room(dict(base, ocr_lines=[["405", 0.9]])) is None
    assert PT.vlm_only_room(dict(base, vlm=dict(base["vlm"], sign_type="floor_directory"))) is None


def _fl(f, c, stage="refresh", exact=False, **kw):
    return dict(floor=f, cost=c, stage=stage, exact_done=exact, **kw)


def test_r40_no_wait_for_all_exact_but_fair_rivals_needed():
    import e7_long as EL
    fl = [_fl(4, 0.03, "exact", True, scale_rel=0.03)] + [_fl(f, 0.09) for f in (1, 2, 3, 5, 6, 7, 8, 9)]
    h = [dict(t_end=1, path_m=70, floors=fl, floors_exact=[4], n_refresh=2), dict(t_end=2, path_m=80, floors=fl, floors_exact=[4], n_refresh=2)]
    assert EL.classify(h, 9)["status"] == "WOULD_CONFIRM"
    fl2 = [fl[0]] + [_fl(f, 0.09, "coarse") for f in (1, 2, 3, 5, 6, 7, 8, 9)]
    assert EL.classify([dict(h[0], floors=fl2), dict(h[1], floors=fl2)], 9)["status"] == "LEADER_UNRESOLVED"


def test_r40_right_floor_wrong_place_blocks_the_confirm():
    import e7_long as EL
    fl = [_fl(4, 0.03, "exact", True, scale_rel=0.03, pos_rival=dict(gap=0.01, dist_m=77.0))] + [_fl(f, 0.09) for f in (1, 2, 3, 5, 6, 7, 8, 9)]
    h = [dict(t_end=1, path_m=70, floors=fl, n_refresh=2), dict(t_end=2, path_m=80, floors=fl, n_refresh=2)]
    c = EL.classify(h, 9)
    assert c["status"] == "LEADER_UNRESOLVED" and "77" in c["why"]


def test_r40_wrong_scale_blocks_the_confirm():
    import e7_long as EL
    fl = [_fl(4, 0.03, "exact", True, scale_rel=0.38)] + [_fl(f, 0.09) for f in (1, 2, 3, 5, 6, 7, 8, 9)]
    assert EL.classify([dict(t_end=1, path_m=70, floors=fl, n_refresh=2), dict(t_end=2, path_m=80, floors=fl, n_refresh=2)], 9)["status"] == "LEADER_UNRESOLVED"


def test_r40_final_verify_checks_place_and_scale():
    good = [dict(floor=4, cost=0.03, inlier=0.6, converged=True, crossing=0.1, scale_rel=0.03), dict(floor=7, cost=0.09, crossing=0.1)]
    assert FV.verdict(dict(floors=good), 4, 0.1, True)[0] == "VERIFIED_FIX"
    assert FV.verdict(dict(floors=[dict(good[0], scale_rel=0.3), good[1]]), 4, 0.1, True)[0] == "VERIFY_FAILED"
    assert FV.verdict(dict(floors=[dict(good[0], pos_rival=dict(gap=0.01, dist_m=70)), good[1]]), 4, 0.1, True)[0] == "VERIFY_FAILED"


def test_r40_final_verify_searches_all_floors_before_deciding(tmp_path):
    e2d = tmp_path / "e2"; e2d.mkdir()
    class Store:
        rows = [(0, 0, 0.0, 0.0), (1, 100, 30.0, 0.0)]
    class Tr:
        store = Store(); e2_dir = e2d
    class Link:
        tr = Tr(); last = {}
    class Long:
        d = dict(status="WOULD_CONFIRM", leader_floor=4)
        asked = 0
        def request_now(self): Long.asked += 1
        def latest_after(self, w): return None
    t = [1000.0]
    V = FV.FinalVerify(tmp_path, Link(), Long(), lambda: dict(stopped=True), clock=lambda: t[0])
    V.tick(t[0]); assert V.d["status"] == "FINAL_VERIFICATION_STOP_REQUESTED"
    V.tick(t[0]); t[0] += 4; V.tick(t[0])
    assert V.d["status"] == "VERIFYING" and (e2d / "e2_refresh_now.json").exists() and Long.asked == 0
    t[0] += 10; V.tick(t[0]); assert Long.asked == 0
    (e2d / "e2_state.json").write_text(json.dumps(dict(floors_total=9, refresh=dict(last_wall=t[0], n_passes=3), candidates_refresh=[dict(floor=f, refresh_n=3) for f in range(1, 10)])))
    V.tick(t[0]); assert Long.asked == 1


def test_r40_settle_gate_follows_a_small_move(tmp_path):
    import e6_settle as ST
    class Clock:
        t = 1000.0
        def time(self): return Clock.t
    moved = [True]
    stops = [dict(stopped=False), dict(stopped=True), dict(stopped=True)]
    g = ST.SettleGate(tmp_path, tmp_path, clock=Clock(), sleep=lambda s: setattr(Clock, "t", Clock.t + 1))
    obs = dict(observation_id="O1", created_timestamp=999.0, stopped_wall=999.5, odom_pose_at_stop=dict(x=0, y=0))
    rb = []
    def rebase(o):
        rb.append(1); moved[0] = False
        return dict(odom_pose_at_stop=dict(x=0.3, y=0), target_note="new")
    n = [0]
    def stop():
        n[0] += 1
        return stops[min(n[0] - 1, 2)]
    def ended():
        return n[0] > 4
    out, why, rec = g.wait(obs, "X", stop, lambda: dict(stale=True), lambda: dict(valid=True), lambda o: (moved[0], "0.30 m"), ended=ended, rebase=rebase)
    assert rb and obs["target_note"] == "new" and out == "EXPERIMENT_ENDED"   # stays in the wait with the new pose, instead of ROBOT_MOVED


def test_r40_settle_gate_without_rebase_ends_as_before(tmp_path):
    import e6_settle as ST
    g = ST.SettleGate(tmp_path, tmp_path, sleep=lambda s: None)
    out, why, rec = g.wait(dict(observation_id="O1"), "X", lambda: dict(stopped=True), lambda: dict(stale=True), lambda: dict(valid=True), lambda o: (True, "0.30 m"))
    assert out == "ROBOT_MOVED"


def test_r40_retry_never_after_a_rejection(tmp_path):
    import types
    import e8_pair as EP
    p = EP.PairManager.__new__(EP.PairManager)
    p.status, p.gid, p.state, p.clock = "MORE_OBSERVATION_REQUIRED", "G", tmp_path, lambda: 1000.0
    p.last_eval = dict(verdict="REGION_CANDIDATE_READY", candidates=[dict(id="A")])
    p.e7 = dict(observation_id="G")
    p.group = [dict(observation_id="a", target_odom_link_xy=[0, 0])]
    p.tr = types.SimpleNamespace(store=types.SimpleNamespace(rows=[(0, 0, 0.0, 0.0), (1, 1, 50.0, 0.0)]), alive=lambda: [types.SimpleNamespace(id="A")])
    asked = []
    p.request_e7 = lambda c, r: asked.append(r)
    p._set = lambda *a: None
    (tmp_path / "e7_semantic_shadow.json").write_text(json.dumps(dict(rejected_by=["PLATE_TEXT_CONFLICT"])))
    assert p.retry_e7() is False and not asked
    (tmp_path / "e7_semantic_shadow.json").write_text(json.dumps(dict(rejected_by=[])))
    assert p.retry_e7() is False      # first call only sets the mark
    p.clock = lambda: 1070.0
    p._e7_retry["path"] = 40.0
    assert p.retry_e7() is True and asked
    n = len(asked)
    p.status = "MORE_OBSERVATION_REQUIRED"; p.e7 = dict(observation_id="G", e7=dict(t_end=1060.0)); p._e7_retry["path"] = 30.0
    assert p.retry_e7() is False and len(asked) == n     # the last window ended 10 s ago


def test_r40_moved_while_settling_asks_for_reposition_instead_of_ending(tmp_path):
    """X20261103_003729 O001: the operator moved after [①]; R39 ended the observation (no head motion had happened). Now: guidance and [재조준] on the same plate. REAL target O006 of X20261102_025326, fake aim console."""
    import test_e8_manual_focus_fake as F
    import test_r36 as T36
    o = T36.real_o006()
    c = F.FakeConsole(F.render())
    p, ch, _ = F.build(tmp_path, c, script=[None] * 3 + [dict(action="reposition_skip")])
    p.envelope = T36.ENV
    p.reposition = dict(K=lambda: T36.K, samples=lambda: T36.moved_back(o, 0.0))
    n = [0]
    def stop():
        n[0] += 1
        return dict(stopped=n[0] == 1, why="MOVING speed 0.3 m/s", speed_mps=0.3)
    p.stop_check = stop
    r = p.run(o, 1)
    phases = [h["phase"] for h in ch.d.get("history") or []]
    assert "REPOSITION_NEEDED" in phases and phases.count("AIMING") == 1 and phases[0] == "AIMING" and r["status"] == "ROBOT_NOT_STOPPED"   # the first entry is the channel's opening label, before the settle wait; no aim follows


def _yaw_rows(run):
    import csv
    X = OPR / run
    t0 = json.loads((X / "experiment.json").read_text())["created_wall"]
    r = list(csv.DictReader(open(X / "archive/live2_store/odom_link.csv")))
    k = "fastlio_yaw_rad"
    return t0, [(float(x["t_sec"]), float(x[k])) for x in r]


@pytest.mark.skipif(not (OPR / "X20261103_003729").exists(), reason="field record")
def test_r40_new_window_fires_after_the_real_turn_and_not_in_the_corridor():
    import e2_worker as W
    t0, rows = _yaw_rows("X20261103_003729")
    told = t0 + 102.0
    assert not W.new_window_due(rows, told, t0 + 125.0, 60.0)     # straight corridor, turn at about +150 s
    assert not W.new_window_due(rows, told, t0 + 135.0, 60.0)
    assert W.new_window_due(rows, told, t0 + 170.0, 60.0)
    assert not W.new_window_due(rows, t0 + 170.0, t0 + 180.0, 60.0)   # too close to the last one


def test_r40_new_structure_window_requeues(tmp_path):
    """Real method, real odometry file of X20261103_003729: queued floors move to the new window, finished ones go again; the running code path has no silent failure on the file format."""
    import shutil, types
    import e2_worker as W
    t0, _ = _yaw_rows("X20261103_003729")
    shutil.copy(OPR / "X20261103_003729/archive/live2_store/odom_link.csv", tmp_path / "odom_link.csv")
    log = []
    C = next(v for v in vars(W).values() if isinstance(v, type) and hasattr(v, "new_structure_window"))
    s = types.SimpleNamespace(store=tmp_path, W=60.0, floors=list(range(1, 10)), state={}, paused=lambda: False, span=lambda: (t0, t0 + 170.0), window_path_m=lambda t: 20.0, a=types.SimpleNamespace(min_window_path_m=2.0), log=lambda k, **kw: log.append(kw))
    LS = types.SimpleNamespace(pin=lambda *a, **k: None)
    q = [("exact", 4, t0 + 102.0, 60.0), ("exact", 9, t0 + 102.0, 60.0), ("refresh", 1, t0 + 100.0, 60.0)]
    t = C.new_structure_window(s, q, t0 + 102.0, LS)
    assert t == t0 + 170.0 and q[0][2] == t and q[1][2] == t and q[2][2] == t0 + 100.0
    assert sorted(x[1] for x in q if x[0] == "exact") == list(range(1, 10)) and all(x[2] == t for x in q if x[0] == "exact")
    assert s.state["exact_new_windows"] == 1 and log[0]["requeued"] == [4, 9]
