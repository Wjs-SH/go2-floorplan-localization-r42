"""R35 unit tests. SYNTHETIC inputs except where named: E7 structural class, GLOBAL_CONFIRMED unchanged, shadow rules and authority, scale survey, E3 lanes, status panel, PAPER_METRICS on the REAL archived loc1 / loc2 records."""
import importlib.util, itertools, json, shutil, sys, types
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(STG))
import e6e7_link as L, e7_semantic_shadow as SH, scale_survey as SV, paper_metrics as PM, status_view as SVW, e3_tracker as E3  # noqa: E401,E402
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"


def R(cid, score, floor=4, conv=True, amb=False, cross=0.05, scale=1.0, ppm=24.0):
    return dict(candidate_id=cid, floor_id=floor, alignment_score=score, convergence=conv, local_ambiguity=dict(ambiguous=amb), wall_crossing=dict(violation=cross), correction=dict(scale=scale), candidate_global_pose=dict(ppm=ppm))


def res(results, best, final="MORE_OBSERVATION_REQUIRED", gap=None, ambiguity=None, veto=False, indist=False):
    return dict(results=results, best=best, final_decision=final, gap_m=gap, ambiguity=ambiguity, crossing_veto=veto, structurally_indistinguishable=indist)


def test_rival_first_by_less_than_gap_is_ambiguous_not_rejected():
    r = res([R("S", 0.0021), R("X", 0.0, floor=2, conv=False)], "X", gap=0.0021, ambiguity=True)
    assert L.structural_class(r, "S")[0] == "STRUCTURALLY_AMBIGUOUS"
    d, why = L.global_decision(r, "S")
    assert d == "AMBIGUOUS" and "geometry disagrees" not in why


def test_real_contradiction_stays_rejected():
    r = res([R("S", 0.27), R("X", 0.0, floor=2)], "X", gap=0.27, ambiguity=False)
    assert L.structural_class(r, "S")[0] == "STRUCTURAL_CONTRADICTION" and L.global_decision(r, "S")[0] == "REJECTED"


def test_no_rival_and_failed_fit_is_contradiction():
    assert L.structural_class(res([R("S", 0.4)], "S", gap=None), "S")[0] == "STRUCTURAL_CONTRADICTION"


def test_compatible_only_when_first_separated_converged_no_veto():
    ok = res([R("S", 0.0), R("X", 0.1, floor=2)], "S", final="GEOMETRY_CONSISTENT_SEMANTIC_PENDING", gap=0.1, ambiguity=False)
    assert L.structural_class(ok, "S")[0] == "STRUCTURALLY_COMPATIBLE"
    assert L.structural_class(dict(ok, crossing_veto=True), "S")[0] == "STRUCTURALLY_AMBIGUOUS"
    assert L.structural_class(res([R("S", 0.0, conv=False), R("X", 0.1, floor=2)], "S", gap=0.1), "S")[0] == "STRUCTURALLY_AMBIGUOUS"


def old_decision(r, sem):
    if not r.get("results"):
        return "MORE_OBSERVATION_REQUIRED"
    if r.get("best") != sem:
        return "REJECTED"
    if r.get("final_decision") == "GLOBAL_CONFIRMED":
        return "GLOBAL_CONFIRMED"
    return "AMBIGUOUS" if r.get("ambiguity") else "MORE_OBSERVATION_REQUIRED"


def test_global_confirmed_is_exactly_the_old_condition_over_all_combinations():
    n = 0
    for best, final, s_score, x_score, conv, amb, veto, empty in itertools.product(("S", "X"), ("GLOBAL_CONFIRMED", "MORE_OBSERVATION_REQUIRED", "GEOMETRY_CONSISTENT_SEMANTIC_PENDING", "GLOBAL_UNKNOWN"),
                                                                                  (0.0, 0.02, 0.2), (0.0, 0.02, 0.2), (True, False), (True, False, None), (True, False), (False, True)):
        r = res([] if empty else [R("S", s_score, conv=conv), R("X", x_score, floor=2)], best, final=final, gap=abs(s_score - x_score), ambiguity=amb, veto=veto)
        new, old = L.global_decision(r, "S")[0], old_decision(r, "S")
        assert (new == "GLOBAL_CONFIRMED") == (old == "GLOBAL_CONFIRMED"), (best, final, new, old)
        if old != "REJECTED":
            assert new in (old, "REJECTED") or (old == "MORE_OBSERVATION_REQUIRED" and new == "REJECTED"), (old, new)    # never softer than before except REJECTED -> AMBIGUOUS
        n += 1
    assert n > 1000


def O(oid, text, conf=.95, target=(0.0, 0.0), sha=None, vlm=None, extra=None):
    return dict(observation_id=oid, ocr_lines=[[text, conf]] + (extra or []), image_sha256=sha or f"s{oid}", target_odom_link_xy=list(target), capture_id=oid, vlm=vlm)


def EV(obs, verdict="REGION_CANDIDATE_READY", floors=(4,), kinds=None):
    per = []
    for o, k in zip(obs, kinds or ["ROOM"] * len(obs)):
        per.append(dict(observation_id=o["observation_id"], usable=True, reading_kind=k, labels=[[o["ocr_lines"][0][0], .7]], category=dict(category="restroom", original_text="TOILETS") if k == "CATEGORY" else None))
    return dict(verdict=verdict, why="x", per_observation=per, floors_explaining=list(floors), candidates=[dict(id="S", residuals_m=[.1], rooms=["a"])], regions=[dict(floor=4, hypotheses=["S"])], pairs=[])


E7OK = lambda **k: dict(res([R("S", 0.002, cross=k.get("cross", 0.05)), R("X", 0.0, floor=2, conv=False, cross=0.05)], "X", gap=0.002, ambiguity=True), structural_class=k.get("cls", "STRUCTURALLY_AMBIGUOUS"))
A = dict(gap_m=.05, xw_veto=.10, floor_px_per_m={4: 24.0})


def S(obs, ev=None, e7=None, kinds=None):
    return SH.evaluate(ev or EV(obs, kinds=kinds), obs, E7OK() if e7 is None else e7, "S", "AMBIGUOUS", **A)


def test_two_rooms_alone_are_not_enough_third_floor_evidence_is_required():
    two = [O("a", "405"), O("b", "408", target=(10, 0))]
    r = S(two)
    assert r["verdict"] == "WOULD_AMBIGUOUS" and "NO_FLOOR_EVIDENCE" in r["blocked_by"] and r["would_fix_2room"] is True
    assert S(two + [O("c", "407", target=(5, 0))])["verdict"] == "WOULD_AMBIGUOUS"            # R38: a third room number has the same digit risk
    assert S([O("a", "405"), O("b", "408", target=(10, 0), extra=[["4F", .9]])])["verdict"] == "WOULD_FIX"   # a floor marker does not depend on the digits


def test_category_does_not_count_as_floor_evidence():
    o = [O("a", "405"), O("b", "408", target=(10, 0)), O("c", "TOILETS", target=(14, 0))]
    r = S(o, kinds=["ROOM", "ROOM", "CATEGORY"])
    assert r["verdict"] == "WOULD_AMBIGUOUS" and "NO_FLOOR_EVIDENCE" in r["blocked_by"] and r["category_readings"]


def test_same_image_same_plate_same_label_are_not_independent():
    a = O("a", "405", target=(0, 0), sha="same")
    assert len(S([a, O("b", "408", target=(10, 0), sha="same")])["room_readings"]) == 1
    assert len(S([a, O("b", "408", target=(1, 0))])["room_readings"]) == 1
    assert len(S([a, O("b", "405", target=(10, 0))])["room_readings"]) == 1


def test_low_confidence_room_is_not_counted():
    assert "ROOM_READINGS_1_OF_2" in S([O("a", "405"), O("b", "408", conf=.7, target=(10, 0))])["blocked_by"]


def test_floor_marker_conflict_rejects():
    r = S([O("a", "505"), O("b", "508", target=(10, 0), extra=[["4F", .9]])], ev=EV([O("a", "505"), O("b", "508")], floors=(5,)))
    assert r["verdict"] == "WOULD_REJECT" and any("FLOOR_MARKER" in x for x in r["rejected_by"])


def test_contradiction_rejects_and_crossing_blocks_but_reports_the_alternative():
    t = [O("a", "405"), O("b", "408", target=(10, 0), extra=[["4F", .9]])]   # R38: a floor marker is the floor evidence; a third room number no longer is
    assert S(t, e7=E7OK(cls="STRUCTURAL_CONTRADICTION"))["verdict"] == "WOULD_REJECT"
    r = S(t, e7=E7OK(cross=0.2))
    assert r["verdict"] == "WOULD_AMBIGUOUS" and "CROSSING_MARGIN" in r["blocked_by"] and r["would_fix_no_crossing"] is True


def test_shadow_has_no_path_to_a_fix():
    src = (STG / "e7_semantic_shadow.py").read_text()
    assert "e7_state" not in src.replace("e7_state.json or to a FIX", "") and "GLOBAL_CONFIRMED" not in src
    link = (STG / "e6e7_link.py").read_text()
    body = link[link.index("def _semantic_shadow"):link.index("def _epoch")]
    assert "global_decision" not in body and "self.last" not in body


def test_scale_survey():
    rows = [dict(name="a", x1=0, y1=0, x2=250, y2=0, measured_m=10.0), dict(name="b", x1=0, y1=0, x2=0, y2=125, measured_m=5.0), dict(name="c", x1=0, y1=0, x2=500, y2=0, measured_m=20.1)]
    r = SV.analyse(rows, 4)
    assert r["status"] == "CONSISTENT" and abs(r["combined_px_per_m"] - 24.9) < 0.1 and r["scale_file_px_per_m"] == 23.96 and abs(r["diff_from_scale_file_pct"] - 3.9) < 0.3
    assert SV.analyse(rows[:2], 4)["status"] == "NEEDS_MORE_SEGMENTS"
    assert SV.analyse(rows[:2] + [dict(name="d", x1=0, y1=0, x2=250, y2=0, measured_m=9.0)], 4)["status"] == "INCONSISTENT"
    assert SV.analyse([dict(name="e", x1=0, y1=0, x2=1, y2=0, measured_m=-1)], 4)["status"] == "NO_DATA"


class H:
    n = 0

    def __init__(self, last, status="ACTIVE", kf=5):
        H.n += 1
        self.id, self.last_kf_t, self.status, self.n_kf, self.used, self.floor = f"H{H.n}", last, status, kf, set(), 4


def tracker(hyps, kfs):
    t = E3.Tracker.__new__(E3.Tracker)
    t.H, t.kf, t.events, t.done = hyps, [(f"s{x}", float(x), 0, 0, 0) for x in kfs], [], []

    def ek(kf):
        al = [h for h in t.alive() if kf[1] > h.last_kf_t and kf[0] not in h.used]
        for h in al:
            h.used.add(kf[0]); h.last_kf_t = kf[1]
        t.done.append((kf[1], len(al)))
    t.eval_kf = ek
    return t


def test_new_hypothesis_replays_only_what_the_score_can_use():
    live, late = H(99.0), H(10.0, kf=0)
    t = tracker([live, late], range(11, 101))
    for _ in range(10):
        t.work_keyframes()
    assert late.last_kf_t == 100.0 and live.last_kf_t == 100.0 and not t.pending()
    assert len(t.done) <= E3.N_KF + 1                         # was 90 keyframes
    assert late.catchup_skipped_s == 70.0 and not getattr(live, "catchup_skipped_s", 0)


def test_a_hypothesis_within_the_window_loses_nothing():
    a = H(90.0)
    t = tracker([a], range(80, 101))
    for _ in range(5):
        t.work_keyframes()
    assert [d[0] for d in t.done] == [float(x) for x in range(91, 101)] and not getattr(a, "catchup_skipped_s", 0)


def test_score_window_is_what_makes_the_skip_safe():
    src = (STG / "e3_tracker.py").read_text()
    assert "self.ev = deque(maxlen=N_KF)" in src and "(W_E2 * self.e2_merit + sum(s)) / (W_E2 + len(s))" in src


def test_backlog_limits_suspended_slots_only():
    src = (STG / "e3_tracker.py").read_text()
    assert "room = min(room, BACKLOG_ROOM)" in src and "first = (pend + act)[:EVAL_PER_KF]" in src


@pytest.mark.parametrize("run", ["X20261101_192409", "X20261101_193139", "X20261031_231940"])
def test_paper_metrics_on_real_runs_no_fix_no_ttf(run, tmp_path):
    ed = tmp_path / "ev"; ed.mkdir()
    for n in ("E8_SUMMARY.json", "time_metrics.json", "gt.jsonl"):
        if (OPR / "evaluation" / run / n).exists():
            shutil.copy(OPR / "evaluation" / run / n, ed / n)
    m = PM.build(OPR / "experiments" / run, ed)
    assert m["global_fix"] is False and m["time_to_fix_raw_s"] is None and m["time_to_fix_normalized_s"] is None and m["fix_error"]["position_error_m"] is None and m["failure_reason"]
    assert m["run_id"] == run and m["e2_floors_total"] == 9 and m["start_gt"] and m["duration_s"] > 100 and m["e5_stop_requests"] >= 0
    assert json.loads((ed / "PAPER_METRICS.json").read_text())["run_id"] == run


def test_paper_metrics_fix_error_only_with_a_gt_mark_near_the_fix(tmp_path):
    rd, ed = tmp_path / "r", tmp_path / "e"; (rd / "state").mkdir(parents=True); ed.mkdir()
    (rd / "experiment.json").write_text(json.dumps(dict(exp_id="X", created_wall=1000.0, ended_wall=1200.0)))
    (rd / "state/e7_state.json").write_text(json.dumps(dict(global_decision="GLOBAL_CONFIRMED", written_wall=1100.0)))
    (rd / "state/fix_overlay.json").write_text(json.dumps(dict(pose_px=[100.0, 0.0], ppm=10.0, heading_deg=10.0)))
    (ed / "time_metrics.json").write_text(json.dumps(dict(TTF_raw_s=100.0, TTF_normalized_s=80.0)))
    (ed / "gt.jsonl").write_text(json.dumps(dict(kind="start", px=0.0, py=0.0, wall=1000.0, gt_id="s")) + "\n")
    m = PM.build(rd, ed)
    assert m["global_fix"] and m["time_to_fix_raw_s"] == 100.0 and m["fix_error"]["position_error_m"] is None    # the start mark is not a GT at the fix
    with open(ed / "gt.jsonl", "a") as f:
        f.write(json.dumps(dict(kind="fix_check", px=90.0, py=0.0, yaw_deg=15.0, wall=1105.0, gt_id="f")) + "\n")
    m = PM.build(rd, ed)
    assert m["fix_error"]["position_error_m"] == 1.0 and m["fix_error"]["heading_error_deg"] == -5.0


def test_status_chips(tmp_path):
    st = tmp_path; (st / "e2").mkdir()
    (st / "e7_state.json").write_text(json.dumps(dict(global_decision="AMBIGUOUS", structural_class="STRUCTURALLY_AMBIGUOUS", why="w", written_wall=10.0)))
    (st / "e7_semantic_shadow.json").write_text(json.dumps(dict(verdict="WOULD_AMBIGUOUS", blocked_by=["NO_FLOOR_EVIDENCE"], rejected_by=[])))
    (st / "e3_state.json").write_text(json.dumps(dict(leader_kf_lag_s=42.0, kf_pending=30)))
    fs = SVW.build(dict(created_wall=1.0, timeline=[]), st, st, None, now=20.0)
    t = " ".join(c["text"] for c in fs["chips"])
    assert "구조로 구별되지 않음" in t and "최종 확정 아님" in t and "SHADOW WOULD_AMBIGUOUS" in t and "기록 전용" in t and "E3 지연" in t
    (st / "e7_state.json").write_text(json.dumps(dict(global_decision="AMBIGUOUS", written_wall=0.5)))
    assert not any(c["code"].startswith("E7_AMBIGUOUS") for c in SVW.build(dict(created_wall=1.0, timeline=[]), st, st, None, now=20.0)["chips"])    # earlier run
