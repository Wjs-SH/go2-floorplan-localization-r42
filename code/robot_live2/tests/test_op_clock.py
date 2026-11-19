"""STAGING (2026-11-01, r33): operator-normalized clock (op_clock.py) — settle -> 1 s, focus -> 3 s, an analysis index only.
REAL: the loc2 record X20261031_231940 (settle 13.0 / 4.7 / 6.6 s, focus 102.2 / 59.1 / 28.9 s, O001 settle ended by ROBOT_MOVED)
and the archived loc1 / loc2 runs through the real e6_settle.time_metrics. SYNTHETIC event files for the edge cases."""
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
import op_clock as OC  # noqa: E402
import e6_settle as ST  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments"
LOC2, LOC1 = OPR / "X20261031_231940", OPR / "X20261031_231209"


def real(x):
    ex = json.loads((x / "experiment.json").read_text())
    return ex, OC.intervals(x / "state", ex["exp_id"])


def test_real_loc2_intervals_and_values():
    ex, iv = real(LOC2)
    t0 = ex["created_wall"]
    got = [(i["kind"], i["observation_id"][-4:], round(i["end"] - i["start"], 1), i["completed"]) for i in iv]
    assert got == [("SETTLE", "O001", 4.2, False), ("SETTLE", "O002", 13.0, True), ("FOCUS", "O002", 102.2, True), ("SETTLE", "O003", 4.7, True),
                   ("FOCUS", "O003", 59.1, True), ("SETTLE", "O005", 6.6, True), ("FOCUS", "O005", 28.9, True)]
    c = OC.clock(iv, t0, t0 + 400.0)
    assert (c["N_settle"], c["N_focus"]) == (3, 3) and c["T_focus_s"] == 190.2 and c["T_settle_s"] == 24.2
    assert c["normalized_s"] == round(400.0 - 214.4 + 3 * 1 + 3 * 3, 1) and c["paused"] is None
    end = OC.clock(iv, t0, ex["ended_wall"])
    assert abs(end["raw_s"] - 729.5) < 0.1 and abs(end["normalized_s"] - (end["raw_s"] - 202.4)) < 0.15


def test_clock_pauses_during_manual_work_and_resumes():
    ex, iv = real(LOC2)
    t0 = ex["created_wall"]
    a, b = OC.clock(iv, t0, t0 + 150.0), OC.clock(iv, t0, t0 + 200.0)        # inside the 102 s O002 focus
    assert b["raw_s"] - a["raw_s"] == 50.0 and b["normalized_s"] == a["normalized_s"] and b["paused"]["kind"] == "FOCUS"
    c = OC.clock(iv, t0, t0 + 226.2)                                         # 10 s after [②]
    assert c["paused"] is None and c["normalized_s"] == round(a["normalized_s"] + 3.0 + 10.0, 1)
    s = OC.clock(iv, t0, t0 + 100.0)                                         # inside the O002 settle wait
    assert s["paused"]["kind"] == "SETTLE"


def test_cancelled_settle_is_real_time():
    ex, iv = real(LOC2)
    t0 = ex["created_wall"]
    c = OC.clock(iv, t0, t0 + 90.0)                                          # after O001 (ROBOT_MOVED at +71.2)
    assert c["normalized_s"] == c["raw_s"] == 90.0 and c["N_settle"] == 0


def ev_files(tmp, settle=(), manual=()):
    st = tmp / "state"; st.mkdir(exist_ok=True)
    (st / "e6_settle_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in settle))
    (st / "e6_manual_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in manual))
    return st


def S(w, oid, **k):
    return dict(wall=w, observation_id=oid, **k)


def test_refused_approval_retake_and_abnormal_order(tmp_path):
    st = ev_files(tmp_path, settle=[S(10, "A", kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL"),
                                    S(14, "A", kind="COMMAND", action="approve", accepted=False),          # refused: still waiting
                                    S(20, "A", kind="COMMAND", action="approve", accepted=True),
                                    S(20.5, "A", kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL"),       # duplicate state: ignored
                                    S(5, "Z", kind="COMMAND", action="approve", accepted=True)],           # no open interval: ignored
                         manual=[S(30, "A", kind="COMMAND", action="capture", accepted=True),              # command before the phase
                                 S(31, "A", kind="PHASE", phase="WAIT_MANUAL_FOCUS"),
                                 S(41, "A", kind="COMMAND", action="capture_anyway", accepted=True),
                                 S(50, "A", kind="PHASE", phase="WAIT_MANUAL_FOCUS"),                       # [다시 촬영]
                                 S(55, "A", kind="COMMAND", action="capture", accepted=True),
                                 S(60, "A", kind="PHASE", phase="HOME_CONFIRMED")])
    iv = OC.intervals(st)
    assert [(i["kind"], i["start"], i["end"], i["completed"]) for i in iv] == [("SETTLE", 10.0, 20.0, True), ("FOCUS", 31.0, 41.0, True),
                                                                               ("FOCUS", 50.0, 55.0, True)]
    c = OC.clock(iv, 0.0, 100.0)
    assert (c["N_settle"], c["N_focus"]) == (1, 2) and c["normalized_s"] == 100 - 10 - 10 - 5 + 1 + 6


def test_cancel_and_abort_before_capture_are_not_normalized(tmp_path):
    st = ev_files(tmp_path, settle=[S(10, "A", kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL"), S(15, "A", kind="STATE", state="USER_CANCELLED"),
                                    S(20, "B", kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL"), S(22, "B", kind="COMMAND", action="approve", accepted=True)],
                         manual=[S(30, "B", kind="PHASE", phase="WAIT_MANUAL_FOCUS"), S(70, "B", kind="PHASE", phase="USER_CANCELLED")])
    c = OC.clock(OC.intervals(st), 0.0, 100.0)
    assert (c["N_settle"], c["N_focus"]) == (1, 0) and c["normalized_s"] == 100 - 2 + 1


def test_other_experiment_events_are_ignored(tmp_path):
    st = ev_files(tmp_path, settle=[S(10, "A", kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL", experiment_id="X_OLD"),
                                    S(20, "A", kind="COMMAND", action="approve", accepted=True, experiment_id="X_OLD")])
    assert OC.intervals(st, "X_NEW") == [] and len(OC.intervals(st, "X_OLD")) == 1


def test_overlap_is_never_subtracted_twice():
    iv = [dict(kind="SETTLE", observation_id="A", start=10.0, end=20.0, completed=True, why=""),
          dict(kind="FOCUS", observation_id="B", start=15.0, end=25.0, completed=True, why="")]
    c = OC.clock(iv, 0.0, 50.0)
    assert c["removed_s"] == 15.0 and c["normalized_s"] == 50 - 15 + 1 + 3


def test_fix_and_no_fix_ttf(tmp_path):
    iv = [dict(kind="SETTLE", observation_id="A", start=10.0, end=20.0, completed=True, why=""),
          dict(kind="FOCUS", observation_id="A", start=25.0, end=85.0, completed=True, why=""),
          dict(kind="FOCUS", observation_id="B", start=150.0, end=None, completed=False, why="open")]
    st = tmp_path / "s"; st.mkdir()
    o = OC.summary(st, 0.0, 300.0)
    assert o["TTF_raw_s"] is None and o["TTF_normalized_s"] is None and "NO_FIX" in o["why"]
    f = OC.clock(iv, 0.0, 120.0)
    assert f["raw_s"] == 120.0 and f["normalized_s"] == 120 - 10 - 60 + 1 + 3


def test_time_metrics_on_the_real_archived_runs_keep_raw_values():
    for x in (LOC1, LOC2):
        tm = ST.time_metrics(x)
        assert tm["TTF_raw_s"] is None and tm["TTF_normalized_s"] is None                       # NO_FIX runs: no time-to-fix at all
        oc = tm["operator_clock"]
        assert tm["T_end_raw_s"] == oc["at_end"]["raw_s"] and tm["T_end_normalized_s"] <= tm["T_end_raw_s"]
        assert "분석값" in tm["normalized_note"] and tm["t_run_start"] == json.loads((x / "experiment.json").read_text())["created_wall"]
    tm2 = ST.time_metrics(LOC2)
    assert tm2["operator_clock"]["at_end"]["N_focus"] == 3 and abs(tm2["T_end_raw_s"] - tm2["T_end_normalized_s"] - 202.4) < 0.15
