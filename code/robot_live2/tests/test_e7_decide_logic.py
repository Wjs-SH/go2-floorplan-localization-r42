"""STAGING (2026-10-30): the structure-path FIX RULE (E6E7Link._decide) on hand-made window results — logic only, no geometry,
no field evidence. The real-record behaviour is in test_e7_final_replay.py."""
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
import e6e7_link as L  # noqa: E402

TC = 1000.0


def link(tmp, mode):
    lk = L.E6E7Link.__new__(L.E6E7Link)
    lk.state, lk.last, lk.job = tmp, None, None
    lk.tr = types.SimpleNamespace(store=types.SimpleNamespace(session=lambda: {"odom_session_id": "S"}))
    lk.struct_policy = dict(mode=mode, enabled=mode == "ENABLED")
    return lk


def sa(passes, mode="ENABLED"):
    return dict(kind="LIVE2_STRUCTURE_PATH", leader="H4", floor=4, status="RUNNING", mode=mode, phase="VERIFY_WHILE_MOVING",
                t_confident=TC, passes=passes, epoch="S", started_wall=0.0, e4_gap=0.2)


def w(dt, ok=None, best="H4", unique=None, live=False, skipped=None):
    if skipped:
        return dict(t_end=TC + dt, ok=False, skipped=skipped, live=live or None)
    unique = ok if unique is None else unique
    return dict(t_end=TC + dt, ok=bool(ok and best == "H4"), unique=unique, best=best, live=live or None, gap_m=0.06 if unique else 0.01)


RUN = [w(-60, True), w(-50, True), w(-40, True), w(-30, True)]          # passing run over 30 s (first / last disjoint)


@pytest.mark.parametrize("mode,status", [("ENABLED", "CONFIRMED"), ("SHADOW", "SHADOW_WOULD_CONFIRM")])
def test_run_plus_live_agreement_confirms_only_when_enabled(tmp_path, mode, status):
    lk = link(tmp_path, mode)
    s = sa(RUN + [w(-20, False), w(10, False, live=True)], mode)
    lk._decide(s, {"best": "H4"})
    assert s["status"] == status
    e7 = tmp_path / "e7_state.json"
    assert e7.exists() == (mode == "ENABLED")
    if mode == "SHADOW":
        assert (tmp_path / "e7_shadow.jsonl").exists()


def test_short_run_is_not_enough(tmp_path):
    lk = link(tmp_path, "ENABLED")
    s = sa([w(-60, True), w(-50, True), w(-40, True), w(10, False, live=True)])          # span 20 s < 30 s
    lk._decide(s, None)
    assert s["status"] == "RUNNING" and s["evidence"]["run"] is None


def test_a_gap_or_skipped_window_breaks_the_run(tmp_path):
    lk = link(tmp_path, "ENABLED")
    s = sa([w(-60, True), w(-50, True), w(-40, skipped="AT_REST"), w(-30, True), w(-20, True), w(10, False, live=True)])
    lk._decide(s, None)
    assert s["status"] == "RUNNING"


def test_any_contradicting_window_ends_the_attempt(tmp_path):
    lk = link(tmp_path, "ENABLED")
    s = sa(RUN + [w(-90, True, best="H7", unique=True), w(10, False, live=True)])
    lk._decide(s, None)
    assert s["status"] == "NOT_CONFIRMED" and "contradiction" in s["why"] and not (tmp_path / "e7_state.json").exists()


def test_no_live_window_agreeing_means_no_fix(tmp_path):
    lk = link(tmp_path, "ENABLED")
    s = sa(RUN + [w(10, False, best="H7", unique=False, live=True)])                      # live best is another candidate
    lk._decide(s, None)
    assert s["status"] == "RUNNING" and s["evidence"]["live_agree"] == 0


def test_too_old_run_is_not_used(tmp_path):
    lk = link(tmp_path, "ENABLED")
    s = sa([w(-300, True), w(-290, True), w(-280, True), w(-270, True), w(10, False, live=True)])
    lk._decide(s, None)
    assert s["status"] == "RUNNING"
