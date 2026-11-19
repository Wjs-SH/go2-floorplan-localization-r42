"""STAGING (2026-10-30): live2/e4_stage.py — display-only staged localisation view. It must never claim more than the existing E4 / E7."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate" / "live2"))
import e4_stage as S  # noqa: E402

E2_PART = dict(floors_total=9, floors=list(range(1, 10)), floors_done=[4, 1])
E2_FULL = dict(floors_total=9, floors=list(range(1, 10)), floors_done=list(range(1, 10)))
E3 = dict(hypotheses=[dict(id="H1_F4", floor=4, status="ACTIVE", score=0.3, n_kf=20), dict(id="H2_F7", floor=7, status="ACTIVE", score=0.5, n_kf=20)])


def st(e2=E2_PART, e3=E3, e4="SEARCH_INCOMPLETE", e7=None, now=1000.0, **kw):
    return S.stage(e2, e3, dict(state=e4, reasons=["r"], metrics={}), e7, now, **kw)


def test_searching_without_hold():
    assert st(lead_since=990.0)["stage"] == "SEARCHING"                      # leader only 10 s old


def test_provisional_needs_hold_and_no_recent_ambiguity():
    r = st(lead_since=900.0)
    assert r["stage"] == "PROVISIONAL" and "미탐색 층" in r["why"] and r["fix_allowed_by_this_view"] is False
    assert st(lead_since=900.0, last_amb=990.0)["stage"] == "SEARCHING"       # AMBIGUOUS 10 s ago


def test_provisional_needs_keyframes():
    e3 = dict(hypotheses=[dict(E3["hypotheses"][0], n_kf=5)])
    assert st(e3=e3, lead_since=900.0)["stage"] == "SEARCHING"


def test_ambiguous_passthrough():
    assert st(e4="AMBIGUOUS", lead_since=900.0)["stage"] == "AMBIGUOUS"


def test_ready_only_from_e4_confident():
    for s4 in ("SEARCH_INCOMPLETE", "DOMINANT_UNVERIFIED", "AMBIGUOUS", "INITIALIZING"):
        assert st(e2=E2_FULL, e4=s4, lead_since=0.0)["stage"] != "READY_FOR_VERIFY"
    assert st(e2=E2_FULL, e4="CONFIDENT_GEOMETRY")["stage"] == "READY_FOR_VERIFY"


def test_fixed_only_from_e7():
    assert st(e2=E2_FULL, e4="CONFIDENT_GEOMETRY", e7=dict(global_decision="GEOMETRY_CONSISTENT_SEMANTIC_PENDING"))["stage"] == "READY_FOR_VERIFY"
    assert st(e2=E2_FULL, e4="CONFIDENT_GEOMETRY", e7=dict(global_decision="GLOBAL_CONFIRMED"))["stage"] == "FIXED"


def test_unresolved_after_full_search():
    assert st(e2=E2_FULL, e4="AMBIGUOUS", full_since=800.0)["stage"] == "UNRESOLVED"
    assert st(e2=E2_FULL, e4="AMBIGUOUS", full_since=950.0)["stage"] == "AMBIGUOUS"     # < 120 s after the full search
    assert st(e2=E2_PART, e4="AMBIGUOUS", full_since=None)["stage"] == "AMBIGUOUS"
    assert st(e2=E2_FULL, e4="GEOMETRY_INCONSISTENT")["stage"] == "UNRESOLVED"


def test_independent_views_counts_fresh_path_only():
    rows = [(f"s{i}", float(i), 0.5 * i, 0.0, 0.0) for i in range(30)]          # 0.5 m per second, straight
    kf = [float(i) for i in range(0, 30, 1)]                                    # a keyframe every 0.5 m
    assert S.independent_views(kf, rows, sep_m=3.0) == 5                         # 0, 6, 12, 18, 24 s
