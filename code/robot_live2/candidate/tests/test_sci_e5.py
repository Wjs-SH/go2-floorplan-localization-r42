"""Unit tests for the SCI E5 layer (policy gates, event FSM, reading semantics/fusion, pan/tilt planner safety)."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
for p in ("sci_visual_observation", "sci_text_reading", "sci_pantilt", "sci_scheduler"):
    sys.path.insert(0, str(SRC / p))

import policy as P  # noqa: E402
import semantic as S  # noqa: E402
from features import Geometry, predicted_obliquity  # noqa: E402

GEOM = Geometry()


def region(**kw):
    r = dict(box=[300, 100, 340, 130], tile=[300, 100, 340, 130], sim=0.12, msim=0.12, z=3.0, xyz=[3.0, 1.0, 1.5],
             spot=1, nobs=3, nscene=3, board=-0.05, adv=0.02, hgt=1.6, fired=False,
             neg={c: 0.02 for c in ("poster", "board", "glare", "light", "elevator", "directory", "other")},
             ctx={c: 0.0 for c in ("poster", "board", "directory", "elevator", "glare")}, ctx_plate=0.05)
    r.update(kw)
    return r


def feat(**kw):
    f = dict(glare=dict(sat=0.0, spec=0.0, contrast=100, sharp=200), plane=dict(ok=True, normal=[0, 0, -1], obliquity_deg=20,
             valid_in=0.9), edge=50, speed=0.3, bearing=0.0, up=[0, -1, 0], pantilt=dict(ok=True, pan=95, tilt=158),
             text_px=60, z_tile=3.0, obliquity_pred=25.0, aim_px=[320, 115])
    f.update(kw)
    return f


def stage(**flags):
    return P.StagePolicy("T", P.load_cfg(None), GEOM, **flags)


def run_scenes(sp, regs_feats, n=3, e4=None, t0=10.0):
    d, first = None, None
    for k in range(n):
        d, stop = sp.step(t0 + k, [r for r, _ in regs_feats], [f for _, f in regs_feats], thr=0.08, e4=e4)
        first = first or stop
    return d, first


def test_positive_stops_after_persistence():
    sp = stage()
    d, stop = sp.step(10.0, [region()], [feat()], 0.08, None)
    assert stop is None and d[0]["display"] == "VERIFYING"       # one scene is not enough (A1)
    d, stop = sp.step(11.0, [region()], [feat()], 0.08, None)
    assert stop is not None and d[0]["state"] == "STOP_REQUEST"


def test_single_frame_never_stops():
    sp = stage()
    _, stop = sp.step(10.0, [region(nscene=1)], [feat()], 0.08, None)
    assert stop is None


def test_poster_rejected():
    r = region(neg=dict(region()["neg"], poster=0.20))
    d, stop = run_scenes(stage(), [(r, feat())])
    assert stop is None and d[0]["display"] == "REJECTED_POSTER"


def test_board_context_rejected():
    r = region(ctx=dict(region()["ctx"], board=0.12), ctx_plate=0.05)
    d, stop = run_scenes(stage(), [(r, feat())])
    assert stop is None and d[0]["display"] == "REJECTED_POSTER"


def test_glare_rejected():
    d, stop = run_scenes(stage(), [(region(), feat(glare=dict(sat=0.3, spec=0.4, contrast=50, sharp=10)))])
    assert stop is None and d[0]["display"] == "REJECTED_GLARE"


def test_oblique_deferred_not_rejected():
    d, stop = run_scenes(stage(), [(region(), feat(obliquity_pred=78.0))])
    assert stop is None and d[0]["state"] == "DEFER" and d[0]["display"] == "REJECTED_ANGLE"


def test_too_far_and_too_close():
    d, _ = run_scenes(stage(), [(region(z=6.5), feat(z_tile=6.5))])
    assert d[0]["display"] == "REJECTED_DISTANCE"
    d, _ = run_scenes(stage(), [(region(z=0.9), feat(z_tile=0.9, speed=0.0))])
    assert d[0]["display"] == "REJECTED_DISTANCE"


def test_pantilt_out_of_range_deferred():
    d, stop = run_scenes(stage(), [(region(), feat(pantilt=dict(ok=False, why="out_of_range", pan=30, tilt=150)))])
    assert stop is None and d[0]["display"] == "REJECTED_PANTILT"


def test_dedup_same_spot():
    sp = stage()
    _, s1 = run_scenes(sp, [(region(), feat())])
    assert s1 is not None
    d, s2 = run_scenes(sp, [(region(), feat())], t0=20.0)
    assert s2 is None and d[0]["display"] == "ALREADY_OBSERVED"


def test_e4_confident_suppresses_but_unavailable_does_not():
    _, stop = run_scenes(stage(), [(region(), feat())], e4=dict(available=True, level="CONFIDENT"))
    assert stop is None
    _, stop = run_scenes(stage(), [(region(), feat())], e4=dict(available=False, status="E4_UNAVAILABLE"))
    assert stop is not None


def test_busy_blocks_new_stop():
    sp = stage()
    sp.step(10.0, [region()], [feat()], 0.08, None)
    d, stop = sp.step(11.0, [region()], [feat()], 0.08, None, busy=True)
    assert stop is None and "busy:event_in_progress" in d[0]["reasons"]


def test_ablation_flags():
    # A1-only stage ignores glare; A2 rejects it
    r, f = region(), feat(glare=dict(sat=0.3, spec=0.4, contrast=50, sharp=10))
    _, s = run_scenes(stage(a2=False, a3=False, a4=False), [(r, f)])
    assert s is not None
    _, s = run_scenes(stage(a3=False, a4=False), [(r, f)])
    assert s is None


def test_fsm_waits_for_stop_and_times_out():
    t = [0.0]
    cfg = P.load_cfg(None)
    f = P.EventFSM(1, dict(t=0.0), cfg, clock=lambda: t[0])
    f.on_odom(0.5, 0.0)
    assert f.state == "WAIT_STABLE"
    t[0] = 0.5; f.on_odom(0.0, 0.0)
    assert f.state == "WAIT_STABLE"
    t[0] = 1.3; f.on_odom(0.0, 0.0)
    assert f.state == "STOP_CONFIRMED"
    g = P.EventFSM(2, dict(t=0.0), cfg, clock=lambda: t[0])
    t[0] = 0.0; g.on_odom(0.4, 0.0)
    t[0] = 25.0; g.on_odom(0.4, 0.0)
    assert g.state == "FAILED" and "timeout" in g.log[-1]["reason"]
    assert all("t" in x for x in g.log)


def test_fsm_moving_robot_resets_stability():
    t = [0.0]
    f = P.EventFSM(1, dict(t=0.0), P.load_cfg(None), clock=lambda: t[0])
    f.on_odom(0.0, 0.0)
    t[0] = 0.5; f.on_odom(0.0, 0.0)
    t[0] = 0.6; f.on_odom(0.3, 0.0)        # moved again
    t[0] = 1.0; f.on_odom(0.0, 0.0)
    assert f.state == "WAIT_STABLE"


def test_predicted_obliquity_decreases_when_approaching_side_sign():
    # side wall to the left (normal +x in camera frame), sign ahead-left
    plane = dict(ok=True, normal=[1.0, 0.0, 0.0])
    box = [100, 200, 140, 230]
    o0 = predicted_obliquity(plane, box, GEOM.fx, GEOM.fy, GEOM.cx, GEOM.cy, 4.0, 0.0)
    o1 = predicted_obliquity(plane, box, GEOM.fx, GEOM.fy, GEOM.cx, GEOM.cy, 4.0, 2.0)
    assert o1 < o0


@pytest.mark.parametrize("lines,typ,room", [
    ([("408", 0.99), ("LINC3.0 사업단", 0.6)], "room_number_plate", "408"),
    ([("TOEIC", 0.9), ("2026.10.20", 0.8), ("접수 기간", 0.7), ("408", 0.9)], "poster_or_notice", None),
    ([("1F", 0.9), ("2F", 0.9), ("3F", 0.9), ("4F", 0.9), ("층별 안내", 0.8)], "floor_directory", None),
    ([("엘리베이터", 0.9), ("3", 0.9), ("4", 0.9), ("5", 0.9), ("6", 0.9)], "elevator_sign_or_button", None),
    ([("인공지능 연구실", 0.9)], "lab_or_office_name_sign", None),
])
def test_classify_text(lines, typ, room):
    c = S.classify_text(lines)
    fz = S.fuse(c, None)
    assert c["sign_type"] == typ
    got = fz["label_alternatives"][0]["label"] if fz["label_alternatives"] else None
    assert got == room


def test_fusion_conflict_keeps_both():
    o = S.classify_text([("408", 0.95)])
    fz = S.fuse(o, dict(sign_type="room_number_plate", room_number="406"))
    assert fz["status"] == "CONFLICT" and {a["label"] for a in fz["label_alternatives"]} == {"408", "406"}
    assert all(a["p"] == 0.5 for a in fz["label_alternatives"])


def test_fusion_vlm_says_poster_blocks_label():
    o = S.classify_text([("408", 0.95)])
    fz = S.fuse(o, dict(sign_type="poster_or_notice", room_number=""))
    assert fz["status"] == "NOT_A_LOCATION_SIGN" and not fz["label_alternatives"]


def test_planner_never_executes_without_approval(monkeypatch):
    from planner import PanTiltPlanner
    monkeypatch.delenv("SCI_PANTILT_APPROVED", raising=False)
    p = PanTiltPlanner(GEOM, console="http://127.0.0.1:9", execute=True)
    plan = p.plan(320, 240, 3.0)
    r = p.aim(plan)
    assert r["mode"] == "DRY_RUN" and not r["ok"] and p.fetch_capture() is None
    assert abs(plan["pan_cmd"] - GEOM.calib["servo"]["pan_zero_deg"]) < 2.0     # centre pixel ~ calibrated zero


def test_planner_refuses_out_of_range():
    from planner import PanTiltPlanner
    p = PanTiltPlanner(GEOM)
    plan = p.plan(5, 5, 0.5)                    # near + extreme corner -> tilt/pan beyond limits
    r = p.aim(plan)
    assert (not plan["reachable"]) == (r["mode"] == "REFUSED")


def test_admission_control():
    from resource_guard import admit_gpu_job
    ok, _ = admit_gpu_job(dict(mem_avail_mb=9000, gpu_pct=10, temps={"a": 60}), 3500)
    assert ok
    ok, why = admit_gpu_job(dict(mem_avail_mb=5000, gpu_pct=10, temps={"a": 60}), 3500)
    assert not ok and "memory" in why


def test_vlm_only_number_is_not_a_location_label():
    fz = S.fuse(None, dict(sign_type="room_number_plate", room_number="201"))
    assert fz["status"] == "VLM_ONLY_UNCONFIRMED" and fz["label_alternatives"] == []


def test_low_confidence_ocr_alone_is_unconfirmed():
    fz = S.fuse(S.classify_text([("409", 0.6)]), None)
    assert fz["status"] == "OCR_ONLY_UNCONFIRMED" and fz["label_alternatives"] == []
    fz = S.fuse(S.classify_text([("409", 0.6)]), dict(sign_type="room_number_plate", room_number="409"))
    assert fz["status"] == "AGREE" and fz["label_alternatives"][0]["label"] == "409"
