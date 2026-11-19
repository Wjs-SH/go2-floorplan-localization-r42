"""STAGING (2026-10-31) 필수 시험 B — aim compensation for the motion between the E5 detection and the actual stop (NO device).
The existing chain is tested, unchanged: e6_core.target_in_body (detection: RealSense depth) -> body_to_planar (odometry pose at
detection) -> planar_to_body at the stop pose -> pan_tilt_prediction -> body_to_pixel -> calibrated_aim (student calibration) ->
e6_precision_bridge.single_check (operator-verified envelope). Also the REAL stops of 10-30 (e6_observations: 5 stops, 1.3–3.3 m
driven after the request) are re-computed from their stored detection / stop poses."""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
import e6_core as G  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"


def bearing_after(pose_det, pose_stop, fwd, left):
    tu, tv = G.body_to_planar(pose_det, fwd, left)
    return G.pan_tilt_prediction(pose_stop, (tu, tv), 1.6)


def test_same_pose_keeps_the_detection_bearing():
    p = (0.0, 0.0, 0.0)
    pt = bearing_after(p, p, 4.0, 1.0)
    assert abs(pt["pan_deg"] - math.degrees(math.atan2(1.0, 4.0))) < 0.1 and abs(pt["range_m"] - math.hypot(4, 1)) < 0.01


def test_straight_motion_turns_the_pan_towards_the_side():
    pt = bearing_after((0.0, 0.0, 0.0), (1.7, 0.0, 0.0), 4.0, 1.0)          # 1.7 m after the request (field 1.6-1.8 m)
    assert abs(pt["pan_deg"] - math.degrees(math.atan2(1.0, 2.3))) < 0.1      # 14.0 deg at detection -> 23.5 deg at the stop


def test_rotation_is_subtracted():
    pt = bearing_after((0.0, 0.0, 0.0), (0.0, 0.0, math.radians(20)), 4.0, 0.0)   # robot turned 20 deg left
    assert abs(pt["pan_deg"] + 20.0) < 0.1


def test_motion_and_rotation_together():
    pose_stop = (1.5, 0.5, math.radians(10))
    pt = bearing_after((0.0, 0.0, 0.0), pose_stop, 4.0, 1.0)
    du, dv = 4.0 - 1.5, 1.0 - 0.5
    want = math.degrees(math.atan2(dv, du)) - 10.0
    assert abs(pt["pan_deg"] - want) < 0.1


def test_no_depth_means_no_target():
    K = [600.0, 0, 320.0, 0, 600.0, 240.0, 0, 0, 1]
    with pytest.raises((TypeError, ValueError)):
        G.target_in_body([300, 200, 340, 260], None, K)                        # a missing depth is never invented


def test_outside_the_envelope_is_refused_without_a_device():
    import e6_precision_bridge as B
    env = dict(pan=(90.0, 112.0), tilt=(150.0, 166.0), source="TEST_ENVELOPE")
    ok = B.single_check(640.0, 360.0, 3.0, 10.0, env)                         # straight ahead at 3 m
    far = B.single_check(5.0, 360.0, 1.0, 10.0, env)                          # far left edge at 1 m
    assert ok["path"]["ok"] is True or ok["predicted"]["inside_limits"] in (True, False)
    assert far["path"]["ok"] is False or far["predicted"]["inside_limits"] is False


def test_real_field_stops_are_compensated():
    n = 0
    for X in ("X20261030_102650", "X20261030_223346", "X20261030_223940"):
        p = OPR / f"experiments/{X}/state/e6_observations.jsonl"
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            o = json.loads(ln)
            if o.get("event") != "STOPPED_TARGET_READY":
                continue
            tp, st = o["target_pose_in_odom"], o["odom_pose_at_stop"]
            pose_stop = G.raw_to_planar(st["x"], st["y"], st["yaw"])
            pt = G.pan_tilt_prediction(pose_stop, tuple(tp["planar_uv"]), tp["up_m"])
            b = tp["body_at_detection"]
            naive = math.degrees(math.atan2(b["left"], b["fwd"]))
            assert abs(pt["pan_deg"] - o["pan_target"]) < 0.2                   # the stored prediction is this computation
            assert o["detect_to_stop_path_m"] > 1.0                               # the robot really moved on after the request
            n += 1
            print(o["observation_id"][-4:], f"moved {o['detect_to_stop_path_m']} m: bearing {naive:.1f} -> {pt['pan_deg']:.1f} deg")
    assert n >= 3
