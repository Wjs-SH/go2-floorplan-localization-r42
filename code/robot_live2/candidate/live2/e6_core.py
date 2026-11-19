"""e6_core.py — LIVE2 E6 observation geometry (pure python + numpy). PREDICTION ONLY: nothing here moves a device.

Contract per observation (AUTO_E5 or OPERATOR_ASSISTED):
  observation_id, source, target_track_id, detection_timestamp, image_frame_id, bbox, depth, odom_pose_at_detection,
  odom_pose_at_stop, camera_extrinsics, target_pose_in_odom, target_pose_at_stop, pan_target, tilt_target,
  observation_confidence, semantic_map_candidates, state, why.

Frames (2026-10-20 field analysis, tools/odom display work):
  raw      FAST-LIO /Odometry camera_init -> base_link: the IMU frame at start, IMU mounted upside down -> mirrored plane.
  planar   (u, v, h): u = robot forward at FAST-LIO start, v = left, h = heading CCW (left turn +). Obtained by the
           reflection u = x cosA + y sinA, v = x sinA - y cosA, h = -yaw_raw with A = -47.3 deg (extrinsic chain;
           measured -45.3 deg). Distances and left/right are physical in this frame; it is NOT a map frame.
  odom_link  E2/E3 frame (-x, y, -yaw) of raw; used only to hand targets to E3 hypotheses (map projection).
Camera mounting (body frame: x fwd, y left, z up): UNVERIFIED offline estimate (pitch from floor fit 5-6.5 deg up, yaw from
motion +2 deg, height 0.41 m, lever arm unknown -> 0). Every derived angle carries extrinsics_status.
"""
import json
import math
from pathlib import Path

import numpy as np

PLANAR_A = math.radians(-47.3)
CAMERA_EXTRINSICS = dict(height_m=0.41, pitch_up_deg=5.75, yaw_left_deg=2.0, lever_fwd_m=0.0, lever_left_m=0.0,
                         status="UNVERIFIED (offline estimate 2026-10-20 from recorded frames; needs frame_probe + tape measure)")
STOP_SPEED_MPS, STOP_YAWRATE_DPS, STOP_HOLD_S = 0.05, 3.0, 1.5
ODOM_FRESH_S = 1.0                     # a stop is only confirmed from odometry that is still arriving
WAIT_STOP_TIMEOUT_S = 45.0


def raw_to_planar(x, y, yaw):
    c, s = math.cos(PLANAR_A), math.sin(PLANAR_A)
    return x * c + y * s, x * s - y * c, -yaw


def planar_to_raw(u, v, h):
    c, s = math.cos(PLANAR_A), math.sin(PLANAR_A)        # the reflection is its own inverse
    return u * c + v * s, u * s - v * c, -h


def target_in_body(bbox, depth_m, K, ext=CAMERA_EXTRINSICS):
    """Box centre + depth (metres along the optical axis) -> point in the robot body frame (fwd, left, up from floor)."""
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    uc, vc = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    xo, yo, zo = (uc - cx) / fx * depth_m, (vc - cy) / fy * depth_m, depth_m     # optical: x right, y down, z forward
    fwd, left, up = zo, -xo, -yo                                                  # level camera looking forward
    p = math.radians(ext["pitch_up_deg"])                                        # camera pitched up by p
    fwd, up = fwd * math.cos(p) - up * math.sin(p), fwd * math.sin(p) + up * math.cos(p)
    w = math.radians(ext["yaw_left_deg"])
    fwd, left = fwd * math.cos(w) - left * math.sin(w), fwd * math.sin(w) + left * math.cos(w)
    return fwd + ext["lever_fwd_m"], left + ext["lever_left_m"], up + ext["height_m"]


def body_to_planar(pose_uvh, fwd, left):
    u, v, h = pose_uvh
    return u + fwd * math.cos(h) - left * math.sin(h), v + fwd * math.sin(h) + left * math.cos(h)


def planar_to_body(pose_uvh, tu, tv):
    u, v, h = pose_uvh
    du, dv = tu - u, tv - v
    return du * math.cos(h) + dv * math.sin(h), -du * math.sin(h) + dv * math.cos(h)


def stop_state(samples, now, hold_s=STOP_HOLD_S, wall_now=None):
    """samples = [(t, x, y, yaw)] raw odometry. STOPPED when over the last hold_s the planar speed < 0.05 m/s and the yaw
    rate < 3 deg/s (FAST-LIO odometry, not a command echo). wall_now: if given, the newest sample must be <= 1 s old —
    odometry that stopped arriving looks perfectly still and must NOT be taken as a stop."""
    if wall_now is not None and (not samples or wall_now - samples[-1][0] > ODOM_FRESH_S):
        return dict(stopped=False, why="ODOM_STALE" if samples else "NO_ODOMETRY",
                    odom_age_s=None if not samples else round(wall_now - samples[-1][0], 2))
    w = [s for s in samples if now - hold_s <= s[0] <= now]
    if len(w) < 3 or w[-1][0] - w[0][0] < 0.8 * hold_s:
        return dict(stopped=False, why="NOT_ENOUGH_ODOMETRY", n=len(w))
    dt = w[-1][0] - w[0][0]
    # R36: net displacement over the window, not the sum of 10 Hz steps: FAST-LIO jitter counted as travel called a STANDING robot MOVING 15.2 % of the time (30 % in X20261102_023712: five refused ① presses, one
    # observation cancelled "moved during focus"). On 4 field runs, 10 Hz FAST-LIO states, 12528 windows: standing called MOVING 15.2 % -> 0.0 %, walking called STOPPED 0 % -> 0 % (tools/stop_state_compare.py).
    dist = math.hypot(w[-1][1] - w[0][1], w[-1][2] - w[0][2])
    dyaw = sum(abs((b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(w, w[1:]))
    speed, yr = dist / dt, math.degrees(dyaw) / dt
    ok = speed < STOP_SPEED_MPS and yr < STOP_YAWRATE_DPS
    return dict(stopped=ok, speed_mps=round(speed, 3), yaw_rate_dps=round(yr, 2), window_s=round(dt, 2),
                why="STOPPED" if ok else f"MOVING speed {speed:.2f} m/s yaw {yr:.1f} deg/s")


def path_between(samples, t0, t1):
    """Odometry path length [m] between two stamps (detection -> stop); None if the ring does not cover t0."""
    w = [s for s in samples if t0 <= s[0] <= t1]
    if not samples or samples[0][0] > t0 + 0.3 or len(w) < 2:
        return None
    return round(sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(w, w[1:])), 3)


def pan_tilt_prediction(pose_stop_uvh, target_uv, target_up_m, ext=CAMERA_EXTRINSICS):
    """Angles from the robot (camera position assumed at the body origin + height) to the target at the stop pose.
    pan + = left. PREDICTION ONLY (pan-tilt mount offset and frame contract unverified)."""
    fwd, left = planar_to_body(pose_stop_uvh, *target_uv)
    rng = math.hypot(fwd, left)
    return dict(pan_deg=round(math.degrees(math.atan2(left, fwd)), 1),
                tilt_deg=round(math.degrees(math.atan2(target_up_m - ext["height_m"], rng)), 1),
                range_m=round(rng, 2), target_fwd_m=round(fwd, 2), target_left_m=round(left, 2),
                status="PREDICTION_ONLY", extrinsics_status=ext["status"])


def odom_link_point(tu, tv):
    """Planar target -> the E2/E3 odom_link frame (point only)."""
    x, y, _ = planar_to_raw(tu, tv, 0.0)
    return -x, y


def map_projection(hyp_rec, xo, yo):
    """odom_link point -> floor-plan px through an E3 hypothesis transform (same as e2_core.pose_in_map, point only)."""
    phi = math.radians(hyp_rec["phi"])
    A = np.array([[math.cos(phi), -math.sin(phi)], [-math.sin(phi), -math.cos(phi)]])
    p = np.asarray(hyp_rec["pos"], float) + hyp_rec["ppm"] * (A @ (np.array([xo, yo]) - np.asarray(hyp_rec["anchor"], float)))
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------- calibrated pan-tilt command (student pantilt_aim.Aim)
PANTILT_CALIB = Path(__file__).resolve().parents[1] / "config" / "pantilt_calib_multi_20260928_trim1.REF_COPY.json"
PAN_LIM, TILT_LIM = (50.0, 150.0), (115.0, 180.0)     # calibration limits ∩ console limits (planner rule)


def body_to_pixel(fwd, left, up, K, ext=CAMERA_EXTRINSICS):
    """Inverse of target_in_body: body point -> (u, v, depth along the optical axis) in the RealSense colour image."""
    fwd, left, up = fwd - ext["lever_fwd_m"], left - ext["lever_left_m"], up - ext["height_m"]
    w = math.radians(ext["yaw_left_deg"])
    fwd, left = fwd * math.cos(w) + left * math.sin(w), -fwd * math.sin(w) + left * math.cos(w)
    p = math.radians(ext["pitch_up_deg"])
    fwd, up = fwd * math.cos(p) + up * math.sin(p), -fwd * math.sin(p) + up * math.cos(p)
    zo, xo, yo = fwd, -left, -up
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    return cx + fx * xo / zo, cy + fy * yo / zo, zo


def calibrated_aim(u, v, z, calib=PANTILT_CALIB):
    """Student pantilt_aim.Aim (2026-09-28 calibration, LOO mean 1.28 deg inside its sampled range) -> servo commands.
    strict semantics of the console: a command outside the limits is reported, not clipped. Pure computation."""
    import sys as _s
    code = Path(__file__).resolve().parents[1] / "external" / "sci_event" / "code"
    if str(code) not in _s.path:
        _s.path.insert(0, str(code))
    import pantilt_aim as PA
    cfg = json.loads(Path(calib).read_text())
    pan, tilt = PA.Aim(cfg, fill_missing=True).aim(float(u), float(v), float(z))
    inside = PAN_LIM[0] <= pan <= PAN_LIM[1] and TILT_LIM[0] <= tilt <= TILT_LIM[1]
    return dict(pan_cmd=round(pan, 2), tilt_cmd=round(tilt, 2), inside_limits=inside,
                strict_result="OK" if inside else "target_out_of_pantilt_range",
                calibration=str(calib), calibration_status="STUDENT_CALIBRATION_2026-09-28 (RealSense pixel+depth -> servo)",
                in_image=0 <= u < 640 and 0 <= v < 480 and z > 0)
