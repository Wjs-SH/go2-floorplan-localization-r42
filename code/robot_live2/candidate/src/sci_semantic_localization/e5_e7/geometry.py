"""geometry.py — explicit frame chain for E5 -> E6 -> E7 (thesis research, 2026-10-12).

Frames (all right-handed unless stated):
  RS optical   : x right, y down, z forward (RealSense colour camera, intrinsics fx fy cx cy)
  RS body-axes : x forward, y left, z up — same origin as RS optical. Conversion [zo, -xo, -yo]
                 (identical to ~/kci_event_handoff_20261009/code/pantilt_aim.py pixel_to_point, l.89-95)
  body         : RS body-axes rotated by the RealSense mount rotation R_mount = Rz(yaw)Ry(pitch)Rx(roll)
                 from the pan/tilt calibration file key `realsense_mount_deg` (0,0,0 in the 09-28 file =
                 "RealSense level and facing forward"; this is an ASSUMPTION carried as bearing uncertainty).
                 Origin = RealSense optical centre (its offset from base_link is not calibrated; it shifts the
                 observation point by < 0.3 m and is covered by the range tolerance).
  map metric   : x right, y up (semantic-map `map_frame`, bottom-left origin), unit = pixel/ppm
  map pixel    : x right, y down, top-left origin (semantic-map `pixel_frame` == floor_id localisation map
                 pixel frame, see e6_region provenance check)
  heading psi  : angle of the body +x axis in the map metric frame, CCW from +x.

Map-pixel offset of a body vector b at heading psi (scale ppm px/m):
  d_px = ppm * [cos psi*bx - sin psi*by, -(sin psi*bx + cos psi*by)]
This equals floor_id's `_A(phi) @ b * ppm` with _A = [[c,-s],[-s,-c]] (rt_localizer.py:33-34), i.e. the same
convention the localiser uses (verified in tests/test_e5_e7.py::test_convention_matches_floor_id).

Pan/tilt: the observation bearing is NOT computed as candidate_yaw + pan. The pan/tilt servo angles are
recorded for evidence and inverted only for a consistency check (servo_to_rs_direction).
"""
from __future__ import annotations

import math

import numpy as np


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def mount_rotation(mount_deg) -> np.ndarray:
    yaw, pitch, roll = (math.radians(float(mount_deg.get(k, 0.0) or 0.0)) for k in ("yaw", "pitch", "roll")) \
        if isinstance(mount_deg, dict) else (math.radians(float(v)) for v in mount_deg)
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


class InvalidDepth(ValueError):
    """R1: depth z is 0 / NaN / Inf / negative — it is NOT a distance and is never clamped into a valid range."""
    reason_code = "GEOMETRY_UNAVAILABLE"


def depth_valid(depth_m) -> bool:
    try:
        z = float(depth_m)
    except (TypeError, ValueError):
        return False
    return math.isfinite(z) and z > 0.0


def rs_pixel_to_body(u: float, v: float, depth_m: float, intr: dict, mount_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    """RealSense colour pixel + z-depth (m, along the optical axis, NOT the ray length) -> 3-D point in the body axes (m).
    The horizontal range used by E6 is body_bearing_range(...)[1] = hypot(x_body, y_body) — computed once, here; consumers
    must not convert z->range again. Invalid depth raises InvalidDepth (R1)."""
    if not depth_valid(depth_m):
        raise InvalidDepth(f"invalid depth z={depth_m!r}")
    xo = (u - intr["cx"]) / intr["fx"] * depth_m
    yo = (v - intr["cy"]) / intr["fy"] * depth_m
    zo = depth_m
    return mount_rotation(mount_deg) @ np.array([zo, -xo, -yo])


def body_bearing_range(p_body: np.ndarray):
    """Horizontal bearing (rad, CCW from body +x, left positive) and horizontal range (m)."""
    return math.atan2(p_body[1], p_body[0]), math.hypot(p_body[0], p_body[1])


def bearing_sigma(u_half_px: float, intr: dict, mount_yaw_sigma_deg: float = 3.0, min_deg: float = 1.0) -> float:
    """1-sigma bearing uncertainty (rad): half the candidate box angular half-width combined with the
    uncalibrated mount yaw. Engineering prior, fixed before any trial."""
    box = math.atan2(max(u_half_px, 0.0), intr["fx"]) / 2.0
    return math.hypot(max(box, math.radians(min_deg)), math.radians(mount_yaw_sigma_deg))


def range_sigma(z_m: float) -> float:
    """1-sigma horizontal-range uncertainty (m): D435i depth error ~2 % of distance (Intel spec, <=2 % at 2 m)
    + 0.05 m for the camera-origin vs body-origin offset. Engineering prior, fixed before any trial."""
    return 0.05 + 0.02 * max(z_m, 0.0) ** 1.0


def body_to_map_px(p_px, psi: float, b_xy, ppm: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    bx, by = float(b_xy[0]), float(b_xy[1])
    return np.asarray(p_px, float) + ppm * np.array([c * bx - s * by, -(s * bx + c * by)])


def map_px_heading(p_px, q_px) -> float:
    """Heading (map metric, CCW from +x) of the vector p -> q given in map pixels (y down)."""
    return math.atan2(-(q_px[1] - p_px[1]), q_px[0] - p_px[0])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def floor_id_A(phi: float) -> np.ndarray:
    """Copy of floor_id rt_localizer._A for convention tests (not used in computation)."""
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [-s, -c]])


def servo_to_rs_direction(pan_cmd: float, tilt_cmd: float, servo: dict) -> np.ndarray:
    """Invert the servo mapping of pantilt_aim.py (pan_cmd = pan_zero + pan_gain*pan_deg, tilt likewise) to a unit
    direction of the pan/tilt optical axis in RealSense body-axes, IGNORING the axis offsets (pan_axis_from_realsense).
    Used only to cross-check the commanded aim against the RealSense-derived bearing; never as the map bearing."""
    pan = math.radians((pan_cmd - servo["pan_zero_deg"]) / servo["pan_deg_per_left_deg"])
    tilt = math.radians((tilt_cmd - servo["tilt_zero_deg"]) / servo["tilt_deg_per_up_deg"])
    return np.array([math.cos(tilt) * math.cos(pan), math.cos(tilt) * math.sin(pan), math.sin(tilt)])
