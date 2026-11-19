"""R1 pure stop / geometry-input policies for E5 (no ROS, no device).

BASE_STOPPED : body progress has stopped. Primary evidence = a FRESH body-speed sample whose source and unit are
               declared (e.g. a stored/fake /sportmodestate hypot(vx,vy) adapter, m/s). A stale sample - even an exact
               0.0 - is not evidence. Without fresh primary evidence the existing FAST-LIO odom rule is used UNCHANGED
               (0.05 m/s, 3 deg/s, hold 0.7 s) and labelled as the auxiliary source.
IMAGE_STABLE : optical stability right before exposure (fresh body evidence still stopped + low yaw rate, servo
               settled when reported). Separate state, separate finite timeout.
Every wait state checks its timeout on EVERY call (the legacy FSM only checked it while 'moving').
All thresholds here are development values until measured (UNVALIDATED); none loosens the legacy FAST-LIO rule.
"""
from __future__ import annotations

import math

DEFAULT_STOP_CFG = dict(
    # FAST-LIO auxiliary rule: identical to config/e5_policy.json event.* (not loosened)
    aux_lin_mps=0.05, aux_yaw_dps=3.0, aux_hold_s=0.7,
    # primary body-speed rule (development values, UNVALIDATED): same speed threshold and hold as the legacy rule
    base_lin_mps=0.05, base_hold_s=0.7, sample_stale_s=0.3, verified_sources=("sportmodestate.velocity_hypot_xy",),
    wait_base_stop_timeout_s=20.0, wait_image_stable_timeout_s=5.0,
    img_yaw_dps=3.0, img_hold_s=0.3,
)


class BodySample:
    """value_mps: horizontal body speed; rx_s: receive time on the policy clock; source/unit must be declared."""

    def __init__(self, value_mps, rx_s, source, unit="m/s", yaw_dps=None):
        self.value, self.rx, self.source, self.unit, self.yaw = value_mps, rx_s, source, unit, yaw_dps


def _fresh_primary(sample, now, cfg):
    if sample is None:
        return None, "no_primary_sample"
    if sample.unit != "m/s" or sample.source not in cfg["verified_sources"]:
        return None, f"unverified_source:{sample.source}/{sample.unit}"
    if now - sample.rx > cfg["sample_stale_s"]:
        return None, f"stale_primary:{now - sample.rx:.2f}s"
    if sample.value is None or not math.isfinite(sample.value):
        return None, "invalid_primary"
    return sample, None


class StopMonitor:
    """WAIT_BASE_STOP -> BASE_STOPPED -> WAIT_IMAGE_STABLE -> IMAGE_STABLE, each wait with its own finite timeout.
    Terminal failures: BASE_STOP_TIMEOUT, IMAGE_STABLE_TIMEOUT."""

    def __init__(self, t_start, cfg=None):
        self.cfg = dict(DEFAULT_STOP_CFG, **(cfg or {}))
        self.state, self.t_enter = "WAIT_BASE_STOP", t_start
        self._since, self.basis, self.log = None, None, [dict(state="WAIT_BASE_STOP", t=t_start)]

    def _to(self, state, now, **kw):
        self.state, self.t_enter, self._since = state, now, None
        self.log.append(dict(state=state, t=now, **kw))

    def update(self, now, primary=None, aux_lin=None, aux_yaw=None, aux_fresh=True, servo_settled=None):
        c = self.cfg
        if self.state == "WAIT_BASE_STOP":
            p, why = _fresh_primary(primary, now, c)
            if p is not None:
                still = p.value <= c["base_lin_mps"] and (p.yaw is None or p.yaw <= c["aux_yaw_dps"])
                basis, hold = f"primary:{p.source}", c["base_hold_s"]
            elif aux_lin is not None and aux_fresh:
                still = aux_lin <= c["aux_lin_mps"] and (aux_yaw is not None and aux_yaw <= c["aux_yaw_dps"])
                basis, hold = f"aux:fastlio_odom({why})", c["aux_hold_s"]
            else:
                still, basis, hold = False, f"no_fresh_evidence({why})", None
            if still:
                self._since = self._since if self._since is not None else now
                if now - self._since >= hold:
                    self.basis = basis
                    self._to("BASE_STOPPED", now, basis=basis)
                    return self.state
            else:
                self._since = None
            if now - self.t_enter > c["wait_base_stop_timeout_s"]:            # checked on every branch
                self._to("BASE_STOP_TIMEOUT", now, basis=basis)
            return self.state
        if self.state == "BASE_STOPPED":
            self._to("WAIT_IMAGE_STABLE", now)
        if self.state == "WAIT_IMAGE_STABLE":
            p, why = _fresh_primary(primary, now, c)
            if p is not None:
                still = p.value <= c["base_lin_mps"] and (p.yaw is None or p.yaw <= c["img_yaw_dps"])
            elif aux_lin is not None and aux_fresh and aux_yaw is not None:
                still = aux_lin <= c["aux_lin_mps"] and aux_yaw <= c["img_yaw_dps"]
            else:
                still = False
            if servo_settled is False:
                still = False
            if still:
                self._since = self._since if self._since is not None else now
                if now - self._since >= c["img_hold_s"]:
                    self._to("IMAGE_STABLE", now, servo_settled=servo_settled)
                    return self.state
            else:
                self._since = None
            if now - self.t_enter > c["wait_image_stable_timeout_s"]:         # checked on every branch
                self._to("IMAGE_STABLE_TIMEOUT", now)
        return self.state


# ------------------------------------------------------------------------------------------ geometry inputs
DEFAULT_GEO_CFG = dict(
    max_depth_gap_s=0.10,          # UNCALIBRATED development value (legacy code had no bound)
    max_pose_gap_s=0.30,           # same as RosSource.pose max_gap
    max_frame_odom_skew_s=3.0,     # beyond this the two stamp streams are treated as unsynchronised
    odom_jump_m=2.0, odom_jump_dt_s=0.5,
)


def odom_reset_detected(hist, cfg=None):
    """hist: [(t, x, y, yaw)]. Time going backwards or a position jump faster than physically plausible."""
    c = dict(DEFAULT_GEO_CFG, **(cfg or {}))
    for (ta, xa, ya, _), (tb, xb, yb, _) in zip(hist, hist[1:]):
        if tb < ta:
            return True, f"odom_time_backwards:{ta:.3f}->{tb:.3f}"
        if tb - ta <= c["odom_jump_dt_s"] and math.hypot(xb - xa, yb - ya) > c["odom_jump_m"]:
            return True, f"odom_jump:{math.hypot(xb - xa, yb - ya):.2f}m/{tb - ta:.2f}s"
    return False, None


def geometry_input_check(frame_stamp, depth_stamp=None, pose_stamp=None, frame_clock="ros_header_s",
                         odom_clock="ros_header_s", depth_valid=True, odom_hist=None, need_depth=True, cfg=None):
    """Returns (ok, reasons). Any reason => GEOMETRY_UNAVAILABLE / DEFER for the stop & aim geometry."""
    c, why = dict(DEFAULT_GEO_CFG, **(cfg or {})), []
    if frame_clock != odom_clock:
        why.append(f"clock_domain_mismatch:{frame_clock}!={odom_clock}")
    if pose_stamp is None:
        why.append("pose_missing")
    elif abs(pose_stamp - frame_stamp) > c["max_pose_gap_s"]:
        why.append(f"pose_gap:{abs(pose_stamp - frame_stamp):.2f}s")
    if need_depth:
        if depth_stamp is None:
            why.append("depth_missing")
        elif abs(depth_stamp - frame_stamp) > c["max_depth_gap_s"]:
            why.append(f"depth_time_gap:{abs(depth_stamp - frame_stamp):.3f}s>{c['max_depth_gap_s']}(uncalibrated)")
        if not depth_valid:
            why.append("depth_invalid")
    if odom_hist:
        if abs(odom_hist[-1][0] - frame_stamp) > c["max_frame_odom_skew_s"]:
            why.append(f"frame_odom_unsynced:{odom_hist[-1][0] - frame_stamp:.1f}s")
        rst, w = odom_reset_detected(odom_hist, c)
        if rst:
            why.append(w)
    return (not why), why


def valid_depth_m(z):
    """0 / NaN / Inf / negative depth is invalid (never clamped to a 'normal' distance)."""
    return z is not None and isinstance(z, (int, float)) and math.isfinite(z) and z > 0.0


def decision_reference(odom_hist, now_stamp, frame_stamp, max_pose_gap_s=0.3):
    """Reference pose for the P2 stop prediction = the latest valid odom sample at decision time (never a sample from
    the future of now_stamp). frame_age_s is reported, NOT added again (the latest pose already contains the motion
    during the frame age). Returns dict(ok, pose, stamp, frame_age_s, why)."""
    past = [o for o in (odom_hist or []) if o[0] <= now_stamp + 1e-9]
    if not past:
        return dict(ok=False, why="no_odom_at_decision_time", frame_age_s=None)
    t, x, y, w = past[-1]
    if now_stamp - t > max_pose_gap_s:
        return dict(ok=False, why=f"latest_odom_stale:{now_stamp - t:.2f}s", frame_age_s=now_stamp - frame_stamp)
    return dict(ok=True, pose=(x, y, w), stamp=t, frame_age_s=now_stamp - frame_stamp, why=None)


def feasible_window(p2, tr, pose, basis, max_adv_m=6.0, step_m=0.1):
    """Forward advances (m) along the current heading at which the track is readable AND pan/tilt reachable with the
    UNCHANGED limits. Empty list => NO_FEASIBLE_STOP_WINDOW. Never widens limits, never rotates the robot."""
    x, y, w = pose
    out, k = [], 0
    while k * step_m <= max_adv_m + 1e-9:
        a = k * step_m
        v = p2.view_from(tr, (x + a * math.cos(w), y + a * math.sin(w), w), basis)
        ok, _ = p2.readable(v, 0.0)
        if ok:
            out.append(round(a, 2))
        k += 1
    return out
