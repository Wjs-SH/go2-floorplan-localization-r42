"""det_pose.py — raw FAST-LIO odometry at an image stamp (LIVE2 detection pose record). Pure python, no ROS."""
import math

ODOM_GAP_MAX_S, ODOM_NEAREST_MAX_S = 0.25, 0.30


def odom_at(samples, ts):
    """Linear interpolation between the bracketing odometry samples (gap <= 0.25 s, yaw unwrapped); else nearest within 0.3 s."""
    if not samples or ts is None:
        return dict(status="NO_ODOMETRY")
    s = sorted(samples)
    after = next((i for i, o in enumerate(s) if o[0] >= ts), None)
    if after is not None and after > 0:
        a_, b_ = s[after - 1], s[after]
        if b_[0] - a_[0] <= ODOM_GAP_MAX_S:
            w_ = (ts - a_[0]) / max(b_[0] - a_[0], 1e-9)
            dyaw = (b_[3] - a_[3] + math.pi) % (2 * math.pi) - math.pi
            return dict(status="INTERPOLATED", x=a_[1] + w_ * (b_[1] - a_[1]), y=a_[2] + w_ * (b_[2] - a_[2]), yaw=a_[3] + w_ * dyaw,
                        t_before=a_[0], t_after=b_[0], gap_s=round(b_[0] - a_[0], 3))
    near = min(s, key=lambda o: abs(o[0] - ts))
    if abs(near[0] - ts) <= ODOM_NEAREST_MAX_S:
        return dict(status="NEAREST", x=near[1], y=near[2], yaw=near[3], dt_s=round(near[0] - ts, 3))
    return dict(status="NO_ODOMETRY_NEAR_STAMP", dt_nearest_s=round(near[0] - ts, 3))
