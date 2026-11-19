#!/usr/bin/env python3
"""e6_rgb_only.py (R34, STAGED, NOT WIRED TO ANY BUTTON): RGB-only aim for when depth is unavailable and the operator confirms a plate while the robot stands still.
Gives only a direction (ray) from the pixel. No distance and no 3D point are produced, because none is known; it is therefore not a target for E8 / E7. Refused unless enabled, robot stopped and inside the verified range. Hardware use
needs field approval, so nothing here sends a command: the result is a proposal. Enable: LIVE2_RGB_ONLY_AIM=1 (not set anywhere)."""
import math
import os

STATUS = "STAGED_NOT_ENABLED"


def propose(u, v, K, robot_stopped, envelope, pan_zero=95.352, tilt_zero=157.189, enabled=None):
    enabled = os.environ.get("LIVE2_RGB_ONLY_AIM") == "1" if enabled is None else enabled
    if not enabled:
        return dict(ok=False, status=STATUS, why="RGB-only aim is staged only (field approval needed)", command_sent=False)
    if not robot_stopped:
        return dict(ok=False, status="REFUSED_MOVING", why="the operator confirms the plate only while the robot is stopped", command_sent=False)
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    pan_rel, tilt_rel = math.degrees(math.atan2(cx - u, fx)), math.degrees(math.atan2(cy - v, fy))
    pan, tilt = pan_zero + pan_rel, tilt_zero - tilt_rel
    ok = envelope["pan"][0] <= pan <= envelope["pan"][1] and envelope["tilt"][0] <= tilt <= envelope["tilt"][1]
    return dict(ok=ok, status="PROPOSAL_ONLY" if ok else "OUTSIDE_VERIFIED_ENVELOPE", pan_proposal=round(pan, 1), tilt_proposal=round(tilt, 1), distance_m=None, target_3d=None,
                basis="ray only; parallax between the cameras uncorrected because the distance is unknown", usable_for_e8=False, command_sent=False)
