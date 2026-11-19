#!/usr/bin/env python3
"""Per-candidate image-quality and geometry features for the SCI Observation Policy (CPU, numpy/OpenCV only).

All geometry uses the RealSense colour intrinsics from the pan/tilt calibration (read-only copy).
Pan/tilt reachability uses the legacy pantilt_aim.Aim (unchanged) with the calibration's servo limits, intersected
with the aim_console limits PAN_LIM/TILT_LIM (both are hard stops; the stricter wins).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

LEGACY = Path(__file__).resolve().parents[1] / "sci_legacy_e5"
sys.path.insert(0, str(LEGACY))
from pantilt_aim import Aim  # noqa: E402

CALIB = Path(os.environ.get("SCI_PANTILT_CALIB", str(Path.home() / "pantilt_calib" / "configs" / "pantilt_calib_multi_20260928.json")))
CONSOLE_PAN_LIM, CONSOLE_TILT_LIM = (20.0, 180.0), (100.0, 180.0)      # aim_console.py PAN_LIM/TILT_LIM (read)
# IMX477 effective focal length at 1920 px width: ESTIMATED from 18 cm ArUco captures (aim_log depth + 0.21 m axis
# offset); calibration file has focal_mm = null. Used only for the readability estimate.
IMX_FPX_1920 = float(os.environ.get("SCI_IMX_FPX", "4400"))


def load_calib(path=CALIB):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Geometry:
    def __init__(self, calib=None):
        self.calib = calib or load_calib()
        c = self.calib["realsense_color"]
        self.fx, self.fy, self.cx, self.cy = c["fx"], c["fy"], c["cx"], c["cy"]
        self.W, self.H = c["width"], c["height"]
        self.aim = Aim(self.calib)
        sv = self.calib.get("servo", {})
        pl, tl = sv.get("pan_limit_deg", CONSOLE_PAN_LIM), sv.get("tilt_limit_deg", CONSOLE_TILT_LIM)
        self.pan_lim = (max(pl[0], CONSOLE_PAN_LIM[0]), min(pl[1], CONSOLE_PAN_LIM[1]))
        self.tilt_lim = (max(tl[0], CONSOLE_TILT_LIM[0]), min(tl[1], CONSOLE_TILT_LIM[1]))
        # SCI first supervised trial: the console runs with narrowed limits -> plan with the same limits
        for env, attr in (("SCI_PAN_LIM", "pan_lim"), ("SCI_TILT_LIM", "tilt_lim")):
            if os.environ.get(env):
                lo, hi = (float(v) for v in os.environ[env].split(","))
                cur = getattr(self, attr)
                setattr(self, attr, (max(cur[0], lo), min(cur[1], hi)))

    # ---------------------------------------------------------------- pan/tilt
    def pantilt(self, u, v, z, margin_deg=2.0):
        try:
            pan, tilt = self.aim.aim(u, v, z)
        except ValueError as e:
            return dict(ok=False, why=f"aim_error:{e}", pan=None, tilt=None)
        ok = (self.pan_lim[0] + margin_deg <= pan <= self.pan_lim[1] - margin_deg
              and self.tilt_lim[0] + margin_deg <= tilt <= self.tilt_lim[1] - margin_deg)
        return dict(ok=bool(ok), why=None if ok else "out_of_range", pan=float(pan), tilt=float(tilt))

    # ---------------------------------------------------------------- depth plane
    def plane(self, dimg, box, grow=0.6, step=2):
        """Fit a plane to depth points around the box (grown) -> normal (camera optical frame), obliquity vs the
        viewing ray to the box centre (0 deg = frontal), depth-valid fraction inside the box, residual RMS."""
        if dimg is None:
            return dict(ok=False)
        x0, y0, x1, y1 = box[:4]
        w, h = x1 - x0, y1 - y0
        X0, X1 = int(max(0, x0 - grow * w)), int(min(self.W - 1, x1 + grow * w))
        Y0, Y1 = int(max(0, y0 - grow * h)), int(min(self.H - 1, y1 + grow * h))
        inner = dimg[int(max(0, y0)):int(max(0, y1)) + 1, int(max(0, x0)):int(max(0, x1)) + 1]
        valid_in = float((inner > 0).mean()) if inner.size else 0.0
        z = dimg[Y0:Y1 + 1:step, X0:X1 + 1:step].astype(np.float32) / 1000.0
        vs, us = np.mgrid[Y0:Y1 + 1:step, X0:X1 + 1:step]
        ok = (z > 0.2) & (z < 10.0)
        if ok.sum() < 30:
            return dict(ok=False, valid_in=valid_in)
        P = np.stack([(us[ok] - self.cx) / self.fx * z[ok], (vs[ok] - self.cy) / self.fy * z[ok], z[ok]], 1)
        zm = np.median(P[:, 2])
        P = P[np.abs(P[:, 2] - zm) < max(0.35, 0.25 * zm)]          # drop background/foreground (door frames, people)
        if len(P) < 30:
            return dict(ok=False, valid_in=valid_in)
        c = P.mean(0)
        _, s, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[2] / np.linalg.norm(vt[2])
        ray = np.array([((x0 + x1) / 2 - self.cx) / self.fx, ((y0 + y1) / 2 - self.cy) / self.fy, 1.0])
        ray /= np.linalg.norm(ray)
        obl = math.degrees(math.acos(min(1.0, abs(float(n @ ray)))))
        rms = float(s[2] / math.sqrt(len(P)))
        return dict(ok=True, normal=[float(v) for v in n], obliquity_deg=obl, rms_m=rms, valid_in=valid_in, z_med=float(zm))

    # ---------------------------------------------------------------- photometric
    @staticmethod
    def glare(bgr, box):
        x0, y0, x1, y1 = (int(round(v)) for v in box[:4])
        roi = bgr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if roi.size == 0:
            return dict(sat=0.0, spec=0.0, contrast=0.0, sharp=0.0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sat = float((g >= 245).mean())
        spec = float(((hsv[..., 2] >= 235) & (hsv[..., 1] <= 40)).mean())     # bright & colourless = specular
        contrast = float(np.percentile(g, 95) - np.percentile(g, 5))
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        return dict(sat=sat, spec=spec, contrast=contrast, sharp=sharp)

    # ---------------------------------------------------------------- size / readability
    def size_m(self, box, z):
        if z is None:
            return None, None
        return (box[2] - box[0]) * z / self.fx, (box[3] - box[1]) * z / self.fy

    def readable_px(self, box, z, obliquity_deg, pan_axis_back_m=0.21, text_frac=0.35):
        """Estimated text height (px) in a 1920-wide IMX477 capture after aiming: box physical height x text_frac,
        foreshortened by cos(obliquity) only horizontally (so height unchanged), at range z + axis offset."""
        if z is None:
            return None
        _, hm = self.size_m(box, z)
        return float(hm * text_frac * IMX_FPX_1920 / (z + pan_axis_back_m))

    def edge_margin(self, box):
        return float(min(box[0], box[1], self.W - 1 - box[2], self.H - 1 - box[3]))


def predicted_obliquity(plane, box, fx, fy, cx, cy, z, advance_m):
    """Obliquity (deg) of the sign plane seen from a camera that moved `advance_m` forward along its optical axis
    (robot keeps walking during operator reaction). None if the plane is unknown."""
    if not plane.get("ok") or z is None:
        return None
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    P = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z - max(0.0, advance_m)])
    if P[2] <= 0.05:
        return 90.0
    n = np.asarray(plane["normal"])
    return float(math.degrees(math.acos(min(1.0, abs(float(n @ (P / np.linalg.norm(P))))))))


def odom_speed(hist, t, win=0.5):
    """hist: list of (t, x, y, yaw). Linear (m/s) and yaw rate (deg/s) over the last `win` s ending at t."""
    h = [p for p in hist if t - win - 1e-6 <= p[0] <= t + 1e-6]
    if len(h) < 2:
        return None, None
    (ta, xa, ya, wa), (tb, xb, yb, wb) = h[0], h[-1]
    dt = max(tb - ta, 1e-3)
    dw = math.atan2(math.sin(wb - wa), math.cos(wb - wa))
    return math.hypot(xb - xa, yb - ya) / dt, math.degrees(abs(dw)) / dt
