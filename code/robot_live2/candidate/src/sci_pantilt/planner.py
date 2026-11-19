#!/usr/bin/env python3
"""SCI pan/tilt aim planner + guarded console interface.

Planning (always allowed, no hardware): RealSense pixel/range of the retargeted sign -> servo command angles through the
UNCHANGED legacy pantilt_aim.Aim and the read-only calibration pantilt_calib_multi_20260928.json; reachability against
calibration + aim_console limits.

Actuation (PHYSICAL, needs operator approval): the only command path is the existing aim_console HTTP endpoint
  GET /aim?u=&v=&focus=&z=&strict=1&req=<request_id>&owner=<owner>   (r1-aim-1; strict=1 refuses out-of-range)
It is sent ONLY when BOTH are true: env SCI_PANTILT_APPROVED=1 and the planner was created with execute=True.
Otherwise the planner returns mode=DRY_RUN with the exact URL it would have sent (state PENDING_APPROVAL).
Read-only console probes (/state) are allowed for interface verification.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sci_visual_observation"))
from features import Geometry  # noqa: E402
from aim_client import AimClient, urllib_get  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sci_common"))
import r1_contracts as C  # noqa: E402

CONSOLE = os.environ.get("SCI_AIM_CONSOLE", "http://127.0.0.1:8088")


class PanTiltPlanner:
    def __init__(self, geom: Geometry | None = None, console=CONSOLE, execute=False, http_get=None):
        self.geom = geom or Geometry()
        self.console = console.rstrip("/")
        self.client = AimClient(self.console, http_get=http_get or urllib_get)
        self.execute = bool(execute) and os.environ.get("SCI_PANTILT_APPROVED") == "1"

    def plan(self, u, v, z, focus=True):
        pt = self.geom.pantilt(u, v, z, margin_deg=0.0)
        url = f"{self.console}/aim?u={int(round(u))}&v={int(round(v))}&focus={1 if focus else 0}&z={z:.3f}&strict=1"
        return dict(u=float(u), v=float(v), z=float(z), pan_cmd=pt["pan"], tilt_cmd=pt["tilt"], reachable=pt["ok"],
                    why=pt["why"], pan_lim=self.geom.pan_lim, tilt_lim=self.geom.tilt_lim, url=url,
                    calibration=str(Path(os.environ.get("SCI_PANTILT_CALIB", "")) or "default"))

    def plan_point(self, p_cam, focus=True):
        """Plan for a 3D point in the RealSense optical frame (may be outside the image, e.g. a verified sign the robot
        has walked past). The console /aim takes pixel + range, so the point is expressed as its (possibly off-image)
        pixel projection and optical depth; pantilt_aim reconstructs the same 3D point from it."""
        x, y, z = (float(c) for c in p_cam)
        if z <= 0.05:
            return dict(reachable=False, why="target_behind_camera_plane", url=None, pan_cmd=None, tilt_cmd=None, p_cam=list(p_cam))
        u, v = self.geom.fx * x / z + self.geom.cx, self.geom.fy * y / z + self.geom.cy
        d = self.plan(u, v, z, focus)
        d["p_cam"], d["off_image"] = list(p_cam), not (0 <= u < self.geom.W and 0 <= v < self.geom.H)
        return d

    def refine_from_capture(self, offset_px, plan, fpx=4400.0, width=1920, max_step_deg=5.0):
        """One bounded fine-aim step from the target's pixel offset in the IMX477 capture (e.g. OCR text-box centre minus
        image centre). Returns new servo command angles (dry-run unless execute is enabled)."""
        sx = fpx * width / 1920.0
        d_pan = -math.degrees(math.atan2(offset_px[0], sx)) * self.geom.calib["servo"]["pan_deg_per_left_deg"]
        d_tilt = -math.degrees(math.atan2(offset_px[1], sx)) * abs(self.geom.calib["servo"]["tilt_deg_per_up_deg"]) * \
            (1 if self.geom.calib["servo"]["tilt_deg_per_up_deg"] > 0 else -1)
        d_pan, d_tilt = (max(-max_step_deg, min(max_step_deg, d)) for d in (d_pan, d_tilt))
        pan, tilt = plan["pan_cmd"] + d_pan, plan["tilt_cmd"] + d_tilt
        ok = self.geom.pan_lim[0] <= pan <= self.geom.pan_lim[1] and self.geom.tilt_lim[0] <= tilt <= self.geom.tilt_lim[1]
        return dict(pan_cmd=pan, tilt_cmd=tilt, d_pan=d_pan, d_tilt=d_tilt, within_limits=ok,
                    url=f"{self.console}/goto?pan={pan:.1f}&tilt={tilt:.1f}" if ok else None,
                    note="estimated IMX477 focal length; /goto is the existing console endpoint (clipped to its limits)")

    def console_state(self, timeout=1.5):
        """Read-only probe. Returns dict or {'reachable': False, 'error': ...}."""
        try:
            d = json.loads(self.client.http_get(self.console + "/state", timeout))
            return dict(reachable=True, status=d.get("status"), job=d.get("job"), protocol=d.get("protocol"),
                        device_state=d.get("device_state"), active_request=d.get("active_request"))
        except Exception as e:
            return dict(reachable=False, error=f"{type(e).__name__}: {e}")

    def aim(self, plan, run_id="adhoc", event_id=0, attempt=0, timeout_s=25.0, poll_s=0.25, kind="aim"):
        """R1: send the aim (only if approved) as an r1-aim-1 request and follow THAT request_id to a terminal state.
        The legacy 'console job counter advanced = success' rule is removed (10-15 race: another job's photo was taken
        as this event's capture). Legacy consoles are refused with PROTOCOL_MISMATCH; there is no fallback."""
        if kind == "aim" and not plan["reachable"]:
            return dict(mode="REFUSED", ok=False, why="target_out_of_pantilt_range", plan=plan)
        if not self.execute:
            return dict(mode="DRY_RUN", ok=False, why="PENDING_APPROVAL (SCI_PANTILT_APPROVED!=1 or execute=False)", plan=plan)
        params = {} if kind == "home" else dict(u=int(round(plan["u"])), v=int(round(plan["v"])), z=f"{plan['z']:.3f}",
                                                 focus=1 if plan.get("focus", True) else 0, strict=1)
        r = self.client.run(kind, params, run_id, event_id, attempt, deadline_s=timeout_s, poll_s=poll_s,
                            cancel_grace_s=float(os.environ.get("SCI_AIM_CANCEL_GRACE_S", "5.0")))
        ok = r["status"] == C.ReqStatus.COMPLETED and (kind == "home" or r.get("capture") is not None)
        return dict(r, mode="EXECUTE", ok=ok, why=None if ok else r.get("reason_code"), plan=plan)

    def fetch_capture(self, capture_id=None, timeout=5):
        """Bytes of exactly this capture_id. The global full.jpg is never used (R1)."""
        if not self.execute or not capture_id:
            return None
        return self.client.fetch_capture(capture_id, timeout)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="plan only (no movement): pixel+range -> servo angles")
    ap.add_argument("u", type=float); ap.add_argument("v", type=float); ap.add_argument("z", type=float)
    a = ap.parse_args()
    p = PanTiltPlanner()
    print(json.dumps(dict(plan=p.plan(a.u, a.v, a.z), console=p.console_state(), execute_enabled=p.execute), indent=1))
