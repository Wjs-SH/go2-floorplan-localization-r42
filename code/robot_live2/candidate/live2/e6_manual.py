"""e6_manual.py — LIVE2 E6 manual-focus observation (2026-10-26 directive: the DC focus motor is OUT of the E8 path).

focus_mode MANUAL_FIXED_NO_MOTOR + operator UI: after STOP_CONFIRMED and the student coarse aim, the 8770 page shows the
IMX477 live view; the operator turns the lens focus ring BY HAND and presses [초점 조절 완료 및 촬영]. Then: capture (full
resolution, sha256 verified on disk) -> HOME -> HOME confirmed -> RESUME_READY (two-tone on the tablet). No F+/F- command,
no autofocus, no focus pulse is ever sent on this path (the console refuses /focus in this mode as well).

Channel (files only; the 8770 console never talks to the device):
  <state>/e6_manual.json                 phase snapshot written by the bridge (tablet reads it through live1_web)
  <state>/e6_manual_events.jsonl         one line per phase change / operator command (append-only)
  <state>/control/e6_manual_cmd.jsonl    operator commands appended by the 8770 console (auth + CSRF + request_id there);
                                         the bridge reads only lines written after it started, for the CURRENT observation
                                         and experiment; every command is answered in the snapshot (accepted / refused why)
Phases: AIMING, AIM_READY, WAIT_MANUAL_FOCUS, CAPTURE_REQUESTED, CAPTURING, CAPTURE_SAVED, RETURNING_HOME, HOME_CONFIRMED
        | CAPTURE_FAILED (stays waiting: no HOME, no drive-on), HOME_NOT_CONFIRMED, USER_CANCELLED (HOME, not counted),
        USER_ABORTED (no HOME, E6 held), ROBOT_MOVED, MANUAL_TIMEOUT, AIM_FAILED, DEVICE_ERROR (no HOME).
        RESUME_READY itself comes from obs_flow (HOME confirmed AND the record handled by e8_pair AND sensors / safety OK).
HOME confirmed (no angle sensor on this head — an OPERATIONAL definition, stated on the page and in the record):
  student HOME job COMPLETED with device IDLE, then the Nano reports the commanded target == HOME (+-0.15 deg), and a NEW
  IMX477 frame arrived after the settle time. The physical pose is not measured.
Sharpness: student AfCam formula (frame -> 1280x720 gray, Laplacian variance). ROI / student centre ROI / full frame are
three different scales and are never compared with each other.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

PHASES = ("REPOSITION_NEEDED", "WAIT_BODY_SETTLE", "WAIT_AIM_SETTLE", "AIMING", "AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_REQUESTED", "CAPTURING", "CAPTURE_SAVED", "CAPTURE_REVIEW", "RETURNING_HOME",
          "HOME_CONFIRMED", "CAPTURE_FAILED", "HOME_NOT_CONFIRMED", "USER_CANCELLED", "USER_ABORTED", "ROBOT_MOVED",
          "MANUAL_TIMEOUT", "AIM_FAILED", "DEVICE_ERROR")
ACTIVE = ("WAIT_BODY_SETTLE", "WAIT_AIM_SETTLE", "AIMING", "AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_REQUESTED", "CAPTURING", "CAPTURE_SAVED", "CAPTURE_REVIEW", "RETURNING_HOME",
          "CAPTURE_FAILED")
ACTIONS = ("capture", "capture_anyway", "cancel", "abort", "accept_capture", "retake", "reposition_done", "reposition_skip")
# 2026-11-01 (field X20261031_231940: 2 of 3 captures unreadable, no way to look at the photo before it went to OCR):
# CAPTURE_SAVED -> CAPTURE_REVIEW (head stays where it is, nothing moves) -> [사진 확인 · 판독 진행] accept_capture (HOME, then
# the record goes to E8 OCR) | [다시 촬영] retake (back to WAIT_MANUAL_FOCUS in the SAME observation; the rejected photo is kept
# as <oid>_tryN.jpg with its sha256, never read). One observation = one accepted photo, however many attempts.
MAX_ATTEMPTS = 5               # captures per observation (the 5th can only be accepted or the observation cancelled)
WAIT_MAX_S = 600.0             # operator focus budget: a NOTICE only (2026-10-28) — never ends the observation, never HOMEs
POLL_S = 0.25
FRAME_MAX_AGE_S = 1.5          # capture refused when the newest IMX477 frame is older (camera stalled)
HOME_TOL_DEG = 0.15
HOME_SETTLE_S = 1.0
SHARP_LINE = 30.0              # student min_sharp_converged (ROI scale): below -> [다시 조절] / [현재 상태로 촬영]
CENTRE = (320, 180, 960, 540)  # student af_roi 0.5 in 1280x720
ROI_TEXTURE_MIN = 6.0          # gray std inside the ROI below this -> "looks like plain background"
SAT_LEVEL, GLARE_FRAC = 250, 0.05   # PROVISIONAL (one field frame, X20261027_235345 O002: plate 21 % >= 250): glare flag only
CONTRAST_LOW = 40.0            # ROI p95 - p5 gray below this -> LOW_TEXT_CONTRAST flag (provisional, not validated)


# ------------------------------------------------------------------ metrics (pure, no device)
def gray720(frame):
    import cv2
    return cv2.cvtColor(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def clamp_roi(roi):
    x0, y0, x1, y1 = [int(round(float(v))) for v in roi]
    x0, x1 = sorted((max(0, min(1279, x0)), max(1, min(1280, x1))))
    y0, y1 = sorted((max(0, min(719, y0)), max(1, min(720, y1))))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError(f"ROI too small {roi}")
    return x0, y0, x1, y1


def parse_roi(s):
    """'x0,y0,x1,y1' (1280x720 coordinates) -> tuple or None."""
    try:
        v = [float(x) for x in str(s).split(",")]
        return clamp_roi(v) if len(v) == 4 else None
    except (TypeError, ValueError):
        return None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def measure(frame, roi=None, auto=True):
    """-> dict(sharp=dict(roi, centre, full), roi, roi_source, roi_texture, luma, auto_roi, roi_warning).
    roi None -> the sign ROI found in this frame (e6_focus_v2.sign_roi) or the student centre."""
    import cv2
    g = gray720(frame)
    lv = lambda im: round(float(cv2.Laplacian(im, cv2.CV_64F).var()), 2)        # noqa: E731
    auto_roi, auto_src = None, None
    if auto or roi is None:
        import e6_focus_v2 as F2
        auto_roi, auto_src = F2.sign_roi(frame)
    if roi is None:
        roi, src = tuple(auto_roi), auto_src
    else:
        roi, src = clamp_roi(roi), "OPERATOR_SELECTED"
    x0, y0, x1, y1 = roi
    r = g[y0:y1, x0:x1]
    c = g[CENTRE[1]:CENTRE[3], CENTRE[0]:CENTRE[2]]
    tex = round(float(r.std()), 1)
    warn = []
    if tex < ROI_TEXTURE_MIN:
        warn.append(f"선택 영역에 무늬가 거의 없음 (밝기 표준편차 {tex} < {ROI_TEXTURE_MIN}) — 표지판이 아닌 배경일 수 있음")
    if src == "OPERATOR_SELECTED" and auto_src == "SIGN_COMPONENT" and iou(roi, auto_roi) < 0.1:
        warn.append("선택 영역이 자동 검출된 표지판 영역과 겹치지 않음")
    sat = round(float((r >= SAT_LEVEL).mean()), 4)
    if sat >= GLARE_FRAC:
        warn.append(f"표지판 영역 포화 픽셀 {100 * sat:.1f}% — 반사광 의심 (선명도 수치가 초점과 무관하게 커질 수 있음)")
    return dict(sharp=dict(roi=lv(r), centre=lv(c), full=lv(g)), roi=list(roi), roi_source=src, roi_texture=tex,
                roi_saturated_frac=sat, roi_contrast=round(float(np.percentile(r, 95) - np.percentile(r, 5)), 1),
                luma=round(float(r.mean()), 1), auto_roi=None if auto_roi is None else list(auto_roi), auto_roi_source=auto_src,
                roi_warning=warn, scale_note="roi / centre / full are different scales — never compared with each other")


def quality_flags(m, operator_accepted=False):
    """Image-quality flags of a MANUAL capture (several may hold). They never block the capture; they describe it."""
    m = m or {}
    f = ["MANUAL_ACCEPTED"] if operator_accepted else []
    roi = (m.get("sharp") or {}).get("roi")
    if roi is None or roi < SHARP_LINE:
        f.append("LOW_SHARPNESS")
    if (m.get("roi_saturated_frac") or 0) >= GLARE_FRAC:
        f += ["SATURATED_ROI", "SUSPECTED_GLARE"]
    if m.get("roi_contrast") is not None and m["roi_contrast"] < CONTRAST_LOW:
        f.append("LOW_TEXT_CONTRAST")
    if m.get("roi_source") not in ("OPERATOR_SELECTED", "SIGN_COMPONENT"):
        f.append("TARGET_ROI_NOT_DETECTED")
    return f


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def save_verified(data, path, expect_sha=None):
    """Write the capture bytes atomically, read back, check sha256 and that it decodes. -> (ok, info)."""
    import cv2
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        back = path.read_bytes()
    except OSError as e:
        return False, dict(why=f"WRITE_FAILED {type(e).__name__}: {e}")
    sha = sha256_bytes(back)
    if expect_sha and sha != expect_sha:
        return False, dict(why=f"SHA256_MISMATCH console {expect_sha[:12]} disk {sha[:12]}", sha256=sha)
    img = cv2.imdecode(np.frombuffer(back, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False, dict(why="IMAGE_NOT_DECODABLE", sha256=sha)
    return True, dict(sha256=sha, bytes=len(back), resolution=[img.shape[1], img.shape[0]], file=str(path), image=img)


def home_confirmed(home_res, pose, frame_after, home_cmd, tol=HOME_TOL_DEG):
    """Operational HOME confirmation (no angle sensor). -> (ok, checks)."""
    checks = dict(job=(home_res or {}).get("status"), device_state=(home_res or {}).get("device_state"),
                  nano_target=None if not pose else [pose.get("pan"), pose.get("tilt")], home_cmd=list(home_cmd),
                  new_frame_after_settle=bool(frame_after),
                  basis="commanded target + new frame; physical angle NOT measured (no sensor)")
    ok_job = checks["job"] == "COMPLETED" and str(checks["device_state"]) == "IDLE"
    ok_pose = bool(pose) and pose.get("pan") is not None and abs(float(pose["pan"]) - home_cmd[0]) <= tol + 0.05 \
        and abs(float(pose["tilt"]) - home_cmd[1]) <= tol + 0.05       # Nano reports 0.1 deg resolution
    checks.update(job_ok=ok_job, pose_ok=ok_pose)
    return ok_job and ok_pose and bool(frame_after), checks


# ------------------------------------------------------------------ channel (bridge side)
class Channel:
    def __init__(self, state_dir, clock=time.time):
        self.state = Path(state_dir)
        self.clock = clock
        self.snap_file = self.state / "e6_manual.json"
        self.events = self.state / "e6_manual_events.jsonl"
        self.cmd_file = self.state / "control" / "e6_manual_cmd.jsonl"
        try:
            self.off = self.cmd_file.stat().st_size          # commands from before this bridge are never executed
        except OSError:
            self.off = 0
        self.d = {}

    def _write(self):
        self.state.mkdir(parents=True, exist_ok=True)
        tmp = self.snap_file.with_name(f".{self.snap_file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False, default=str))
        os.replace(tmp, self.snap_file)

    def event(self, kind, **kw):
        rec = dict(kind=kind, wall=self.clock(), observation_id=self.d.get("observation_id"), experiment_id=self.d.get("experiment_id"))
        rec.update(kw)
        with open(self.events, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def start(self, observation_id, experiment_id, **kw):
        self.d = dict(kind="LIVE2_E6_MANUAL", observation_id=observation_id, experiment_id=experiment_id, phase=None,
                      phase_since=None, history=[], replies=[], started_wall=self.clock(), focus_mode="MANUAL_FIXED_NO_MOTOR",
                      dc_motor="NOT_USED (no F+/F- command on this path)", **kw)
        self.phase("AIMING")

    def phase(self, ph, why=None, **kw):
        assert ph in PHASES, ph
        now = self.clock()
        changed = ph != self.d.get("phase")
        self.d.update(kw, phase=ph, why=why, wall=now)
        if changed:
            self.d["phase_since"] = now
            self.d["history"] = (self.d.get("history") or []) + [dict(phase=ph, wall=round(now, 3), why=why)]
            self.event("PHASE", phase=ph, why=why)
        self._write()

    def update(self, **kw):
        self.d.update(kw, wall=self.clock())
        self._write()

    def reply(self, cmd, accepted, why):
        r = dict(request_id=cmd.get("request_id"), action=cmd.get("action"), accepted=accepted, why=why, wall=self.clock())
        self.d["replies"] = (self.d.get("replies") or [])[-9:] + [r]
        self.event("COMMAND", **r, roi=cmd.get("roi"), actor=cmd.get("actor"))
        self._write()

    def commands(self):
        """New command lines -> list of dicts (malformed lines skipped)."""
        try:
            with open(self.cmd_file) as f:
                f.seek(self.off)
                lines = f.readlines()
                self.off = f.tell()
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                c = json.loads(ln)
            except ValueError:
                continue
            if isinstance(c, dict):
                out.append(c)
        return out


def append_command(state_dir, cmd):
    """8770 console side: one operator command line (validated there)."""
    p = Path(state_dir) / "control" / "e6_manual_cmd.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    return p


def read_snapshot(state_dir):
    try:
        return json.loads((Path(state_dir) / "e6_manual.json").read_text())
    except (OSError, ValueError):
        return None
