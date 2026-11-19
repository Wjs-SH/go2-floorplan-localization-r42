#!/usr/bin/env python3
"""e6_precision_bridge.py — LIVE2 E6 precision observation: E6 target -> student aim console (pan-tilt + IMX477 + AF) -> ROI OCR
(+ optional VLM) -> <state>/e6_precision_results.jsonl (consumed by e6e7_link in e34).

Reuses the student stack unchanged: src/sci_visual_observation/aim_client.AimClient (r1-aim-1 HTTP: /aim strict, poll,
cancel, fetch_capture) talking to the student aim_console (sole owner of /dev/ttyUSB0 + IMX477), and
src/sci_text_reading (RapidOCR CPU; VLM only on request). This program never opens a serial port or camera itself.

Modes
  --plan             print, per ready E6 observation, the exact /aim request it WOULD send (no network, no device)
  --execute          REAL hardware via the aim console. Refused unless: hardware approval B record valid for 'pantilt'
                     (live1_common.approval_status('B', 'pantilt')), env SCI_PANTILT_APPROVED=1, console /state reachable,
                     robot still STOPPED on fresh odometry right before every aim (e6_core.stop_state, wall clock).
  --offline-image F  ROI OCR on an existing image file only (evaluation; source OFFLINE_IMAGE, never fed to localisation).
  --execute --focus-mode from_approval   E8 automatic mode (2026-10-22): started by the operator console at [실험 시작] (ctl
                     module 'e6') when approval B is valid; the focus mode comes from approval B (AUTOFOCUS_V2 -> student AF,
                     MANUAL_FIXED_NO_MOTOR -> no focus motor). State for the tablet in <state>/e6_bridge.json
                     (RUNNING / BLOCKED / ENDED, heartbeat); after a device / focus block it stays BLOCKED (no command, no retry).
Aim pixel: the E6 RE-DETECTION in the current RealSense scene when available (odometry/mount error drops out), else the
prediction if inside the image and the calibrated limits; otherwise NOT_AIMABLE. strict=1 always (out-of-range -> refused).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
for p in (HERE, CAND / "live1", CAND / "src" / "sci_visual_observation", CAND / "src" / "sci_text_reading", CAND / "src"):
    sys.path.insert(0, str(p))

ROI_FRAC = 0.5
QUALITY_MIN_SHARP = 30.0       # student AF min_sharp_converged (af_profile_kci_20261017); below -> re-observe, no OCR label                 # aim centres the target: OCR the central ROI (student reader uses the full frame)


# 2026-10-28 field (X20261027_235345 O001, 10-27 O001/O002/O004): the operator stops after the STOP sound, the plate is then
# beside the robot and outside the RealSense view; the old rule refused every such stop. Now the stop-pose prediction (detection
# RGB-D target carried through FAST-LIO odometry, e6_tracker) is also used OUT of the RealSense view when its angular uncertainty
# is small enough for the IMX477 view. Uncertainty model (stated assumptions, NOT tuned on ground truth): position error =
# PRED_ERR_ODOM x detection->stop path + PRED_ERR_DEPTH x detection depth + PRED_ERR_MOUNT; angle = atan(error / range at stop);
# allowed <= PRED_MAX_DEG (IMX477 half view ~12 deg at DEG_PER_PX_EST 0.01266 x 1920 px). Success is decided on the IMX477 view.
PRED_ERR_ODOM, PRED_ERR_DEPTH, PRED_ERR_MOUNT, PRED_MAX_DEG = 0.05, 0.02, 0.10, 8.0


def prediction_uncertainty(o):
    """-> (deg or None, detail). None = not computable (no stop range / depth)."""
    rng = (o.get("target_pose_at_stop") or {}).get("range_m")
    z = (o.get("pantilt_command_prediction") or {}).get("predicted_depth_m") or o.get("depth")
    if not rng or not z:
        return None, "no stop range / depth"
    path = float(o.get("detect_to_stop_path_m") or 0.0)
    e = PRED_ERR_ODOM * path + PRED_ERR_DEPTH * float(o.get("depth") or z) + PRED_ERR_MOUNT
    deg = math.degrees(math.atan2(e, float(rng)))
    return round(deg, 2), f"err {e:.2f} m (odom {PRED_ERR_ODOM:.0%} x {path:.2f} m + depth + mount) at {float(rng):.2f} m"


def aim_target(o):
    """-> (u, v, z, basis) or None. Prefers the re-detection (calibration-only chain), then the stop-pose prediction (in the
    RealSense view, or out of it within PRED_MAX_DEG uncertainty and in front of the robot)."""
    rd = o.get("redetection") or {}
    if rd.get("status") == "REDETECTED" and (rd.get("pantilt_command") or {}).get("inside_limits"):
        return rd["pixel"][0], rd["pixel"][1], float(rd["depth_m"]), "REDETECTED"
    pc = o.get("pantilt_command_prediction") or {}
    if pc.get("in_image") and pc.get("inside_limits"):
        return pc["predicted_pixel"][0], pc["predicted_pixel"][1], float(pc["predicted_depth_m"]), "PREDICTED_UNVERIFIED_MOUNT"
    ts = o.get("target_pose_at_stop") or {}
    deg, _ = prediction_uncertainty(o)
    if pc.get("inside_limits") and pc.get("predicted_pixel") and pc.get("predicted_depth_m") and (ts.get("fwd") or 0) > 0 \
            and deg is not None and deg <= PRED_MAX_DEG:
        return pc["predicted_pixel"][0], pc["predicted_pixel"][1], float(pc["predicted_depth_m"]), "PREDICTED_OUT_OF_VIEW"
    return None


def roi_ocr(img_bgr, frac=ROI_FRAC):
    """Sign-ROI OCR (e6_ocr.sign_read): sign ROI + full frame through the persistent OCR venv; sharpness = student AfCam
    formula on the sign ROI (gate) — the old centre-50 % crop missed off-centre signs (AIM #1/#2: sign up-left)."""
    import e6_ocr
    lines, meta = e6_ocr.sign_read(img_bgr)
    return lines, dict(meta, roi=meta["sign_roi_px"])


def home_from_calibration(default=(95.352, 157.189)):
    """student 2026-11-13: HOME = the pan / tilt zero of the calibration the aim planner uses (e6_core.PANTILT_CALIB; the aim
    console is started with the same file). It used to be a number here: the 2026-11-12 trim (zero 95.872 / 156.193) moved the
    console's HOME but not this number, and every capture ended HOME_NOT_CONFIRMED (0.55 / 0.99 deg off, 0.15 allowed).
    Unreadable calibration -> the old number."""
    try:
        import e6_core as G
        sv = json.loads(Path(G.PANTILT_CALIB).read_text())["servo"]
        return (round(float(sv["pan_zero_deg"]), 3), round(float(sv["tilt_zero_deg"]), 3))
    except Exception:
        return default


HOME_CMD = home_from_calibration()               # student pan_zero / tilt_zero (aim_console HOME)


LASH_DEG = 5.0                                   # student move_to: both axes go to (target - lash) first, then to target
STUDENT_LIMITS = dict(pan=(50.0, 150.0), tilt=(115.0, 180.0))   # calibration limits passed to the console (strict)
ENVELOPE_FILE = Path(__file__).resolve().parents[2] / "runtime" / "pantilt_envelope.json"


def load_envelope(path=ENVELOPE_FILE):
    """Operator-verified mechanical envelope (servo degrees, cable/mount interference checked on site). Absent or invalid ->
    the conservative test box HOME +-10 deg. Always intersected with the student calibration limits."""
    env = dict(pan=(HOME_CMD[0] - 10, HOME_CMD[0] + 10), tilt=(HOME_CMD[1] - 10, HOME_CMD[1] + 10), source="DEFAULT_HOME_PM10")
    try:
        r = json.loads(Path(path).read_text())
        env = dict(pan=tuple(map(float, r["pan"])), tilt=tuple(map(float, r["tilt"])), source=f"OPERATOR_VERIFIED:{r.get('verified_at')}")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    for k in ("pan", "tilt"):
        env[k] = (max(env[k][0], STUDENT_LIMITS[k][0]), min(env[k][1], STUDENT_LIMITS[k][1]))
    return env


def path_check(pan, tilt, env, start=HOME_CMD, lash=LASH_DEG):
    """Joint-space path of one student aim: start -> (target - lash) -> target -> HOME. The envelope is an axis box (convex),
    so every straight segment stays inside iff its end points do. No angle feedback: 'start' is the last COMMANDED pose."""
    pts = dict(start=start, lash_pre=(pan - lash, tilt - lash), target=(pan, tilt), home=HOME_CMD)
    bad = [k for k, (p_, t_) in pts.items() if not (env["pan"][0] <= p_ <= env["pan"][1] and env["tilt"][0] <= t_ <= env["tilt"][1])]
    span = dict(pan=[round(min(q[0] for q in pts.values()), 2), round(max(q[0] for q in pts.values()), 2)],
                tilt=[round(min(q[1] for q in pts.values()), 2), round(max(q[1] for q in pts.values()), 2)])
    step = max(max(abs(a_[0] - b_[0]), abs(a_[1] - b_[1])) for a_, b_ in zip(list(pts.values())[:-1], list(pts.values())[1:]))
    return dict(ok=not bad, outside=bad, waypoints={k: [round(x, 2) for x in v] for k, v in pts.items()}, span=span,
                max_step_deg=round(step, 2), envelope=dict(pan=list(env["pan"]), tilt=list(env["tilt"]), source=env["source"]))


def single_check(u, v, z, max_dev, env=None):
    """--single geometry (no device): student calibrated command, deviation from HOME, expected head direction.
    max_dev = the B2 test box (HOME +-max_dev); env = an operator-verified envelope replaces it (path incl. lash checked)."""
    import e6_core as G
    pc = G.calibrated_aim(u, v, z)
    dp, dt = pc["pan_cmd"] - HOME_CMD[0], pc["tilt_cmd"] - HOME_CMD[1]
    dev = max(abs(dp), abs(dt))
    box = env or dict(pan=(HOME_CMD[0] - max_dev, HOME_CMD[0] + max_dev), tilt=(HOME_CMD[1] - max_dev, HOME_CMD[1] + max_dev),
                      source=f"TEST_BOX_HOME_PM{max_dev:g}")
    pa = path_check(pc["pan_cmd"], pc["tilt_cmd"], box)
    why = None if pc["inside_limits"] else "outside student calibration limits"
    why = why or (None if pa["ok"] else f"path points {pa['outside']} outside {box['source']}")
    return dict(predicted=pc, d_pan=round(dp, 2), d_tilt=round(dt, 2), dev_deg=round(dev, 2), path=pa,
                target_side=("LEFT" if u < 320 else "RIGHT" if u > 320 else "CENTRE") + "/" + ("UP" if v < 240 else "DOWN" if v > 240 else "MID"),
                refused=why)


def single_outcome(res, sharp, sharp_min=QUALITY_MIN_SHARP):
    """Student order aim -> focus -> capture -> HOME. HOME only after a clean device run (IDLE, no hardware error,
    not cancelled / unknown); a focus failure or blurry capture is NOT a success but still a normal end."""
    import e6_pipeline as P
    st, ds, rc = res.get("status"), str(res.get("device_state")), res.get("reason_code")
    fst = (res.get("focus") or {}).get("status")
    focus_home, _ = P.after_focus(fst, ds)        # 2026-10-22: same rule as the E8 pipeline (418 test sent HOME after a fault)
    home_ok = st in ("COMPLETED", "FAILED") and rc != "hardware_error" and focus_home
    if st != "COMPLETED" or not (res.get("capture") or {}).get("capture_id"):
        q = "AIM_OR_CAPTURE_FAILED"
    elif sharp is None:
        q = "CAPTURE_UNREADABLE"
    else:
        q = "CAPTURED_SHARP" if sharp >= sharp_min else "CAPTURED_BLURRY_REOBSERVE"
    return home_ok, q


AUDIT = Path(__file__).resolve().parents[2] / "runtime" / "audit" / "hw_commands.jsonl"


def audit(event, **kw):
    """Append-only record of every pan-tilt / focus / console command decision (sent or refused)."""
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT, "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), wall=round(time.time(), 3), event=event, **kw),
                           ensure_ascii=False, default=str) + "\n")


FAULT_FILE = Path(__file__).resolve().parents[2] / "runtime" / "e6_focus_fault.json"


def focus_allowed(ap):
    """Any focus-motor motion (/focus, /aim focus=1, Focus V2) needs approval B limits focus_mode=AUTOFOCUS_V2 AND
    focus_mechanical_check=PASSED, and with a recorded fault also focus_fault_cleared=true. Otherwise only NO_MOTOR paths."""
    lim = (ap or {}).get("limits") or {}
    if lim.get("focus_mode") != "AUTOFOCUS_V2" or lim.get("focus_mechanical_check") != "PASSED":
        return False, f"focus motor not approved (focus_mode={lim.get('focus_mode')}, mechanical_check={lim.get('focus_mechanical_check')})"
    if FAULT_FILE.exists() and lim.get("focus_fault_cleared") is not True:
        return False, "recorded focus fault (runtime/e6_focus_fault.json) not cleared in approval B"
    try:                                     # 2026-10-22: the actuator profile must be named, and verified or under test
        prof = json.loads((Path(__file__).resolve().parent / "e6_af_profile_kci.json").read_text())
    except (OSError, ValueError):
        return False, "actuator profile e6_af_profile_kci.json missing"
    if lim.get("af_profile_id") != prof.get("profile_id"):
        return False, f"approval B af_profile_id {lim.get('af_profile_id')} != profile {prof.get('profile_id')}"
    if not prof.get("verified") and lim.get("af_profile_verification_run") is not True:
        return False, "profile not verified: needs approval B af_profile_verification_run=true (one supervised run)"
    return True, None


def hw_gate(kind, **kw):
    """Hardware approval B must be valid AT THE MOMENT of each command (not only at program start). -> (ok, why)."""
    import os
    import live1_common as C
    ap = C.approval_status("B", "pantilt")
    ok = bool(ap.get("valid")) and os.environ.get("SCI_PANTILT_APPROVED") == "1"
    why = None if ok else f"approval B {ap.get('why') or 'valid'} / SCI_PANTILT_APPROVED={os.environ.get('SCI_PANTILT_APPROVED')}"
    if ok and (kind == "focus" or kw.get("focus") == 1):
        ok, why = focus_allowed(ap)
    audit("SENT" if ok else "REFUSED_NO_VALID_APPROVAL", kind=kind, approval_file=ap.get("file"),
          approval_expires=ap.get("expires_epoch"), why=why, **kw)
    return ok, why


def focus_mode_from_approval(ap):
    """Approval B -> (pipeline focus mode, None) or (None, why). AUTOFOCUS_V2 only with every focus_allowed condition."""
    lim = (ap or {}).get("limits") or {}
    mode = lim.get("focus_mode")
    if mode == "MANUAL_FIXED_NO_MOTOR":
        return "MANUAL_FIXED_NO_MOTOR", None
    if mode == "AUTOFOCUS_V2":
        ok, why = focus_allowed(ap)
        return ("AUTO", None) if ok else (None, why)
    return None, f"approval B focus_mode {mode!r}: needs AUTOFOCUS_V2 (with mechanical check) or MANUAL_FIXED_NO_MOTOR"


def bridge_state(state_dir, **kw):
    """<state>/e6_bridge.json for the console / tablet (atomic)."""
    out = dict(kind="LIVE2_E6_BRIDGE", wall=time.time(), **kw)
    tmp = Path(state_dir) / f".e6_bridge.json.{__import__('os').getpid()}.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False, default=str))
    __import__("os").replace(tmp, Path(state_dir) / "e6_bridge.json")
    return out


STALE_STOP_S, STOP_MOVED_M, STOP_TURNED_DEG, REBASE_MAX_M = 60.0, 0.25, 5.0, 3.0


def stop_still_valid(o, odom_now, now):
    """(ok, why): the observation's target was computed at ITS stop pose; aim only while the robot is still there.
    2026-10-22: the bridge picks up every STOPPED_TARGET_READY in e6_state 'recent' — one from an earlier stop (robot moved
    since) would aim at a stale pixel. odom_now = (t, x, y, yaw[rad]) newest FAST-LIO pose (same raw frame as odom_pose_at_stop)."""
    sp = o.get("odom_pose_at_stop") or {}
    sw = o.get("stopped_wall")
    if sw is None or now - float(sw) > STALE_STOP_S:
        return False, f"stop is {now - float(sw or 0):.0f} s old (> {STALE_STOP_S:.0f} s)"
    if odom_now is None or sp.get("x") is None:
        return False, "no odometry to compare with the stop pose"
    d = math.hypot(odom_now[1] - sp["x"], odom_now[2] - sp["y"])
    dyaw = abs(math.degrees((odom_now[3] - sp["yaw"] + math.pi) % (2 * math.pi) - math.pi))
    if d > STOP_MOVED_M or dyaw > STOP_TURNED_DEG:
        return False, f"robot moved {d:.2f} m / {dyaw:.1f} deg since this stop"
    return True, None


def ready(state):
    try:
        st = json.loads((state / "e6_state.json").read_text())
    except (OSError, ValueError):
        return []
    return [o for o in st.get("recent") or [] if o.get("state") == "STOPPED_TARGET_READY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--hub", default="/dev/shm/live1_hub")
    ap.add_argument("--console", default="http://127.0.0.1:8088")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--offline-image")
    g.add_argument("--single", nargs=3, type=float, metavar=("U", "V", "Z"),
                   help="supervised first test: ONE aim->focus->capture->HOME at a RealSense pixel/depth chosen by the operator")
    ap.add_argument("--max-dev", type=float, default=10.0, help="--single: refuse if the command is farther than this from HOME (deg)")
    ap.add_argument("--envelope", nargs="?", const=str(ENVELOPE_FILE), default=None,
                    help="--single: use the operator-verified envelope (+ lash path) instead of the HOME +-max-dev test box")
    ap.add_argument("--single-dry", action="store_true", help="--single: geometry only, no device contact")
    ap.add_argument("--ocr", action="store_true", help="--single: also run ROI OCR on the capture (B4 scope; off by default)")
    ap.add_argument("--focus-mode", choices=["auto", "manual_fixed", "from_approval"], default="auto",
                    help="manual_fixed: lens set by hand with the power off; /aim focus=0 and /focus is never sent (NO_MOTOR); "
                         "from_approval: approval B focus_mode decides (E8 automatic mode)")
    ap.add_argument("--reading", choices=["immediate", "deferred"], default="immediate",
                    help="deferred (E8 pair policy): store each capture + sha256; e8_pair reads both after the next observation")
    ap.add_argument("--no-manual-ui", dest="manual_ui", action="store_false",
                    help="MANUAL_FIXED_NO_MOTOR without the operator focus UI: capture right after the aim (old behaviour); "
                         "default = wait for the operator's focus ring + [완료] on the 8770 page")
    ap.add_argument("--settle-mode", choices=["AUTO", "MANUAL_SETTLE"], default="AUTO",
                    help="MANUAL_SETTLE (2026-10-30): after the confirmed stop wait for the operator's [진동 안정 확인 · 팬틸트 실행] "
                         "(e6_settle); AUTO = existing path")
    ap.add_argument("--deadline-s", type=float, default=70.0)
    ap.add_argument("--max-s", type=float, default=1800.0)
    a = ap.parse_args()
    state = Path(a.state)
    if a.offline_image:
        import cv2
        lines, meta = roi_ocr(cv2.imread(a.offline_image))
        print(json.dumps(dict(source="OFFLINE_IMAGE", image=a.offline_image, ocr_lines=lines, **meta), ensure_ascii=False))
        return 0
    if a.single and a.single_dry:
        print(json.dumps(dict(mode="SINGLE_DRY_NO_DEVICE", request=a.single, **single_check(*a.single, a.max_dev, load_envelope(a.envelope) if a.envelope else None)), ensure_ascii=False, default=str))
        return 0
    if a.plan:
        for o in ready(state):
            t = aim_target(o)
            print(json.dumps(dict(observation_id=o["observation_id"], source=o.get("source"),
                                  would_send=None if t is None else dict(path="/aim", u=int(t[0]), v=int(t[1]), z=round(t[2], 2), focus=1,
                                                                          strict=1, basis=t[3]),
                                  status="NOT_AIMABLE" if t is None else "PLAN_ONLY"), ensure_ascii=False))
        return 0
    import os
    import live1_common as C
    ap_b = C.approval_status("B", "pantilt")
    if not ap_b["valid"] or os.environ.get("SCI_PANTILT_APPROVED") != "1":
        if a.execute:
            state.mkdir(parents=True, exist_ok=True)
            bridge_state(state, state="REFUSED", why=f"approval B {ap_b.get('why')}")
        sys.exit(f"REFUSED --execute: hardware approval B {ap_b.get('why')} / SCI_PANTILT_APPROVED={os.environ.get('SCI_PANTILT_APPROVED')}")
    import cv2
    import numpy as np
    import e6_core as G
    from aim_client import AimClient
    cl = AimClient(a.console)
    if a.single:
        u, v, z = a.single
        chk = single_check(u, v, z, a.max_dev, load_envelope(a.envelope) if a.envelope else None)
        if chk["refused"]:
            sys.exit(f"REFUSED --single: {chk['refused']}")
        pc = chk["predicted"]
        t0 = time.time()
        ok_, why_ = hw_gate("aim", mode="single", pan=pc["pan_cmd"], tilt=pc["tilt_cmd"], u=u, v=v, z=z,
                            focus=0 if a.focus_mode == "manual_fixed" else 1)
        if not ok_:
            sys.exit(f"REFUSED --single aim: {why_}")
        res = cl.run("aim", dict(u=int(u), v=int(v), z=round(z, 2), focus=0 if a.focus_mode == "manual_fixed" else 1, strict=1),
                     "live2test", int(t0) % 1000, 1, a.deadline_s)
        rec = dict(kind="SINGLE_SUPERVISED_TEST", t=time.strftime("%Y-%m-%d %H:%M:%S"), request=dict(u=u, v=v, z=z), predicted=pc,
                   d_pan=chk["d_pan"], d_tilt=chk["d_tilt"], dev_deg=chk["dev_deg"], expected_side=chk["target_side"], path=chk["path"],
                   status=res.get("status"), device_state=res.get("device_state"), reason=res.get("reason_code"), focus=res.get("focus"),
                   capture=res.get("capture"), plan=(res.get("record") or {}).get("plan"), angle_feedback="NONE",
                   aim_error="NOT_MEASURED", observed_direction="OPERATOR_TO_REPORT", total_s=round(time.time() - t0, 1),
                   focus_mode="MANUAL_FIXED_NO_MOTOR" if a.focus_mode == "manual_fixed" else "AUTO")
        cap, sharp = res.get("capture") or {}, None
        if cap.get("capture_id"):
            raw = cl.fetch_capture(cap["capture_id"]) or b""
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            if img is not None:
                f_img = state / "e6_captures" / f"single_{int(t0)}.jpg"; f_img.parent.mkdir(exist_ok=True); cv2.imwrite(str(f_img), img)
                h, w = img.shape[:2]
                import e6_focus_v2 as F2
                x0, y0 = int(1280 * (1 - ROI_FRAC) / 2), int(720 * (1 - ROI_FRAC) / 2)
                centre_sharp, _ = F2.roi_metric(img, (x0, y0, 1280 - x0, 720 - y0))     # student metric, centre ROI
                roi, roi_src = F2.sign_roi(img)
                sharp, _ = F2.roi_metric(img, roi)                                       # same metric on the sign ROI
                cx, cy = (roi[0] + roi[2]) / 2 * w / 1280, (roi[1] + roi[3]) / 2 * h / 720
                rec.update(capture_file=str(f_img), capture_bytes=len(raw), resolution=[w, h], roi_laplacian_var=round(sharp, 1),
                           sharp_roi=list(roi), sharp_roi_source=roi_src, centre_laplacian_var=round(centre_sharp, 1),
                           sign_centre_px=[round(cx), round(cy)], sign_offset_px=[round(cx - w / 2), round(cy - h / 2)])
        home_ok, rec["image_quality"] = single_outcome(res, sharp)
        if a.ocr and rec["image_quality"] == "CAPTURED_SHARP":       # OCR only on a sharp sign ROI (>= QUALITY_MIN_SHARP)
            lines, meta = roi_ocr(img)
            rec.update(ocr_lines=lines, ocr_meta=meta)
        elif a.ocr:
            rec.update(ocr_lines=[], ocr_skipped=f"image_quality {rec['image_quality']} (sharp {sharp}) < {QUALITY_MIN_SHARP}")
        if home_ok:
            home_ok, why_ = hw_gate("home", mode="single")
            if not home_ok:
                rec["home"] = f"NOT_SENT: {why_}"
        if home_ok:
            h_ = cl.run("home", {}, "live2test", int(t0) % 1000, 2, 15.0)
            rec["home"] = dict(status=h_.get("status"), device_state=h_.get("device_state"))
        else:
            rec["home"] = "NOT_SENT_DEVICE_NOT_CLEAN -> operator checks the head; no further commands"
        with open(state / "e6_single_tests.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(rec, ensure_ascii=False, default=str)[:1800])
        return 0 if home_ok else 6
    state.mkdir(parents=True, exist_ok=True)
    if a.focus_mode == "from_approval":
        fm, why_fm = focus_mode_from_approval(ap_b)
        if fm is None:
            bridge_state(state, state="REFUSED", why=why_fm)
            sys.exit(f"REFUSED --execute: {why_fm}")
    else:
        fm = "MANUAL_FIXED_NO_MOTOR" if a.focus_mode == "manual_fixed" else "AUTO"
    if not cl.probe().get("ok", True):
        bridge_state(state, state="REFUSED", why="aim console not reachable")
        sys.exit("REFUSED --execute: aim console not reachable")
    import signal
    import e6_pipeline as P
    try:
        con = (Path(__file__).resolve().parents[2] / "runtime" / "owned" / "e6_aim_console.pid").read_text().split()
        af_strategy = con[2] if len(con) > 2 else None
    except OSError:
        af_strategy = None
    stop_req = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop_req.append(1))
    base = dict(focus_mode=fm, af_strategy=af_strategy, reading=a.reading, approval_file=ap_b.get("file"), approval_expires=ap_b.get("expires_epoch"),
                started_wall=time.time(), pid=__import__("os").getpid(),
                manual_focus_ui=fm == "MANUAL_FIXED_NO_MOTOR" and a.manual_ui, settle_mode=a.settle_mode,
                dc_focus_motor="NOT_USED" if fm == "MANUAL_FIXED_NO_MOTOR" else "STUDENT_AF")
    try:
        exp_id = json.loads((state.parent / "ctl" / "experiment.json").read_text()).get("exp_id")
    except (OSError, ValueError):
        exp_id = None
    def stop_check():
        idx = json.loads((Path(a.hub) / "index.json").read_text())
        return G.stop_state([tuple(s[:4]) for s in idx.get("odom") or []], (idx.get("odom") or [[0]])[-1][0], wall_now=time.time())

    vlm_fn, vlm_admit = P.vlm_endpoint(state_dir=state) if os.environ.get("SCI_E6_VLM") == "1" else (None, None)

    def moved_check(o):                          # manual focus wait: the stop this observation was aimed from still holds?
        try:
            odom = (json.loads((Path(a.hub) / "index.json").read_text()).get("odom") or [None])[-1]
        except (OSError, ValueError):
            return False, None                   # unknown -> the stop_check (ODOM_STALE) refuses the capture instead
        ok, why = stop_still_valid(dict(o, stopped_wall=time.time()), odom, time.time())
        return (not ok and "moved" in str(why)), why

    manual = None
    if fm == "MANUAL_FIXED_NO_MOTOR" and a.manual_ui:   # 2026-10-26: operator focus ring + [완료] button on the 8770 page
        import e6_manual as M
        manual = M.Channel(state)
    settle = None
    if a.settle_mode == "MANUAL_SETTLE":         # operator approval after the body settled (never replaces approval B)
        import e6_settle as ST
        settle = dict(gate=ST.SettleGate(state, C.RUNTIME), approval=lambda: C.approval_status("B", "pantilt"),
                      moved=lambda o: moved_check(o), ended=lambda: bool(stop_req))
    def _hub_idx():
        return json.loads((Path(a.hub) / "index.json").read_text())

    reposition = dict(K=lambda: (_hub_idx().get("caminfo") or {}).get("k"), samples=lambda: [tuple(s_[:4]) for s_ in _hub_idx().get("odom") or []],
                      e5_status=state.parent / "e5_shm" / "status.json") if manual else None
    if settle is not None and reposition is not None:   # R40: moved a little before ①: same plate, new stop, new prediction. Left the place (> REBASE_MAX_M): the observation ends as before
        def _rebase(o):
            import e6_tracker as T
            smp = reposition["samples"]()
            sp = o.get("odom_pose_at_stop") or {}
            if not smp or sp.get("x") is None or math.hypot(smp[-1][1] - sp["x"], smp[-1][2] - sp["y"]) > REBASE_MAX_M:
                return None
            return T.repredict(o, smp, reposition["K"](), e5_status=reposition.get("e5_status"))
        settle["rebase"] = _rebase
    pipe = P.E6Pipeline(cl, hw_gate, stop_check, state / "e6_captures", vlm=vlm_fn, vlm_admit=vlm_admit, settle=settle, reposition=reposition,
                        focus_mode=fm, deadline_s=a.deadline_s, clock=time, defer_reading=a.reading == "deferred",
                        manual=manual, moved_check=moved_check if manual else None, experiment_id=exp_id,
                        heartbeat=lambda: bridge_state(state, state="RUNNING", why=None, n_observations=n_obs_box[0],
                                                       manual_phase=(manual.d.get("phase") if manual else None), **base))
    done, t_end = set(), time.time() + a.max_s
    n_obs_box = [0]
    for o in ready(state):                       # stops that existed before this bridge started are never aimed at
        done.add(o["observation_id"])
    n_obs = 0
    while time.time() < t_end and not stop_req:
        if pipe.blocked:                         # no command, no retry: the operator ends the experiment and checks the head
            bridge_state(state, state="BLOCKED", why=pipe.blocked, n_observations=n_obs, **base)
            time.sleep(1.0)
            continue
        bridge_state(state, state="RUNNING", why=None, n_observations=n_obs, **base)
        if manual is not None:                   # SigLIP STOP request pending -> open the IMX477 now (pan-tilt does NOT move)
            try:
                rec_ = json.loads((state / "e6_state.json").read_text()).get("recent") or []
            except (OSError, ValueError):
                rec_ = []
            if any(o.get("state") == "WAIT_STOP" or (o.get("state") == "STOPPED_TARGET_READY" and o.get("observation_id") not in done)
                   for o in rec_):
                cam = cl._get_json("/live2/camera", dict(arm=1, hold=90), timeout=2.0)
                base["imx477"] = {k: cam.get(k) for k in ("on_demand", "open", "armed_left_s", "last_open_s", "opens")}
        for o in ready(state):
            oid = o["observation_id"]
            if oid in done:
                continue
            done.add(oid)
            try:
                odom = (json.loads((Path(a.hub) / "index.json").read_text()).get("odom") or [None])[-1]
            except (OSError, ValueError):
                odom = None
            ok_s, why_s = stop_still_valid(o, odom, time.time())
            if not ok_s:                         # no device command for a stale stop; the link sees a non-evidence record
                rec = dict(kind="LIVE2_E6_PRECISION", observation_id=oid, source=o.get("source"), status="SKIPPED_STALE_STOP",
                           why=why_s, wall=time.time(), e6_trace=[], ocr_lines=[], vlm=None,
                           target_odom_link_xy=(o.get("target_pose_in_odom") or {}).get("odom_link_xy"))
            else:
                rec = pipe.run(o, len(done))
                if settle is not None and rec.get("settle", {}).get("times", {}).get("t_manual_approval"):
                    settle["gate"].mark(t_image_capture=rec.get("capture_wall"),
                                        t_semantic_complete=next((t["t"] + (t.get("dur_ms") or 0) / 1000.0 for t in reversed(rec["e6_trace"])
                                                                  if t["state"] in ("OCR", "VLM_IF_NEEDED")), None))
            rec.update(experiment_id=exp_id, settle_mode=a.settle_mode, target_track_id=o.get("target_track_id"), stop_basis=o.get("stop_basis"),
                       detect_to_stop_path_m=o.get("detect_to_stop_path_m"),
                       command_predicted=(o.get("pantilt_command_prediction") or {}).get("pan_cmd"),
                       command_redetected=((o.get("redetection") or {}).get("pantilt_command") or {}))
            with open(state / "e6_precision_results.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n_obs += 1
            n_obs_box[0] = n_obs
            if manual is not None and ok_s:      # stops created while the operator focused belong to THIS stop: never aimed
                for o2 in ready(state):
                    if o2["observation_id"] not in done:
                        done.add(o2["observation_id"])
                        with open(state / "e6_precision_results.jsonl", "a") as f:
                            f.write(json.dumps(dict(kind="LIVE2_E6_PRECISION", observation_id=o2["observation_id"], source=o2.get("source"),
                                                    status="SKIPPED_SAME_STOP", why=f"requested during observation {oid} at the same stop",
                                                    wall=time.time(), e6_trace=[], ocr_lines=[], vlm=None, experiment_id=exp_id,
                                                    target_odom_link_xy=(o2.get("target_pose_in_odom") or {}).get("odom_link_xy")),
                                               ensure_ascii=False, default=str) + "\n")
            if pipe.blocked or stop_req:
                break
        time.sleep(1.0)
    bridge_state(state, state="BLOCKED_ENDED" if pipe.blocked else "ENDED", why=pipe.blocked, n_observations=n_obs, **base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
