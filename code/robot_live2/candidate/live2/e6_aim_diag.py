"""e6_aim_diag.py — why the pan-tilt did / did not aim at THIS observation, in numbers (display + audit; no device access).

2026-10-27 field (X20261027_205016 / _205536): five SigLIP STOP requests, the operator stopped every time, the pan-tilt never
moved and the tablet only said "정지 확인됨". Causes found in the logs: E6 was OFF (approval B of an earlier boot), and every
corrected target needed pan 67 / 116-133 deg and tilt 141-151 deg, outside the HOME +-10 deg box (+ 5 deg lash point).
This module turns one E6 tracker record (+ the precision record, bridge state, approval B, envelope) into:
  * times: detection (RealSense frame stamp), stop request, stop confirmed; detection -> stop path / net / heading change
  * target relative to the robot at detection and at the stop (fwd, left, bearing, range) — the detection-time target is
    carried through FAST-LIO odometry to the stop pose by e6_tracker (raw odom frame, same session)
  * aim: uncorrected (student calibration on the DETECTION pixel), corrected (stop-pose prediction), re-detected (current
    RealSense scene), and which one E6 would use
  * one status code + Korean text: AIMED_CAPTURED / E6_OFF_NO_APPROVAL_B / OUTSIDE_ENVELOPE / OUT_OF_REALSENSE_VIEW / PREDICTION_UNCERTAIN /
    TARGET_BEHIND / DEPTH_OR_ODOM_INVALID / DEVICE_ERROR / CAMERA_NO_FRAME / ... (never just "aim failed").
Angles are COMMANDED servo degrees (student calibration); nothing here is a measured head angle.
"""
import math
import os
from pathlib import Path

LASH_DEG = 5.0
BYID = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_IMX477_HQ_Camera_UC517-video-index0"

KO = dict(
    WAIT_STOP="정지 대기 — 조이스틱으로 정지하십시오",
    E6_OFF_NO_APPROVAL_B="팬틸트 미동작: 이번 부팅의 승인 B 없음 → E6 꺼짐 (정지는 관측 전용)",
    E6_OFF="팬틸트 미동작: E6 모듈 꺼짐",
    E6_BLOCKED="팬틸트 미동작: E6 중단 상태 (이전 장치·HOME 오류) — 헤드 점검 필요",
    DEPTH_OR_ODOM_INVALID="조준 불가: 검출 시 깊이·FAST-LIO 자세가 유효하지 않음",
    TARGET_BEHIND="조준 불가: 정지 후 표지판이 로봇 뒤쪽",
    OUT_OF_REALSENSE_VIEW="조준 불가: 정지 후 표지판이 RealSense 화면 밖이고 예측 조준 조건 미충족 (학생 보정 한계 초과 등)",
    PREDICTION_UNCERTAIN="조준 보류: 검출 후 이동이 커서 예측 방향 불확실성이 큼 (IMX477 시야 밖일 가능성)",
    OUTSIDE_ENVELOPE="조준 거부: 필요한 팬·틸트가 확인된 안전 범위 밖",
    E6_RUNNING="E6 조준·관측 진행 중",
    AIMED_CAPTURED="조준·촬영 완료",
    REFUSED_NO_VALID_APPROVAL="조준 거부: 명령 시점에 승인 B 무효",
    DEVICE_ERROR="팬틸트 장치 오류 (Nano 응답 이상) — 헤드 점검",
    CAMERA_NO_FRAME="IMX477 영상 수신 실패 (조준 후 프레임 없음)",
    TARGET_NOT_IN_IMX477="조준 후 IMX477 화면에서 표지판을 찾지 못함",
    ROBOT_NOT_STOPPED="E6 시작 시 로봇이 정지 상태가 아님",
    STALE_STOP="정지 기록이 오래됐거나 정지 후 로봇이 움직임 — 조준 안 함",
    USER_CANCELLED="운용자가 촬영 취소", USER_ABORTED="운용자가 관측 중단 (HOME 안 함)",
    MANUAL_FOCUS_TIMEOUT="수동 초점 시간 초과", ROBOT_MOVED_DURING_FOCUS="수동 초점 중 로봇 이동",
    E6_PENDING="E6 결과 대기",
)
PREC_CODE = dict(OUTSIDE_VERIFIED_ENVELOPE="OUTSIDE_ENVELOPE", NOT_AIMABLE="OUT_OF_REALSENSE_VIEW",
                 REFUSED_NO_VALID_APPROVAL="REFUSED_NO_VALID_APPROVAL", DEVICE_ERROR_BLOCKED="DEVICE_ERROR",
                 NO_CAPTURE="CAMERA_NO_FRAME", TARGET_NOT_IN_IMX477="TARGET_NOT_IN_IMX477", ROBOT_NOT_STOPPED="ROBOT_NOT_STOPPED",
                 SKIPPED_STALE_STOP="STALE_STOP", SKIPPED_SAME_STOP="STALE_STOP", CAPTURE_STORED="AIMED_CAPTURED",
                 CAPTURED="AIMED_CAPTURED", CAPTURED_BLURRY_REOBSERVE="AIMED_CAPTURED", USER_CANCELLED="USER_CANCELLED",
                 USER_ABORTED="USER_ABORTED", MANUAL_FOCUS_TIMEOUT="MANUAL_FOCUS_TIMEOUT",
                 ROBOT_MOVED_DURING_FOCUS="ROBOT_MOVED_DURING_FOCUS", BLOCKED="E6_BLOCKED")


def effective_box(env, lash=LASH_DEG):
    """Commandable aim targets inside the envelope: the student move_to first goes to (target - lash) on both axes."""
    return dict(pan=[round(env["pan"][0] + lash, 2), round(env["pan"][1], 2)],
                tilt=[round(env["tilt"][0] + lash, 2), round(env["tilt"][1], 2)], source=env.get("source"))


def _rel(fwd, left):
    return dict(fwd=round(fwd, 2), left=round(left, 2), bearing_deg=round(math.degrees(math.atan2(left, fwd)), 1),
                range_m=round(math.hypot(fwd, left), 2))


def report(o, prec=None, bridge=None, approval_b=None, env=None):
    """o = e6_tracker record (WAIT_STOP / STOPPED_TARGET_READY / INVALID_INPUT ...). -> dict (see module doc)."""
    import e6_core as G
    out = dict(observation_id=o.get("observation_id"), source=o.get("source"), tracker_state=o.get("state"),
               angle_kind="COMMANDED servo degrees (student calibration), not a measured head angle")
    out["times"] = dict(detected=o.get("detection_timestamp"), stop_requested=o.get("created_wall"), stop_confirmed=o.get("stopped_wall"),
                        detect_to_stop_s=o.get("detect_to_stop_s"), detect_to_stop_path_m=o.get("detect_to_stop_path_m"))
    a, s = o.get("odom_pose_at_detection") or {}, o.get("odom_pose_at_stop") or {}
    if a.get("x") is not None and s.get("x") is not None:
        out["motion"] = dict(net_m=round(math.hypot(s["x"] - a["x"], s["y"] - a["y"]), 2),
                             dyaw_deg=round(math.degrees((s["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi), 1),
                             path_m=o.get("detect_to_stop_path_m"), frame="FAST-LIO raw odometry (same session)")
    tb = (o.get("target_pose_in_odom") or {}).get("body_at_detection")
    if tb:
        out["target_at_detection"] = dict(_rel(tb["fwd"], tb["left"]), depth_m=o.get("depth"))
    ts = o.get("target_pose_at_stop")
    if ts:
        out["target_at_stop"] = _rel(ts["fwd"], ts["left"])
    b = o.get("bbox")
    if b and o.get("depth"):
        try:
            r = G.calibrated_aim((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(o["depth"]))
            out["aim_uncorrected"] = dict(pan=round(r["pan_cmd"], 2), tilt=round(r["tilt_cmd"], 2),
                                          note="detection pixel as if the robot had not moved (NOT used)")
        except Exception:                                                    # noqa: BLE001
            pass
    pc, rd = o.get("pantilt_command_prediction") or {}, o.get("redetection") or {}
    if pc.get("pan_cmd") is not None:
        out["aim_corrected"] = dict(pan=round(pc["pan_cmd"], 2), tilt=round(pc["tilt_cmd"], 2), in_image=pc.get("in_image"),
                                    predicted_pixel=pc.get("predicted_pixel"))
    rpc = rd.get("pantilt_command") or {}
    if rd.get("status") == "REDETECTED" and rpc.get("pan_cmd") is not None:
        out["aim_redetected"] = dict(pan=round(rpc["pan_cmd"], 2), tilt=round(rpc["tilt_cmd"], 2), pixel=rd.get("pixel"),
                                     depth_m=rd.get("depth_m"))
    out["redetection"] = rd.get("status")
    import e6_precision_bridge as B
    out["prediction_uncertainty_deg"], out["prediction_uncertainty"] = B.prediction_uncertainty(o)
    out["prediction_max_deg"] = B.PRED_MAX_DEG
    t = B.aim_target(o) if o.get("state") == "STOPPED_TARGET_READY" else None      # the SAME rule E6 uses
    use = None if t is None else (out.get("aim_redetected") if t[3] == "REDETECTED" else out.get("aim_corrected"))
    out["aim_used"] = None if use is None else dict(use, basis=t[3])
    if env:
        box = effective_box(env)
        out["envelope"] = box
        need = use or out.get("aim_corrected")
        if need:
            out["inside_envelope"] = box["pan"][0] <= need["pan"] <= box["pan"][1] and box["tilt"][0] <= need["tilt"] <= box["tilt"][1]
    code = _status(o, out, prec, bridge or {}, approval_b or {})
    out.update(status=code, ko=KO.get(code, code))
    if code == "OUTSIDE_ENVELOPE" and out.get("envelope"):
        need = out.get("aim_used") or out.get("aim_corrected") or {}
        out["ko"] += (f" — 필요 팬 {need.get('pan')}° · 틸트 {need.get('tilt')}° / 허용 팬 {out['envelope']['pan'][0]}–{out['envelope']['pan'][1]}°"
                      f" · 틸트 {out['envelope']['tilt'][0]}–{out['envelope']['tilt'][1]}° ({out['envelope']['source']})")
    if code == "E6_OFF_NO_APPROVAL_B":
        out["ko"] += f" ({(approval_b or {}).get('why')})"
    if prec:
        out["precision"] = {k: prec.get(k) for k in ("status", "why", "pan_command", "tilt_command", "aim", "home_confirmed")}
    return out


def _status(o, out, prec, bridge, apb):
    st = o.get("state")
    if st == "WAIT_STOP":
        return "WAIT_STOP"
    if st == "INVALID_INPUT":
        return "DEPTH_OR_ODOM_INVALID"
    if prec and prec.get("status"):
        return PREC_CODE.get(prec["status"], prec["status"])
    if not bridge.get("alive") and bridge.get("state") in (None, "OFF", "REFUSED", "ENDED"):
        return "E6_OFF_NO_APPROVAL_B" if not apb.get("valid") else "E6_OFF"
    if bridge.get("blocked"):
        return "E6_BLOCKED"
    ts = out.get("target_at_stop") or {}
    if ts and ts.get("fwd", 1) <= 0:
        return "TARGET_BEHIND"
    if out.get("aim_used") is None and out.get("aim_corrected") is not None:
        if (out.get("prediction_uncertainty_deg") or 0) > (out.get("prediction_max_deg") or 1e9):
            return "PREDICTION_UNCERTAIN"
        return "OUT_OF_REALSENSE_VIEW"
    if out.get("inside_envelope") is False:
        return "OUTSIDE_ENVELOPE"
    return "E6_RUNNING" if bridge.get("alive") else "E6_PENDING"


def readiness(approval_b, env, port="/dev/ttyUSB0", byid=BYID, holders=None):
    """What E6 needs for THIS boot, checked before [실험 시작] (no device contact): approval B scope, focus mode, envelope,
    Nano port + IMX477 present, nothing else holding them. -> dict(ok, items[], effective box)."""
    lim = (approval_b or {}).get("limits") or {}
    items = [dict(key="approval_b", ok=bool((approval_b or {}).get("valid")),
                  ko="승인 B (이번 부팅, 팬틸트)" + ("" if (approval_b or {}).get("valid") else f" 없음: {(approval_b or {}).get('why')}")),
             dict(key="focus_mode", ok=lim.get("focus_mode") == "MANUAL_FIXED_NO_MOTOR",
                  ko=f"초점 모드 {lim.get('focus_mode') or '—'} (필요: MANUAL_FIXED_NO_MOTOR, 초점 모터 권한 불필요)"),
             dict(key="nano", ok=os.path.exists(port), ko=f"Nano {port}" + ("" if os.path.exists(port) else " 없음")),
             dict(key="imx477", ok=os.path.exists(byid), ko="IMX477 (by-id)" + ("" if os.path.exists(byid) else " 없음")),
             dict(key="free", ok=not holders, ko="장치 점유 없음" if not holders else f"점유 중: {holders}")]
    box = effective_box(env) if env else None
    return dict(ok=all(i["ok"] for i in items), items=items, effective_box=box,
                required_approval=dict(module="pantilt", focus_mode="MANUAL_FIXED_NO_MOTOR", boot_bound=True,
                                       envelope=None if not env else env.get("source")),
                note="승인 B 는 부팅마다 새로 필요 (현장 확인). 없으면 실험은 시작되지만 E6 팬틸트·IMX477 은 꺼진 채 관측 전용")


def latest(e6_recent):
    obs = [o for o in (e6_recent or []) if o.get("observation_id")]
    return max(obs, key=lambda x: float(x.get("created_wall") or 0)) if obs else None


def event_image(state_dir, e5_status, oid):
    """RealSense scene image of the SigLIP detection that created this observation (observe run events/), or None."""
    run = (e5_status or {}).get("run")
    if not run:
        return None
    import json
    p = Path(run) / "live2_stop_requests.jsonl"
    try:
        for ln in p.read_text().splitlines()[::-1]:
            r = json.loads(ln)
            if r.get("observation_id") == oid and r.get("image_file"):
                f = Path(run) / r["image_file"]
                return f if f.is_file() else None
    except (OSError, ValueError):
        return None
    return None
