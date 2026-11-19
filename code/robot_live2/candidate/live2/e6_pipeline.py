"""e6_pipeline.py — LIVE2 E6 precision-observation state machine (device + reading half; runs in e6_precision_bridge).

States (each recorded with start time, duration and outcome in record["e6_trace"]):
  STOP_CONFIRMED        fresh odometry says the robot is stopped (stop_check)              else END ROBOT_NOT_STOPPED
  TARGET_REDETECTION    target pixel/depth in the CURRENT RealSense scene (e6_tracker redetection; prediction only if
                        in-image, flagged UNVERIFIED_MOUNT)                                else END NOT_AIMABLE
  COARSE_AIM            student calibration (pantilt_aim.Aim 2026-09-28) -> path check inside the operator envelope
                        (start, 5 deg lash point, target, HOME) -> approval B re-checked -> student /aim focus=0
                        (move + one IMX477 frame, no focus motion)                         device error -> END, no HOME
  IMX477_TARGET_CHECK   sign ROI in that IMX477 frame: CENTERED / OFF_CENTER / NOT_FOUND, pixel offset and an
                        angle estimate (deg/px from the AIM #1 sign size — UNVERIFIED)
  FINE_AIM_IF_NEEDED    OFF_CENTER -> correction PROPOSED and recorded, NOT applied (FINE_AIM_ENABLED = False until the
                        pixel->angle model is verified on hardware). NOT_FOUND -> HOME, END TARGET_NOT_IN_IMX477
  FOCUS_CHECK           student-formula sharpness on the sign ROI >= 30 -> no focus motion
  AUTOFOCUS_IF_NEEDED   approval re-checked -> student /focus (console focus search: Focus V2 when started with it) ->
                        new capture. SUSPECTED_MECHANICAL_FAULT / AF_BLOCKED -> blocked: no HOME, no further commands
  HIGH_RES_CAPTURE      the final 1920x1080 frame saved; sharp_ok gate
  HOME                  student /home, only after a clean device run and with a valid approval at that moment
  OCR                   e6_ocr.sign_read (sign ROI + full frame, persistent ~/sci_ws/.venv RapidOCR) only if sharp_ok
  VLM_IF_NEEDED         student OCR-first rule (r1_jobs.ReaderCore.want_vlm); only if a VLM endpoint is up and admitted;
                        a VLM-only number never becomes a label (student fuse: VLM_ONLY_UNCONFIRMED)
  SEMANTIC_MAP_MATCH, HYPOTHESIS_UPDATE   -> done in e34 by e6e7_link (the E3 tracker lives there); this record carries
                        everything they need (DEFERRED_TO_E34_LINK in the trace)
No GT anywhere. Command success != physical success: pan/tilt are commanded values (no feedback).

MANUAL FOCUS (2026-10-26 directive, focus_mode MANUAL_FIXED_NO_MOTOR + e6_manual.Channel): the DC focus motor is not used.
STOP_CONFIRMED -> TARGET_REDETECTION -> COARSE_AIM (same envelope / lash path check / approval gate, /aim focus=0) ->
IMX477_TARGET_CHECK (NOT_FOUND does not end it: a blurry lens hides the sign; the operator decides on the live view) ->
WAIT_MANUAL_FOCUS (operator turns the focus ring; nothing moves; commands from the 8770 page; robot must stay stopped) ->
HIGH_RES_CAPTURE (one fresh full-resolution frame through the console owner, sha256 checked on disk; failure keeps waiting,
no HOME) -> HOME (+ operational confirmation, e6_manual.home_confirmed) -> CAPTURE_STORED (OCR / VLM asynchronous in e8_pair).
Cancel -> HOME, not counted. Abort -> no HOME, E6 held. Robot moved / timeout -> HOME, not counted.
"""
import json
import os
import time
from pathlib import Path

FINE_AIM_ENABLED = False
DEG_PER_PX_EST = 0.01266          # IMX477 1920 px, from the AIM #1 sign size (RealSense metric width / IMX477 px) — UNVERIFIED
CENTER_TOL_FRAC = 0.20            # sign ROI centre within +-20 % of the frame size from the image centre -> CENTERED
SHARP_MIN = 30.0
FOCUS_FAULTS = ("SUSPECTED_MECHANICAL_FAULT", "SUSPECTED_DRIVE_FAULT", "AF_BLOCKED_AFTER_FAULT",    # latched -> no HOME / commands
                "FOCUS_COMM_ERROR")                                                                # serial failed: no HOME either
# 2026-10-22 (field 418 test): ONE rule for "may anything move after this focus result" — used by this pipeline, the bridge
# --single test and the console stop. NO_HOME: the focus drive is faulty / unconfirmed (no response at all, serial error,
# latched fault) -> no HOME, no further command; the operator checks the head. AF_DRIVE_LIMIT: the drive DID respond but the
# cap was reached -> HOME is allowed, then no further AF / aim in this experiment. Device not IDLE -> no HOME either.
FOCUS_NO_HOME = FOCUS_FAULTS + ("AF_NO_RESPONSE",)
FOCUS_SESSION_BLOCK = ("AF_DRIVE_LIMIT",)


def after_focus(status, device_state="IDLE"):
    """-> (home_allowed, block_kind) block_kind: None | 'DEVICE' | 'FOCUS_FAULT' | 'FOCUS_SESSION'."""
    if str(device_state) != "IDLE":
        return False, "DEVICE"
    if status in FOCUS_NO_HOME:
        return False, "FOCUS_FAULT"
    if status in FOCUS_SESSION_BLOCK:
        return True, "FOCUS_SESSION"
    return True, None
FOCUS_MODES = ("AUTO", "MANUAL_FIXED_NO_MOTOR")   # MANUAL: the operator set the lens by hand with the power off; /focus is
#   never sent (no F+/- command), a blurry frame stays CAPTURED_BLURRY_REOBSERVE. Never counted as an autofocus success.
STATES = ("STOP_CONFIRMED", "TARGET_REDETECTION", "COARSE_AIM", "IMX477_TARGET_CHECK", "FINE_AIM_IF_NEEDED", "FOCUS_CHECK",
          "AUTOFOCUS_IF_NEEDED", "HIGH_RES_CAPTURE", "HOME", "OCR", "VLM_IF_NEEDED", "SEMANTIC_MAP_MATCH", "HYPOTHESIS_UPDATE")


def target_check(img):
    """IMX477 frame -> where the sign is relative to the image centre (no motion)."""
    import e6_focus_v2 as F2
    h, w = img.shape[:2]
    roi, src = F2.sign_roi(img)
    sharp = round(F2.roi_metric(img, roi)[0], 1)
    sx, sy = w / 1280.0, h / 720.0
    cx, cy = (roi[0] + roi[2]) / 2 * sx, (roi[1] + roi[3]) / 2 * sy
    dx, dy = cx - w / 2, cy - h / 2
    touches = roi[0] <= 2 or roi[1] <= 2 or roi[2] >= 1278 or roi[3] >= 718
    if src != "SIGN_COMPONENT":
        state = "NOT_FOUND"
    elif abs(dx) <= CENTER_TOL_FRAC * w and abs(dy) <= CENTER_TOL_FRAC * h:
        state = "CENTERED"
    else:
        state = "OFF_CENTER"
    return dict(state=state, roi_source=src, sign_roi_1280=list(roi), sign_center_px=[round(cx), round(cy)],
                offset_px=[round(dx), round(dy)], touches_edge=touches, sharpness=sharp, resolution=[w, h],
                offset_deg_est=dict(right=round(dx * DEG_PER_PX_EST, 2), down=round(dy * DEG_PER_PX_EST, 2),
                                    model="UNVERIFIED deg/px from the AIM #1 sign size"))


REPOSITION_STATUSES, REPOSITION_MAX, REPOSITION_WAIT_S = ("OUTSIDE_VERIFIED_ENVELOPE", "NOT_AIMABLE"), 2, 120.0   # R40: 60 s ran out on X20261103_000434 O001


class E6Pipeline:
    def __init__(self, client, gate, stop_check, save_dir, ocr=None, vlm=None, vlm_admit=None, envelope=None, clock=time,
                 deadline_s=70.0, log=print, focus_mode="AUTO", defer_reading=False, manual=None, moved_check=None,
                 experiment_id=None, sleep=time.sleep, heartbeat=None, settle=None, reposition=None):
        if focus_mode not in FOCUS_MODES:
            raise ValueError(f"focus_mode {focus_mode} not in {FOCUS_MODES}")
        self.focus_mode = focus_mode
        self.cl, self.gate, self.stop_check, self.save_dir = client, gate, stop_check, Path(save_dir)
        self.ocr, self.vlm, self.vlm_admit, self.envelope = ocr, vlm, vlm_admit, envelope
        self.clock, self.deadline_s, self.log = clock, deadline_s, log
        self.blocked = None          # device / focus fault: no further commands from this pipeline instance
        self.after_home_block = None  # FOCUS_SESSION_BLOCK: applied after this observation's HOME
        self.defer_reading = defer_reading   # E8 pair policy: store the capture; OCR / VLM after the next observation (e8_pair)
        # 2026-10-26: MANUAL_FIXED_NO_MOTOR + operator UI (e6_manual.Channel): wait for the operator's focus-ring adjustment
        # and [초점 조절 완료 및 촬영] instead of capturing right after the aim. Never with the DC motor.
        if manual is not None and focus_mode != "MANUAL_FIXED_NO_MOTOR":
            raise ValueError("manual focus UI only with focus_mode MANUAL_FIXED_NO_MOTOR")
        self.manual, self.moved_check, self.experiment_id, self.sleep = manual, moved_check, experiment_id, sleep
        # 2026-10-30 MANUAL_SETTLE (e6_settle): dict(gate=SettleGate, approval=fn, moved=fn, ended=fn) -> after STOP_CONFIRMED wait for
        # the operator's [진동 안정 확인 · 팬틸트 실행]; nothing of the aim / focus / capture before. None = AUTO (unchanged path).
        self.settle = settle
        self.reposition = reposition          # R36: dict(K=fn, samples=fn) -> after a refused aim, guide the operator and re-aim from the new stop (needs the manual flow)
        self.heartbeat = heartbeat or (lambda: None)  # bridge liveness while the operator focuses (live1_web: > 10 s silent = LOST)

    # ---- bookkeeping
    def _enter(self, rec, state):
        rec["e6_state"] = state
        self._t0 = self.clock.time()
        rec["e6_trace"].append(dict(state=state, t=round(self._t0, 3)))

    def _leave(self, rec, outcome, **kw):
        tr = rec["e6_trace"][-1]
        tr.update(outcome=outcome, dur_ms=round(1000 * (self.clock.time() - self._t0), 1), **kw)

    def _end(self, rec, status, why=None):
        rec.update(status=status, why=why, e6_end_state=rec.get("e6_state"))
        rec["e6_timings_ms"] = {t["state"]: t.get("dur_ms") for t in rec["e6_trace"]}
        return rec

    def _fetch(self, res):
        import cv2
        import numpy as np
        cap = (res or {}).get("capture") or {}
        if not cap.get("capture_id"):
            return None, cap
        raw = self.cl.fetch_capture(cap["capture_id"]) or b""
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        return img, cap

    def _device_error(self, res):
        return str((res or {}).get("device_state")) == "UNKNOWN" or "hardware" in str((res or {}).get("reason_code")) \
            or (res or {}).get("status") in ("UNKNOWN",)

    def _home(self, rec, idx):
        self._enter(rec, "HOME")
        if self.blocked:
            self._leave(rec, "NOT_SENT_BLOCKED", why=self.blocked)
            return
        ok, why = self.gate("home", observation_id=rec["observation_id"])
        if not ok:
            self.blocked = f"no valid approval at HOME time: {why}"
            self._leave(rec, "NOT_SENT_NO_VALID_APPROVAL", why=why)
            return
        h = self.cl.run("home", {}, "live2", idx, 9, 15.0)
        self._leave(rec, h.get("status"), device_state=h.get("device_state"),
                    note="commanded; actual pose not measured (no feedback) — operator / image confirms")
        if self._device_error(h):
            self.blocked = f"HOME device error {h.get('device_state')} {h.get('reason_code')}"

    # ---- main
    def run(self, obs, idx):
        """obs = e6_tracker STOPPED_TARGET_READY record (observation_id, source, redetection / prediction, target pose).
        -> precision record (written by the bridge to e6_precision_results.jsonl)."""
        import cv2
        import e6_core as G
        import e6_precision_bridge as B
        rec = dict(kind="LIVE2_E6_PRECISION", observation_id=obs["observation_id"], source=obs.get("source"),
                   target_odom_link_xy=(obs.get("target_pose_in_odom") or {}).get("odom_link_xy"), wall=self.clock.time(),
                   angle_feedback="NONE (commanded angles only)", e6_trace=[], ocr_lines=[], vlm=None,
                   odom_pose_at_stop=obs.get("odom_pose_at_stop"), stopped_wall=obs.get("stopped_wall"))
        if self.blocked:
            return self._end(rec, "BLOCKED", self.blocked)

        self._enter(rec, "STOP_CONFIRMED")
        st = self.stop_check()
        self._leave(rec, "OK" if st.get("stopped") else "NOT_STOPPED", stop=st)
        rec["odometry_at_e6"] = dict(st, checked_wall=self.clock.time())         # freshness of the stop at E6 start
        if not st.get("stopped"):
            return self._end(rec, "ROBOT_NOT_STOPPED")
        if self.settle is not None:                    # MANUAL_SETTLE: operator approval after the body settled; no device command before
            self._enter(rec, "WAIT_SETTLE_APPROVAL")
            out, why, srec = self.settle["gate"].wait(obs, self.experiment_id, self.stop_check, self._shake, self.settle["approval"],
                                                      self.settle["moved"], ended=self.settle.get("ended", lambda: False), rebase=self.settle.get("rebase"),
                                                      heartbeat=self.heartbeat)
            self._leave(rec, out, why=why)
            rec["settle"] = {k: srec.get(k) for k in ("mode", "times", "approval_check", "replies")}
            if out != "E6_APPROVED":
                return self._end(rec, out, why)
            self.settle["gate"].mark(t_e6_start=self.clock.time())
        if self.manual is not None:                    # rule 1: nothing of the manual flow before STOP_CONFIRMED
            self.manual.start(rec["observation_id"], self.experiment_id, stop=st, source=obs.get("source"))
            cam = self._camera(300, wait_s=4.0)        # armed at the STOP request already; make sure frames flow before the aim
            rec["imx477_on_demand"] = {k: cam.get(k) for k in ("on_demand", "open", "first_frame", "last_open_s", "opens", "device",
                                                               "last_error")}
            self.manual.update(imx477=rec["imx477_on_demand"])
            try:
                self.manual.phase("WAIT_BODY_SETTLE", why="로봇 정지 확인 — 카메라 흔들림 안정화 중")
                bad = self._settle(rec, "WAIT_BODY_SETTLE", self.SETTLE["body_min_s"], self.SETTLE["body_max_s"])
                if bad == "ROBOT_NOT_STOPPED" and self.reposition is not None:   # R40 X20261103_003729 O001: moved after ①. The head has not moved: the plate is still in odometry; stop and press [재조준]
                    obs2, end2 = self._reposition_wait(rec, obs, dict(status="ROBOT_MOVED_BEFORE_AIM", why="robot moved while settling"))
                    if obs2 is None:
                        self._camera(20, exact=True)
                        rec["manual"] = dict(history=self.manual.d.get("history"))
                        return self._end(rec, bad, end2[1])
                    obs = obs2
                    rec.update(odom_pose_at_stop=obs.get("odom_pose_at_stop"))
                    self.manual.phase("WAIT_BODY_SETTLE", why="재배치 후 정지 확인 — 카메라 흔들림 안정화 중")
                    bad = self._settle(rec, "WAIT_BODY_SETTLE", self.SETTLE["body_min_s"], self.SETTLE["body_max_s"])
                if bad:
                    self.manual.phase("AIM_FAILED", why=f"{bad} — 조준 안 함 (팬틸트 움직이지 않음)")
                    self._camera(20, exact=True)
                    rec["manual"] = dict(history=self.manual.d.get("history"))
                    return self._end(rec, bad, "body / camera did not settle before the aim" if bad != "ROBOT_NOT_STOPPED"
                                     else "robot moved while settling")
                self.manual.phase("AIMING", why="팬틸트 자동 조준 중")
                aimed = self._aim_part(rec, obs, idx)
                n_rep = 0
                while isinstance(aimed, dict) and aimed.get("status") in REPOSITION_STATUSES and self.reposition is not None and n_rep < REPOSITION_MAX:
                    n_rep += 1
                    obs2, bad = self._reposition_wait(rec, obs, aimed)      # no device command: the operator moves the robot with the joystick
                    if obs2 is None:
                        self._camera(20, exact=True)
                        rec["manual"] = dict(history=self.manual.d.get("history"))
                        return self._end(rec, bad[0], bad[1])
                    obs = obs2
                    rec.update(odom_pose_at_stop=obs.get("odom_pose_at_stop"), repositions=n_rep)
                    self.manual.phase("WAIT_BODY_SETTLE", why="재배치 후 정지 확인 — 카메라 흔들림 안정화 중")
                    bad2 = self._settle(rec, "WAIT_BODY_SETTLE", self.SETTLE["body_min_s"], self.SETTLE["body_max_s"])
                    if bad2:
                        self.manual.phase("AIM_FAILED", why=f"{bad2} — 조준 안 함")
                        self._camera(20, exact=True)
                        rec["manual"] = dict(history=self.manual.d.get("history"))
                        return self._end(rec, bad2, "robot moved while settling after the reposition")
                    self.manual.phase("AIMING", why="재배치 후 팬틸트 자동 조준 중")
                    aimed = self._aim_part(rec, obs, idx)
                return self._manual_tail(aimed, rec, obs, idx)
            except Exception as e:                     # noqa: BLE001  a crash here must leave a visible, non-resume state
                self.blocked = f"MANUAL_FLOW_ERROR {type(e).__name__}: {e}"
                self.manual.phase("DEVICE_ERROR", why=self.blocked + " — HOME 안 함, 헤드 점검")
                try:
                    self._camera(20, exact=True)
                except Exception:                      # noqa: BLE001  the camera release must not mask the error
                    pass
                return self._end(rec, "BLOCKED", self.blocked)
        return self._run_auto(rec, obs, idx)

    def _aim_part(self, rec, obs, idx):
        """TARGET_REDETECTION .. IMX477_TARGET_CHECK for the manual flow. -> (img, cap, tc) or a finished record."""
        import e6_core as G
        import e6_precision_bridge as B
        self._enter(rec, "TARGET_REDETECTION")
        t = B.aim_target(obs)
        self._leave(rec, "NONE" if t is None else t[3], target=None if t is None else [round(t[0]), round(t[1]), round(t[2], 2)])
        rec["prediction_uncertainty_deg"], rec["prediction_uncertainty"] = B.prediction_uncertainty(obs)
        if t is None:
            ts = obs.get("target_pose_at_stop") or {}
            why = ("target behind the robot at the stop" if (ts.get("fwd") or 1) <= 0 else
                   f"prediction uncertainty {rec['prediction_uncertainty_deg']} deg > {B.PRED_MAX_DEG} ({rec['prediction_uncertainty']})"
                   if (rec["prediction_uncertainty_deg"] or 0) > B.PRED_MAX_DEG else
                   "student calibration limits exceeded" if not (obs.get("pantilt_command_prediction") or {}).get("inside_limits") else
                   "no re-detection and no usable prediction")
            return self._end(rec, "NOT_AIMABLE", why)
        self._enter(rec, "COARSE_AIM")
        pc = G.calibrated_aim(t[0], t[1], t[2])
        pa = B.path_check(pc["pan_cmd"], pc["tilt_cmd"], self.envelope or B.load_envelope())
        rec.update(path=pa, pan_command=pc["pan_cmd"], tilt_command=pc["tilt_cmd"], aim_basis=t[3])
        self.manual.update(pan_command=pc["pan_cmd"], tilt_command=pc["tilt_cmd"], aim_basis=t[3], path=pa)
        if not pc["inside_limits"] or not pa["ok"]:   # the -5 deg lash pre-point is one of the checked path points
            self._leave(rec, "OUTSIDE_VERIFIED_ENVELOPE")
            return self._end(rec, "OUTSIDE_VERIFIED_ENVELOPE", f"path points {pa['outside']} outside {pa['envelope']['source']}")
        ok, why = self.gate("aim", observation_id=rec["observation_id"], pan=pc["pan_cmd"], tilt=pc["tilt_cmd"])
        if not ok:
            self._leave(rec, "REFUSED_NO_VALID_APPROVAL", why=why)
            self.blocked = why
            return self._end(rec, "REFUSED_NO_VALID_APPROVAL", why)
        res = self.cl.run("aim", dict(u=int(t[0]), v=int(t[1]), z=round(t[2], 2), focus=0, strict=1), "live2", idx, 1, self.deadline_s)
        rec["aim"] = {k: res.get(k) for k in ("status", "device_state", "reason_code", "request_id")}
        self._leave(rec, res.get("status"), plan=(res.get("record") or {}).get("plan"))
        if self._device_error(res):
            self.blocked = f"DEVICE {res.get('device_state')} {res.get('reason_code')}"
            return self._end(rec, "DEVICE_ERROR_BLOCKED", self.blocked + " — operator checks the head; no HOME")
        img, cap = self._fetch(res)
        self._enter(rec, "IMX477_TARGET_CHECK")
        if img is None:                                # aim not completed / no frame: HOME, not counted
            self._leave(rec, "NO_FRAME", reason=res.get("reason_code"))
            self._home(rec, idx)
            return self._end(rec, "NO_CAPTURE", res.get("reason_code"))
        tc = target_check(img)
        rec["imx477_target_check"] = tc
        self._leave(rec, tc["state"], offset_px=tc["offset_px"], sharpness=tc["sharpness"])
        self._enter(rec, "FINE_AIM_IF_NEEDED")
        if tc["state"] == "OFF_CENTER":
            rec["fine_aim"] = dict(d_pan_deg=round(-tc["offset_deg_est"]["right"], 2), d_tilt_deg=round(tc["offset_deg_est"]["down"], 2),
                                   applied=False, why="FINE_AIM_ENABLED=False: pixel->angle model / tilt axis unverified on hardware")
            self._leave(rec, "PROPOSED_NOT_APPLIED", proposal=rec["fine_aim"])
        else:
            self._leave(rec, "NOT_NEEDED" if tc["state"] == "CENTERED" else "NOT_FOUND_OPERATOR_DECIDES")
        return img, cap, tc

    def _reposition_wait(self, rec, obs, failed):
        """R36: the target is outside the verified envelope (or the robot passed it). Nothing moves by itself: guidance on the tablet, the operator repositions with the joystick and presses
        [재조준], or skips this plate. The envelope is not widened. -> (new observation, None) or (None, (status, why))."""
        import e6_manual as M
        import e6_tracker as T
        ch = self.manual
        R = self.reposition
        K, env = R["K"](), self.envelope or __import__("e6_precision_bridge").load_envelope()
        g = T.reposition_guidance(obs, K, env)
        rec.setdefault("reposition", []).append(dict(failed_status=failed.get("status"), failed_why=failed.get("why"), guidance=g, wall=self.clock.time()))
        ch.phase("REPOSITION_NEEDED", why=g["text"], guidance=g, refused=dict(status=failed.get("status"), why=failed.get("why"), pan=rec.get("pan_command"), tilt=rec.get("tilt_command")))
        self._enter(rec, "REPOSITION_WAIT")
        t0 = self.clock.time()
        while True:
            if self.clock.time() - t0 > REPOSITION_WAIT_S:
                self._leave(rec, "TIMEOUT")
                ch.phase("AIM_FAILED", why=f"재배치 {REPOSITION_WAIT_S:.0f} s 초과 — 이번 표지판은 건너뜀")
                return None, (failed.get("status"), f"{failed.get('why')}; no reposition within {REPOSITION_WAIT_S:.0f} s")
            self.heartbeat()
            # R37 (11-02 X20261102_052047 O002, O003, O005): without this write e6_manual.json went stale: the console refused [재조준] after 5 s (E6_BRIDGE_SILENT) and the panel vanished after 30 s. All 3 ended "no reposition within 60 s".
            ch.update(waited_s=round(self.clock.time() - t0, 1), wait_max_s=REPOSITION_WAIT_S)
            for c in ch.commands():
                act = c.get("action")
                if c.get("observation_id") != rec["observation_id"] or (self.experiment_id and c.get("experiment_id") != self.experiment_id):
                    ch.reply(c, False, f"SESSION_MISMATCH (current {rec['observation_id']})")
                elif act == "reposition_skip":
                    ch.reply(c, True, "이번 표지판 건너뜀 — 팬틸트 움직이지 않음")
                    self._leave(rec, "OPERATOR_SKIPPED")
                    ch.phase("AIM_FAILED", why="운용자가 건너뜀 — 이동 재개 가능")
                    return None, (failed.get("status"), f"{failed.get('why')}; operator skipped after the guidance")
                elif act != "reposition_done":
                    ch.reply(c, False, "REPOSITION_PENDING — [재조준] 또는 [건너뛰기]")
                else:
                    st = self.stop_check()
                    if not st.get("stopped"):
                        ch.reply(c, False, f"ROBOT_NOT_STOPPED ({st.get('why')}) — 완전히 멈춘 뒤 다시")
                        continue
                    ch.reply(c, True, "재조준")
                    self._leave(rec, "OPERATOR_READY")
                    return T.repredict(obs, R["samples"](), K, e5_status=R.get("e5_status")), None
            self.sleep(M.POLL_S)

    def _camera(self, hold, exact=False, wait_s=0.0):
        """IMX477 on demand (console /live2/camera): arm for `hold` s; optionally wait for the first frame. -> status dict."""
        f = getattr(self.cl, "camera", None)
        q = dict(arm=1, hold=int(hold), **({"set": 1} if exact else {}))
        st = f(q) if f is not None else self.cl._get_json("/live2/camera", q, timeout=2.0)
        t0 = self.clock.time()
        while wait_s and st.get("on_demand") and not st.get("first_frame") and self.clock.time() - t0 < wait_s:
            self.sleep(0.2)
            st = f(dict(arm=0)) if f is not None else self.cl._get_json("/live2/camera", dict(arm=0), timeout=2.0)
        return st

    # 2026-10-28 field (X20261028_060633 / _061121): the aim started 0.6-1.4 s after the stop while the operator saw the head
    # shake for 2-7 s. Body settle = the FAST-LIO stop rule keeps holding; camera settle = IMX477 whole-frame shift between two
    # fresh frames (console /live2/shake). Thresholds are PROVISIONAL first field values, recorded with every sample.
    SETTLE = dict(body_min_s=2.0, body_max_s=10.0, aim_min_s=0.8, aim_max_s=5.0, ok_px=8.0, ok_n=2, poll_s=0.2)

    def _shake(self):
        f = getattr(self.cl, "shake", None)
        try:
            return f() if f is not None else self.cl._get_json("/live2/shake", dict(gap=0.15), timeout=3.0)
        except Exception as e:                                                # noqa: BLE001
            return dict(stale=True, why=f"{type(e).__name__}: {e}")

    def _settle(self, rec, label, min_s, max_s):
        """-> None (settled) | 'ROBOT_NOT_STOPPED' | 'STABILIZATION_TIMEOUT'. Never moves anything."""
        P = self.SETTLE
        self._enter(rec, label)
        t0, samples, streak = self.clock.time(), [], 0
        unmeasured = False
        while True:
            el = self.clock.time() - t0
            st = self.stop_check()
            if not st.get("stopped"):
                self._leave(rec, "ROBOT_NOT_STOPPED", samples=samples[-20:], stop=st)
                return "ROBOT_NOT_STOPPED"
            m = self._shake()
            if m.get("stale") or m.get("_error") or m.get("shift_px") is None:
                unmeasured = True
                samples.append(dict(t=round(el, 2), stale=True, why=m.get("why") or m.get("_error")))
            else:
                samples.append(dict(t=round(el, 2), px=m["shift_px"], resp=m.get("response")))
                streak = streak + 1 if m["shift_px"] <= P["ok_px"] else 0
            if el >= min_s and (streak >= P["ok_n"] or (unmeasured and not any("px" in x for x in samples[-5:]) and el >= min_s)):
                out = "STABLE" if streak >= P["ok_n"] else "STABILIZATION_UNMEASURED (camera shake not measurable: time-only)"
                self._leave(rec, out, settle_s=round(el, 2), samples=samples[-30:])
                rec.setdefault("stabilization", {})[label] = dict(result=out, settle_s=round(el, 2), samples=samples[-30:],
                                                                  thresholds=dict(P))
                return None
            if el >= max_s:
                self._leave(rec, "STABILIZATION_TIMEOUT", settle_s=round(el, 2), samples=samples[-30:])
                rec.setdefault("stabilization", {})[label] = dict(result="STABILIZATION_TIMEOUT", settle_s=round(el, 2),
                                                                  samples=samples[-30:], thresholds=dict(P))
                return "STABILIZATION_TIMEOUT"
            if self.manual is not None:
                self.manual.update(stabilization=dict(label=label, elapsed_s=round(el, 1), last=samples[-1], ok_px=P["ok_px"]))
                self.heartbeat()
            self.sleep(P["poll_s"])

    def _metrics(self, roi=None):
        f = getattr(self.cl, "metrics", None)
        if f is not None:
            return f(roi)
        q = dict(auto=1)
        if roi:
            q["roi"] = ",".join(str(int(v)) for v in roi)
        return self.cl._get_json("/live2/metrics", q, timeout=3.0)

    def _manual_tail(self, aimed, rec, obs, idx):
        """WAIT_MANUAL_FOCUS -> operator command -> HIGH_RES_CAPTURE (verified on disk) -> HOME -> HOME confirmed."""
        import e6_manual as M
        ch = self.manual
        if isinstance(aimed, dict):                    # ended before the aim was ready
            ch.phase("AIM_FAILED", why=f"{aimed.get('status')}: {aimed.get('why') or ''}")
            self._camera(20, exact=True)               # 2026-10-28: the IMX477 stayed armed 300 s after a refused aim
            rec["manual"] = dict(history=ch.d.get("history"))
            return aimed
        img, cap, tc = aimed
        ch.phase("WAIT_AIM_SETTLE", why="조준 후 카메라 안정화 중")
        bad = self._settle(rec, "WAIT_AIM_SETTLE", self.SETTLE["aim_min_s"], self.SETTLE["aim_max_s"])
        if bad == "ROBOT_NOT_STOPPED":
            ch.phase("ROBOT_MOVED", why="조준 후 안정화 중 로봇 이동")
            self._manual_home(rec, idx)                # nobody has touched the lens yet (focus not started)
            rec["manual"] = dict(history=ch.d.get("history"))
            return self._end(rec, "ROBOT_MOVED_DURING_FOCUS", "robot moved after the aim")
        if bad:                                        # still shaking after the aim: stated, the operator decides on the view
            ch.update(aim_settle_warning=f"조준 후 {self.SETTLE['aim_max_s']:.0f} s 동안 흔들림이 기준 이하로 줄지 않음")
        seen = tc["state"] != "NOT_FOUND"
        ch.phase("AIM_READY", why=None if seen else "표지판 자동 검출 안 됨 — 흐림 때문일 수 있음: 영상을 보고 판단",
                 target_check={k: tc.get(k) for k in ("state", "sign_roi_1280", "offset_px", "sharpness", "roi_source")},
                 target_seen_at_aim=seen, aim_capture_id=(cap or {}).get("capture_id"))
        rec["focus_mode"] = self.focus_mode
        self._enter(rec, "FOCUS_CHECK")
        self._leave(rec, "MANUAL_BY_OPERATOR", sharpness_at_aim=tc["sharpness"])
        self._enter(rec, "AUTOFOCUS_IF_NEEDED")
        self._leave(rec, "NOT_SENT_MANUAL_FIXED_FOCUS", why="DC focus motor excluded: the operator turns the focus ring")
        self._enter(rec, "WAIT_MANUAL_FOCUS")
        ch.phase("WAIT_MANUAL_FOCUS", why="렌즈 초점링을 손으로 돌려 맞춘 뒤 [초점 조절 완료 및 촬영]")
        t_wait = self.clock.time()
        moving_since, done, end = None, None, None
        pending, attempts, t_rev, review_s = None, [], None, 0.0      # 2026-11-01 post-capture review / retake
        while done is None and end is None:
            now = self.clock.time()
            if now - t_wait > M.WAIT_MAX_S and not ch.d.get("over_budget"):   # 2026-10-28: budget notice only; the operator
                ch.update(over_budget=True, why=f"수동 초점 {M.WAIT_MAX_S:.0f} s 초과 — 촬영 또는 관측 취소를 선택하십시오")  # decides
            st = self.stop_check()
            moved = self.moved_check(obs) if self.moved_check else (False, None)
            if st.get("stopped"):
                moving_since = None
            elif str(st.get("why", "")).startswith("MOVING"):
                moving_since = moving_since or now
            ch.update(robot_stopped=bool(st.get("stopped")), stop=st, waited_s=round(now - t_wait, 1))
            self.heartbeat()
            if now - getattr(self, "_cam_t", 0) > 10.0:   # keep the IMX477 open while the operator focuses
                self._cam_t = now
                self._camera(60)
            if moved[0] or (moving_since is not None and now - moving_since >= 3.0):
                end = ("ROBOT_MOVED", "ROBOT_MOVED_DURING_FOCUS", moved[1] or f"robot moving for 3 s: {st.get('why')}")
                break
            for c in ch.commands():
                act = c.get("action")
                if done is not None or end is not None:     # rule 6: one decision per observation; later ones are answered
                    ch.reply(c, False, "DUPLICATE / ALREADY_DECIDED for this observation")
                    continue
                if c.get("observation_id") != rec["observation_id"] or (self.experiment_id and c.get("experiment_id") != self.experiment_id):
                    ch.reply(c, False, f"SESSION_MISMATCH (current {rec['observation_id']} / {self.experiment_id})")
                    continue
                if act not in M.ACTIONS:
                    ch.reply(c, False, "UNKNOWN_ACTION")
                    continue
                if act == "cancel" and c.get("hands_clear") is not True:
                    ch.reply(c, False, "HANDS_CLEAR_REQUIRED — 관측 취소는 HOME 이동을 포함합니다: 렌즈에서 손을 뗀 뒤 다시")
                    continue
                if act == "cancel":
                    ch.reply(c, True, "관측 취소 — HOME 후 이 관측은 세지 않음")
                    notvis = c.get("target_visible") is False
                    end = ("USER_CANCELLED", "TARGET_NOT_IN_IMX477" if notvis else "USER_CANCELLED",
                           "operator: sign NOT visible in the IMX477 view (aim failed)" if notvis else "operator cancelled the capture")
                    continue
                if act == "abort":
                    ch.reply(c, True, "관측 중단 — HOME 안 함, E6 정지 (운용자 점검)")
                    end = ("USER_ABORTED", "USER_ABORTED", "operator stopped the observation (no HOME)")
                    continue
                if act in ("accept_capture", "retake") and pending is None:
                    ch.reply(c, False, "NO_CAPTURE_TO_REVIEW")
                    continue
                if pending is not None:                       # CAPTURE_REVIEW: only accept / retake / cancel / abort
                    if act in ("capture", "capture_anyway"):
                        ch.reply(c, False, "REVIEW_PENDING — 촬영된 사진을 [사진 확인 · 판독 진행] 또는 [다시 촬영]으로 먼저 결정")
                        continue
                    if act == "retake":
                        if len(attempts) + 1 >= M.MAX_ATTEMPTS:
                            ch.reply(c, False, f"MAX_ATTEMPTS {M.MAX_ATTEMPTS} — 이 사진으로 판독하거나 관측 취소")
                            continue
                        review_s += self.clock.time() - t_rev
                        attempts.append(self._archive_attempt(rec, pending, len(attempts) + 1, "RETAKE", c))
                        pending = None
                        ch.reply(c, True, f"다시 촬영 — 같은 관측 {len(attempts) + 1}번째 시도 (이전 사진은 판독하지 않음)")
                        ch.phase("WAIT_MANUAL_FOCUS", why="재촬영 — 초점 확인 후 [② 초점 확인 완료 · 사진 촬영]", review=None,
                                 attempts=[{k: a.get(k) for k in ("n", "file", "sha256", "decision")} for a in attempts])
                        continue
                    if c.get("hands_clear") is not True:      # accept -> the head moves to HOME
                        ch.reply(c, False, "HANDS_CLEAR_REQUIRED — 사진 확인 후 팬틸트가 HOME 으로 움직입니다: 렌즈에서 손을 뗀 뒤 다시")
                        continue
                    review_s += self.clock.time() - t_rev
                    pc, pm, pr = pending
                    attempts.append(dict(n=len(attempts) + 1, file=pr[1]["file"], sha256=pr[1]["sha256"], capture_id=pr[2]["capture_id"],
                                         action=pc.get("action"), decision="ACCEPTED", request_id=c.get("request_id"), wall=self.clock.time()))
                    ch.reply(c, True, "사진 확인 — HOME 후 판독 진행")
                    done = (dict(pc, hands_clear=True, review_request_id=c.get("request_id")), pm, pr)
                    pending = None
                    continue
                if not st.get("stopped"):
                    ch.reply(c, False, f"ROBOT_NOT_STOPPED ({st.get('why')}) — 촬영 거부")
                    continue
                if c.get("hands_clear") is not True:          # the head moves to HOME right after the capture
                    ch.reply(c, False, "HANDS_CLEAR_REQUIRED — 렌즈에서 손을 뗐는지 확인 후 다시 누르십시오")
                    continue
                roi = M.parse_roi(c.get("roi")) if c.get("roi") else None
                m = self._metrics(roi)
                if not m or m.get("_error") or m.get("status") or m.get("frame_age_s") is None or m["frame_age_s"] > M.FRAME_MAX_AGE_S:
                    ch.reply(c, False, f"FRAME_NOT_FRESH ({(m or {}).get('frame_age_s') or (m or {}).get('_error') or (m or {}).get('status')})")
                    continue
                # 2026-10-28 field (X20261027_235345): MANUAL mode never refuses on the AF sharpness line — the operator's
                # decision stands; the numbers are kept as quality flags (the page warns before sending capture_anyway)
                q = M.quality_flags(m, operator_accepted=True)
                ch.reply(c, True, "촬영 요청 수락" + (f" — 품질 경고 {q}" if len(q) > 1 else ""))
                r = self._manual_capture(rec, idx, c, m, roi)
                if r is None:
                    continue                           # CAPTURE_FAILED: stays waiting, no HOME, no drive-on
                pending, t_rev = (c, m, r), self.clock.time()
                capd_ = r[2]
                ch.phase("CAPTURE_REVIEW", why="촬영된 사진을 확인하십시오 — [사진 확인 · 판독 진행] 또는 [다시 촬영] (팬틸트는 그대로)",
                         review=dict(n=len(attempts) + 1, max_attempts=M.MAX_ATTEMPTS, file=Path(r[1]["file"]).name, sha256=r[1]["sha256"],
                                     capture_id=capd_.get("capture_id"), resolution=r[1]["resolution"], action=c.get("action"),
                                     roi=(capd_.get("metrics") or m or {}).get("roi"),
                                     roi_source=(capd_.get("metrics") or m or {}).get("roi_source"),
                                     flags=M.quality_flags(dict(capd_.get("metrics") or m or {}), operator_accepted=True)))
            if done is None and end is None:
                self.sleep(M.POLL_S)
        if pending is not None:                        # ended (moved / cancel / abort) while a photo was under review: never read
            review_s += self.clock.time() - t_rev
            attempts.append(self._archive_attempt(rec, pending, len(attempts) + 1, f"NOT_ACCEPTED_{end[0] if end else 'END'}", None))
        wait_s = round(self.clock.time() - t_wait, 1)
        wt = next(t for t in reversed(rec["e6_trace"]) if t["state"] == "WAIT_MANUAL_FOCUS")   # capture tries may follow it
        wt.update(outcome=end[0] if end else "OPERATOR_CONFIRMED", dur_ms=round(1000 * wait_s, 1), wait_s=wait_s)
        rec["manual"] = dict(wait_s=wait_s, target_seen_at_aim=seen, history=None, review_s=round(review_s, 1),
                             attempts=attempts, n_attempts=len(attempts))
        if end:
            ph, status, why = end
            ch.phase(ph, why=why)
            if ph in ("USER_ABORTED", "MANUAL_TIMEOUT"):
                self._camera(20, exact=True)
            if ph == "USER_ABORTED":
                self.blocked = "USER_ABORTED: observation stopped by the operator without HOME — head not at HOME"
            elif ph == "MANUAL_TIMEOUT":              # nobody confirmed the hand is off the lens: no automatic HOME
                self.blocked = "MANUAL_TIMEOUT: no operator decision — HOME NOT sent (hand on the lens not excluded); operator checks"
            else:
                self._manual_home(rec, idx)
            rec["manual"]["history"] = ch.d.get("history")
            return self._end(rec, status, why)
        c, m, (img2, info, capd) = done
        return self._manual_finish(rec, idx, c, m, img2, info, capd, seen, wait_s)

    def _archive_attempt(self, rec, pending, n, decision, cmd):
        """A reviewed-and-rejected photo: renamed <oid>_try<n>.jpg (sha256 unchanged, re-verified) so that only the accepted
        photo carries the observation's file name; recorded in rec['manual']['attempts'] and the manual event log."""
        import e6_manual as M
        pc, pm, (img, info, capd) = pending
        src = Path(info["file"])
        dst = src.with_name(f"{rec['observation_id']}_try{n}.jpg")
        try:
            os.replace(src, dst)
            sha = M.sha256_bytes(dst.read_bytes())
        except OSError as e:
            dst, sha = src, f"ARCHIVE_FAILED {type(e).__name__}"
        a = dict(n=n, file=str(dst), sha256=info["sha256"], sha256_after_archive=sha, capture_id=capd.get("capture_id"),
                 action=pc.get("action"), decision=decision, request_id=(cmd or {}).get("request_id"), wall=self.clock.time())
        self.manual.event("ATTEMPT_ARCHIVED", **a)
        return a

    def _manual_capture(self, rec, idx, c, m, roi):
        """-> (image, save info, console capture dict) or None (CAPTURE_FAILED, reason in the channel)."""
        import e6_manual as M
        ch = self.manual
        ch.phase("CAPTURE_REQUESTED", why=c.get("action"))
        self._enter(rec, "HIGH_RES_CAPTURE")
        ch.phase("CAPTURING")
        params = dict(roi=",".join(str(int(v)) for v in roi)) if roi else {}
        res = self.cl.run("live2/manual_capture", params, "live2", idx, 3, 15.0)
        capd = (((res.get("record") or {}).get("result") or {}).get("detail") or {}).get("capture")
        why = None
        if self._device_error(res):
            why = f"DEVICE {res.get('device_state')} {res.get('reason_code')}"
        elif res.get("status") != "COMPLETED" or not capd:
            why = f"CONSOLE_CAPTURE {res.get('status')} {(((res.get('record') or {}).get('result') or {}).get('detail') or {}).get('why') or res.get('reason_code')}"
        info, img = None, None
        if why is None:
            raw = self.cl.fetch_capture(capd["capture_id"]) or b""
            ok, info = M.save_verified(raw, self.save_dir / f"{rec['observation_id']}.jpg", expect_sha=capd.get("sha256"))
            if not ok:
                why = f"SAVE_NOT_VERIFIED {info.get('why')}"
            else:
                img = info.pop("image")
        if why:
            self._leave(rec, "CAPTURE_FAILED", why=why)
            ch.phase("CAPTURE_FAILED", why=f"{why} — 저장 실패: HOME·재출발 안 함. 다시 [촬영] 또는 [촬영 취소]")
            if self._device_error(res):
                raise RuntimeError(why)                # device UNKNOWN: no HOME, no further command (handled as BLOCKED)
            return None
        self._leave(rec, "SAVED_VERIFIED", sha256=info["sha256"], file_bytes=info["bytes"], resolution=info["resolution"])
        ch.phase("CAPTURE_SAVED", why=f"sha256 {info['sha256'][:12]} · {info['resolution'][0]}x{info['resolution'][1]}",
                 capture=dict(file=info["file"], sha256=info["sha256"], resolution=info["resolution"], capture_id=capd["capture_id"]))
        return img, info, capd

    def _manual_home(self, rec, idx):
        """HOME + operational confirmation (commanded target == HOME, IDLE, a new frame after settle)."""
        import e6_manual as M
        import e6_precision_bridge as B
        ch = self.manual
        ch.phase("RETURNING_HOME")
        self._home(rec, idx)
        h = rec["e6_trace"][-1]
        home_res = dict(status=h.get("outcome"), device_state=h.get("device_state"))
        pose, fresh = None, False
        if home_res["status"] == "COMPLETED" and not self.blocked:
            self.sleep(M.HOME_SETTLE_S)
            pr = self.cl.run("live2/pose", {}, "live2", idx, 8, 10.0)
            pose = ((pr.get("record") or {}).get("result") or {}).get("detail")
            m0 = self._metrics(None) or {}
            self.sleep(0.3)
            m1 = self._metrics(None) or {}
            fresh = m1.get("seq") is not None and m0.get("seq") is not None and m1["seq"] > m0["seq"] and \
                (m1.get("frame_age_s") or 9) <= M.FRAME_MAX_AGE_S
        ok, checks = M.home_confirmed(home_res, pose, fresh, B.HOME_CMD)
        self._camera(20, exact=True)                   # observation over: the IMX477 stream is released 20 s later
        rec["home"] = dict(confirmed=ok, checks=checks)
        rec["home_confirmed"] = ok
        if ok:
            ch.phase("HOME_CONFIRMED", why="운영상 복귀 확인 (Nano 명령각 = HOME, 새 영상 수신) — 실제 각도 센서 없음", home=rec["home"])
        else:
            if not self.blocked:
                self.blocked = f"HOME_NOT_CONFIRMED {checks}"
            ch.phase("HOME_NOT_CONFIRMED", why="HOME 복귀 미확인 — 재출발 금지, 헤드 점검", home=rec["home"])
        return ok

    def _manual_finish(self, rec, idx, c, m, img, info, capd, seen, wait_s):
        import e6_manual as M
        cm = capd.get("metrics") or {}
        sharp = (cm.get("sharp") or m.get("sharp") or {})
        line_met = (sharp.get("roi") or 0) >= M.SHARP_LINE
        seen_cap = cm.get("auto_roi_source") == "SIGN_COMPONENT"
        rec.update(capture_file=info["file"], capture_id=capd["capture_id"], resolution=info["resolution"],
                   image_sha256=info["sha256"], capture_wall=self.clock.time(),
                   image_quality=dict(ok=True, basis="OPERATOR_CONFIRMED_MANUAL_FOCUS", roi_laplacian_var=sharp.get("roi"),
                                      centre_laplacian_var=sharp.get("centre"), full_laplacian_var=sharp.get("full"),
                                      roi=cm.get("roi"), roi_source=cm.get("roi_source"), roi_warning=cm.get("roi_warning"),
                                      sharp_line=M.SHARP_LINE, sharp_line_met=line_met, min_sharp=M.SHARP_LINE,
                                      operator_override_low_sharpness=c.get("action") == "capture_anyway",
                                      flags=M.quality_flags(dict(cm, sharp=sharp), operator_accepted=True),
                                      roi_saturated_frac=cm.get("roi_saturated_frac"), roi_contrast=cm.get("roi_contrast")),
                   target_in_view=("SIGN_DETECTED_AT_AIM" if seen else "SIGN_DETECTED_AT_CAPTURE" if seen_cap
                                   else "NOT_DETECTED (operator decided)"))
        rec["manual"].update(operator_action=c.get("action"), operator_request_id=c.get("request_id"), operator_wall=c.get("wall"),
                             hands_clear_confirmed=c.get("hands_clear") is True, target_visible=c.get("target_visible"),
                             operator_actor=c.get("actor"), best_seen_roi=c.get("best_seen"), operator_verdict=c.get("verdict"))
        home_ok = self._manual_home(rec, idx)
        rec["manual"]["history"] = self.manual.d.get("history")
        if not home_ok:
            return self._end(rec, "CAPTURE_STORED", "image stored; HOME NOT confirmed -> no drive-on, operator checks the head")
        for s_ in ("OCR", "VLM_IF_NEEDED", "SEMANTIC_MAP_MATCH", "HYPOTHESIS_UPDATE"):
            self._enter(rec, s_)
            self._leave(rec, "DEFERRED_TO_E8_PAIR")
        rec["reading"] = "DEFERRED_TO_E8_PAIR"
        return self._end(rec, "CAPTURE_STORED", "image + sha256 stored after the operator's manual focus; OCR/VLM asynchronous (e8_pair)")

    def _run_auto(self, rec, obs, idx):
        import cv2
        import e6_core as G
        import e6_precision_bridge as B

        self._enter(rec, "TARGET_REDETECTION")
        t = B.aim_target(obs)
        self._leave(rec, "NONE" if t is None else t[3], target=None if t is None else [round(t[0]), round(t[1]), round(t[2], 2)])
        if t is None:
            return self._end(rec, "NOT_AIMABLE", "no re-detection and no in-image prediction")

        self._enter(rec, "COARSE_AIM")
        pc = G.calibrated_aim(t[0], t[1], t[2])
        pa = B.path_check(pc["pan_cmd"], pc["tilt_cmd"], self.envelope or B.load_envelope())
        rec.update(path=pa, pan_command=pc["pan_cmd"], tilt_command=pc["tilt_cmd"], aim_basis=t[3])
        if not pc["inside_limits"] or not pa["ok"]:
            self._leave(rec, "OUTSIDE_VERIFIED_ENVELOPE")
            return self._end(rec, "OUTSIDE_VERIFIED_ENVELOPE", f"path points {pa['outside']} outside {pa['envelope']['source']}")
        ok, why = self.gate("aim", observation_id=rec["observation_id"], pan=pc["pan_cmd"], tilt=pc["tilt_cmd"])
        if not ok:
            self._leave(rec, "REFUSED_NO_VALID_APPROVAL", why=why)
            self.blocked = why
            return self._end(rec, "REFUSED_NO_VALID_APPROVAL", why)
        res = self.cl.run("aim", dict(u=int(t[0]), v=int(t[1]), z=round(t[2], 2), focus=0, strict=1), "live2", idx, 1, self.deadline_s)
        rec["aim"] = {k: res.get(k) for k in ("status", "device_state", "reason_code", "request_id")}
        self._leave(rec, res.get("status"), plan=(res.get("record") or {}).get("plan"))
        if self._device_error(res):
            self.blocked = f"DEVICE {res.get('device_state')} {res.get('reason_code')}"
            return self._end(rec, "DEVICE_ERROR_BLOCKED", self.blocked + " — operator checks the head; no HOME")
        img, cap = self._fetch(res)

        self._enter(rec, "IMX477_TARGET_CHECK")
        if img is None:
            self._leave(rec, "NO_FRAME", reason=res.get("reason_code"))
            self._home(rec, idx)
            return self._end(rec, "NO_CAPTURE", res.get("reason_code"))
        tc = target_check(img)
        rec["imx477_target_check"] = tc
        self._leave(rec, tc["state"], offset_px=tc["offset_px"], sharpness=tc["sharpness"])

        self._enter(rec, "FINE_AIM_IF_NEEDED")
        if tc["state"] == "NOT_FOUND":
            self._leave(rec, "TARGET_NOT_IN_IMX477")
            self._home(rec, idx)
            return self._end(rec, "TARGET_NOT_IN_IMX477", "no sign component in the IMX477 frame")
        if tc["state"] == "OFF_CENTER":
            prop = dict(d_pan_deg=round(-tc["offset_deg_est"]["right"], 2), d_tilt_deg=round(tc["offset_deg_est"]["down"], 2),
                        applied=False, why="FINE_AIM_ENABLED=False: pixel->angle model unverified on hardware")
            rec["fine_aim"] = prop
            self._leave(rec, "PROPOSED_NOT_APPLIED", proposal=prop)
        else:
            self._leave(rec, "NOT_NEEDED")

        self._enter(rec, "FOCUS_CHECK")
        sharp_now = tc["sharpness"]
        need_af = sharp_now < SHARP_MIN
        self._leave(rec, "AF_NEEDED" if need_af else "ALREADY_SHARP", sharpness=sharp_now)

        self._enter(rec, "AUTOFOCUS_IF_NEEDED")
        final_img, final_cap = img, cap
        rec["focus_mode"] = self.focus_mode
        if need_af and self.focus_mode == "MANUAL_FIXED_NO_MOTOR":
            self._leave(rec, "NOT_SENT_MANUAL_FIXED_FOCUS", why="no focus motor command in this mode; blurry -> re-observe")
        elif need_af:
            ok, why = self.gate("focus", observation_id=rec["observation_id"])
            if not ok:
                self._leave(rec, "REFUSED_NO_VALID_APPROVAL", why=why)
                self.blocked = why
            else:
                fr = self.cl.run("focus", {}, "live2", idx, 2, self.deadline_s)
                foc = fr.get("focus") or {}
                rec["focus"] = {k: foc.get(k) for k in ("status", "reason", "strategy", "start", "best", "final", "sec", "drive_ms",
                                                        "blocked", "budget_hit", "pulses", "roi_source", "profile_id",
                                                        "profile_verified", "student_status", "estimated_command_offset_ms")}
                self._leave(rec, foc.get("status") or fr.get("status"), focus_s=foc.get("sec"))
                if self._device_error(fr):
                    self.blocked = f"DEVICE {fr.get('device_state')} {fr.get('reason_code')}"
                elif after_focus(foc.get("status"))[1] == "FOCUS_FAULT":
                    self.blocked = f"FOCUS_{foc.get('status')} — no HOME / further commands until the focus drive is checked"
                else:
                    if after_focus(foc.get("status"))[1] == "FOCUS_SESSION":
                        self.after_home_block = (f"FOCUS_{foc.get('status')} — {foc.get('reason') or ''} — no further AF / aim in "
                                                 "this experiment; operator checks the focus drive")
                    im2, cap2 = self._fetch(fr)
                    if im2 is not None:
                        final_img, final_cap = im2, cap2
        else:
            self._leave(rec, "SKIPPED_ALREADY_SHARP")

        self._enter(rec, "HIGH_RES_CAPTURE")
        import e6_focus_v2 as F2
        roi, roi_src = F2.sign_roi(final_img)
        sharp = round(F2.roi_metric(final_img, roi)[0], 1)
        sharp_ok = sharp >= SHARP_MIN
        self.save_dir.mkdir(parents=True, exist_ok=True)
        f_img = self.save_dir / f"{rec['observation_id']}.jpg"
        cv2.imwrite(str(f_img), final_img)
        rec.update(capture_file=str(f_img), capture_id=(final_cap or {}).get("capture_id"),
                   resolution=[final_img.shape[1], final_img.shape[0]],
                   image_quality=dict(roi_laplacian_var=sharp, roi_source=roi_src, min_sharp=SHARP_MIN, ok=sharp_ok))
        self._leave(rec, "SHARP" if sharp_ok else "BLURRY", sharpness=sharp, file_bytes=f_img.stat().st_size)

        self._home(rec, idx)
        if self.after_home_block and not self.blocked:
            self.blocked = self.after_home_block
        if self.blocked and not sharp_ok:
            rec["ocr_skipped"] = "blurry"
            return self._end(rec, "FOCUS_FAULT_BLOCKED" if "FOCUS_" in self.blocked else "BLOCKED", self.blocked)

        if self.defer_reading:
            import hashlib
            rec.update(image_sha256=hashlib.sha256(f_img.read_bytes()).hexdigest(), capture_wall=self.clock.time())
            if sharp_ok:
                for s_ in ("OCR", "VLM_IF_NEEDED", "SEMANTIC_MAP_MATCH", "HYPOTHESIS_UPDATE"):
                    self._enter(rec, s_)
                    self._leave(rec, "DEFERRED_TO_E8_PAIR")
                rec["reading"] = "DEFERRED_TO_E8_PAIR"
                return self._end(rec, "CAPTURE_STORED", "image + sha256 stored; OCR/VLM after the next observation")
        self._enter(rec, "OCR")
        if not sharp_ok:
            self._leave(rec, "SKIPPED_BLURRY", sharpness=sharp)
            return self._end(rec, "CAPTURED_BLURRY_REOBSERVE", f"sign ROI sharpness {sharp} < {SHARP_MIN}")
        lines, meta = (self.ocr or __import__("e6_ocr").sign_read)(final_img)
        rec.update(ocr_lines=lines, ocr_meta={k: meta.get(k) for k in ("ocr_roi", "ocr_full", "engine", "ocr_s", "sign_roi_px")})
        self._leave(rec, "LINES" if lines else "EMPTY", n=len(lines), ocr_s=meta.get("ocr_s"))

        self._enter(rec, "VLM_IF_NEEDED")
        need, reason = want_vlm(lines)
        if not need:
            self._leave(rec, "NOT_NEEDED", reason=reason)
        elif self.vlm is None:
            self._leave(rec, "UNAVAILABLE", reason=reason)
        else:
            adm, why = (self.vlm_admit or (lambda: (True, "no admission rule")))()
            if not adm:
                self._leave(rec, "NOT_ADMITTED", reason=reason, why=why)
            else:
                try:
                    v = self.vlm(final_img)
                    rec["vlm"] = (v or {}).get("parsed")
                    rec["vlm_meta"] = dict(sec=(v or {}).get("sec"), raw=str((v or {}).get("raw"))[:300])
                    self._leave(rec, "ASKED", reason=reason, sec=(v or {}).get("sec"))
                except Exception as e:  # noqa: BLE001  VLM failure is not a device failure
                    self._leave(rec, "ERROR", reason=reason, error=f"{type(e).__name__}: {e}")
        for s_ in ("SEMANTIC_MAP_MATCH", "HYPOTHESIS_UPDATE"):
            self._enter(rec, s_)
            self._leave(rec, "DEFERRED_TO_E34_LINK")
        return self._end(rec, "CAPTURED")


def want_vlm(lines):
    """Student OCR-first rule (r1_jobs.ReaderCore.want_vlm) on classify_text of the OCR lines."""
    import e6_semantic  # noqa: F401  (puts the student sci_text_reading on sys.path)
    import semantic as S
    from r1_jobs import ReaderCore
    r = S.classify_text([tuple(x) for x in lines]) if lines else dict(rooms=[], sign_type="other")
    return ReaderCore.want_vlm(dict(result=r), None)


def siglip_idle(state_dir, now=None, fresh_s=10.0):
    """(ok, why): the GPU is free for the VLM only while SigLIP is paused by the operator or E4 is not requesting inference
    (siglip_request.json mode != ACTIVE, or stale). VLM and SigLIP never share the GPU (student rule, now enforced in code)."""
    import time as _t
    sd = Path(state_dir)
    now = _t.time() if now is None else now
    try:
        if json.loads((sd / "control" / "pause.json").read_text()).get("paused"):
            return True, "SigLIP paused by operator"
    except (OSError, ValueError):
        pass
    try:
        r = json.loads((sd / "siglip_request.json").read_text())
    except (OSError, ValueError):
        return True, "no SigLIP request file (SigLIP not running)"
    if r.get("mode") == "ACTIVE" and now - float(r.get("stamp") or r.get("wall") or 0) <= fresh_s:
        return False, "SigLIP ACTIVE (E4 request) — VLM waits; OCR result stands"
    return True, f"SigLIP request {r.get('mode')} / stale"


def vlm_endpoint(url="http://127.0.0.1:8097", min_mem_mb=6200, state_dir=None):
    """(vlm_fn, admit_fn). R36: the server is started on demand by vlm_ondemand (it was never started before) and shares the GPU with SigLIP (measured, see vlm_ondemand.py)."""
    import vlm_ondemand as VO
    v = VO.OnDemandVLM(state_dir)
    return v.ask, v.admit
