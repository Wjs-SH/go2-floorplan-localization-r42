"""e6_af_student.py — AUTOFOCUS_V2 default path: the STUDENT KCI stepwise autofocus, unchanged, behind LIVE2 safety adapters.

Why (FLoViS-Loc 2026-10-22 directive): the student's stepwise search (autofocus.stepwise via aim_console Focuser.autofocus,
strategy kci_stepwise_20260919) is the one with real successes on this lens; LIVE2 must reuse it, not replace it. The
LIVE2 Focus V2 search (e6_focus_v2) stays available as an alternative, never as a silent substitute.

How (no student file is edited): install(AC) substitutes, IN MEMORY, the classes the student code constructs at call time
  AC.AfCam            -> RoiAfCam        student metric (1280x720 resize, Laplacian variance of the gray ROI, median of 2,
                                         160x120 motion image) on the SIGN ROI fixed at the start of the search (fallback =
                                         student centre ROI, labelled FALLBACK_ROI); fresh frames only (stale -> AF_FRAME_STALE,
                                         the student Stats.fresh silently returned the old frame); luma / saturation logged;
                                         low exposure -> AF_LOW_EXPOSURE; every measurement recorded.
  AC.AfFocus          -> GuardedAfFocus  same F<pwm>,<ms> pulse + settle; before each pulse: persistent fault lock, pulse length
                                         inside the profile range, pulse / total-drive caps, time budget (-> student FocusBudget,
                                         which the student code already handles = stop where it is); serial errors ->
                                         FocusCommError (device error, no further command); pulses with no image change are
                                         summed per direction -> both directions >= profile limit = SUSPECTED_DRIVE_FAULT
                                         (mechanical OR Nano / wiring / power — not concluded), latched in runtime/e6_focus_fault.json.
                                         2026-10-22 early stop: an image RESPONSE after a pulse = motion image change >=
                                         response_moved_min OR sharpness change >= response_rel_sharp (from a frame >= blurry_floor).
                                         >= no_response_stop_ms of drive without a response, counted only from frames >= blurry_floor
                                         (below it the student itself notes one step barely changes the image) -> SUSPECTED_DRIVE_FAULT
                                         (latched). The drive cap (unverified profile: unverified_max_total_drive_ms) reached with NO
                                         response at all in the call -> AF_NO_RESPONSE (session block, not latched: may be a very
                                         blurry start at an end stop).
  AC.R1ConsoleCamera  -> RoiR1Camera     the student's skip-if-sharp and fresh final check measure the same SIGN ROI.
  R1Hardware._focus_kci_stepwise is wrapped: pre-checks (lock, frame, exposure, budget feasibility) -> the student method
  itself -> post (target still there, status mapping, record). Outside an AF session every substitute behaves like the
  student original. Lens position is never measured: estimated_command_offset = signed sum of commanded pulse ms.
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROFILE_FILE = HERE / "e6_af_profile_kci.json"
STRATEGY = "student_kci_stepwise_20260919+live2_roi_guard"


class St:
    CONVERGED = "AF_CONVERGED"
    NOT_CONVERGED = "AF_NOT_CONVERGED"
    BUDGET = "AF_BUDGET_EXHAUSTED"
    INFEASIBLE = "AF_BUDGET_INFEASIBLE"
    STALE = "AF_FRAME_STALE"
    LOST = "AF_TARGET_LOST"
    DARK = "AF_LOW_EXPOSURE"
    LIMIT = "AF_DRIVE_LIMIT"
    NORESP = "AF_NO_RESPONSE"
    PROFILE = "AF_PULSE_OUT_OF_PROFILE"
    DRIVE = "SUSPECTED_DRIVE_FAULT"
    BLOCKED = "AF_BLOCKED_AFTER_FAULT"
    COMM = "FOCUS_COMM_ERROR"


class AfGuard(Exception):
    def __init__(self, status, reason):
        super().__init__(f"{status}: {reason}")
        self.status, self.reason = status, reason


def load_profile(path=PROFILE_FILE):
    return json.loads(Path(path).read_text())


CTX = dict(active=False)


def _reset(profile, roi, roi_source, budget_s, clock):
    CTX.clear()
    CTX.update(active=True, profile=profile, roi=tuple(roi), roi_source=roi_source, budget_s=budget_s, t0=clock(), clock=clock,
               pulses=0, drive_ms=0, offset_ms=0, last_dir=None, last_ms=None, pending=None, nochange={1: 0, -1: 0},
               measurements=[], guard=None, last_sharp=None, ever_resp=False, noresp_ms=0)


def _crop(frame, roi):
    f = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = [int(v) for v in roi]
    return f[y0:y1, x0:x1]


def _measure_roi(frame, roi):
    """Student AfCam._one on a given 1280x720-frame ROI (+ luma, saturation fraction)."""
    c = _crop(frame, roi)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return (float(cv2.Laplacian(gray, cv2.CV_64F).var()), cv2.resize(gray, (160, 120)).astype(np.float32),
            float(gray.mean()), float(np.mean(gray >= 250)))


def install(AC, F2=None, profile=None, clock=None, log=None):
    """Patch the student console module in memory (files unchanged). F2 = e6_focus_v2 (sign_roi, FAULT_FILE, LATCH).
    clock defaults to a LATE-bound time.time(): the student deadline (AfCam.deadline = time.time() + budget) and the guards
    must read the same clock."""
    clock = clock or (lambda: time.time())
    if F2 is None:
        import e6_focus_v2 as F2
    profile = profile or load_profile()
    log = log or AC.log
    orig_cam, orig_focus, orig_r1cam = (getattr(c, "_live2_orig", c) for c in (AC.AfCam, AC.AfFocus, AC.R1ConsoleCamera))  # re-install
                                                                     # replaces the adapters, never stacks them
    orig_method = getattr(AC.R1Hardware._focus_kci_stepwise, "__wrapped_student__", AC.R1Hardware._focus_kci_stepwise)

    class RoiAfCam(orig_cam):
        def _one(self):
            if not CTX.get("active"):
                return orig_cam._one(self)
            raise RuntimeError("RoiAfCam._one is not used inside a LIVE2 AF session")

        def measure(self):
            if not CTX.get("active"):
                return orig_cam.measure(self)
            if self.deadline is not None and clock() > self.deadline:
                raise AC.FocusBudget()
            P = CTX["profile"]
            vals, img = [], None
            seq, _ = self.stats.get()
            for _ in range(self.samples):
                fr = self.stats.fresh_frame(seq, float(P.get("frame_timeout_s", 1.0)))
                if fr is None:
                    raise AfGuard(St.STALE, f"no fresh IMX477 frame after seq {seq} within {P.get('frame_timeout_s', 1.0)} s")
                seq, frame = fr[0], fr[1]
                s, img, luma, sat = _measure_roi(frame, CTX["roi"])
                vals.append(s)
                CTX["measurements"].append(dict(frame_seq=int(seq), t_mono=round(time.monotonic(), 3), res=[frame.shape[1], frame.shape[0]],
                                                roi=list(CTX["roi"]), sharp=round(s, 2), luma=round(luma, 1), sat=round(sat, 4),
                                                pulse=CTX.get("pending")))
                if luma < float(P.get("low_luma", 25.0)):
                    raise AfGuard(St.DARK, f"ROI luma {luma:.1f} < {P.get('low_luma', 25.0)}")
                time.sleep(0.06)
            m = float(np.median(vals))
            self.best_seen = max(self.best_seen, m)
            pend = CTX.get("pending")
            if pend is not None and CTX.get("last_img") is not None:          # image change caused by the last pulse
                moved = float(np.mean(np.abs(img - CTX["last_img"])))
                pend["moved"] = round(moved, 2)
                d = pend["dir"]
                if moved < float(P.get("change_thresh", 1.5)):
                    CTX["nochange"][d] += pend["ms"]
                else:
                    CTX["nochange"][d] = 0
                lim = float(P["drive_fault_nochange_ms_each_dir"])
                if CTX["nochange"][1] >= lim and CTX["nochange"][-1] >= lim:
                    raise AfGuard(St.DRIVE, f"no image change after {CTX['nochange'][1]:.0f} ms (+) and {CTX['nochange'][-1]:.0f} ms (-) "
                                            "of commanded drive: mechanical jam/slip OR Nano/L9110S/wiring/power — not concluded")
                prev, floor = CTX.get("last_sharp"), float(P.get("blurry_floor", 8.0))
                resp = moved >= float(P.get("response_moved_min", 2.0)) or \
                    (prev is not None and prev >= floor and abs(m - prev) >= float(P.get("response_rel_sharp", 0.15)) * prev)
                pend["response"] = bool(resp)
                if resp:
                    CTX["ever_resp"], CTX["noresp_ms"] = True, 0
                elif prev is not None and prev >= floor:
                    CTX["noresp_ms"] += pend["ms"]
                if CTX["noresp_ms"] >= float(P.get("no_response_stop_ms", 3000)):
                    raise AfGuard(St.DRIVE, f"no image response for {CTX['noresp_ms']:.0f} ms of commanded drive from a frame "
                                            f"sharper than {floor:g} (moved < {P.get('response_moved_min', 2.0)}, sharpness change "
                                            f"< {100 * float(P.get('response_rel_sharp', 0.15)):.0f} %): mechanical OR Nano/L9110S/"
                                            "wiring/power — not concluded")
            CTX["pending"], CTX["last_img"], CTX["last_sharp"] = None, img, m
            if self.accept is not None and m >= self.accept:
                raise AC.FocusAccept(m)
            return m, img

    class GuardedAfFocus(orig_focus):
        def move(self, direction, ms):
            if not CTX.get("active"):
                return orig_focus.move(self, direction, ms)
            P = CTX["profile"]
            if F2.LATCH.get("fault") or F2.FAULT_FILE.exists():
                raise AfGuard(St.BLOCKED, f"persistent focus fault lock: {F2.LATCH.get('fault') or F2.FAULT_FILE}")
            lo, hi = P["pulse_ms_range"]
            if not lo <= ms <= hi:
                raise AfGuard(St.PROFILE, f"pulse {ms} ms outside profile {lo}-{hi} ms")
            if CTX["pulses"] + 1 > int(P["max_pulses"]) or CTX["drive_ms"] + ms > int(P["max_total_drive_ms"]):
                if not CTX.get("ever_resp"):
                    raise AfGuard(St.NORESP, f"drive cap {P['max_total_drive_ms']} ms reached ({CTX['drive_ms']} ms, {CTX['pulses']} "
                                             "pulses) without ANY image response: focus drive not confirmed — operator checks it")
                raise AfGuard(St.LIMIT, f"pulse/drive cap: {CTX['pulses']} pulses, {CTX['drive_ms']} ms (+{ms}) of "
                                        f"{P['max_total_drive_ms']} ms")
            if clock() + ms / 1000.0 + float(P["settle_s"]) > CTX["t0"] + CTX["budget_s"]:
                raise AC.FocusBudget()                                      # student path: stop where it is
            if int(self.speed) > int(P["pwm_max"]):
                raise AfGuard(St.PROFILE, f"pwm {self.speed} > profile max {P['pwm_max']}")
            CTX["pending"] = dict(dir=int(math.copysign(1, direction)), ms=int(ms), pwm=int(self.speed), n=CTX["pulses"] + 1,
                                  t_mono=round(time.monotonic(), 3))
            ser = getattr(self.nano, "ser", None)                          # 2026-10-22: the student Nano.cmd keeps only
            lines, orig_rl = [], getattr(ser, "readline", None)             # pan/tilt of the reply; tap the raw lines (incl.
            if orig_rl is not None:                                         # the firmware 'focus=' field) as evidence that
                def _tap(*a_, **k_):                                        # the Nano received / ran the F command
                    b_ = orig_rl(*a_, **k_)
                    lines.append(b_.decode(errors="replace").strip() if isinstance(b_, bytes) else str(b_))
                    return b_
                ser.readline = _tap
            try:
                orig_focus.move(self, direction, ms)
            except Exception as e:                                          # noqa: BLE001
                raise F2.FocusCommError(f"focus pulse failed: {type(e).__name__}: {e}")
            finally:
                if orig_rl is not None:
                    try:
                        del ser.readline
                    except AttributeError:
                        ser.readline = orig_rl
                CTX["pending"]["nano_reply"] = [x for x in lines if x][:3]
            CTX["pulses"] += 1
            CTX["drive_ms"] += int(ms)
            CTX["offset_ms"] += int(ms) * CTX["pending"]["dir"]

    class RoiR1Camera(orig_r1cam):
        def measure(self, after_seq, timeout_s):
            if not CTX.get("active"):
                return orig_r1cam.measure(self, after_seq, timeout_s)
            fr = self.stats.fresh_frame(after_seq, timeout_s or self.timeout)
            if fr is None:
                return None
            s, _, luma, sat = _measure_roi(fr[1], CTX["roi"])
            return dict(seq=fr[0], sharp=s, luma=luma, sat=sat, target_present=None)

    def focus(hw, cancel):
        """Replacement for R1Hardware._focus_kci_stepwise: LIVE2 pre/post checks around the STUDENT method."""
        cancel.check()
        P = profile
        base = dict(strategy=STRATEGY, profile_id=P.get("profile_id"), profile_verified=bool(P.get("verified")),
                    position_semantics="estimated_command_offset (sum of signed commanded pulse ms; no lens position sensor)")
        if F2.LATCH.get("fault") or F2.FAULT_FILE.exists():
            return dict(base, status=St.BLOCKED, ok=False, reason=f"persistent focus fault lock ({F2.FAULT_FILE.name}): no motion",
                        pulses=0, drive_ms=0, sec=0.0)
        budget = float(min(float(hw.budget_s or P["budget_s"]), float(P["budget_s"])))
        if budget < float(P["min_budget_s"]):
            return dict(base, status=St.INFEASIBLE, ok=False, reason=f"budget {budget:.0f} s < profile minimum {P['min_budget_s']} s",
                        pulses=0, drive_ms=0, sec=0.0)
        lim = approval_limits()                                             # approval B may only TIGHTEN the profile
        P = dict(P)
        if not P.get("verified") and P.get("unverified_max_total_drive_ms") is not None:   # first runs on unverified hardware
            P["max_total_drive_ms"] = min(int(P["max_total_drive_ms"]), int(P["unverified_max_total_drive_ms"]))
        if lim.get("drive_ms_max") is not None:
            P["max_total_drive_ms"] = min(int(P["max_total_drive_ms"]), int(lim["drive_ms_max"]))
        if lim.get("pwm_max") is not None:
            P["pwm_max"] = min(int(P["pwm_max"]), int(lim["pwm_max"]))
        if lim.get("af_s_max") is not None and float(lim["af_s_max"]) < budget:
            budget = float(lim["af_s_max"])
            if budget < float(P["min_budget_s"]):
                return dict(base, status=St.INFEASIBLE, ok=False, pulses=0, drive_ms=0, sec=0.0,
                            reason=f"approval B af_s_max {budget:.0f} s < profile minimum {P['min_budget_s']} s")
        st = hw.a.stats
        first = st.fresh_frame(st.get()[0], float(P.get("frame_timeout_s", 1.0)))
        if first is None:
            return dict(base, status=St.STALE, ok=False, reason="no fresh IMX477 frame before the search", pulses=0, drive_ms=0, sec=0.0)
        roi, src = F2.sign_roi(first[1])
        src = "SIGN_ROI" if src == "SIGN_COMPONENT" else "FALLBACK_ROI"
        _reset(P, roi, src, budget, clock)
        hw.budget_s = budget
        t0 = clock()
        status, out = None, None
        try:
            out = orig_method(hw, cancel)                                   # the student KCI stepwise, unchanged
        except AfGuard as g:
            status, CTX["guard"] = g.status, g.reason
        finally:
            try:
                hw.a.nano.cmd("F0")
            except Exception:                                               # noqa: BLE001
                pass
            CTX["active"] = False
        rec = dict(base, roi=list(CTX["roi"]), roi_source=CTX["roi_source"], pulses=CTX["pulses"], drive_ms=CTX["drive_ms"],
                   effective_limits=dict(max_total_drive_ms=P["max_total_drive_ms"], pwm_max=P["pwm_max"], budget_s=budget,
                                         no_response_stop_ms=P.get("no_response_stop_ms"), pulse_ms_range=P["pulse_ms_range"]),
                   ever_response=CTX.get("ever_resp"),
                   estimated_command_offset_ms=CTX["offset_ms"], measurements=CTX["measurements"][-80:],
                   budget_s=budget, sec=round(clock() - t0, 2))
        if status is not None:                                              # a LIVE2 guard stopped the student search
            rec.update(status=status, ok=False, reason=CTX["guard"])
            if status == St.DRIVE:
                F2.LATCH["fault"] = f"{St.DRIVE} @ {time.strftime('%H:%M:%S')}"
                try:
                    F2.FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
                    F2.FAULT_FILE.write_text(json.dumps(dict(fault=F2.LATCH["fault"], status=St.DRIVE, reason=CTX["guard"], wall=time.time(),
                                                             note="clear only after a physical check, through approval B")))
                except OSError:
                    pass
            log(f"  초점(LIVE2 가드): {status} — {CTX['guard']}")
            return rec
        rec.update({k: v for k, v in out.items() if k not in rec})
        rec["student_status"] = out.get("status")
        if CTX["roi_source"] == "SIGN_ROI" and out.get("status") == "AF_CONVERGED":   # same target still in view?
            fr = st.fresh_frame(st.get()[0], float(P.get("frame_timeout_s", 1.0)))
            if fr is not None:
                r2, s2 = F2.sign_roi(fr[1])
                iou = _iou(roi, r2) if s2 == "SIGN_COMPONENT" else 0.0
                rec["final_target_iou"] = round(iou, 2)
                if iou < float(P.get("target_iou_min", 0.3)):
                    rec.update(status=St.LOST, ok=False, reason=f"sign ROI not found again at the end (IoU {iou:.2f})")
        if rec.get("status") == "AF_CONVERGED" and CTX["roi_source"] == "FALLBACK_ROI":
            rec["note"] = "converged on the FALLBACK (centre) ROI only — not evidence that the room sign is sharp"
        return rec

    focus.__wrapped_student__ = orig_method
    RoiAfCam._live2_orig, GuardedAfFocus._live2_orig, RoiR1Camera._live2_orig = orig_cam, orig_focus, orig_r1cam
    AC.AfCam, AC.AfFocus, AC.R1ConsoleCamera = RoiAfCam, GuardedAfFocus, RoiR1Camera
    AC.R1Hardware._focus_kci_stepwise = focus
    log(f"LIVE2: focus = {STRATEGY} (student stepwise unchanged; profile {profile.get('profile_id')} verified={profile.get('verified')})")
    return focus


def approval_limits():
    """Hardware approval B limits valid NOW (empty when unreadable — the bridge already refused focus without approval)."""
    try:
        import sys
        sys.path.insert(0, str(HERE.parent / "live1"))
        import live1_common as C
        ap = C.approval_status("B", "pantilt")
        return dict(ap.get("limits") or {}) if ap.get("valid") else {}
    except Exception:                                                       # noqa: BLE001
        return {}


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / u if u > 0 else 0.0
