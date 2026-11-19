"""e6_focus_v2.py — LIVE2 candidate autofocus for the student pan-tilt IMX477 DC focus motor.

Hardware facts (student code / records): the focus ring is driven by a DC motor with timed pulses (Nano `F<pwm>,<ms>`,
firmware cap 3 s), open loop, no lens position sensor, no end switch. PWM 140-180 grips, 255 slips, 160 barely turns
(2026-09-19); full travel 11-12 s; backlash 60-320 ms (2026-09-16).

Why a new search (AIM #1, 2026-10-20 06:56, reports/E6_AIM1_AF_ANALYSIS_20261020.md): the student KCI stepwise
  - stepped over the peak with 300 ms steps (11.4 -> 15.5 -> 6.1; the peak is narrower than one step),
  - after reversing, drove ~1.16 s without any image change and kept alternating directions ("end reached -> flip"),
  - spent ~25 s at PWM 160 (barely moves) until the 45 s budget ran out, and
  - judged sharpness on the image centre while the sign sat up-left of it.

This module replaces ONLY that search (R1Hardware._focus_kci_stepwise) through the LIVE2 launcher
(e6_aim_console_live2.py); the student files are imported unchanged. Same sharpness formula as the student AfCam
(1280x720 resize, Laplacian variance of the gray ROI, median of 2 fresh frames, 160x120 ROI image for motion), but on
the sign ROI found in the IMX477 frame (fallback: the student centre ROI 0.5).

State machine (every pulse: cancel check, time budget, travel limits; motor stopped on every exit):
  INIT     link check ('?' must answer), sign ROI, 2 measurements -> noise floor; already sharp -> CONVERGED, no motion
  CLIMB    pulses of STEP_MS in the current direction; rise -> remember best; 2 drops below DROP_RATIO*best after a rise
           = peak passed -> RETURN. No rise after NO_IMPROVE_STEPS changed pulses -> try the other direction once,
           then AF_NO_IMPROVEMENT.
  RETURN   reverse: take up backlash with short probes until the image changes (cap LASH_CAP_MS per reversal; cap hit
           -> that direction is blocked). Then FINE_MS pulses back toward the best position; stop at
           >= max(CONVERGED, 0.85*best) or ACCEPT_STOP; passed again -> reverse with half steps (MAX_REVERSALS).
  STALL    STALL_PULSES consecutive (non-backlash) pulses without image change -> that direction is blocked
           (end stop / jam suspected); never pushed further. Both directions blocked, or the return direction
           blocked after the peak was passed -> SUSPECTED_MECHANICAL_FAULT (latched: no focus motion until the
           operator clears it by restarting the console).
  END      F0, fresh final measurement on the ROI: AF_CONVERGED only if >= CONVERGED (30, student threshold for
           readable text). Otherwise the status says why; the caller records the capture as blurry (no OCR success).
Comm failure (serial exception / '?' without answer) -> FocusCommError (the student DeviceOwner turns it into
hardware_error -> device UNKNOWN -> no further commands, no HOME). Cancel -> the owner's Cancelled propagates.
Positions are commanded pulse sums (ms), not lens positions.
"""
import statistics
import json
import time
from pathlib import Path

import numpy as np

STRATEGY = "live2_focus_v2_20261020"
PWM = 180                 # verified grip band 140-180; never 160 for fine steps (barely turns)
STEP_MS = 150             # <= 150 ms (user requirement); student 300 ms stepped over the peak
FINE_MS = 50
MIN_FINE_MS = 30
LASH_PROBE_MS = 50
LASH_CAP_MS = 400         # per reversal (recorded backlash 60-320 ms)
STALL_PULSES = 3
BUDGET_S = 20.0
NET_TRAVEL_MS = 3000      # |sum of commanded ms| from the start position (full travel 11-12 s)
TOTAL_DRIVE_MS = 6000     # sum of all pulse lengths in one request
SETTLE_S = 0.35           # student AfFocus settle after a pulse
CONVERGED = 30.0          # student min_sharp_converged (text readable, 2026-09-16)
ACCEPT_STOP = 40.0        # student KCI accept_sharp: stop searching here (the peak is narrower than a step)
DROP_RATIO = 0.75
NO_IMPROVE_STEPS = 8      # x STEP_MS = 1.2 s of travel past the start in each direction
RISE_ABS = 1.5            # a rise must clear the start sharpness by max(RISE_ABS, RISE_REL*start)
RISE_REL = 0.25
MAX_REVERSALS = 3
MOVE_MIN = 1.5            # student run_pass "no image change" threshold (160x120 mean abs diff)
METRIC = dict(type="laplacian_var_gray", resolution="1280x720 resize", samples="median of 2 fresh frames",
              motion="160x120 mean abs diff", version="r1-af-1 (student AfCam formula)")


class St:
    CONVERGED = "AF_CONVERGED"
    NOT_CONVERGED = "AF_NOT_CONVERGED"
    NO_IMPROVEMENT = "AF_NO_IMPROVEMENT"
    BUDGET = "AF_BUDGET_EXHAUSTED"
    TRAVEL = "AF_TRAVEL_LIMIT"
    MECH = "SUSPECTED_MECHANICAL_FAULT"
    BLOCKED = "AF_BLOCKED_AFTER_FAULT"
    FRAME = "AF_FRAME_UNAVAILABLE"


class FocusCommError(RuntimeError):
    pass


class FrameUnavailable(RuntimeError):
    pass


class _Stop(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


LATCH = dict(fault=None)  # process-wide: SUSPECTED_MECHANICAL_FAULT blocks every later focus request
# 2026-10-22: the latch lived only in memory, so a console stop / reboot silently re-enabled focus motion. It is now also a
# file; a new console (any boot) starts latched until approval B states the mechanical check passed and clears it.
FAULT_FILE = Path(__file__).resolve().parents[2] / "runtime" / "e6_focus_fault.json"
try:
    LATCH["fault"] = json.loads(FAULT_FILE.read_text()).get("fault")
except (OSError, ValueError):
    pass


# ------------------------------------------------------------------------------------------------ ROI + metric
def sign_roi(frame_bgr):
    """Sign / text plate ROI in 1280x720 coordinates -> ((x0, y0, x1, y1), source). Bright, rectangular, textured
    component closest to the image centre; fallback = student centre ROI 0.5."""
    import cv2
    f = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    centre = (320, 180, 960, 540)
    thr = max(130.0, float(np.percentile(g, 70)))
    m = (g > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    best, best_score = None, 0.0
    for x, y, w, h, a in st[1:]:
        frac, rect, asp = a / (1280 * 720), a / max(w * h, 1), w / max(h, 1)
        if not (0.02 <= frac <= 0.6 and rect >= 0.6 and 0.4 <= asp <= 4.0):
            continue
        texture = float(g[y:y + h, x:x + w].std())
        dist = np.hypot(x + w / 2 - 640, y + h / 2 - 360) / np.hypot(640, 360)
        score = a * rect * (1.0 + texture) / (1.0 + 2.0 * dist)
        if score > best_score:
            best, best_score = (int(x), int(y), int(x + w), int(y + h)), score
    if best is None:
        return centre, "CENTRE_FALLBACK"
    x0, y0, x1, y1 = best
    mx, my = int(0.08 * (x1 - x0)), int(0.08 * (y1 - y0))
    x0, y0, x1, y1 = x0 + mx, y0 + my, x1 - mx, y1 - my
    if x1 - x0 < 64 or y1 - y0 < 48:
        return centre, "CENTRE_FALLBACK_SMALL"
    return (x0, y0, x1, y1), "SIGN_COMPONENT"


def roi_metric(frame_bgr, roi):
    """Student AfCam formula on an arbitrary ROI -> (sharpness, 160x120 float32 image)."""
    import cv2
    f = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = roi
    gray = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), cv2.resize(gray, (160, 120)).astype(np.float32)


class FrameMeter:
    """measure() over a frame source with fresh_frame(after_seq, timeout) -> (seq, frame, t) | None and frame_seq()."""

    def __init__(self, fresh_frame, frame_seq, roi, clock, samples=2):
        self.fresh_frame, self.frame_seq, self.roi, self.clock, self.samples = fresh_frame, frame_seq, roi, clock, samples

    def measure(self):
        seq, vals, img = self.frame_seq(), [], None
        for _ in range(self.samples):
            r = self.fresh_frame(seq, 1.0)
            if r is None:
                raise FrameUnavailable("no fresh IMX477 frame within 1 s")
            seq = r[0]
            s, img = roi_metric(r[1], self.roi)
            vals.append(s)
            self.clock.sleep(0.06)
        return statistics.median(vals), img


class NanoMotor:
    """Focus pulses on the console's Nano (same command as the student AfFocus/R1NanoFocus)."""

    def __init__(self, nano, clock, pwm=PWM):
        self.nano, self.clock, self.pwm = nano, clock, min(int(pwm), PWM)

    def check_link(self):
        try:
            r = self.nano.cmd("?")
        except Exception as e:  # noqa: BLE001
            raise FocusCommError(f"status query failed: {type(e).__name__}: {e}")
        if not r:
            raise FocusCommError("status query: no answer from the Nano")
        return r

    def pulse(self, direction, ms):
        try:
            self.nano.cmd(f"F{self.pwm * direction},{int(ms)}")
        except Exception as e:  # noqa: BLE001
            raise FocusCommError(f"focus pulse failed: {type(e).__name__}: {e}")
        self.clock.sleep(ms / 1000.0)

    def stop(self):
        self.nano.cmd("F0")


class RealClock:
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


# ------------------------------------------------------------------------------------------------ search
class FocusV2:
    def __init__(self, meter, motor, cancel=None, clock=RealClock, log=print, first_dir=1, budget_s=BUDGET_S,
                 roi=None, roi_source=None):
        self.meter, self.motor, self.cancel, self.clock, self.log = meter, motor, cancel, clock, log
        self.first_dir, self.budget_s, self.roi, self.roi_source = first_dir, min(float(budget_s), BUDGET_S), roi, roi_source
        self.pos = 0                       # commanded ms sum (not a lens position)
        self.drive = 0
        self.lash_ms = 0
        self.stall = {1: 0, -1: 0}
        self.blocked = {}
        self.reversals = 0
        self.trace = []

    # ---- one pulse + measurement
    def _check_limits(self, ms, d, lash):
        if self.cancel is not None:
            self.cancel.check()
        if self.clock.now() - self.t0 + ms / 1000.0 + SETTLE_S > self.budget_s:
            raise _Stop(St.BUDGET, f"time budget {self.budget_s:g} s")
        if abs(self.pos + d * ms) > NET_TRAVEL_MS:
            raise _Stop(St.TRAVEL, f"net commanded travel would exceed {NET_TRAVEL_MS} ms")
        if self.drive + ms > TOTAL_DRIVE_MS:
            raise _Stop(St.TRAVEL, f"total drive would exceed {TOTAL_DRIVE_MS} ms")

    def _pulse(self, d, ms, kind):
        self._check_limits(ms, d, kind == "lash")
        self.motor.pulse(d, ms)
        self.pos += d * ms
        self.drive += ms
        if kind == "lash":
            self.lash_ms += ms
        self.clock.sleep(SETTLE_S)
        s, img = self.meter.measure()
        # change is judged against the last frame where motion was confirmed (cumulative): a 50 ms pulse moves the
        # ring too little for a frame-to-frame test (AIM #1: 50-120 ms steps gave ~1.1 vs threshold 1.5)
        moved = float(np.mean(np.abs(img - self.ref_img)))
        changed = moved >= self.move_thr or abs(s - self.ref_s) > max(1.0, 0.2 * self.ref_s)
        if changed:
            self.ref_s, self.ref_img = s, img
        self.prev_s, self.prev_img = s, img
        self.best_seen = max(self.best_seen, s)
        self.trace.append(dict(t=round(self.clock.now() - self.t0, 2), kind=kind, dir=d, ms=ms, pos=self.pos,
                               sharp=round(s, 1), moved=round(moved, 2), changed=changed))
        if kind != "lash":
            if changed:
                self.stall[d] = 0
            else:
                self.stall[d] += 1
                if self.stall[d] >= STALL_PULSES:
                    self.blocked[d] = f"{STALL_PULSES} pulses without image change (end stop / jam suspected)"
        self.log(f"  [AFv2] {kind:5s} {'+' if d > 0 else '-'}{ms:3d}ms  sharp {s:6.1f}  moved {moved:5.2f}"
                 f"{'' if changed else '  (no change)'}")
        if s >= ACCEPT_STOP:
            raise _Stop(St.CONVERGED, f"sharp {s:.1f} >= accept {ACCEPT_STOP:g}")
        return s, changed

    def _takeup(self, d):
        """After a reversal: short probes until the image changes. Cap -> direction blocked (never pushed on)."""
        self.ref_s, self.ref_img = self.prev_s, self.prev_img
        used = 0
        while used < LASH_CAP_MS:
            ms = min(LASH_PROBE_MS, LASH_CAP_MS - used)
            _, changed = self._pulse(d, ms, "lash")
            used += ms
            if changed:
                self.stall[d] = 0
                return True
        self.blocked[d] = f"no image change after {LASH_CAP_MS} ms backlash take-up"
        return False

    def _mech_if_stuck(self, must_return=False, d=None):
        if (1 in self.blocked and -1 in self.blocked) or (must_return and d in self.blocked):
            raise _Stop(St.MECH, "; ".join(f"{'+' if k > 0 else '-'}: {v}" for k, v in sorted(self.blocked.items())))

    # ---- phases
    def _climb(self, d):
        """-> (direction of travel, passed_peak). Raises _Stop on accept / no improvement / limits.
        A rise counts only when it clears the start by max(RISE_ABS, RISE_REL*start) (noise is ~3 %); 'no improvement'
        = NO_IMPROVE_STEPS*STEP_MS of travel PAST THE START position in the current direction without such a rise."""
        tried_other, rose, drops = False, False, 0
        rise_floor = self.start + max(RISE_ABS, RISE_REL * self.start)
        while True:
            if d in self.blocked:
                self._mech_if_stuck()
                if rose:             # peak region is behind us in the other direction -> RETURN handles it
                    return d, True
                if tried_other:
                    raise _Stop(St.NO_IMPROVEMENT, "no rise; the other direction is blocked")
                d, tried_other = -d, True
                self.reversals += 1
                if not self._takeup(d):
                    self._mech_if_stuck()
                    raise _Stop(St.NO_IMPROVEMENT, "other direction blocked during backlash take-up")
                continue
            s, _ = self._pulse(d, STEP_MS, "climb")
            if s > self.best * 1.05 and s >= rise_floor:
                self.best, self.best_pos, rose, drops = s, self.pos, True, 0
            elif rose and s < self.best * DROP_RATIO:
                drops += 1
                if drops >= 2:
                    return d, True
            else:
                drops = 0
            if not rose and d * self.pos >= NO_IMPROVE_STEPS * STEP_MS:
                if tried_other:
                    raise _Stop(St.NO_IMPROVEMENT, f"no rise within {NO_IMPROVE_STEPS * STEP_MS} ms of travel each way")
                d, tried_other = -d, True
                self.reversals += 1
                if not self._takeup(d):
                    self._mech_if_stuck()
                    raise _Stop(St.NO_IMPROVEMENT, "other direction blocked during backlash take-up")

    def _return(self, d):
        """Peak passed while travelling d: go back with fine steps; stop near the best."""
        step = FINE_MS
        while True:
            d = -d
            self.reversals += 1
            if self.reversals > MAX_REVERSALS + 1:
                raise _Stop(St.NOT_CONVERGED, f"peak not re-found after {MAX_REVERSALS} reversals")
            if d in self.blocked or not self._takeup(d):
                self._mech_if_stuck(must_return=True, d=d)
            target = max(CONVERGED, 0.85 * self.best)
            pass_best, drops, budget_ms = self.prev_s, 0, abs(self.pos - self.best_pos) + 2 * STEP_MS
            used = 0
            while used < budget_ms:
                if d in self.blocked:
                    self._mech_if_stuck(must_return=True, d=d)
                s, _ = self._pulse(d, step, "fine")
                used += step
                if s > self.best:
                    self.best, self.best_pos = s, self.pos
                if s >= target:
                    raise _Stop(St.CONVERGED, f"sharp {s:.1f} >= target {target:.1f} (best {self.best:.1f})")
                if s > pass_best:
                    pass_best, drops = s, 0
                elif s < pass_best * 0.85:
                    drops += 1
                    if drops >= 2:
                        break
            step = max(MIN_FINE_MS, step // 2)

    def run(self):
        self.t0 = self.clock.now()
        res = dict(strategy=STRATEGY, profile_verified=True, budget_s=self.budget_s, pwm=getattr(self.motor, "pwm", PWM),
                   metric=dict(METRIC, roi=self.roi, roi_source=self.roi_source),
                   position_semantics="commanded pulse sum ms (open loop, no lens position sensor)")
        if LATCH["fault"]:
            return dict(res, status=St.BLOCKED, ok=False, reason=f"focus latched after: {LATCH['fault']}", start=None,
                        best=None, final=None, sec=0.0, budget_hit=False, drive_ms=0, samples=[])
        status, reason = None, None
        self.start = self.best = self.best_seen = 0.0
        try:
            self.motor.check_link()
            s1, img1 = self.meter.measure()
            s2, img2 = self.meter.measure()
            noise = float(np.mean(np.abs(img1 - img2)))
            self.move_thr = max(MOVE_MIN, 2.5 * noise)
            self.start = self.best = self.best_seen = min(s1, s2)
            self.best_pos, self.prev_s, self.prev_img = 0, s2, img2
            self.ref_s, self.ref_img = s2, img2
            self.log(f"  [AFv2] start sharp {self.start:.1f} (roi {self.roi_source}), motion noise {noise:.2f} -> thr {self.move_thr:.2f}")
            if self.start >= CONVERGED:
                raise _Stop(St.CONVERGED, f"already sharp {self.start:.1f} >= {CONVERGED:g} (no motor motion)")
            d, passed = self._climb(self.first_dir)
            if passed:
                self._return(d)
            status, reason = St.NOT_CONVERGED, "search ended without reaching the threshold"
        except _Stop as e:
            status, reason = e.status, e.reason
        except FrameUnavailable as e:
            status, reason = St.FRAME, str(e)
        finally:
            try:
                self.motor.stop()
            except Exception:  # noqa: BLE001  comm already broken: the caller sees FocusCommError / hardware_error
                pass
        if status == St.MECH:
            LATCH["fault"] = f"{reason} @ {time.strftime('%H:%M:%S')}"
            try:
                FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
                FAULT_FILE.write_text(json.dumps(dict(fault=LATCH["fault"], wall=time.time(), status="SUSPECTED_MECHANICAL_FAULT",
                                                      note="clear only after a mechanical check, through approval B")))
            except OSError:
                pass
        final = None
        if status != St.FRAME:
            try:
                final, _ = self.meter.measure()
            except FrameUnavailable:
                status, reason = St.FRAME, "no fresh frame for the final check"
        if status == St.CONVERGED and (final is None or final < CONVERGED):
            status, reason = St.NOT_CONVERGED, f"{reason}; fresh final {final} < {CONVERGED:g}"
        sec = self.clock.now() - self.t0
        self.log(f"  [AFv2] {status}: {reason} · final {final} · best {self.best_seen:.1f} · drive {self.drive} ms · {sec:.1f} s")
        return dict(res, status=status, ok=status == St.CONVERGED, reason=reason, start=self.start, best=self.best_seen,
                    final=final, sec=sec, budget_hit=status == St.BUDGET, drive_ms=self.drive, lash_ms=self.lash_ms,
                    net_cmd_ms=self.pos, reversals=self.reversals,
                    blocked={("+" if k > 0 else "-"): v for k, v in self.blocked.items()}, samples=self.trace[-60:])


# ------------------------------------------------------------------------------------------------ console hook
def console_focus(hw, cancel, AC):
    """Replacement for student R1Hardware._focus_kci_stepwise(cancel) (hw = R1Hardware, AC = aim_console module)."""
    cancel.check()
    a = hw.a
    clock = RealClock
    st = a.stats
    first = st.fresh_frame(st.get()[0], 1.0)
    if first is None:
        return dict(status=St.FRAME, ok=False, reason="no IMX477 frame", strategy=STRATEGY, start=None, best=None,
                    final=None, sec=0.0, budget_hit=False, samples=[], metric=METRIC, profile_verified=True,
                    position_semantics="commanded pulse sum ms (open loop, no lens position sensor)")
    roi, src = sign_roi(first[1])
    meter = FrameMeter(st.fresh_frame, lambda: st.get()[0], roi, clock)
    motor = NanoMotor(a.nano, clock, pwm=min(int(hw.profile.get("pwm", PWM)), PWM))
    fv = FocusV2(meter, motor, cancel=cancel, clock=clock, log=AC.log, roi=list(roi), roi_source=src,
                 budget_s=min(float(hw.budget_s or BUDGET_S), BUDGET_S))
    try:
        return fv.run()
    finally:
        try:
            a.nano.cmd("F0")
        except Exception:  # noqa: BLE001
            pass


def install(AC):
    """Patch the student console module in memory (files unchanged)."""
    AC.R1Hardware._focus_kci_stepwise = lambda self, cancel: console_focus(self, cancel, AC)
    AC.log(f"LIVE2: focus search = {STRATEGY} (student KCI stepwise replaced in memory; student files unchanged)")
