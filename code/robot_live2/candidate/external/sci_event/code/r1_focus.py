"""R1: bounded, bidirectional focus search state machine for the time-driven DC focus motor (pure stdlib).

Hardware facts (2026-10-16 audit): the focus ring is driven by a DC motor with timed pulses (Nano `F<pwm>,<ms>`),
there is NO position sensor and NO end-stop switch. Therefore:
  - no hard-stop homing, no "push to one end" reference; a sharpness plateau is NOT evidence of an end-stop;
  - all positions are `estimated_command_offset` (sum of commanded pulses), never an absolute lens position;
  - returning to the best offset is open-loop (backlash/slip) -> AF_CONVERGED only after a NEW frame re-evaluates it;
  - device AF is blocked (FOCUS_PROFILE_UNVERIFIED) unless a verified actuator profile is configured: a safe pulse
    budget around a supervised start reference.
Success on a simulated curve is not a physical focus success.

Interfaces (injected):
  camera.measure(after_seq, timeout_s) -> None | dict(seq, sharp, luma, target_present (True/False/None))
      a frame whose seq > after_seq (captured after the actuator call returned); None = no fresh frame
  camera.frame_seq() -> int
  actuator.pulse(direction (+1/-1), ms) -> None   (returns when the drive time ended)
  actuator.stop() -> None
  clock.now() -> float seconds, clock.sleep(s)
"""
from __future__ import annotations

import time

METRIC = dict(type="laplacian_var_gray", roi="centre fraction (see roi)", resolution="1280x720 resize",
              version="r1-af-1")


class St:
    FOCUS_PROFILE_UNVERIFIED = "FOCUS_PROFILE_UNVERIFIED"
    AF_CONVERGED = "AF_CONVERGED"
    AF_NOT_CONVERGED = "AF_NOT_CONVERGED"
    AF_BUDGET_EXHAUSTED = "AF_BUDGET_EXHAUSTED"
    AF_TARGET_LOST = "AF_TARGET_LOST"
    AF_LOW_EXPOSURE = "AF_LOW_EXPOSURE"
    AF_BUDGET_INFEASIBLE = "AF_BUDGET_INFEASIBLE"
    CANCELLED = "CANCELLED"


class RealClock:
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


# Verified profile example shape (values must come from a supervised bench test; none is verified as of 2026-10-16):
#   dict(verified=True, verified_by="...", verified_at="...", start_reference="supervised: sign sharp at 2.0 m",
#        pwm=180, step_ms=300, settle_s=0.35, backlash_ms=120, max_steps_each_dir=6, frame_timeout_s=0.5,
#        measure_s=0.2, min_sharp_converged=40.0, low_luma=25.0, plateau_rel=0.10)
UNVERIFIED_PROFILE = dict(verified=False, why="no supervised actuator profile (pulse budget/start reference) recorded")


class FocusSearch:
    def __init__(self, camera, actuator, profile=None, clock=None, cancel=None, roi=0.5):
        self.cam, self.act, self.clock, self.cancel = camera, actuator, clock or RealClock(), cancel
        self.p = dict(profile or UNVERIFIED_PROFILE)
        self.roi = roi
        self.pulses = []          # (direction, ms, kind) actually commanded
        self.samples = []
        self.offset = 0           # estimated_command_offset in steps

    # ------------------------------------------------------------------ planning
    def plan(self, budget_s):
        """Max k steps each side such that: baseline measure, +1..+k measured, back to 0, -1..-k measured,
        return to best (<= 2k), final fresh measure, fit in budget. k is capped by the safe profile range."""
        p = self.p
        move = p["step_ms"] / 1000.0 + p["settle_s"]
        meas = p.get("measure_s", 0.2)                      # expected fresh-frame + metric time
        rev = p.get("backlash_ms", 0) / 1000.0 + p["settle_s"]
        best_k = 0
        for k in range(1, int(p["max_steps_each_dir"]) + 1):
            cost = (5 * k) * move + (2 * k + 2) * meas + 3 * rev
            if cost <= budget_s:
                best_k = k
        return best_k

    # ------------------------------------------------------------------ steps
    def _check_cancel(self):
        if self.cancel is not None and self.cancel.is_set():
            raise _Stop(St.CANCELLED, "cancel_requested")

    def _check_deadline(self):
        if self.clock.now() > self.deadline:
            raise _Stop(St.AF_BUDGET_EXHAUSTED, "deadline_passed")

    def _step(self, d, kind="search"):
        self._check_cancel()
        self._check_deadline()
        if self.last_dir is not None and d != self.last_dir and self.p.get("backlash_ms"):
            self.act.pulse(d, self.p["backlash_ms"])          # take up gear lash; not counted as offset
            self.pulses.append((d, self.p["backlash_ms"], "backlash"))
            self.clock.sleep(self.p["settle_s"])
        self.act.pulse(d, self.p["step_ms"])
        self.pulses.append((d, self.p["step_ms"], kind))
        self.clock.sleep(self.p["settle_s"])
        self.offset += d
        self.last_dir = d

    def _measure(self, tag):
        self._check_cancel()
        seq0 = self.cam.frame_seq()
        m = self.cam.measure(seq0, self.p.get("frame_timeout_s", 0.5))
        if m is None or m.get("seq") is None or m["seq"] <= seq0:
            raise _Stop(St.AF_NOT_CONVERGED, "FRESH_FRAME_UNAVAILABLE")
        rec = dict(tag=tag, estimated_command_offset=self.offset, seq=m["seq"], sharp=float(m["sharp"]),
                   luma=m.get("luma"), target_present=m.get("target_present"), t=self.clock.now() - self.t0)
        self.samples.append(rec)
        if m.get("target_present") is False:
            raise _Stop(St.AF_TARGET_LOST, "target_not_in_roi")
        if m.get("luma") is not None and m["luma"] < self.p.get("low_luma", 0):
            raise _Stop(St.AF_LOW_EXPOSURE, f"luma {m['luma']:.1f} < {self.p.get('low_luma')}")
        return rec

    def _goto(self, target):
        while self.offset != target:
            self._step(1 if target > self.offset else -1, kind="return")

    # ------------------------------------------------------------------ run
    def run(self, budget_s):
        self.t0 = self.clock.now()
        self.deadline = self.t0 + budget_s
        self.last_dir = None
        base = dict(metric=dict(METRIC, roi=self.roi), budget_s=budget_s, profile_verified=bool(self.p.get("verified")),
                    position_semantics="estimated_command_offset (open loop, no lens position sensor)",
                    plateau_is_end_stop_evidence=False)
        if not self.p.get("verified"):
            return dict(base, status=St.FOCUS_PROFILE_UNVERIFIED, reason=self.p.get("why", "unverified"),
                        pulses=[], samples=[], ok=False)
        k = self.plan(budget_s)
        if k < 1:
            return dict(base, status=St.AF_BUDGET_INFEASIBLE, planned_k=0, pulses=[], samples=[], ok=False,
                        reason="budget cannot hold baseline+bidirectional probe+return+final check")
        status, reason = None, None
        try:
            self._measure("baseline")
            for _ in range(k):
                self._step(+1)
                self._measure("probe+")
            self._goto(0)
            for _ in range(k):
                self._step(-1)
                self._measure("probe-")
            sh = [s["sharp"] for s in self.samples]
            hi, lo = max(sh), min(sh)
            if hi <= 0 or (hi - lo) / hi < self.p.get("plateau_rel", 0.10):
                self._goto(0)
                status, reason = St.AF_NOT_CONVERGED, "PLATEAU_NO_CONTRAST (not an end-stop; scene/lens range unknown)"
            else:
                best = max(self.samples, key=lambda s: s["sharp"])
                self._goto(best["estimated_command_offset"])
                fin = self._measure("final_after_return")          # NEW frame after the open-loop return
                if fin["sharp"] >= self.p["min_sharp_converged"] and fin.get("target_present") is not False:
                    status, reason = St.AF_CONVERGED, "fresh_final_frame_meets_threshold"
                else:
                    status, reason = St.AF_NOT_CONVERGED, (f"final {fin['sharp']:.1f} < min {self.p['min_sharp_converged']}"
                                                           f" (best seen {best['sharp']:.1f}; open-loop return)")
        except _Stop as e:
            status, reason = e.status, e.reason
        finally:
            try:
                self.act.stop()
            except Exception:  # noqa: BLE001
                pass
        best = max(self.samples, key=lambda s: s["sharp"]) if self.samples else None
        final = self.samples[-1] if self.samples else None
        return dict(base, status=status, reason=reason, ok=status == St.AF_CONVERGED, planned_k=k,
                    pulses=list(self.pulses), samples=self.samples,
                    best=best, final=final, estimated_command_offset_final=self.offset,
                    sec=self.clock.now() - self.t0)


class _Stop(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


# ---------------------------------------------------------------------- fakes (tests / dry simulation only)
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class FakeFocusActuator:
    """True lens position in 'step units'; slip makes the real motion differ from the command (open loop)."""

    def __init__(self, clock, pos=0.0, slip=0.0, step_ms=300, slip_dirs=(1, -1)):
        self.clock, self.pos, self.slip, self.step_ms, self.slip_dirs = clock, pos, slip, step_ms, slip_dirs
        self.calls, self.stops = [], 0

    def pulse(self, d, ms):
        self.calls.append((d, ms))
        self.clock.sleep(ms / 1000.0)
        self.pos += d * (ms / self.step_ms) * (1.0 - (self.slip if d in self.slip_dirs else 0.0))

    def stop(self):
        self.stops += 1


class FakeCamera:
    """sharp = curve(true lens position). stale=True -> seq never advances (no fresh frame)."""

    def __init__(self, clock, actuator, curve, luma=120.0, target_present=True, stale_after=None, frame_s=0.05):
        self.clock, self.act, self.curve = clock, actuator, curve
        self.luma, self.target_present, self.stale_after, self.frame_s = luma, target_present, stale_after, frame_s
        self.seq = 0
        self.n_measure = 0

    def frame_seq(self):
        return self.seq

    def measure(self, after_seq, timeout_s):
        self.n_measure += 1
        if self.stale_after is not None and self.n_measure > self.stale_after:
            self.clock.sleep(timeout_s)
            return dict(seq=self.seq, sharp=self.curve(self.act.pos), luma=self.luma)   # old seq: not fresh
        self.clock.sleep(self.frame_s)
        self.seq += 1
        tp = self.target_present(self.n_measure) if callable(self.target_present) else self.target_present
        return dict(seq=self.seq, sharp=float(self.curve(self.act.pos)), luma=self.luma, target_present=tp)
