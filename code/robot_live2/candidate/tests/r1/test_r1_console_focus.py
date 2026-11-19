"""R1 W-B: aim console r1-aim-1 protocol, single device owner, focus search (SIMULATED fakes + STATIC wiring).

T05/T07/T08/T09/T11 (server side), T29/T30/T31. No serial port, camera, socket, subprocess or model is used:
DeviceOwner gets an in-memory FakeHW, FocusSearch gets FakeCamera/FakeFocusActuator/FakeClock.
"""
import ast
import os
import sys
import threading
import time
from pathlib import Path

import pytest

TREE = Path(os.environ["SCI_WS"])
CODE = TREE / "external" / "sci_event" / "code"
sys.path.insert(0, str(CODE))

R1O = pytest.importorskip("r1_device_owner")
R1F = pytest.importorskip("r1_focus")


# ------------------------------------------------------------------------------------------------ fakes
class FakeHW:
    """Records every hardware step with monotonic times; `gate` holds a step 'running' until released."""

    def __init__(self, stale=False, fail=False, pixels=b"\xff\xd8same-pixels\xff\xd9"):
        self.events, self.seq, self.stale, self.fail, self.pixels = [], 0, stale, fail, pixels
        self.gate = threading.Event()
        self.gate.set()
        self.running, self.max_running = 0, 0
        self.lock = threading.Lock()

    def _hw(self, name, *a):
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        self.events.append((name + "_start", time.monotonic(), a))
        self.gate.wait(5.0)                      # hardware step cannot be pre-empted by a logical cancel
        if self.fail:
            with self.lock:
                self.running -= 1
            raise IOError("nano did not answer")
        self.events.append((name + "_end", time.monotonic(), a))
        with self.lock:
            self.running -= 1

    def plan_aim(self, p):
        if p.get("refuse"):
            return dict(refused="target_out_of_pantilt_range")
        return dict(refused=None, pan=100.0 + p["u"] / 100, tilt=150.0, clipped=False)

    def move(self, pan, tilt, cancel):
        self._hw("move", pan, tilt)

    def home(self, cancel):
        self._hw("home")

    def focus(self, cancel, p):
        self._hw("focus")
        return dict(status="AF_CONVERGED", ok=True)

    def frame_seq(self):
        return self.seq

    def capture_fresh(self, after_seq, timeout_s):
        if self.stale:
            return dict(jpeg=self.pixels, seq=self.seq, rx_mono_ns=time.monotonic_ns(), w=4, h=3)  # old frame
        self.seq += 1
        return dict(jpeg=self.pixels, seq=self.seq, rx_mono_ns=time.monotonic_ns(), w=4, h=3)

    def stop_actuators(self):
        self.events.append(("stop_actuators", time.monotonic(), ()))


def owner(tmp_path, **kw):
    hw = FakeHW(**kw)
    return R1O.DeviceOwner(hw, str(tmp_path / "caps"), fresh_timeout_s=0.2, log=lambda *a: None), hw


def wait_status(o, rid, statuses, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = o.get(rid)
        if r["status"] in statuses:
            return r
        time.sleep(0.01)
    return o.get(rid)


AIM = dict(u=320, v=240, z=2.0, focus=True, strict=True)


# ------------------------------------------------------------------------------------------------ T05 / T07
def test_T05_result_is_bound_to_request_not_a_counter(tmp_path):
    o, hw = owner(tmp_path)
    a = o.submit("aim", "e5:run1:ev001:a1:x", "e5:run1:ev001:a1", AIM)
    ra = wait_status(o, a["request_id"], R1O.TERMINAL)
    m_id, m_own = o.manual_id("focus")
    m = o.submit("focus", m_id, m_own, {})                     # operator presses manual /focus afterwards
    rm = wait_status(o, m["request_id"], R1O.TERMINAL)
    ra2 = o.get(a["request_id"])
    assert ra["status"] == "COMPLETED" and rm["status"] == "COMPLETED"
    assert ra2["capture"]["capture_id"] == ra["capture"]["capture_id"]          # A's result unchanged by M
    assert ra2["capture"]["request_id"] == a["request_id"] and rm["capture"]["request_id"] == m_id
    assert ra2["capture"]["capture_id"] != rm["capture"]["capture_id"]
    assert m_own.startswith("manual:") and ra2["owner"].startswith("e5:")
    st = o.state()
    assert st["protocol"] == "r1-aim-1" and "job" not in st                    # no global job counter exposed


def test_T07_route_requires_protocol_fields_and_reports_protocol(tmp_path):
    o, hw = owner(tmp_path)
    code, ctype, body = R1O.route(o, "/aim", {"u": ["10"], "v": ["10"], "req": ["e5:r:ev1:a1:n"], "owner": ["e5:r:ev1:a1"]})
    import json
    j = json.loads(body)
    assert j["protocol"] == "r1-aim-1" and j["request_id"] == "e5:r:ev1:a1:n" and j["status"] in ("ACCEPTED", "REJECTED_BUSY")
    assert R1O.route(o, "/full.jpg", {}) is None                               # no global-image result path in r1
    bad = o.submit("aim", "", "", AIM)
    assert bad["status"] == "PROTOCOL_ERROR"
    o.wait_idle()


# ------------------------------------------------------------------------------------------------ T08
def test_T08_single_owner_serialises_and_rejects_while_busy(tmp_path):
    o, hw = owner(tmp_path)
    hw.gate.clear()
    a = o.submit("aim", "A", "e5:r:ev1:a1", AIM)
    wait_status(o, "A", {"RUNNING"})
    h = o.submit("home", "H", "e5:r:ev1:home")
    f = o.submit("focus", *o.manual_id("focus"), {})
    assert h["status"] == "REJECTED_BUSY" and f["status"] == "REJECTED_BUSY"
    dup = o.submit("aim", "A", "e5:r:ev1:a1", AIM)                              # duplicate: no second sequence
    assert dup["duplicate"] is True
    hw.gate.set()
    wait_status(o, "A", R1O.TERMINAL)
    assert sum(1 for e in hw.events if e[0] == "move_start") == 1
    h2 = o.submit("home", "H2", "e5:r:ev1:home")
    wait_status(o, "H2", R1O.TERMINAL)
    assert hw.max_running == 1


def test_T08_cancelled_home_cannot_move_during_next_request(tmp_path):
    o, hw = owner(tmp_path)
    hw.gate.clear()
    o.submit("home", "H", "e5:r:ev1:home")
    wait_status(o, "H", {"RUNNING"})
    c = o.cancel("H", why="next_event")
    assert c["status"] == "CANCELLING" and c["hw_done"] is False
    b = o.submit("aim", "B", "e5:r:ev2:a1", AIM)                                # still CANCELLING -> refused
    assert b["status"] == "REJECTED_BUSY" and b["reason_code"] == "device_cancelling"
    hw.gate.set()
    rh = wait_status(o, "H", R1O.TERMINAL)
    assert rh["status"] == "CANCELLED" and rh["hw_done"] is True
    b2 = o.submit("aim", "B2", "e5:r:ev2:a2", AIM)
    wait_status(o, "B2", R1O.TERMINAL)
    t_home_end = max(t for n, t, _ in hw.events if n == "home_end")
    t_b_move = min(t for n, t, _ in hw.events if n == "move_start")
    assert t_home_end <= t_b_move                                             # HOME ended before B moved
    assert not any(n == "home_start" and t > t_b_move for n, t, _ in hw.events)


# ------------------------------------------------------------------------------------------------ T09
def test_T09_cancel_or_timeout_is_not_device_idle(tmp_path):
    o, hw = owner(tmp_path)
    hw.gate.clear()
    o.submit("aim", "A", "e5:r:ev1:a1", AIM)
    wait_status(o, "A", {"RUNNING"})
    c = o.cancel("A", why="client_timeout", logical_expiry=True)
    assert c["status"] == "CANCELLING" and c["t_logical_expired"] is not None and c["hw_done"] is False
    assert o.state()["device_state"] == "CANCELLING"                          # NOT idle: hardware still in its step
    time.sleep(0.05)
    assert o.get("A")["t_device_idle"] is None
    hw.gate.set()
    r = wait_status(o, "A", R1O.TERMINAL)
    assert r["status"] == "CANCELLED" and r["hw_done"] is True and r["capture"] is None
    assert r["t_cancel_requested"]["mono_ns"] < r["t_hw_end"]["mono_ns"] <= r["t_device_idle"]["mono_ns"]
    assert o.state()["device_state"] == "IDLE"


def test_T09_late_result_quarantined_not_given_to_next(tmp_path):
    o, hw = owner(tmp_path)
    hw.gate.clear()
    o.submit("focus", "F1", "manual:1", {})
    wait_status(o, "F1", {"RUNNING"})
    o.cancel("F1")
    hw.gate.set()
    r1 = wait_status(o, "F1", R1O.TERMINAL)
    o.submit("aim", "A2", "e5:r:ev2:a1", AIM)
    r2 = wait_status(o, "A2", R1O.TERMINAL)
    assert r1["status"] == "CANCELLED" and r1["capture"] is None
    if r1["quarantine"]:
        assert r1["quarantine"]["capture"]["capture_id"] != r2["capture"]["capture_id"]
    assert r2["capture"]["request_id"] == "A2"


def test_T09_hardware_error_leaves_device_unknown(tmp_path):
    o, hw = owner(tmp_path, fail=True)
    o.submit("aim", "A", "e5:r:ev1:a1", AIM)
    r = wait_status(o, "A", R1O.TERMINAL)
    assert r["status"] == "FAILED" and r["device_state"] == "UNKNOWN"
    b = o.submit("aim", "B", "e5:r:ev2:a1", AIM)
    assert b["status"] == "REJECTED_BUSY" and b["reason_code"] == "device_unknown"
    assert o.reset_unknown("supervised check: servos idle") is True


# ------------------------------------------------------------------------------------------------ T11
def test_T11_same_pixels_new_seq_is_new_capture(tmp_path):
    o, hw = owner(tmp_path)
    o.submit("aim", "A", "e5:r:ev1:a1", AIM)
    ra = wait_status(o, "A", R1O.TERMINAL)
    o.submit("aim", "B", "e5:r:ev2:a1", AIM)
    rb = wait_status(o, "B", R1O.TERMINAL)
    ca, cb = ra["capture"], rb["capture"]
    assert ca["sha256"] == cb["sha256"]                                       # identical pixels...
    assert ca["frame_seq"] != cb["frame_seq"] and ca["capture_id"] != cb["capture_id"]   # ...distinct exposures
    assert Path(ca["path"]).exists() and Path(cb["path"]).exists()
    assert ca["source_t"] is None and ca["source_t_null_reason"]
    code, ctype, body = R1O.route(o, "/capture", {"capture_id": [ca["capture_id"]]})
    assert code == 200 and body == hw.pixels


def test_T11_no_fresh_frame_is_fresh_capture_unavailable(tmp_path):
    o, hw = owner(tmp_path, stale=True)
    o.submit("aim", "A", "e5:r:ev1:a1", AIM)
    r = wait_status(o, "A", R1O.TERMINAL)
    assert r["status"] == "FAILED" and r["reason_code"] == "FRESH_CAPTURE_UNAVAILABLE" and r["capture"] is None


def test_refused_plan_moves_nothing(tmp_path):
    o, hw = owner(tmp_path)
    o.submit("aim", "A", "e5:r:ev1:a1", dict(AIM, refuse=True))
    r = wait_status(o, "A", R1O.TERMINAL)
    assert r["status"] == "FAILED" and r["reason_code"] == "target_out_of_pantilt_range"
    assert not any(n.startswith("move") for n, _, _ in hw.events)


# ------------------------------------------------------------------------------------------------ focus
PROFILE = dict(verified=True, verified_by="TEST_FAKE_ONLY", start_reference="simulated", pwm=180, step_ms=300,
               settle_s=0.35, backlash_ms=100, max_steps_each_dir=6, frame_timeout_s=0.5, measure_s=0.1,
               min_sharp_converged=40.0, low_luma=25.0, plateau_rel=0.10)


def peak(at, top=200.0, floor=2.0, width=1.5):
    return lambda x: floor + top / (1.0 + ((x - at) / width) ** 2)


def rig(curve, slip=0.0, slip_dirs=(1, -1), **cam):
    clk = R1F.FakeClock()
    act = R1F.FakeFocusActuator(clk, slip=slip, step_ms=PROFILE["step_ms"], slip_dirs=slip_dirs)
    return clk, act, R1F.FakeCamera(clk, act, curve, **cam)


def test_T29_no_profile_no_actuation():
    clk, act, cam = rig(peak(2))
    r = R1F.FocusSearch(cam, act, profile=None, clock=clk).run(15.0)
    assert r["status"] == "FOCUS_PROFILE_UNVERIFIED" and r["ok"] is False and act.calls == []


def test_T29_plateau_is_not_end_stop_and_stays_in_safe_range():
    clk, act, cam = rig(lambda x: 3.0)
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(60.0)
    assert r["status"] == "AF_NOT_CONVERGED" and "PLATEAU" in r["reason"]
    assert r["plateau_is_end_stop_evidence"] is False
    offs = [s["estimated_command_offset"] for s in r["samples"]]
    assert max(abs(x) for x in offs) <= PROFILE["max_steps_each_dir"]
    assert r["estimated_command_offset_final"] == 0                          # returned to the start reference
    assert "estimated_command_offset" in r["position_semantics"]


def test_T30_budget_infeasible_declared_up_front():
    clk, act, cam = rig(peak(2))
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(2.0)
    assert r["status"] == "AF_BUDGET_INFEASIBLE" and act.calls == []


def test_T30_bidirectional_within_budget_converges_on_negative_side():
    clk, act, cam = rig(peak(-2))
    budget = 30.0
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(budget)
    tags = {s["tag"] for s in r["samples"]}
    assert {"baseline", "probe+", "probe-", "final_after_return"} <= tags
    assert r["status"] == "AF_CONVERGED" and r["ok"] is True
    assert r["sec"] <= budget                                                 # plan fits the fake-clock budget
    assert r["best"]["estimated_command_offset"] == -2


def test_T30_slower_than_planned_stops_at_deadline():
    clk, act, cam = rig(peak(3), frame_s=1.5)                                 # real frames much slower than profile
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(15.0)
    assert r["status"] == "AF_BUDGET_EXHAUSTED" and r["ok"] is False and act.stops >= 1


def test_T31_return_without_fresh_frame_is_not_converged():
    clk, act, cam = rig(peak(2))
    k = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).plan(30.0)
    clk, act, cam = rig(peak(2), stale_after=1 + 2 * k)                         # final measurement gets an old frame
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(30.0)
    assert r["status"] == "AF_NOT_CONVERGED" and r["reason"] == "FRESH_FRAME_UNAVAILABLE" and r["ok"] is False


def test_T31_open_loop_slip_return_misses_peak():
    clk, act, cam = rig(peak(2, width=0.4), slip=0.45, slip_dirs=(-1,))   # gear slips only when driven backwards
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(30.0)
    assert r["samples"][-1]["tag"] == "final_after_return"
    assert r["status"] == "AF_NOT_CONVERGED" and r["ok"] is False             # best seen, but not re-achieved


def test_T31_target_lost_and_low_exposure_stop_af():
    clk, act, cam = rig(peak(2), target_present=lambda n: n < 3)
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(30.0)
    assert r["status"] == "AF_TARGET_LOST" and len(r["samples"]) == 3
    clk, act, cam = rig(peak(2), luma=8.0)
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk).run(30.0)
    assert r["status"] == "AF_LOW_EXPOSURE" and act.calls == []


def test_focus_cancel_stops_motor():
    clk, act, cam = rig(peak(2))
    tok = R1O.CancelToken()
    tok.set()
    r = R1F.FocusSearch(cam, act, profile=PROFILE, clock=clk, cancel=tok).run(30.0)
    assert r["status"] == "CANCELLED" and act.stops == 1


# ------------------------------------------------------------------------------------------------ STATIC wiring
SRC = (CODE / "aim_console.py").read_text(encoding="utf-8")
TREE_AST = ast.parse(SRC)


def _func(cls, name):
    for n in ast.walk(TREE_AST):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            for f in n.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return ast.get_source_segment(SRC, f)
    return None


def test_STATIC_console_wiring_uses_r1_owner():
    do_get = _func("Handler", "do_GET")
    assert "R1O.route(OWNER[\"o\"], url.path, q)" in do_get
    assert '"job"' not in do_get and "a.job" not in do_get                    # /state has no job counter
    assert "best * 0.5" not in SRC and "self.job" not in SRC
    assert "이미 작업 중입니다" not in do_get                                 # no legacy plain-text busy reply
    assert "R1F.FocusSearch" in _func("R1Hardware", "focus")
    assert "R1O.DeviceOwner(R1Hardware(" in SRC
    for path in ("/goto", "/pin_test", "/tilt_sweep", "/trim", "/focus_profile"):
        assert path in do_get
    assert "busy.acquire(blocking=False)" not in do_get                       # diagnostics go through the owner
