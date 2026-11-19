"""R1 W-D tests: E5 node identity/event store, r1-aim-1 client, capture/release states, async device work,
frame lineage, stop policy, P2 reference, search gate.  Evidence type: SIMULATED (fakes) unless noted.
Integration tests drive the REAL candidate Node/GatedNode call chain (event_step -> tick_event -> aim -> device worker
-> on_device_result -> store -> status) with fakes injected; nothing touches ROS/devices/models/network.
"""
import ast
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_common", WS / "src" / "sci_pantilt",
          WS / "external" / "sci_event" / "e5_onboard", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import r1_contracts as C  # noqa: E402
import e5_node as N  # noqa: E402
import stop_policy as SP  # noqa: E402
from device_worker import DeviceWorker, LatestSlot  # noqa: E402
from event_store import EventStore  # noqa: E402
from planner import PanTiltPlanner  # noqa: E402
from aim_client import AimClient  # noqa: E402
from siglip_transport import PipeTransport, TransportError, TransportTimeout  # noqa: E402
import r1_fakes_e5 as F  # noqa: E402

CS, RS = C.CaptureStatus, C.ReleaseStatus
BASE_NODE = N.Node          # e5_onboard rebinds N.Node to GatedNode on import; base tests use the SCI Node


# ------------------------------------------------------------------------------------------------ helpers
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "WS", tmp_path / "ws")
    monkeypatch.setattr(N, "SHM", tmp_path / "shm")
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    monkeypatch.setenv("SCI_AIM_TIMEOUT_S", "0.6")
    monkeypatch.setenv("SCI_AIM_CANCEL_GRACE_S", "0.4")
    return tmp_path


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_node(tmp, console, run="r1test", execute=True, primary_zero=True, **kw):
    clk = Clock()
    planner = PanTiltPlanner(N.Geometry(), console="http://fake-console", execute=execute, http_get=console.http_get)
    body = (lambda now: SP.BodySample(0.0, now, "sportmodestate.velocity_hypot_xy")) if primary_zero else None
    deps = dict(frontend=F.FakeFrontend(), monitor=F.FakeMon(), planner=planner, clock=clk, body_speed=body, threaded=False,
                trial_pointer=kw.pop("trial_pointer", None))
    node = BASE_NODE(F.args(tmp, run_name=run, pantilt_execute=execute, **kw), deps=deps)
    return node, clk


def drive(node, src, clk, seconds, dt=0.1, until=None):
    t_end = clk.t + seconds
    while clk.t < t_end:
        clk.t += dt
        node.event_step(src, None)
        time.sleep(0.005)                                            # let the device worker thread run
        if until and until(node):
            break
    for _ in range(200):                                             # settle pending device results
        if not node.device.busy and node.device.results.empty():
            break
        time.sleep(0.01)
    node.event_step(src, None)


def run_event(node, clk, stop=None):
    src = F.FakeSource()
    node.start_event(stop or F.a3_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None,
                     latest_frame=dict(frame_id=9, header_stamp=101.0))
    drive(node, src, clk, 6.0, until=lambda n: n.fsm is not None and not n.fsm.active and not n.device.busy)
    drive(node, src, clk, 0.5)
    return src


def rec(node):
    return node.store.get(node.fsm.key)


def write_done(node, key, cap, rev=1, terminal=True, reason="NO_TEXT_DETECTED", name=None):
    d = dict(schema_version=C.SCHEMA_VERSION, event_key=key, capture_id=cap, terminal=terminal, reason_code=reason,
             result_revision=rev, ocr=dict(lines=[]))
    C.atomic_write_json(node.run / "jobs" / "done" / (name or f"{N._safe(key)}.json"), d)


# =============================================================================================== T03 T04
def test_T03_reading_attached_after_fsm_idle(env):
    con = F.FakeConsole()
    node, clk = make_node(env, con)
    src = run_event(node, clk)
    r = rec(node)
    assert r["capture_status"] == CS.CAPTURE_QUALITY_OK and r["release"] == RS.RELEASE_READY_CAPTURE_OK
    assert not node.fsm.active                                       # FSM idle BEFORE the reading arrives
    write_done(node, r["event_key"], r["capture_id"])
    node.event_step(src, None)
    ev = [e for e in node.status["events"] if e["event_key"] == r["event_key"]][0]
    assert ev["reading"]["reason_code"] == "NO_TEXT_DETECTED" and ev["state"] == "READ_DONE"
    st = json.loads((N.SHM / "status.json").read_text())
    assert st["events"][-1]["reading"]["capture_id"] == r["capture_id"]


def test_T04_same_ev001_in_two_runs_never_mix(env):
    con_a, con_b = F.FakeConsole(), F.FakeConsole()
    a, ca = make_node(env, con_a, run="runA")
    run_event(a, ca)
    b, cb = make_node(env, con_b, run="runB")
    srcb = run_event(b, cb)
    ka, kb = rec(a)["event_key"], rec(b)["event_key"]
    assert ka.endswith("#ev001") and kb.endswith("#ev001") and ka != kb
    write_done(b, ka, rec(a)["capture_id"], name="x.json")            # run A's result dropped into run B's done dir
    write_done(b, kb, "cap-of-someone-else", name="y.json")           # right event, wrong capture
    b.event_step(srcb, None)
    assert rec(b)["reading"] is None
    assert b.status["reader_attach"] == {"x.json": "NO_SUCH_EVENT", "y.json": "CAPTURE_MISMATCH"}
    C.atomic_write_json(b.run / "jobs" / "done" / "ev001.json", dict(ev=1, status="NO_ROOM_LABEL"))   # legacy, no identity
    b.event_step(srcb, None)
    assert b.status["reader_attach"]["ev001.json"] == "LEGACY_NO_IDENTITY" and rec(b)["reading"] is None


def test_T04b_job_file_identity_contract(env):
    node, clk = make_node(env, F.FakeConsole())
    run_event(node, clk)
    jobs = list((node.run / "jobs" / "pending").glob("*.json"))
    assert len(jobs) == 1 and not jobs[0].name.startswith("ev0")      # legacy evNNN.json names never written
    j = json.loads(jobs[0].read_text())
    for k in ("schema_version", "run_id", "event_id", "event_key", "capture_id", "request_id", "capture_status", "pantilt_ok",
              "image", "image_sha256", "quality", "target_roi_verified", "odom_frame"):
        assert k in j, k
    assert j["trial_id"] is None and "trial_id_null_reason" in j and j["target_roi"] is None


# =============================================================================================== T05 T06 T07
def test_T05_manual_focus_and_foreign_capture_not_success(env):
    con = F.FakeConsole()
    cl = AimClient("http://c", http_get=con.http_get)
    con.manual("focus")                                               # manual job runs and completes (old job counter +1)
    r = cl.run("aim", dict(u=320, v=240, z="2.0", focus=1, strict=1), "run", 1, 1, deadline_s=0.5, poll_s=0.01)
    assert r["status"] == "COMPLETED" and r["capture"]["request_id"] == r["request_id"]
    manual_ids = [k for k in con.reqs if k.startswith("manual")]
    assert manual_ids and r["request_id"] not in manual_ids
    # a /request record carrying another request's capture is never accepted
    out = AimClient._final(dict(request_id="mine"), dict(request_id="mine", status="COMPLETED", hw_done=True, device_state="IDLE",
                                                         capture=dict(capture_id="c", request_id="other")), True)
    assert out["capture"] is None


def test_T06_late_result_after_timeout_not_given_to_next_event(env):
    con = F.FakeConsole(job_s=1.5)                                    # hardware keeps going after the logical deadline
    node, clk = make_node(env, con)
    run_event(node, clk)
    r1 = rec(node)
    assert r1["capture_status"] == CS.NO_CAPTURE and r1["capture_id"] is None
    node.on_device_result(F.FakeSource(), dict(key="r1test#ev000", ok=True, result=dict(kind="aim", aim={})))
    assert any(f["kind"] == "LATE_DEVICE_RESULT_QUARANTINED" for f in node.status["faults"])
    time.sleep(1.6)                                                   # old hardware step ends -> CANCELLED + quarantine
    q = [r for r in con.reqs.values() if r["kind"] == "aim"][0]
    con.http_get("http://c/state", 1)
    assert q["status"] == "CANCELLED" and q["capture"] is None and q["quarantine"]["capture"] is not None
    con.job_s = 0.0
    run_event(node, clk)                                              # next event gets its OWN request/capture
    r2 = rec(node)
    assert r2["event_key"].endswith("ev002") and r2["capture_id"] and r2["capture_id"] != (q["quarantine"]["capture"] or {}).get("capture_id")
    assert r2["request_id"].split(":")[1] == "ev002"


def test_T07_legacy_console_refused_no_fallback(env):
    con = F.LegacyConsole()
    node, clk = make_node(env, con)
    run_event(node, clk)
    r = rec(node)
    assert r["capture_status"] == CS.NO_CAPTURE and "PROTOCOL_MISMATCH" in r["reason_code"] or \
        (node.fsm.result["aim"]["reason_code"] == C.PROTOCOL_MISMATCH)
    assert "/aim" not in con.calls and "/full.jpg" not in con.calls      # nothing sent, no global full.jpg
    assert r["release"] == RS.RELEASE_READY_OBSERVATION_FAILED          # no actuator command was issued


# =============================================================================================== T08 T09
def test_T08_single_device_owner_and_no_go_home_thread(env):
    import e5_onboard as OB
    assert not hasattr(OB.GatedNode, "_go_home")                       # 60 s /home retry thread removed
    w = DeviceWorker()
    ev = threading.Event()
    assert w.submit("a", lambda: ev.wait(1.0)) is True
    assert w.submit("b", lambda: None) is False                         # no second physical job while one runs
    ev.set()
    time.sleep(0.1)
    assert w.drain()[0]["key"] == "a"
    con = F.FakeConsole()
    node, clk = make_node(env, con)
    run_event(node, clk)
    kinds = [r["kind"] for r in con.reqs.values()]
    assert kinds == ["aim", "home"]                                     # HOME through the same r1-aim-1 owner, after capture
    home = [r for r in con.reqs.values() if r["kind"] == "home"][0]
    assert home["owner"].startswith("e5:r1test:ev001") and node.fsm.result["home"]["status"] == "COMPLETED"


def test_T08b_cancelled_home_does_not_move_next(env):
    con = F.FakeConsole(job_s=0.0)
    cl = AimClient("http://c", http_get=con.http_get)
    con.job_s = 2.0
    h = cl.run("home", {}, "run", 1, 2, deadline_s=0.05, poll_s=0.01, cancel_grace_s=0.05)
    assert h["logical_expired"] and h["device_state"] in ("CANCELLING", "UNKNOWN") and not h.get("device_idle_confirmed")
    con.job_s = 0.0
    a = cl.run("aim", dict(u=1, v=1, z="1"), "run", 2, 1, deadline_s=0.2, poll_s=0.01)
    assert a["status"] == "REJECTED_BUSY"                                # cancelled HOME still owns the device


def test_T09_timeout_alone_never_releases(env):
    con = F.FakeConsole(job_s=30.0)                                   # hardware never ends within the test
    node, clk = make_node(env, con)
    run_event(node, clk)
    r = rec(node)
    assert r["release"] == RS.RELEASE_BLOCKED_ACTUATOR_UNKNOWN and not r["terminal"]
    assert node.fsm.result["aim"]["logical_expired"] and node.fsm.result["aim"]["device_state"] != "IDLE"
    assert "home" not in [x["kind"] for x in con.reqs.values()]       # no HOME while the device state is unknown


# =============================================================================================== T10 T11 T12
def test_T10_blurred_capture_is_quality_fail_with_fresh_retries(env):
    con = F.FakeConsole(sharp=False)
    node, clk = make_node(env, con)
    run_event(node, clk)
    r = rec(node)
    assert r["capture_status"] == CS.CAPTURE_QUALITY_FAIL and r["release"] == RS.RELEASE_READY_OBSERVATION_FAILED
    assert r["capture"]["quality"]["threshold"] == 40.0 and not r["capture"]["quality"]["ok"]
    aims = [x for x in con.reqs.values() if x["kind"] == "aim"]
    assert len(aims) == 1 + node.cfg["event"]["capture_retries"]      # each retry = a NEW request = a new exposure
    assert len({x["capture"]["capture_id"] for x in aims}) == len(aims)
    j = json.loads(next((node.run / "jobs" / "pending").glob("*.json")).read_text())
    assert j["capture_status"] == CS.CAPTURE_QUALITY_FAIL               # reader must not send it to E6 (W-C contract)


def test_T11_same_capture_is_not_fresh_but_new_seq_with_same_bytes_is(env):
    con = F.FakeConsole(sharp=False, same_capture_on_retry=True)
    node, clk = make_node(env, con)
    run_event(node, clk)
    assert rec(node)["capture_status"] == CS.FRESH_CAPTURE_UNAVAILABLE
    con2 = F.FakeConsole(sharp=False, identical_bytes=True)          # identical pixels, new seq/id -> distinct captures
    n2, c2 = make_node(env, con2, run="r1b")
    run_event(n2, c2)
    aims = [x for x in con2.reqs.values() if x["kind"] == "aim"]
    assert len(aims) == 3 and rec(n2)["capture_status"] == CS.CAPTURE_QUALITY_FAIL
    assert len({x["capture"]["frame_seq"] for x in aims}) == 3


def test_T12_proxy_only_in_diagnostic_mode(env):
    node, clk = make_node(env, F.FakeConsole(), execute=False)
    node.p2_crops[7] = np.random.RandomState(1).randint(0, 255, (60, 60, 3)).astype(np.uint8)
    run_event(node, clk)
    assert rec(node)["capture_status"] == CS.NO_CAPTURE and not list((node.run / "jobs" / "pending").glob("*.json"))
    n2, c2 = make_node(env, F.FakeConsole(), run="diag", execute=False, diagnostic_proxy=True)
    n2.p2_crops[7] = node.p2_crops[7]
    run_event(n2, c2)
    r = rec(n2)
    assert r["capture_status"] == CS.CAPTURE_PROXY and r["capture"]["pantilt_ok"] is False
    j = json.loads(next((n2.run / "jobs" / "pending").glob("*.json")).read_text())
    assert j["pantilt_ok"] is False and j["capture_status"] == CS.CAPTURE_PROXY


# =============================================================================================== T19 T20
def test_T19_old_box_not_drawn_on_newer_frame(env, monkeypatch):
    import e5_onboard as OB
    meta = dict(frame_id=10, boxes=[([10, 10, 60, 60], "sign", 2)], t_drawn=time.time())
    assert OB.live_boxes_for(11, meta) == [] and OB.live_boxes_for(10, meta) == meta["boxes"]
    g = OB.GatedNode.__new__(OB.GatedNode)
    g._live_meta = meta
    frame = np.zeros((480, 640, 3), np.uint8)
    newer = g.render_live((0.0, 11, frame))
    assert int(newer[10:61, 10:61].sum()) == 0                           # nothing painted on a non-source frame
    same = g.render_live((0.0, 10, frame))
    assert int(same[10:61, 10:61].sum()) > 0


def test_T20_missing_source_frame_not_substituted(env):
    node, clk = make_node(env, F.FakeConsole())
    latest = np.full((480, 640, 3), 77, np.uint8)
    node.start_event(F.a3_stop(), 100.0, (0, 0, 0), latest, None, latest_frame=dict(frame_id=9, header_stamp=101.0))
    ev = rec(node)["evidence"]
    assert ev["source_image"] is None and "SOURCE_FRAME_UNAVAILABLE" in ev["source_image_null_reason"]
    assert ev["latest_image"].endswith("stop_latest_frame_no_box.jpg") and not (N.SHM / "last_stop.jpg").exists()
    node.scene_frames[5] = np.zeros((480, 640, 3), np.uint8)             # now the source frame is retained
    n2, _ = make_node(env, F.FakeConsole(), run="r1c")
    n2.scene_frames[5] = node.scene_frames[5]
    n2.start_event(F.a3_stop(), 100.0, (0, 0, 0), latest, None, latest_frame=dict(frame_id=9, header_stamp=101.0))
    ev2 = n2.store.get(n2.fsm.key)["evidence"]
    assert ev2["source_image"].endswith("stop_source_frame_with_box.jpg") and ev2["source_frame"]["frame_id"] == 5


# =============================================================================================== T21
def test_T21_geometry_inputs_rejected():
    ok, why = SP.geometry_input_check(100.0, depth_stamp=100.5, pose_stamp=100.0)
    assert not ok and any(w.startswith("depth_time_gap") and "uncalibrated" in w for w in why)
    assert not SP.geometry_input_check(100.0, depth_stamp=100.0, pose_stamp=101.0)[0]
    assert "clock_domain_mismatch" in SP.geometry_input_check(100.0, 100.0, 100.0, frame_clock="wall_unix_s")[1][0]
    hist = [(99.8, 0, 0, 0), (99.9, 5.0, 0, 0)]
    ok, why = SP.geometry_input_check(100.0, 100.0, 100.0, odom_hist=hist)
    assert not ok and any(w.startswith("odom_jump") for w in why)
    assert any("odom_time_backwards" in w for w in SP.geometry_input_check(100.0, 100.0, 100.0, odom_hist=[(100, 0, 0, 0), (99, 0, 0, 0)])[1])
    for z in (0.0, -1.0, float("nan"), float("inf"), None):
        assert not SP.valid_depth_m(z)


def test_T21b_p2_tick_defers_on_odom_reset(env):
    node, clk = make_node(env, F.FakeConsole(), stage="P2")
    src = F.FakeSource()
    src.odom_hist = [(100.0 + 0.1 * i, 0.3 * 0.1 * i, 0.0, 0.0) for i in range(5)] + [(100.5, 9.0, 0.0, 0.0)]
    node.last_bgr, node.last_tick = np.zeros((480, 640, 3), np.uint8), -1e9
    node.p2_tick(src, 100.5, 10.0)
    assert any("odom_jump" in w for w in node.status["p2_defer"]["why"])


# =============================================================================================== T22 T23
def test_T22_event_loop_runs_during_long_device_job(env):
    con = F.FakeConsole(job_s=1.8)                                    # 18 s aim scaled 1/10 (real threads)
    node, clk = make_node(env, con)
    node.clock = time.monotonic
    os.environ["SCI_AIM_TIMEOUT_S"] = "5"
    src = F.FakeSource()
    node.threaded = True
    node.start_event_thread(src)
    try:
        node.proposals.put(dict(id=7, stop=F.a3_stop(), stamp=100.0, pose=[0, 0, 0], bgr=np.zeros((480, 640, 3), np.uint8),
                                latest_frame=None))
        t0 = time.time()
        while time.time() - t0 < 3 and not node.device.busy:
            time.sleep(0.02)
        assert node.device.busy
        seen = []
        t1 = time.time()
        while node.device.busy and time.time() - t1 < 4:
            seen.append(json.loads((N.SHM / "status.json").read_text())["status_mono_ns"])
            time.sleep(0.2)
        assert len(set(seen)) >= 5                                    # status kept publishing during the 1.8 s job
        assert node.propose_event(F.a3_stop(), 0, (0, 0, 0), None, None) is False   # no new STOP while busy
    finally:
        node.stop_event_thread()
        os.environ["SCI_AIM_TIMEOUT_S"] = "0.6"


def test_T23_bounded_queues_with_drop_reasons(env):
    s = LatestSlot()
    for i in range(10):
        s.put(dict(id=i))
    assert s.depth() == 1 and s.dropped == 9 and s.drop_log[-1]["why"] == "superseded_by_newer"
    node, _ = make_node(env, F.FakeConsole())
    src = F.FakeSource()
    for i in range(1000):
        node.note_frame(src, 100 + i * 0.07, i)
    assert len(node.frame_meta) == 200
    st = EventStore(env / "es", "run", hot_max=20)
    for i in range(30):
        st.new_event(i + 1, dict(t=i))
    assert len(st.hot()) == 20 and st.get("run#ev001")["event_id"] == 1     # disk record exact after eviction


# =============================================================================================== T24
def _pipes():
    r1, w1 = os.pipe()        # client reads r1 (worker writes w1)
    r2, w2 = os.pipe()        # client writes w2 (worker reads r2)
    return PipeTransport(r1, w2), w1, r2


def test_T24_siglip_transport_deadlines_and_late_reply():
    import pickle
    import struct
    tr, worker_w, worker_r = _pipes()
    t0 = time.monotonic()
    with pytest.raises(TransportTimeout):
        tr.request(dict(x=1), 0.3)                                       # read stall
    assert time.monotonic() - t0 < 1.0
    b = pickle.dumps(dict(sim=[1], sec=0.1))
    os.write(worker_w, struct.pack("<I", len(b)) + b)                    # the LATE reply arrives
    with pytest.raises(TransportError):
        tr.request(dict(x=2), 0.3)                                       # poisoned: late reply never consumed as answer
    tr2, ww2, wr2 = _pipes()
    t0 = time.monotonic()
    with pytest.raises(TransportTimeout):
        tr2.send(dict(blob=b"x" * (4 << 20)), 0.3)                       # write stall (nobody reads the worker stdin)
    assert time.monotonic() - t0 < 1.0
    tr3, ww3, wr3 = _pipes()
    os.write(ww3, struct.pack("<I", 5) + b"\x00\x01garb")
    with pytest.raises(TransportError):
        tr3.recv(0.5)                                                    # invalid reply
    src = (WS / "src" / "sci_visual_observation" / "frontend.py").read_text()
    assert "PipeTransport(self.p.stdout.fileno(), self.p.stdin.fileno())" in src and "self.tr.request(" in src


# =============================================================================================== T25 T26
def test_T25_stale_sport_zero_is_not_stopped_and_states_are_separate():
    m = SP.StopMonitor(0.0)
    stale = SP.BodySample(0.0, -5.0, "sportmodestate.velocity_hypot_xy")
    for t in np.arange(0.1, 3.0, 0.1):
        m.update(t, primary=stale, aux_lin=0.3, aux_yaw=0.0)          # aux odom says moving
    assert m.state == "WAIT_BASE_STOP"
    unverified = SP.BodySample(0.0, 3.0, "leg_joint_velocity")
    m.update(3.0, primary=unverified, aux_lin=0.3, aux_yaw=0.0)
    assert m.state == "WAIT_BASE_STOP"
    for t in np.arange(3.1, 4.0, 0.1):
        m.update(t, primary=SP.BodySample(0.0, t, "sportmodestate.velocity_hypot_xy"))
    assert m.state in ("BASE_STOPPED", "WAIT_IMAGE_STABLE")
    m.update(4.0, primary=SP.BodySample(0.0, 4.0, "sportmodestate.velocity_hypot_xy"), servo_settled=False)
    assert m.state == "WAIT_IMAGE_STABLE"                                # separate optical condition
    assert SP.DEFAULT_STOP_CFG["aux_lin_mps"] == 0.05 and SP.DEFAULT_STOP_CFG["aux_hold_s"] == 0.7


def test_T26_wait_timeouts_apply_on_every_branch():
    import policy as P
    t = [0.0]
    f = P.EventFSM(1, dict(t=0.0), P.load_cfg(None), clock=lambda: t[0])
    for k in range(0, 260):                                            # stable but never held 0.7 s (flapping)
        t[0] = k * 0.1
        f.on_odom(0.0 if k % 5 else 0.3, 0.0)
        if f.state != "WAIT_STABLE":
            break
    assert f.state == "FAILED" and "timeout" in f.log[-1]["reason"]
    g = P.EventFSM(2, dict(t=0.0), P.load_cfg(None), clock=lambda: t[0])
    for k in range(260):                                               # 10-15 ev5 pattern: still at 19.8 s, held past 20 s
        t[0] = k * 0.1
        g.on_odom(0.3 if t[0] < 19.8 else 0.0, 0.0)
    assert g.state == "FAILED" and not any(x["state"] == "STOP_CONFIRMED" for x in g.log)
    m = SP.StopMonitor(0.0)
    for k in range(260):
        m.update(k * 0.1, aux_lin=0.0 if k % 5 else 0.3, aux_yaw=0.0)
    assert m.state == "BASE_STOP_TIMEOUT"
    m2 = SP.StopMonitor(0.0)
    for k in range(10):
        m2.update(k * 0.1, primary=SP.BodySample(0.0, k * 0.1, "sportmodestate.velocity_hypot_xy"))
    for k in range(10, 80):
        m2.update(k * 0.1, primary=SP.BodySample(0.0, k * 0.1, "sportmodestate.velocity_hypot_xy"), servo_settled=False)
    assert m2.state == "IMAGE_STABLE_TIMEOUT"


# =============================================================================================== T27 T28
def test_T27_prediction_from_latest_pose_once_no_future():
    hist = [(100.0, 0.0, 0, 0), (100.5, 0.25, 0, 0), (101.0, 0.5, 0, 0), (102.0, 1.0, 0, 0)]
    ref = SP.decision_reference(hist, now_stamp=101.0, frame_stamp=99.8)
    assert ref["pose"] == (0.5, 0, 0) and ref["stamp"] == 101.0           # 102.0 is in the future: not used
    assert abs(ref["frame_age_s"] - 1.2) < 1e-9
    import timing as T
    from features import Geometry
    p2 = T.P2Policy(T.P.load_cfg(None), T.load_p2(None), Geometry())
    pose, adv = p2.predict_pose(ref["pose"], 0.5)
    assert abs(adv - (0.5 * 1.5 + 0.3)) < 1e-9                              # no extra speed*frame_age added
    assert abs(pose[0] - (0.5 + adv)) < 1e-9


def test_T28_no_feasible_window_defers_and_limits_unchanged():
    import timing as T
    from features import Geometry
    g = Geometry()
    lim = tuple(g.pan_lim)
    p2 = T.P2Policy(T.P.load_cfg(None), T.load_p2(None), g)
    # sign on a far-left wall 4.05 m away: range<=4 m needs bearing ~90 deg, which is outside the UNCHANGED pan range
    tr = dict(spot=1, state="VERIFIED", n_verify=3, xy=np.array([5.0, 4.05]), up=0.3, h_m=0.15, n_xy=np.array([0.0, -1.0]),
              last_t=0.0, box=[0, 0, 1, 1], tile=[0, 0, 1, 1], z_last=2.0, sim=0.2, t_verified=0.0)
    p2.tracks[1] = tr
    stop, summ = p2.tick(0.5, (0.0, 0.0, 0.0), 0.4)
    assert stop is None and "NO_FEASIBLE_STOP_WINDOW" in (summ[0]["why"] or []) and tuple(g.pan_lim) == lim
    stop, summ = p2.tick(0.6, (0.0, 0.0, 0.0), 0.4, defer_why=["odom_jump"])
    assert stop is None


# =============================================================================================== T37
def test_T37_wrapper_resolves_candidate_modules_and_guarded_run_dir(env, monkeypatch):
    import e5_onboard as OB
    cand = os.path.realpath(WS)
    for m in (OB.N, OB.SRC, OB.PL, C):
        assert os.path.realpath(m.__file__).startswith(cand), m.__file__
    assert OB.N.Node is OB.GatedNode and OB.N.PanTiltPlanner is OB.Planner
    assert not hasattr(OB.Planner, "_state_full") and "job0" not in Path(OB.__file__).read_text()
    src = ast.parse(Path(OB.N.__file__).read_text())
    names = {n.name for n in ast.walk(src) if isinstance(n, ast.FunctionDef)}
    assert {"event_step", "on_device_result", "attach_readings", "_release", "enqueue_reader_job"} <= names
    # the real GatedNode builds with fakes and runs one observation end-to-end through the candidate code
    # E5RT2 replacement: SCI_ARM_WITHOUT_UI=1 no longer arms (RP01-03). The run needs an explicit, live operator grant in
    # the gate file; this test checks module wiring, so the hand-off geometry gate (tested in tests/e5rt2) is set off here.
    import e5rt.gates as RG
    gp = env / "gate_t37.json"
    gp.write_text(json.dumps(RG.gate_payload("walking", time.time(), grant=dict(id="t37", expires_t=time.time() + 600),
                                             operator_abort=False)))
    monkeypatch.setattr(OB, "GATE", gp)
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    monkeypatch.setattr(OB, "E5RT2_REQUIRE_CONFIRMED_GEOMETRY", False)
    con = F.FakeConsole()
    planner = OB.Planner(N.Geometry(), console="http://fake", execute=True, http_get=con.http_get)
    clk = Clock()
    g = OB.GatedNode(F.args(env, run_name="gated"), deps=dict(frontend=F.FakeFrontend(), monitor=F.FakeMon(), planner=planner,
                                                                 trial_pointer=None,
                                                                 clock=clk, threaded=False,
                                                                 body_speed=lambda now: SP.BodySample(0.0, now, "sportmodestate.velocity_hypot_xy")))
    run_event(g, clk)
    assert g.store.get(g.fsm.key)["capture_status"] == CS.CAPTURE_QUALITY_OK
    assert json.loads((g.run / "code_identity.json").read_text())["modules"]
    monkeypatch.setattr(N, "WS", Path("/home/unitree/sci_ws"))
    with pytest.raises(RuntimeError):
        N._guard_run_dir(Path("/home/unitree/sci_ws/experiments/x"))


def test_T37b_gate_heartbeat_blocks_physical_not_results(tmp_path, monkeypatch):
    import e5_onboard as OB
    monkeypatch.delenv("SCI_ARM_WITHOUT_UI", raising=False)
    p = tmp_path / "gate.json"
    p.write_text(json.dumps(dict(active=True, why="walking")))
    now = time.time()
    g = OB.gate_state(now, path=p, mode="walking", gate_on=True)
    # E5RT2 replacement: legacy 'active' (walking) observes but is NOT an actuator grant (RP01/RP02); was `g["arm"]`.
    assert g["observe"] and not g["arm"] and "NO_EXPLICIT_ACTUATOR_GRANT" in g["reasons"]
    g = OB.gate_state(now + 10, path=p, mode="walking", gate_on=True)            # heartbeat lost: not latched
    assert not g["observe"] and not g["arm"]
    g = OB.gate_state(now + 10, path=p, mode="static_bench", gate_on=True)       # explicit static observation mode
    assert g["observe"] and not g["arm"]
    p.write_text(json.dumps(dict(active=False, armed=True, why="bench")))
    g = OB.gate_state(time.time(), path=p, mode="static_bench", gate_on=True)
    assert g["observe"] and g["arm"]


def test_T04c_trial_pointer_identity(env):
    ptr = env / "rt_live" / "out" / "trial_current.json"
    C.atomic_write_json(ptr, C.trial_record("trial-x", "exp_1", 3, "RUNNING"))
    node, clk = make_node(env, F.FakeConsole(), trial_pointer=str(ptr))
    run_event(node, clk)
    r = rec(node)
    assert r["trial_id"] == "trial-x" and r["generation_id"] == 3
    j = json.loads(next((node.run / "jobs" / "pending").glob("*.json")).read_text())
    assert j["trial_id"] == "trial-x" and j["generation_id"] == 3
    n2, c2 = make_node(env, F.FakeConsole(), run="noptr", trial_pointer=str(env / "absent.json"))
    run_event(n2, c2)
    r2 = n2.store.get(n2.fsm.key)
    assert r2["trial_id"] is None and "absent" in r2["trial_id_null_reason"]
