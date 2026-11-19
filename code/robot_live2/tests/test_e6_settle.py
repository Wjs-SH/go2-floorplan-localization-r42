"""STAGING (2026-10-30): E6 MANUAL_SETTLE gate (e6_settle) + its insertion in the REAL e6_pipeline, with a FAKE aim console (no serial
port, no camera, no robot). Every device command the pipeline would send goes to FakeClient.calls, so 'no pan-tilt before the
approval' is checked on the actual call list. Approval B is a stub callable here (the real record is never written)."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

CAND = Path(__file__).resolve().parents[1] / "candidate"
for p in (CAND / "live2", CAND / "live1", CAND / "src" / "sci_visual_observation", CAND / "src"):
    sys.path.insert(0, str(p))
import e6_settle as ST  # noqa: E402

STOPPED = dict(stopped=True, speed_mps=0.02, yaw_rate_dps=0.3, why="STOPPED")
MOVING = dict(stopped=False, speed_mps=0.45, yaw_rate_dps=4.0, why="MOVING speed 0.45 m/s yaw 4.0 deg/s")
B_OK, B_NO = dict(valid=True, why=None), dict(valid=False, why="APPROVAL_FOR_OTHER_BOOT")


def obs(oid="O1", t=None):
    t = time.time() if t is None else t
    return dict(observation_id=oid, source="AUTO_E5", detection_timestamp=t - 9, created_timestamp=t - 7, stopped_wall=t - 1,
                odom_pose_at_stop=dict(t=t - 1, x=1.0, y=2.0, yaw=0.1),
                redetection=dict(status="REDETECTED", pixel=[320, 200], depth_m=4.0, pantilt_command=dict(inside_limits=True)))


class Env:
    """mutable world the gate reads: stop rule, body-IMU window, camera shake, approval B, robot-moved flag."""
    def __init__(self, tmp, stop=STOPPED, imu=None, shake=None, apb=B_OK):
        self.stop, self.shake, self.apb, self.moved = stop, shake or dict(shift_px=2.0), apb, False
        self.rt = tmp / "runtime"
        self.state = self.rt / "state"
        self.state.mkdir(parents=True)
        self.set_imu(**(imu or dict(gyro_rms_dps=3.0, gyro_peak_dps=6.0, acc_norm_std=0.10)))

    def set_imu(self, **kw):
        rd = self.rt / "fastlio" / "body_test"
        rd.mkdir(parents=True, exist_ok=True)
        import os
        (self.rt / "owned").mkdir(exist_ok=True)
        (self.rt / "owned" / "fastlio.json").write_text(json.dumps(dict(
            pid=os.getpid(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), input="body", body_run_dir=str(rd))))
        self._imu = kw
        self.tick_imu()

    def tick_imu(self):
        rd = self.rt / "fastlio" / "body_test"
        with open(rd / "bridge_health.jsonl", "a") as f:
            f.write(json.dumps(dict(wall_ns=int(time.time() * 1e9), health="READY", publish_hz=499, **self._imu)) + "\n")

    def gate(self, **kw):
        return ST.SettleGate(self.state, self.rt, **kw)

    def wait(self, g, o, exp="X1"):
        def stop_check():
            self.tick_imu()
            return dict(self.stop)
        return g.wait(o, exp, stop_check, lambda: dict(self.shake), lambda: dict(self.apb),
                      lambda _o: (self.moved, "robot moved 0.6 m" if self.moved else None))

    def cmd(self, action, oid="O1", exp="X1", delay=0.6, rid=None):
        def go():
            time.sleep(delay)
            ST.append_command(self.state, dict(request_id=rid or f"r{time.time()}", action=action, observation_id=oid,
                                               experiment_id=exp, actor="test", wall=time.time()))
        threading.Thread(target=go, daemon=True).start()


def replies(env):
    return [json.loads(x) for x in open(env.state / "e6_settle_events.jsonl") if '"COMMAND"' in x]


# ---------------------------------------------------------------- gate
def test_01_waiting_state_written(tmp_path):
    env = Env(tmp_path); g = env.gate()
    env.cmd("cancel", delay=1.0)
    out, why, rec = env.wait(g, obs())
    ev = [json.loads(x) for x in open(env.state / "e6_settle_events.jsonl")]
    assert ev[0]["kind"] == "STATE" and ev[0]["state"] == "WAITING_FOR_SETTLE_APPROVAL" and ev[0]["observation_id"] == "O1"
    assert rec["message"] == ST.MSG_WAIT and rec["times"]["t_stop_request"] and rec["times"]["t_stop_confirmed"]
    assert out == "USER_CANCELLED"


def test_02_approve_while_moving_refused(tmp_path):
    env = Env(tmp_path, stop=MOVING); g = env.gate()
    env.cmd("approve", delay=0.5); env.cmd("cancel", delay=1.6)
    out, _, _ = env.wait(g, obs())
    r = replies(env)
    assert out == "USER_CANCELLED" and r[0]["action"] == "approve" and not r[0]["accepted"] and ST.MSG_UNSTABLE in r[0]["why"]


def test_03_approve_while_shaking_refused(tmp_path):
    env = Env(tmp_path, imu=dict(gyro_rms_dps=12.0, gyro_peak_dps=40.0, acc_norm_std=0.9)); g = env.gate()
    env.cmd("approve", delay=0.5); env.cmd("cancel", delay=1.6)
    out, _, _ = env.wait(g, obs())
    r = replies(env)
    assert not r[0]["accepted"] and "자이로 RMS" in r[0]["why"] and out == "USER_CANCELLED"


def test_03b_camera_shake_refused(tmp_path):
    env = Env(tmp_path, shake=dict(shift_px=20.0)); g = env.gate()
    env.cmd("approve", delay=0.5); env.cmd("cancel", delay=1.6)
    env.wait(g, obs())
    assert "카메라 영상 흔들림" in replies(env)[0]["why"]


def test_04_stable_approve_passes(tmp_path):
    env = Env(tmp_path); g = env.gate()
    env.cmd("approve", delay=0.6)
    out, why, rec = env.wait(g, obs())
    assert out == "E6_APPROVED" and rec["times"]["t_manual_approval"] and rec["times"]["t_settle_ready"]
    assert rec["approval_check"]["approval_b"]["valid"] and rec["approval_check"]["imu"]["available"]


def test_08_stale_event_refused(tmp_path):
    env = Env(tmp_path); g = env.gate()
    env.cmd("approve", oid="O_OLD", delay=0.4); env.cmd("approve", exp="X_PREVIOUS", delay=0.6); env.cmd("cancel", delay=1.4)
    out, _, _ = env.wait(g, obs())
    r = replies(env)
    assert [x["accepted"] for x in r[:2]] == [False, False] and all("STALE_EVENT" in x["why"] for x in r[:2]) and out == "USER_CANCELLED"


def test_09_robot_moved_invalidates(tmp_path):
    env = Env(tmp_path); g = env.gate()
    threading.Timer(0.6, lambda: setattr(env, "moved", True)).start()
    out, why, _ = env.wait(g, obs())
    assert out == "ROBOT_MOVED"


def test_11_no_approval_b_refused(tmp_path):
    env = Env(tmp_path, apb=B_NO); g = env.gate()
    env.cmd("approve", delay=0.5); env.cmd("cancel", delay=1.6)
    out, _, rec = env.wait(g, obs())
    r = replies(env)
    assert not r[0]["accepted"] and "APPROVAL_B_REQUIRED" in r[0]["why"] and out == "USER_CANCELLED"
    assert rec["approval_b"]["valid"] is False


def test_timeout_never_auto_runs(tmp_path):
    env = Env(tmp_path); g = env.gate(wait_max_s=1.2)
    out, why, rec = env.wait(g, obs())
    assert out == "SETTLE_APPROVAL_EXPIRED" and rec["times"]["t_manual_approval"] is None and "자동 실행 안 함" in why


def test_commands_before_bridge_start_ignored(tmp_path):
    env = Env(tmp_path)
    ST.append_command(env.state, dict(request_id="old", action="approve", observation_id="O1", experiment_id="X1", wall=time.time()))
    g = env.gate(wait_max_s=1.0)                     # created AFTER the old line -> it is never read
    out, _, _ = env.wait(g, obs())
    assert out == "SETTLE_APPROVAL_EXPIRED"


def test_imu_not_available_in_l1(tmp_path):
    env = Env(tmp_path)
    (env.rt / "owned" / "fastlio.json").write_text(json.dumps(dict(pid=1, boot_id="other", input="l1")))
    v = ST.imu_vibration(env.rt)
    assert v["available"] is False and "L1" in v["why"]
    ok, bad, det = ST.stability(STOPPED, v, dict(shift_px=3.0))
    assert ok and det["imu"]["available"] is False


def test_stale_body_record_not_used(tmp_path):
    env = Env(tmp_path)
    rec = json.loads((env.rt / "owned" / "fastlio.json").read_text()); rec["pid"] = 999999
    (env.rt / "owned" / "fastlio.json").write_text(json.dumps(rec))
    assert ST.imu_vibration(env.rt)["available"] is False


# ---------------------------------------------------------------- pipeline (fake aim console)
class FakeClient:
    def __init__(self, aim_status="FAILED", capture=False):
        self.calls, self.aim_status, self.capture = [], aim_status, capture

    def run(self, path, payload, *a, **k):
        self.calls.append((path, dict(payload), round(time.time(), 2)))
        if path == "aim":
            return dict(status=self.aim_status, device_state="IDLE", reason_code="FAKE_DEVICE",
                        capture=dict(capture_id="c1") if self.capture else {})
        return dict(status="COMPLETED", device_state="IDLE")

    def fetch_capture(self, cid):
        import cv2
        import numpy as np
        img = np.full((720, 1280, 3), 200, np.uint8)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def probe(self):
        return dict(ok=True)

    def shake(self):
        return dict(shift_px=1.5)

    def _get_json(self, path, q=None, timeout=2.0):
        return dict(shift_px=1.5) if "shake" in path else dict(ok=True)


def pipeline(env, client, apb=None, wait_max_s=300.0):
    import e6_pipeline as P
    gate = env.gate(wait_max_s=wait_max_s)
    settle = dict(gate=gate, approval=lambda: dict(apb or env.apb), moved=lambda o: (env.moved, None), ended=lambda: False)
    hw = lambda kind, **kw: ((apb or env.apb)["valid"], None if (apb or env.apb)["valid"] else "approval B invalid")   # noqa: E731
    pipe = P.E6Pipeline(client, hw, lambda: dict(env.stop), env.state / "caps", experiment_id="X1", settle=settle,
                        defer_reading=True, envelope=None)
    return pipe, gate


def run_bg(pipe, o):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("rec", pipe.run(o, 1)), daemon=True)
    th.start()
    return th, box


def test_05_no_device_command_before_approval_then_once(tmp_path):
    env = Env(tmp_path); cl = FakeClient()
    pipe, gate = pipeline(env, cl)
    th, box = run_bg(pipe, obs())
    time.sleep(1.5)
    assert cl.calls == [], cl.calls                                      # waiting: nothing sent to the pan-tilt
    assert ST.read_snapshot(env.state)["state"] == "WAITING_FOR_SETTLE_APPROVAL"
    env.cmd("approve", delay=0.0, rid="a1"); env.cmd("approve", delay=0.05, rid="a2")   # double tap
    th.join(30)
    aims = [c for c in cl.calls if c[0] == "aim"]
    assert len(aims) == 1, cl.calls                                      # 06: E6 called once
    rec = box["rec"]
    assert rec["settle"]["times"]["t_manual_approval"] <= aims[0][2] + 0.01
    assert rec["e6_trace"][1]["state"] == "WAIT_SETTLE_APPROVAL" and rec["e6_trace"][1]["outcome"] == "E6_APPROVED"
    # 13: fake device returns no capture -> recorded failure, HOME sent, no exception
    assert rec["status"] == "NO_CAPTURE" and any(c[0] == "home" for c in cl.calls)
    # the leftover second approve is answered STALE for the next observation, never a second aim
    env.cmd("cancel", oid="O2", delay=0.5)
    th2, box2 = run_bg(pipe, obs("O2"))
    th2.join(30)
    assert len([c for c in cl.calls if c[0] == "aim"]) == 1 and box2["rec"]["status"] == "USER_CANCELLED"
    # the 2nd tap is answered either in the same poll (DUPLICATE) or by the next observation's gate (STALE_EVENT) — never executed
    assert any(("STALE_EVENT" in (r.get("why") or "") or "DUPLICATE" in (r.get("why") or "")) and r["action"] == "approve" and not r["accepted"]
               for r in replies(env)), replies(env)


def test_09b_cancel_no_device_command(tmp_path):
    env = Env(tmp_path); cl = FakeClient()
    pipe, _ = pipeline(env, cl)
    env.cmd("cancel", delay=0.6)
    rec = pipe.run(obs(), 1)
    assert cl.calls == [] and rec["status"] == "USER_CANCELLED"


def test_10_robot_moves_before_aim(tmp_path):
    env = Env(tmp_path); cl = FakeClient()
    pipe, _ = pipeline(env, cl)
    threading.Timer(0.6, lambda: setattr(env, "moved", True)).start()
    rec = pipe.run(obs(), 1)
    assert cl.calls == [] and rec["status"] == "ROBOT_MOVED"


def test_10b_not_stopped_at_e6_start(tmp_path):
    env = Env(tmp_path, stop=MOVING); cl = FakeClient()
    pipe, _ = pipeline(env, cl)
    rec = pipe.run(obs(), 1)                                             # existing rule: STOP_CONFIRMED fails before the gate
    assert cl.calls == [] and rec["status"] == "ROBOT_NOT_STOPPED"


def test_11b_no_approval_b_no_device_command(tmp_path):
    env = Env(tmp_path, apb=B_NO); cl = FakeClient()
    pipe, _ = pipeline(env, cl, wait_max_s=2.0)
    env.cmd("approve", delay=0.5)
    rec = pipe.run(obs(), 1)
    assert cl.calls == [] and rec["status"] == "SETTLE_APPROVAL_EXPIRED"


def test_12_approval_b_fake_hardware_runs(tmp_path):
    env = Env(tmp_path); cl = FakeClient(aim_status="COMPLETED", capture=True)
    pipe, _ = pipeline(env, cl)
    env.cmd("approve", delay=0.6)
    rec = pipe.run(obs(), 1)
    paths = [c[0] for c in cl.calls]
    assert paths[0] == "aim" and "home" in paths, paths
    assert rec["status"] in ("TARGET_NOT_IN_IMX477", "CAPTURE_STORED", "NO_CAPTURE"), rec["status"]


def test_auto_mode_unchanged(tmp_path):
    """settle=None (AUTO): no WAIT_SETTLE_APPROVAL state, the aim follows STOP_CONFIRMED directly (existing path)."""
    import e6_pipeline as P
    env = Env(tmp_path); cl = FakeClient()
    pipe = P.E6Pipeline(cl, lambda kind, **kw: (True, None), lambda: dict(STOPPED), env.state / "caps", experiment_id="X1",
                        defer_reading=True)
    rec = pipe.run(obs(), 1)
    assert "WAIT_SETTLE_APPROVAL" not in [t["state"] for t in rec["e6_trace"]] and cl.calls[0][0] == "aim"


def test_14_no_reading_no_e7_success(tmp_path):
    """a precision record without any OCR / VLM line is not semantic evidence: the E6->E7 link never reports a FIX from it."""
    import e6e7_link as LK

    class Tr:
        store = type("S", (), dict(rows=[], refresh=lambda self: None, session=lambda self: dict(odom_session_id="s")))()
        def alive(self): return []
        def apply_semantic(self, *a, **k): return 0
        def light(self, d): return None
    st = tmp_path / "st"; st.mkdir()
    (st / "e6_precision_results.jsonl").write_text(json.dumps(dict(kind="LIVE2_E6_PRECISION", observation_id="O1", status="NO_CAPTURE",
                                                                 ocr_lines=[], vlm=None, e6_trace=[], wall=time.time())) + "\n")
    link = LK.E6E7Link(st, Tr(), policy="SINGLE_DIAGNOSTIC")
    link.tick(None, None)
    time.sleep(0.5)
    assert (link.snapshot() or {}).get("global_decision") != "GLOBAL_CONFIRMED"


# ---------------------------------------------------------------- time metrics
def _exp(tmp, waits, fix_at=None, cancel=None, t0=1000.0):
    d = tmp / "X"; (d / "state").mkdir(parents=True)
    (d / "experiment.json").write_text(json.dumps(dict(exp_id="X", created_wall=t0, e6_mode="MANUAL_SETTLE")))
    ob, ev = [], []
    for i, (req, appr) in enumerate(waits):
        oid = f"O{i}"
        ob.append(dict(observation_id=oid, detection_timestamp=req - 2, created_timestamp=req, stopped_wall=req + 3))
        ev.append(dict(kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL", observation_id=oid, wall=req + 3.1, mono=1))
        ev.append(dict(kind="COMMAND", action="approve", accepted=True, observation_id=oid, wall=appr, mono=2))
        ev.append(dict(kind="STATE", state="E6_APPROVED", observation_id=oid, wall=appr, mono=2))
    if cancel:
        ob.append(dict(observation_id="OC", created_timestamp=cancel[0], stopped_wall=cancel[0] + 2))
        ev.append(dict(kind="STATE", state="USER_CANCELLED", observation_id="OC", wall=cancel[1], mono=3))
    (d / "state" / "e6_observations.jsonl").write_text("".join(json.dumps(o) + "\n" for o in ob))
    (d / "state" / "e6_settle_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    if fix_at:
        (d / "state" / "e7_state.json").write_text(json.dumps(dict(global_decision="GLOBAL_CONFIRMED", written_wall=fix_at, path="E8_PAIR")))
    return d


@pytest.mark.parametrize("wait", [0.0, 5.0, 30.0])
def test_tm_single_wait(tmp_path, wait):
    d = _exp(tmp_path, [(1100.0, 1100.0 + wait)], fix_at=1300.0)
    before = (d / "state" / "e6_settle_events.jsonl").read_bytes()
    m = ST.time_metrics(d)
    assert m["TTF_raw_s"] == 300.0 and m["TTF_adjusted_s"] == pytest.approx(300.0 - wait) and m["observations"][0]["T_manual_wait_s"] == wait
    assert (d / "state" / "e6_settle_events.jsonl").read_bytes() == before            # raw records untouched


def test_tm_multiple_overlapping_waits_union(tmp_path):
    d = _exp(tmp_path, [(1100.0, 1120.0), (1110.0, 1130.0), (1200.0, 1210.0)], fix_at=1300.0)
    m = ST.time_metrics(d)
    assert m["sum_manual_wait_s"] == 40.0 and m["TTF_adjusted_s"] == 260.0           # [1100,1130] + [1200,1210]


def test_tm_cancel_not_subtracted(tmp_path):
    d = _exp(tmp_path, [(1100.0, 1105.0)], fix_at=1300.0, cancel=(1150.0, 1170.0))
    m = ST.time_metrics(d)
    c = next(o for o in m["observations"] if o["observation_id"] == "OC")
    assert m["TTF_adjusted_s"] == 295.0 and c["outcome"] == "USER_CANCELLED" and c["T_wait_not_subtracted_s"] == 20.0


def test_tm_no_fix_no_ttf(tmp_path):
    m = ST.time_metrics(_exp(tmp_path, [(1100.0, 1105.0)]))
    assert m["TTF_raw_s"] is None and m["TTF_adjusted_s"] is None and "NO_FIX" in m["why"]


def test_tm_experiment_aborted(tmp_path):
    d = _exp(tmp_path, [])
    (d / "state" / "e6_observations.jsonl").write_text(json.dumps(dict(observation_id="OA", created_timestamp=1100.0, stopped_wall=1102.0)) + "\n")
    (d / "state" / "e6_settle_events.jsonl").write_text(json.dumps(dict(kind="STATE", state="EXPERIMENT_ENDED", observation_id="OA", wall=1110.0, mono=1)) + "\n")
    m = ST.time_metrics(d)
    assert m["TTF_raw_s"] is None and m["observations"][0]["outcome"] == "EXPERIMENT_ENDED"


def test_tm_wait_after_fix_clipped(tmp_path):
    """session restart / later events: only the part of a manual wait before t_fix is subtracted."""
    d = _exp(tmp_path, [(1100.0, 1110.0), (1290.0, 1320.0)], fix_at=1300.0)
    m = ST.time_metrics(d)
    assert m["sum_manual_wait_s"] == 20.0 and m["TTF_adjusted_s"] == 280.0
