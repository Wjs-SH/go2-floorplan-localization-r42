"""R1 main-session cross-worker integration tests (SIMULATED; no device/ROS/model/network).

I1  E5 client (W-D AimClient via the REAL candidate Node) <-> console server logic (W-B DeviceOwner + route) with an
    in-memory transport and fake hardware: request_id/capture_id round trip, manual job never accepted, release state.
I2  E5 reader job (written by the real Node) -> W-C ReaderCore -> done file -> E5 attaches to the exact event.
I3  Blurred capture end to end: CAPTURE_QUALITY_FAIL in E5, terminal CAPTURE_QUALITY_FAIL in reader, E6 never called.
I4  Legacy console (no protocol) through the real chain -> PROTOCOL_MISMATCH, no device job submitted.
I5  Invalid depth packet (W-C e5_packet + W-A geometry/E6) -> null observation with reason -> E6 INPUT_INVALID.
"""
import json
import os
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
HERE = Path(__file__).resolve().parent
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_common", WS / "src" / "sci_pantilt",
          WS / "src" / "sci_text_reading", WS / "src" / "sci_semantic_localization",
          WS / "external" / "sci_event" / "code", WS / "external" / "sci_event" / "e5_onboard", HERE):
    sys.path.insert(0, str(p))

import r1_contracts as C  # noqa: E402
import e5_node as N  # noqa: E402
import r1_device_owner as R1O  # noqa: E402
import r1_jobs as J  # noqa: E402
import r1_fakes_e5 as F  # noqa: E402
import test_r1_e5 as TE  # noqa: E402  (make_node / drive helpers of W-D, real candidate Node)
import test_r1_reader as TR  # noqa: E402  (FakeOcr / ENGINE / CFG of W-C)

CS, RS, RR = C.CaptureStatus, C.ReleaseStatus, C.ReadReason


class ImgHW:
    """Fake hardware for the REAL DeviceOwner: decodable JPEG frames with new sequence numbers."""

    def __init__(self, sharp=True, hold=None):
        self.sharp, self.seq, self.hold, self.steps = sharp, 0, hold, []

    def _step(self, name):
        self.steps.append(name)
        if self.hold is not None:
            self.hold.wait(5.0)

    def plan_aim(self, p):
        return dict(refused=None, pan=100.0, tilt=150.0, clipped=False)

    def move(self, pan, tilt, cancel):
        self._step("move")

    def home(self, cancel):
        self._step("home")

    def focus(self, cancel, p):
        self._step("focus")
        return dict(status="AF_CONVERGED", ok=True)

    def frame_seq(self):
        return self.seq

    def capture_fresh(self, after_seq, timeout_s):
        self.seq += 1
        return dict(jpeg=F.jpeg(sharp=self.sharp, seed=self.seq, w=1280, h=720), seq=self.seq,
                    rx_mono_ns=time.monotonic_ns(), w=1280, h=720)

    def stop_actuators(self):
        self.steps.append("stop_actuators")


class OwnerConsole:
    """In-memory HTTP transport onto the real console routing (aim_console delegates /state to owner.state())."""

    def __init__(self, tmp, hw):
        self.owner = R1O.DeviceOwner(hw, str(tmp / "console_captures"), log=lambda *a: None)
        self.urls = []

    def http_get(self, url, timeout):
        u = urllib.parse.urlparse(url)
        self.urls.append(u.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/state":
            return json.dumps(dict(self.owner.state(), status="idle", log=[])).encode()
        r = R1O.route(self.owner, u.path, q)
        if r is None:
            raise IOError("404 " + u.path)
        return r[2]


def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "WS", tmp_path / "ws")
    monkeypatch.setattr(N, "SHM", tmp_path / "shm")
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    monkeypatch.setenv("SCI_AIM_TIMEOUT_S", "3.0")
    monkeypatch.setenv("SCI_AIM_CANCEL_GRACE_S", "1.0")
    return tmp_path


def _run_event(node, clk):
    src = F.FakeSource()
    node.start_event(F.a3_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None,
                     latest_frame=dict(frame_id=9, header_stamp=101.0))
    for _ in range(400):                                    # real worker thread + real owner thread: wall-time bounded
        clk.t += 0.1
        node.event_step(src, None)
        time.sleep(0.01)
        r = node.store.get(node.fsm.key)
        if r.get("release") and not node.device.busy:
            break
    node.event_step(src, None)
    return src, node.store.get(node.fsm.key)


def _reader(run, lines, localize):
    core = J.ReaderCore(run, TR.CFG, ocr_fn=TR.FakeOcr(lines), engine_info=TR.ENGINE, vlm_fn=None,
                        admit_fn=lambda need: (False, "memory: fake"), localize_fn=localize,
                        clock=TR.Clock(), wall=lambda: 1.7e9, boot_id="BOOT-I")
    for _ in range(20):
        core.step()
        if core.idle():
            break
    return core


def test_I1_client_and_real_console_logic_round_trip(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    con = OwnerConsole(tmp_path, ImgHW(sharp=True))
    # a manual focus job finishes on the console while the event is pending: it must never be taken as ours
    manual = json.loads(con.http_get("http://c/focus", 1).decode())
    assert manual["owner"].startswith("manual:")
    con.owner.wait_idle(5.0)
    node, clk = TE.make_node(tmp_path, con)
    _, r = _run_event(node, clk)
    assert r["capture_status"] == CS.CAPTURE_QUALITY_OK and r["release"] == RS.RELEASE_READY_CAPTURE_OK, r
    srv = con.owner.get(r["request_id"])
    assert srv["owner"].startswith("e5:") and srv["status"] == C.ReqStatus.COMPLETED
    assert srv["capture"]["capture_id"] == r["capture_id"] != manual["request_id"]
    assert "/capture" in con.urls and "/aim" in con.urls
    pend = list((node.run / "jobs" / "pending").glob("*.json"))
    assert len(pend) == 1
    job = json.loads(pend[0].read_text())
    for k in ("schema_version", "run_id", "event_key", "capture_id", "request_id", "capture_status", "pantilt_ok",
              "image", "image_sha256", "target_roi_verified", "odom_frame"):
        assert k in job, k
    assert job["capture_id"] == r["capture_id"] and job["pantilt_ok"] is True
    con.owner.close()


def test_I2_reader_result_attaches_to_exact_event_after_fsm_idle(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    con = OwnerConsole(tmp_path, ImgHW(sharp=True))
    node, clk = TE.make_node(tmp_path, con)
    src, r = _run_event(node, clk)
    assert not node.fsm.active
    calls = []
    _reader(node.run, [("418", 0.95)], localize=lambda st, rec, **k: calls.append(k) or {})
    node.event_step(src, None)
    ev = [e for e in node.status["events"] if e["event_key"] == r["event_key"]][0]
    assert ev["reading"] is not None and ev["reading"]["capture_id"] == r["capture_id"]
    assert ev["reading"]["event_key"] == r["event_key"]
    con.owner.close()


def test_I3_blurred_capture_end_to_end_never_reaches_e6(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    con = OwnerConsole(tmp_path, ImgHW(sharp=False))
    node, clk = TE.make_node(tmp_path, con)
    src, r = _run_event(node, clk)
    assert r["capture_status"] == CS.CAPTURE_QUALITY_FAIL, r
    assert r["release"] == RS.RELEASE_READY_OBSERVATION_FAILED
    calls = []
    core = _reader(node.run, [("418", 0.95)], localize=lambda st, rec, **k: calls.append(dict(k, e6=rec["e6"]["status"])) or {})
    node.event_step(src, None)
    done = [json.loads(p.read_text()) for p in (node.run / "jobs" / "done").glob("*.json")]
    if done:                                                # E5 may decide not to enqueue a quality-failed capture
        assert all(d["reason_code"] == RR.CAPTURE_QUALITY_FAIL and d["terminal"] for d in done)
    # the reader may record a failure packet (packet_only=True) but never runs E6 for a quality-failed capture
    assert all(c.get("packet_only") is True and c["e6"] == "SKIPPED" for c in calls), calls
    assert core.counters["ocr_runs"] == 0
    con.owner.close()


def test_I4_legacy_console_is_protocol_mismatch_through_real_chain(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    con = F.LegacyConsole()
    node, clk = TE.make_node(tmp_path, con)
    _, r = _run_event(node, clk)
    blob = json.dumps(r)
    assert C.PROTOCOL_MISMATCH in blob
    assert r["capture_status"] != CS.CAPTURE_QUALITY_OK and r["release"] != RS.RELEASE_READY_CAPTURE_OK
    assert "/state" in con.calls and "/aim" not in con.calls and "/full.jpg" not in con.calls and con.job == 0


def test_I5_invalid_depth_packet_is_refused_by_e6(tmp_path):
    import e5_e7.e5_packet as P
    import e5_e7.e6_region as E6
    trial = tmp_path / "trial"
    (trial / "runner").mkdir(parents=True)
    ev = dict(t=1.0, stamp=1.7e9, picks=[[300, 200, 340, 240]], z=[0.0], pose=[0, 0, 0], nobs=1)
    (trial / "runner" / "events.jsonl").write_text(json.dumps(ev) + "\n")
    calib = Path(os.environ["SCI_PANTILT_CALIB"])
    pk = P.build_packet(trial, calib, reading=dict(labels=[dict(label="418", p=0.9)]), identity=dict(odom_frame="base"))
    obs = pk["observation"]
    assert obs["range_m"] is None and obs["geometry_status"]["status"] == "GEOMETRY_UNAVAILABLE"
    res = E6.run_e6(pk, ["4F"], tmp_path / "e6", "kci")
    assert res["status"] == "INPUT_INVALID"


def test_T22b_scene_blocking_2s_does_not_stall_event_loop_or_device_job(tmp_path, monkeypatch):
    """Main-session addition for T22: the REAL Node.loop computes a 2 s (fake) scene on the producer thread while the
    event-loop thread keeps publishing status and follows a 1.8 s device job (18 s scaled 1/10) to completion."""
    _env(tmp_path, monkeypatch)
    monkeypatch.setenv("SCI_AIM_TIMEOUT_S", "5")
    con = F.FakeConsole(job_s=1.8)
    node, clk = TE.make_node(tmp_path, con)
    node.clock = time.monotonic
    node.frontend = F.FakeFrontend(scene_delay_s=2.0)
    node.threaded = True
    src = F.FakeSource()
    src.odom_hist = [(100.0 + 0.1 * i, 0.0, 0.0, 0.0) for i in range(80)]
    img = np.zeros((480, 640, 3), np.uint8)
    src.queue = [(100.0 + 0.1 * i, i, img) for i in range(60)]           # > warmup -> at least one 2 s scene
    th = threading.Thread(target=node.loop, args=(src,), daemon=True)
    th.start()
    try:
        for _ in range(100):
            if getattr(node, "_ev_thread", None) is not None:
                break
            time.sleep(0.02)
        node.proposals.put(dict(id=7, stop=F.a3_stop(), stamp=100.0, pose=[0, 0, 0], bgr=img, latest_frame=None))
        seen, busy_seen, t0 = [], False, time.time()
        while time.time() - t0 < 6:
            try:
                seen.append(json.loads((N.SHM / "status.json").read_text())["status_mono_ns"])
            except (OSError, ValueError, KeyError):
                pass
            busy_seen = busy_seen or node.device.busy
            r = node.store.get(node.fsm.key) if node.fsm else {}
            if busy_seen and not node.device.busy and r.get("release"):
                break
            time.sleep(0.2)
        assert busy_seen, "device job never started while the producer was inside a 2 s scene"
        assert len(set(seen)) >= 5, "status stopped while the scene producer was blocked"
        assert r.get("release") == RS.RELEASE_READY_CAPTURE_OK, r
        scenes = [json.loads(x) for x in (node.run / "scenes.jsonl").read_text().splitlines() if x.strip()]
        assert any(sc["timing"].get("scene_ms") == 2000.0 for sc in scenes), "no 2 s scene was actually computed"
    finally:
        src.alive = False
        th.join(timeout=5)
        node.stop_event_thread()


def test_I6_viewer_snapshot_with_floor_progress_present(tmp_path, monkeypatch):
    """Main-session finding (pyflakes): _snapshot used module alias T (r1_trial) before a local `T = np.array(...)`
    assignment in the same function -> UnboundLocalError whenever floor_progress.json exists (computing screen)."""
    import test_r1_trial_geo as TG
    S, root = TG.mk_state(tmp_path, monkeypatch)
    out = root / "out"
    S.start()
    S.stop()
    S.exp.update(state="computing", signal="1100.000")
    tok = TG.T.token_now(out)
    TG.write_tagged(out, "floor_progress.json", dict(signal="1100.000", done=3, total=9), tok)
    snap = S.snapshot(None, "live")                              # baseline-candidate (pre-fix) raised UnboundLocalError
    assert snap["progress"]["done"] == 3
    snap2 = S.snapshot(None, "result")
    assert snap2["exp"]["state"] == "computing"
