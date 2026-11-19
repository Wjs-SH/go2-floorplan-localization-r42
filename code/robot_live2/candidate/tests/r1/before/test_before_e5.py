"""BEFORE (defect reproduction) for W-D. Each test asserts the CORRECT behaviour against the BASELINE E5 code and is
expected to FAIL there (BEFORE_FAILURE). Run with SCI_WS=<baseline tree>. Evidence: SIMULATED fakes of the legacy
console/frames; no device/ROS/model. (The AFTER equivalents are in tests/r1/test_r1_e5.py, against the new API.)"""
import json
import os
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_pantilt", WS / "external" / "sci_event" / "e5_onboard"):
    sys.path.insert(0, str(p))
import e5_node as N  # noqa: E402
import policy as P  # noqa: E402
BASE_NODE = N.Node


def _superseded_on_candidate(after):
    """These reproductions build the BASELINE Node by hand (no injected deps); on the candidate tree the legacy
    construction no longer exists, so they are skipped as superseded by the named AFTER test (main-session edit)."""
    if hasattr(N, "R1_E5") or os.path.basename(os.path.normpath(os.environ["SCI_WS"])) == "candidate":
        pytest.skip(f"legacy construction replaced in candidate; superseded by tests/r1/test_r1_e5.py::{after}")


class _Resp:
    def __init__(self, b):
        self.b = b

    def read(self):
        return self.b


def test_before_manual_focus_job_counter_is_not_this_events_success(monkeypatch):
    """10-15: a different job raising the console counter was accepted as this event's capture."""
    import e5_onboard as OB
    st = dict(job=3)

    def urlopen(url, timeout=None):
        if "/state" in url:
            return _Resp(json.dumps(dict(status="idle", job=st["job"], log=[])).encode())
        if "/aim" in url:
            st["job"] += 1                       # in reality: an operator's manual /focus finished meanwhile
            return _Resp("조준 시작".encode())
        raise IOError("404")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    pl = OB.Planner(N.Geometry(), console="http://fake", execute=True)
    r = pl.aim(pl.plan(320, 240, 2.0), poll_s=0.01, timeout_s=0.3)
    assert not r["ok"], "baseline accepts a job-counter increase it did not prove to be its own"


def _bare_node(tmp, monkeypatch):
    monkeypatch.setattr(N, "WS", tmp)
    monkeypatch.setattr(N, "SHM", tmp / "shm")
    (tmp / "shm").mkdir(parents=True, exist_ok=True)
    n = BASE_NODE.__new__(BASE_NODE)
    n.a = types.SimpleNamespace(source="ros", assume_stopped=False, ui_scale=1.0)
    n.run = tmp / "run"
    for sub in ("jobs/pending", "jobs/done", "trials/ev001/console", "trials/ev001/runner"):
        (n.run / sub).mkdir(parents=True, exist_ok=True)
    n.cfg = P.load_cfg(None)
    n.logs = {k: open(n.run / f"{k}.jsonl", "a") for k in ("scenes", "stops", "fsm", "faults")}
    n.status = dict(events=[dict(id=1, t=1.0, state="READ_ASYNC")], faults=[])
    n.n_ev, n.p2_crops = 1, {}
    return n


def test_before_reading_attached_after_fsm_idle(tmp_path, monkeypatch):
    _superseded_on_candidate("test_T03_reading_attached_after_fsm_idle")
    n = _bare_node(tmp_path, monkeypatch)
    f = P.EventFSM(1, dict(t=1.0), n.cfg)
    f.to("READ_ASYNC", "job_enqueued")
    n.fsm = f
    (n.run / "jobs" / "done" / "ev001.json").write_text(json.dumps(dict(ev=1, status="NO_ROOM_LABEL")))
    n.tick_event(types.SimpleNamespace(odom_hist=[]), None)
    assert n.status["events"][0].get("reading") is not None, "baseline never attaches a reading once the FSM is idle"


def test_before_quality_fail_is_not_captured_and_retry_needs_new_exposure(tmp_path, monkeypatch):
    _superseded_on_candidate("test_T10_blurred_capture_is_quality_fail_with_fresh_retries")
    n = _bare_node(tmp_path, monkeypatch)
    flat = cv2.imencode(".jpg", np.full((90, 160, 3), 120, np.uint8))[1].tobytes()
    fetched = []

    class Pl:
        execute = True

        def aim(self, plan):
            return dict(mode="EXECUTE", ok=True)

        def fetch_capture(self, which="full.jpg"):
            fetched.append(which)
            return flat
    n.planner = Pl()
    f = P.EventFSM(1, dict(t=1.0, stage="A3", spot=1), n.cfg)
    f.trial = n.run / "trials" / "ev001"
    f.legacy_rec = {}
    f.to("AIM", "x")
    n.fsm = f
    n._aim_capture_with_plan(f, dict(url="u", reachable=True, pan_cmd=1, tilt_cmd=1), [1, 1, 2, 2], 1, 1, 2.0,
                             types.SimpleNamespace(pose=lambda s: ((0, 0, 0), s)), 0.0, pose2=(0, 0, 0))
    states = [x["state"] for x in f.log]
    ok_fresh = len(set(fetched)) == len(fetched)
    assert "CAPTURED" not in states and "RELEASE" not in states and ok_fresh, \
        f"baseline: states={states} fetched={fetched} (blurred capture CAPTURED/RELEASED; retries re-download full.jpg)"


def test_before_stale_box_not_drawn_on_newer_frame(tmp_path, monkeypatch):
    import e5_onboard as OB
    monkeypatch.setattr(N, "SHM", tmp_path)
    g = OB.GatedNode.__new__(OB.GatedNode)
    g.a = types.SimpleNamespace(ui_scale=1.0)
    g._live_boxes, g._live_t = [([10, 10, 60, 60], "sign", 2)], time.time()   # boxes computed on frame #10
    class Src:
        lock, latest, new_color, n = threading.Lock(), (0.0, 11, np.zeros((480, 640, 3), np.uint8)), None, 0

        @property
        def alive(self):
            self.n += 1
            return self.n == 1
    src = Src()
    OB.GatedNode._live_loop(g, src)
    im = cv2.imread(str(tmp_path / "frame.jpg"))
    assert int(im[10:61, 10:61].astype(int).sum()) < 5000, "baseline paints frame-#10 boxes on live frame #11"


def test_before_wait_stable_timeout_on_every_branch():
    t = [0.0]
    f = P.EventFSM(1, dict(t=0.0), P.load_cfg(None), clock=lambda: t[0])
    for k in range(260):                         # moving until 19.8 s, then standing still (10-15 ev5: confirmed at 20.38 s)
        t[0] = k * 0.1
        f.on_odom(0.3 if t[0] < 19.8 else 0.0, 0.0)
    assert f.state == "FAILED", f"baseline checks the 20 s timeout only while moving: state={f.state} " \
        f"at {[x['t'] for x in f.log if x['state'] == 'STOP_CONFIRMED']}"


def test_before_aim_does_not_block_event_loop(tmp_path, monkeypatch):
    _superseded_on_candidate("test_T22_event_loop_runs_during_long_device_job")
    n = _bare_node(tmp_path, monkeypatch)
    f = P.EventFSM(1, dict(t=1.0, stage="A3", spot=1, box=[300, 220, 340, 260], z=2.0, xyz=None), n.cfg)
    f.to("STOP_CONFIRMED", "x")
    n.fsm = f

    def slow_aim(src, f, now):
        time.sleep(1.0)                          # stands for the 17.5 s aim+focus of 10-15
    n.aim_and_capture = slow_aim
    n.update_event_status = lambda: None
    t0 = time.time()
    n.tick_event(types.SimpleNamespace(odom_hist=[]), None)
    assert time.time() - t0 < 0.2, "baseline event tick blocks for the whole aim/focus"
