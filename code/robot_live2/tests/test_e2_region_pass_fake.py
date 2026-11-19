"""STAGING (2026-10-30): e2_worker.Coordinator.region_pass glue with a FAKE child pipe (no maps, no fit). Checks: the child gets
(floor, t_now, W, 'exact', region); candidates are ADDED with pass_id '<p>R<n>' + regional_request and stage exact; floors_done is
untouched; a robot at rest / CalibrationUnavailable is a RETRY_WAIT (request kept), other errors FAILED."""
import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(STG))
spec = importlib.util.spec_from_file_location("e2w_fake", STG / "e2_worker.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


class Pipe:
    def __init__(self, replies):
        self.sent, self.replies = [], list(replies)

    def send(self, m):
        self.sent.append(m)

    def recv(self):
        return self.replies.pop(0)


def coord(tmp, path_m=8.0, hold=None):
    c = W.Coordinator.__new__(W.Coordinator)
    c.a = types.SimpleNamespace(min_window_path_m=2.0)
    c.out, c.W, c.pass_id = tmp, 60.0, 1
    c.state = dict(candidates=[dict(floor=7, px=1.0, merit=0.3, twoway=0.5)], candidates_pass=[dict(floor=7, px=1.0, merit=0.3, twoway=0.5)],
                   floors_done=list(range(1, 10)), per_floor={"4": dict(light="/L4", status="DONE")})
    c.span = lambda: (0.0, 200.0)
    c.session = lambda: {"odom_session_id": "S1"}
    c.hold_reason = lambda: hold
    c.window_path_m = lambda t1, W=None: path_m
    c.ledger = W.RL.Ledger(tmp / "led.json"); c.ledger.load(dict(run_id="R", odom_session="S1", maps="m", calib="c"))
    c.region_added, c._resume, c.logs = [], None, []
    real_log = c.log
    c.log = lambda kind, **k: (c.logs.append((kind, k)), real_log(kind, **k))[1]   # the REAL log() runs (R34: never a stub)
    return c


RQ = dict(request_id="PAIR_1", odom_session="S1", region=dict(floor=4, center_px=[750.0, 350.0], radius_px=500.0, rooms=["room_401", "room_405"]))


def test_region_pass_adds_candidates_and_keeps_floors_done(tmp_path):
    c = coord(tmp_path)
    p = Pipe([("progress", 4, {}), ("floor_done", 4, dict(candidates=[dict(floor=4, px=760.0, py=355.0, phi_deg=10.0, ppm=24.0, merit=0.25, twoway=0.6)],
                                                         sec=70.0, light="/L4", H=1825, map_version="mv"))])
    c.region_pass(RQ, {0: dict(pc=p)}, None)
    assert p.sent[0][:4] == (4, 200.0, 60.0, "exact") and p.sent[0][4] == dict(center_px=[750.0, 350.0], radius_px=500.0)
    new = [x for x in c.state["candidates_pass"] if x.get("regional_request") == "PAIR_1"]
    assert len(new) == 1 and new[0]["pass_id"] == "1R1" and new[0]["stage"] == "exact" and new[0]["odom_session"] == "S1"
    assert len(c.state["candidates_pass"]) == 2 and len(c.state["candidates"]) == 2          # added, nothing removed
    assert c.state["floors_done"] == list(range(1, 10))
    assert c.state["regional"][-1]["status"] == "DONE" and c.ledger.get("PAIR_1")["state"] == "COMPLETED" and c.region_pending() is None


def test_robot_at_rest_is_retried_not_failed(tmp_path):
    c = coord(tmp_path, path_m=0.3)
    c.region_pass(RQ, {0: dict(pc=Pipe([]))}, None)
    assert c.state["regional"][-1]["status"] == "RETRY_WAIT" and c.ledger.get("PAIR_1")["state"] == "RETRY_WAIT" and c.ledger.get("PAIR_1")["attempts"] == 0   # waiting costs no attempt
    c2 = coord(tmp_path / "c2" if not (tmp_path / "c2").mkdir() else tmp_path)
    c2.region_pass(RQ, {0: dict(pc=Pipe([("floor_failed", 4, dict(error="CalibrationUnavailable: Not enough overlapping scans"))]))}, None)
    assert c2.state["regional"][-1]["status"] == "RETRY_WAIT"


def test_other_errors_fail_and_attempts_are_bounded(tmp_path):
    c = coord(tmp_path)
    c.region_pass(RQ, {0: dict(pc=Pipe([("floor_failed", 4, dict(error="ValueError: semantic region contains no traversable seed cell"))]))}, None)
    assert c.state["regional"][-1]["status"] == "FAILED"
    (tmp_path / "c2").mkdir()
    c2 = coord(tmp_path / "c2")
    for _ in range(W.RL.MAX_ATTEMPTS):                                       # a fit that keeps being refused
        e = c2.ledger.ingest(RQ); e["not_before"] = 0.0
        c2.region_pass(RQ, {0: dict(pc=Pipe([("floor_failed", 4, dict(error="CalibrationUnavailable: Not enough overlapping scans"))]))}, None)
    assert c2.state["regional"][-1]["status"] == "FAILED" and "attempts" in c2.state["regional"][-1]["why_failed"] and c2.region_pending() is None
