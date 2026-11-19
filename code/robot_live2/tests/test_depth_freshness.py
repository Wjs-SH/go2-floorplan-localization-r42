"""R34 §11 depth faults through the REAL HubWriter + REAL SharedHubSource (real files, real index), with a synthetic sensor feeding the same
source interface. SYNTHETIC inputs; plus the REAL field stamps of 10-31 / 11-01 (OFFLINE_COUNTERFACTUAL). Cases: normal, frozen, reversed stamps,
colour only, temporary loss + recovery, stale depth with new RGB, slot file left over from an older frame."""
import json
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/src/sci_sensor_bridge"))
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
import depth_freshness as DF  # noqa: E402
import shared_hub as SH  # noqa: E402
import ros_readonly as RR  # noqa: E402
import visual_stop as VS  # noqa: E402
import ready_gate as RG  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments"


class Sensor:
    """The real RosReadonlySource.depth() (nearest + gate) over a real _Ring; colour fed by hand."""

    def __init__(self):
        self.lock = threading.Lock()
        self.depths = RR._Ring(8, 10 ** 9)
        self.depth_gate = DF.Tracker()
        self.rx_mono_ns, self.pending = {}, []
        self.seq = 0
        self.odom, self.odom_session, self.subscribed, self.caminfo = deque(), "S", [], None
        self.stats = dict(odom=dict(frames=None, last_rx_wall=None))

    depth = RR.RosReadonlySource.depth

    def _img(self, m):
        return m

    def health(self):
        return dict(depth_fresh=self.depth_gate.snapshot())

    def poll(self):
        out, self.pending = self.pending, []
        return out

    def feed_depth(self, st):
        m = np.full((4, 4), int(st * 10) % 60000, np.uint16)
        self.depths.push((st, 0, m), m.nbytes)

    def feed_color(self, st):
        self.seq += 1
        self.pending.append((st, self.seq, np.zeros((4, 4, 3), np.uint8)))


def rig(tmp_path):
    s = Sensor()
    w = SH.HubWriter(tmp_path, s)
    r = SH.SharedHubSource.__new__(SH.SharedHubSource)
    r.hub, r._idx, r.torn, r.depth_gate = tmp_path, None, 0, DF.Tracker()
    r._load = lambda fr, kind: np.frombuffer((tmp_path / f"slot_{int(fr['slot']):02d}.depth").read_bytes(), "<u2").reshape(fr["h"], fr["w"])
    r._still_valid = lambda fr: True
    return s, w, r


def step(s, w, r, t, depth=True):
    if depth:
        s.feed_depth(t)
    s.feed_color(t)
    w.step()
    r._idx = w.write_index()
    return r.depth(t)


def test_normal_stream_passes_and_reports_ok(tmp_path):
    s, w, r = rig(tmp_path)
    for i in range(10):
        img, st = step(s, w, r, 100.0 + i * 0.1)
        assert img is not None and st == pytest.approx(100.0 + i * 0.1)
    assert r.depth_gate.state == DF.DEPTH_OK and r.depth_gate.stale == 0
    assert DF.hub_depth_state(r._idx)["state"] == DF.DEPTH_OK


def test_frozen_depth_is_refused_within_the_limit_and_no_old_image_returned(tmp_path):
    s, w, r = rig(tmp_path)
    for i in range(5):
        step(s, w, r, 100.0 + i * 0.1)
    got = []
    for i in range(1, 30):
        got.append((100.4 + i * 0.1, step(s, w, r, 100.4 + i * 0.1, depth=False)))
    late = [g for t, g in got if t - 100.4 > 0.5]
    assert late and all(img is None and st is None for img, st in late)
    assert r.depth_gate.state in (DF.DEPTH_STALE, DF.DEPTH_MISSING) and s.depth_gate.state == DF.DEPTH_STALE
    assert DF.hub_depth_state(r._idx)["state"] != DF.DEPTH_OK


def test_slot_file_of_an_older_frame_is_not_used_with_new_rgb(tmp_path):
    s, w, r = rig(tmp_path)
    step(s, w, r, 100.0)
    step(s, w, r, 110.0, depth=False)                         # the colour frame of +10 s has no depth
    fr = r._idx["frames"][-1]
    assert fr["depth_stamp"] is None and DF.frame_depth_state(fr)[0] == DF.DEPTH_MISSING


def test_colour_only_from_the_start(tmp_path):
    s, w, r = rig(tmp_path)
    for i in range(5):
        img, st = step(s, w, r, 50.0 + i, depth=False)
        assert img is None and st is None
    assert r.depth_gate.state == DF.DEPTH_MISSING and DF.hub_depth_state(r._idx)["state"] == DF.DEPTH_MISSING


def test_reversed_depth_stamps_are_never_taken_for_new(tmp_path):
    s, w, r = rig(tmp_path)
    step(s, w, r, 100.0)
    s.feed_depth(90.0)                                        # stamp jumps backwards
    s.feed_color(101.0)
    w.step(); r._idx = w.write_index()
    assert r.depth(101.0) == (None, None) and r.depth_gate.state == DF.DEPTH_STALE


def test_temporary_loss_recovers_by_itself(tmp_path):
    s, w, r = rig(tmp_path)
    step(s, w, r, 100.0)
    for i in range(1, 12):
        step(s, w, r, 100.0 + i * 0.2, depth=False)
    assert r.depth_gate.state != DF.DEPTH_OK
    img, st = step(s, w, r, 103.0)
    assert img is not None and st == pytest.approx(103.0) and r.depth_gate.state == DF.DEPTH_OK


def test_limit_is_tunable_and_bad_values_fall_back(monkeypatch):
    monkeypatch.setenv("LIVE2_DEPTH_MAX_GAP_S", "0.3")
    assert DF.max_gap_s() == 0.3
    for bad in ("0", "-1", "abc", "99"):
        monkeypatch.setenv("LIVE2_DEPTH_MAX_GAP_S", bad)
        assert DF.max_gap_s() == DF.DEFAULT_MAX_GAP_S


def det(dt, z=3.0, sim=0.2):
    return (dict(t=1.0, regions=[dict(box=[1, 2, 3, 4], sim=sim, z=z, spot="s1", dec=dict(P2=dict(state="VERIFIED")))]),
            dict(frame_id=1, odom_raw=dict(x=0.0, y=0.0), depth_dt_s=dt))


def test_visual_stop_refuses_a_distance_from_stale_depth():
    v = VS.VisualStop(None)
    rec, ln = det(538.3)
    assert v.feed(rec, 100.0, ln, now=100.0) is None and v.n_depth_refused == 1 and v.n == 0
    rec, ln = det(None)
    assert v.feed(rec, 100.0, ln, now=100.0) is None and v.n_depth_refused == 2
    rec, ln = det(0.05)
    assert v.feed(rec, 100.0, ln, now=100.0) is not None and v.n == 1


def _field(rid):
    p = next((OPR / rid / "archive/observe_run").rglob("live2_detection_pose.jsonl"))
    return [json.loads(x) for x in open(p)]


def test_real_field_stamps_11_01_frozen_and_10_31_normal():
    a = _field("X20261101_193139")
    bad = [x for x in a if x.get("depth_dt_s") is None or abs(x["depth_dt_s"]) > DF.DEFAULT_MAX_GAP_S]
    assert len(bad) / len(a) > 0.95
    b = _field("X20261031_231940")
    good = [x for x in b if x.get("depth_dt_s") is not None and abs(x["depth_dt_s"]) <= DF.DEFAULT_MAX_GAP_S]
    assert len(good) / len(b) > 0.98


def _snap(state, why=None):
    return dict(rgb=dict(status="LIVE", depth_state=state, depth_why=why),
                lio=dict(status="OK", odom=dict(hz_10s=10.0, last_rx_age_s=0.1, n=5)), siglip=dict(status="LIVE"),
                live2=dict(ages=dict(e2=1.0, e3=1.0, e4=1.0), e2=dict(floors_total=9)))


EXP = dict(state="RUNNING", exp_id="X", boot_id="b", modules=dict(fastlio=dict(state="RUNNING")))


def test_ready_is_blocked_with_the_reason_when_depth_is_not_fresh():
    ok = RG.evaluate(_snap("DEPTH_OK"), EXP, dict(pending=[]), "b")
    assert ok["ready"] is True
    bad = RG.evaluate(_snap("DEPTH_STALE", "newest depth is 900 s away"), EXP, dict(pending=[]), "b")
    assert bad["ready"] is False and bad["missing"] == ["depth"]
    assert "E5" in next(i for i in bad["items"] if i["key"] == "depth")["ko"]


def test_aim_console_no_longer_subscribes_to_depth():
    src = (HERE.parent / "candidate/live2/e6_aim_console_ctl.py").read_text()
    assert '"--use-depth"' not in src
