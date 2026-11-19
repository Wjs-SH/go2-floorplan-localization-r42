"""E5-RT BEFORE/AFTER reproductions. Each test asserts the FIXED behaviour through e5_onboard APIs that exist in both
trees: expected to FAIL on RT_ROOT/baseline (student 10-17 deploy) and PASS on RT_ROOT/candidate."""
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

WS = Path(os.environ["SCI_WS"])
sys.path.insert(0, str(WS / "src" / "sci_visual_observation"))
sys.path.insert(0, str(WS / "external" / "sci_event" / "e5_onboard"))


def test_before_floor_computation_switches_search_off(tmp_path):
    """[멈춤] -> trial 'computing' (221 s on 10-17): the search must keep running (new STOP requests stay off)."""
    import e5_onboard as OB
    p = tmp_path / "gate.json"
    p.write_text(json.dumps(dict(active=False, why="층 계산 중", t=time.time(), trial_state="computing",
                                 global_match_state="computing", schema="e5rt-gate-1")))
    g = OB.gate_state(time.time(), path=p, mode="walking", gate_on=True)
    assert g["observe"], "baseline: observe = heartbeat AND active -> search off during floor computation"
    assert not g["arm"]


def test_before_nan_odom_reaches_policy_history():
    import e5_onboard as OB

    class M:
        pass
    m = M(); m.header = type("H", (), dict(stamp=time.time() - 1))()
    m.pose = M(); m.pose.pose = M()
    m.pose.pose.position = type("V", (), dict(x=float("nan"), y=0.0, z=0.0))()
    m.pose.pose.orientation = type("O", (), dict(x=0.0, y=0.0, z=0.0, w=1.0))()
    s = OB.BaseOdomRosSource.__new__(OB.BaseOdomRosSource)
    s.lock, s.odom, s.n_odom = threading.Lock(), deque(maxlen=50), 0
    s._st = lambda h: h.stamp
    s._on_odom(m)
    assert len(s.odom_hist) == 0, "baseline: a NaN pose is appended to odom_hist (policy/stop speed input)"


def test_before_quality_threshold_hidden_default():
    import e5_onboard as OB
    assert OB.N.QUALITY_SHARP_MIN == 40.0, "baseline: code default silently lowers the capture threshold to 20"


def test_before_home_sent_around_device_owner():
    src = (WS / "external" / "sci_event" / "e5_onboard" / "e5_onboard.py").read_text()
    body = src[src.index("class GatedNode"):src.index("def _source(a):")]
    assert "_console_home" not in body, "baseline: GatedNode starts an HTTP /home thread outside the device owner"


def test_before_status_publish_unserialised():
    src = (WS / "external" / "sci_event" / "e5_onboard" / "e5_onboard.py").read_text()
    body = src[src.index("    def publish_status"):]
    assert "_status_lock" in body and "_rev" in body, "baseline: two threads build/write status.json with no order/revision"


def test_before_live_image_old_boxes():
    import numpy as np
    import e5_onboard as OB
    g = OB.GatedNode.__new__(OB.GatedNode)
    g._live_meta = dict(frame_id=10, boxes=[([10, 10, 60, 60], "sign", 2)], t_drawn=time.time(), t_drawn_mono=time.monotonic())
    os.environ.pop("KCI_LIVE_BOXES", None)
    im = g.render_live((0.0, 11, np.zeros((480, 640, 3), np.uint8)))
    assert int(im[10:61, 10:61].sum()) == 0, "baseline: boxes of frame #10 (<=2.5 s old) painted on live frame #11"
