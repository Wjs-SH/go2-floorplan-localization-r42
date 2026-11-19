"""LIVE-FIELD1 G0 tests: fake inputs only (no ROS graph, no device, no model)."""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

CAND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAND / "live1"))
sys.path.insert(0, str(CAND / "src" / "sci_sensor_bridge"))
import live1_common as C  # noqa: E402
import lio_monitor as LM  # noqa: E402
import frame_probe as FP  # noqa: E402
import live1_web as W  # noqa: E402

MIRROR = dict(frame_id="camera_init", child_frame_id="base_link", verified_by="tester", evidence="unit",
              body_transform=dict(kind="fastlio_mirror_lever_yaw", delta_deg=30.0, lever_m=[0.0, 0.0]))


def _odom(st, x=0.0, y=0.0, yaw=0.0, frame="camera_init", child="base_link"):
    N = types.SimpleNamespace
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    return N(header=N(stamp=N(sec=int(st), nanosec=int(round((st % 1) * 1e9))), frame_id=frame), child_frame_id=child,
             pose=N(pose=N(position=N(x=x, y=y, z=0.0), orientation=N(x=q[0], y=q[1], z=q[2], w=q[3]))))


def _src(contract=None, identity=False):
    import ros_readonly as RR
    s = RR.RosReadonlySource.__new__(RR.RosReadonlySource)
    s._init_state(contract=contract, wall=lambda: 1000.0, visual_identity_pose=identity)
    s.visual_only = False
    return s


# ---------------------------------------------------------------- ros_readonly frame consistency (LIVE1 fix)
def test_odom_hist_and_pose_are_in_the_same_body_frame():
    s = _src(MIRROR)
    for i in range(5):
        s._on_odom(_odom(999.0 + 0.1 * i, x=0.1 * i, y=0.05 * i, yaw=0.2))
    hist = s.odom_hist
    pose, st = s.pose(999.4)
    assert st == pytest.approx(999.4) and hist[-1][0] == pytest.approx(999.4)
    assert hist[-1][1:] == pytest.approx(pose)                                   # P2 reference frame == scene frame
    raw = s.odom_raw_hist()[-1]
    assert raw[1:] != pytest.approx(hist[-1][1:])                                # raw is NOT handed to geometry


def test_no_contract_gives_no_geometry_and_only_a_flagged_identity_pose():
    s = _src(None, identity=True)
    s._on_odom(_odom(999.0, x=1.0))
    assert s.odom_hist == [] and s.pose(999.0) == ((0.0, 0.0, 0.0), 999.0)
    assert s.pose_mode() == "IDENTITY_FLAGGED_CONTRACT_MISSING"
    s2 = _src(None, identity=False)
    s2._on_odom(_odom(999.0, x=1.0))
    assert s2.pose(999.0) == (None, None) and s2.pose_mode() == "NONE_CONTRACT_MISSING"


# ---------------------------------------------------------------- lio monitor
def test_odom_stats_rate_resets_and_frame_change():
    t = [100.0]
    st = LM.OdomStats(wall=lambda: t[0] + 0.05, mono=lambda: t[0])
    for i in range(50):
        t[0] = 100.0 + 0.1 * i
        st.on_odom(t[0], 0.01 * i, 0.0, 0.0, 0, 0, 0, 1, "camera_init", "base_link")
    s = st.summary()
    assert s["hz_10s"] == pytest.approx(10.0, rel=0.01) and s["interarrival_ms"]["p95"] == pytest.approx(100.0, abs=1)
    assert s["header_to_rx_ms"]["p50"] == pytest.approx(50.0, abs=1) and s["path_m"] == pytest.approx(0.49, abs=0.01)
    st.on_odom(t[0], 0, 0, 0, 0, 0, 0, 1, "camera_init", "base_link")             # duplicate stamp
    st.on_odom(t[0] - 5.0, 0, 0, 0, 0, 0, 0, 1, "camera_init", "base_link")       # backwards -> new session
    st.on_odom(t[0] - 4.9, 0, 0, 0, 0, 0, 0, 1, "odom", "base_link")              # producer frame change
    st.on_odom(float("nan"), 0, 0, 0, 0, 0, 0, 1, "odom", "base_link")
    r = st.summary()["rejected"]
    assert r == {"DUPLICATE_OR_REORDERED": 1, "TIME_BACKWARDS_RESET_NEW_SESSION": 1, "FRAME_CHANGED_NEW_SESSION": 1, "NON_FINITE": 1}
    assert st.session == 2


def test_accumulator_is_bounded_and_renders():
    a = LM.Accumulator(res=0.1, max_cells=50, stride=1)
    rng = np.random.default_rng(0)
    a.add_xyz(rng.uniform(-5, 5, (5000, 3)).astype(np.float32), z_ref=0.0)
    s = a.summary()
    assert s["cells"] == 50 and s["dropped_full"] > 0 and s["scans"] == 1
    png = a.render([(0, 0, 0), (1, 1, 1)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_cloud_xyz_decodes_pointcloud2_layout():
    N = types.SimpleNamespace
    pts = np.array([[1, 2, 3, 9], [4, 5, 6, 9]], "<f4")
    m = N(fields=[N(name="x", offset=0), N(name="y", offset=4), N(name="z", offset=8), N(name="intensity", offset=12)],
          width=2, height=1, point_step=16, is_bigendian=False, data=pts.tobytes())
    assert LM.cloud_xyz(m).tolist() == [[1, 2, 3], [4, 5, 6]]


# ---------------------------------------------------------------- frame contract evidence
def _walk(delta_deg, mirror, heading_deg=20.0):
    """Body walks straight 4 m, then turns LEFT 90 deg in place. Raw = inverse of the ros_readonly transform."""
    d, h = math.radians(delta_deg), math.radians(heading_deg)
    out, t, x, y = [], 0.0, 0.0, 0.0
    for i in range(41):
        bx, by = x + 0.1 * i * math.cos(h), y + 0.1 * i * math.sin(h)
        out.append((t, bx, by, h))
        t += 0.1
    for k in range(1, 31):
        out.append((t, bx, by, h + math.radians(3 * k)))
        t += 0.1
    rows = []
    for t, bx, by, yb in out:
        if mirror:
            rx, ry, ryaw = bx, -by, d - yb
        else:
            rx, ry, ryaw = bx, by, yb - d
        rows.append(dict(kind="odom", t=t, x=rx, y=ry, yaw=ryaw, frame_id="camera_init", child_frame_id="base_link"))
    return rows


def test_frame_probe_recovers_delta_and_mirror_from_motion():
    tf = [dict(parent="base_link", child="imu", xyz=[0.026, 0.0, 0.1])]
    d = FP.analyze(_walk(41.0, True), tf, "left")
    assert d["measurements"]["mirror_needed"] is True
    assert d["body_transform"]["kind"] == "fastlio_mirror_lever_yaw" and d["body_transform"]["delta_deg"] == pytest.approx(41.0, abs=0.5)
    assert d["body_transform"]["lever_m"] == [0.026, 0.0] and d["usable"] and d["status"] == "DRAFT_NEEDS_HUMAN_CONFIRMATION"
    assert d["verified_by"] is None                                              # never self-confirmed


def test_frame_probe_refuses_unknown_handedness_and_unsupported_offsets():
    d = FP.analyze(_walk(41.0, True), [], "none")
    assert not d["usable"] and any("HANDEDNESS_UNKNOWN" in p for p in d["problems"]) and any("LEVER" in p for p in d["problems"])
    d2 = FP.analyze(_walk(25.0, False), [dict(parent="base_link", child="imu", xyz=[0.0, 0.0, 0.0])], "left")
    assert d2["measurements"]["mirror_needed"] is False and not d2["usable"]
    assert any("NON_MIRROR_MOUNT_OFFSET" in p for p in d2["problems"])
    d3 = FP.analyze(_walk(41.0, True)[:5], [], "left")
    assert any("TOO_FEW_STRAIGHT_SEGMENTS" in p for p in d3["problems"])


def test_contract_from_probe_matches_ros_readonly_math():
    """A confirmed draft fed to ros_readonly maps the raw walk back to the body walk (heading == motion bearing)."""
    d = FP.analyze(_walk(41.0, True), [dict(parent="base_link", child="imu", xyz=[0.0, 0.0, 0.0])], "left")
    c = dict(frame_id=d["frame_id"], child_frame_id=d["child_frame_id"], body_transform=d["body_transform"], verified_by="t", evidence="u")
    s = _src(c)
    p0 = s.body_pose((0.0, 0.0, math.radians(41.0) - math.radians(20.0)))
    p1 = s.body_pose((1.0 * math.cos(math.radians(20)), -1.0 * math.sin(math.radians(20)), math.radians(21.0)))
    assert math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) == pytest.approx(20.0, abs=0.5)
    assert math.degrees(p1[2]) == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------- web panel (files only)
def test_panel_marks_missing_and_stale_data_never_ok(tmp_path):
    p = W.Panel(tmp_path / "state", tmp_path / "shm")
    snap = p.snapshot()
    assert snap["lio"]["status"] == "DATA_UNAVAILABLE" and snap["e5"]["status"] == "DATA_UNAVAILABLE"
    assert snap["research"]["E7"]["status"] == "NOT_WIRED" and snap["research"]["E3"]["status"] == "NOT_IMPLEMENTED"
    (tmp_path / "state").mkdir()
    C.atomic_json(tmp_path / "state" / "lio_state.json", dict(odom=dict(n=10, hz_10s=10.0, last_rx_age_s=0.05)))
    assert p.lio()["status"] == "OK"
    old = time.time() - 10
    os.utime(tmp_path / "state" / "lio_state.json", (old, old))
    assert p.lio()["status"] == "STALE"
    C.atomic_json(tmp_path / "state" / "lio_state.json", dict(odom=dict(n=10, hz_10s=10.0, last_rx_age_s=1.2)))
    assert p.lio()["status"] == "ODOM_STALE"
    C.atomic_json(tmp_path / "state" / "baseline_A1.json", dict(hz_p50=10.0))
    C.atomic_json(tmp_path / "state" / "lio_state.json", dict(odom=dict(n=10, hz_10s=9.0, last_rx_age_s=0.05)))
    assert p.lio()["status"] == "BELOW_BASELINE_95PCT"


def test_panel_stop_request_ui_only_for_recent_would_stop_and_old_source_flagged(tmp_path):
    shm = tmp_path / "shm"
    st = dict(siglip="OK", scene=dict(stamp=time.time() - 3.0, regions=[], timing={}),
              sem1_would_stop_private=dict(counts={}, last=[dict(kind="WOULD_STOP", spot=4, mono=time.monotonic() - 1.0),
                                                            dict(kind="WOULD_STOP", spot=2, mono=time.monotonic() - 60.0)]))
    C.atomic_json(shm / "status.json", st)
    e = W.Panel(tmp_path / "state", shm).e5()
    assert e["stop_request_ui"]["active"] and [i["spot"] for i in e["stop_request_ui"]["items"]] == [4]
    assert e["scene"]["source_old"] is True and e["mode"] == "E5_STANDALONE"


def test_web_has_no_command_endpoint(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    import urllib.request
    import urllib.error
    srv = ThreadingHTTPServer(("127.0.0.1", 0), W.make_handler(W.Panel(tmp_path, tmp_path, test_only=True)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    r = urllib.request.urlopen(base + "/api/state")
    assert r.headers["X-LIVE1-MODE"] == "TEST_ONLY" and json.loads(r.read())["meta"]["test_only"] is True
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(base + "/api/state", data=b"{}", method="POST"))
    assert e.value.code == 405
    with pytest.raises(urllib.error.HTTPError) as e2:
        urllib.request.urlopen(base + "/img/live.jpg")
    assert e2.value.code == 404
    srv.shutdown()


# ---------------------------------------------------------------- approvals
def test_approval_status_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "APPROVALS", tmp_path)
    assert C.approval_status("A")["why"] == "NO_APPROVAL_RECORD"
    text = "[사용자 승인 A] test text typed by the user"
    rec = dict(user_text=text, user_text_sha256=hashlib.sha256(text.encode()).hexdigest(), boot_id=C.boot_id(),
               expires_epoch=time.time() + 60, modules=["lio"])
    C.atomic_json(tmp_path / "A_1.json", rec)
    assert C.approval_status("A", "lio")["valid"] and C.approval_status("A", "observe")["why"] == "MODULE_NOT_APPROVED:observe"
    C.atomic_json(tmp_path / "A_2.json", dict(rec, expires_epoch=time.time() - 1))
    assert C.approval_status("A", "lio")["why"] == "APPROVAL_EXPIRED"
    C.atomic_json(tmp_path / "A_3.json", dict(rec, boot_id="other"))
    assert C.approval_status("A", "lio")["why"] == "APPROVAL_FOR_OTHER_BOOT"
    C.atomic_json(tmp_path / "A_4.json", dict(rec, user_text="changed after recording"))
    assert C.approval_status("A", "lio")["why"] == "APPROVAL_TEXT_HASH_MISMATCH"


# ---------------------------------------------------------------- supervisor (own child only)
def _sup(tmp_path, extra, child):
    return subprocess.Popen([sys.executable, "-I", str(CAND / "live1" / "live1_supervisor.py"), "--log", str(tmp_path / "s.jsonl"),
                             "--state", str(tmp_path / "st.json"), "--stop-file", str(tmp_path / "stop"), "--module", "t",
                             "--start-temp", "200", "--start-mb", "0", "--wait-start-s", "2", *extra, "--", "bash", "-c", child],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)


def test_supervisor_stop_file_stops_child_tree(tmp_path):
    p = _sup(tmp_path, ["--max-s", "60"], "sleep 30 & sleep 30")
    time.sleep(1.5)
    (tmp_path / "stop").write_text("x")
    p.wait(20)
    st = json.loads((tmp_path / "st.json").read_text())
    assert st["status"] == "ENDED" and st["end_reason"] == "STOP_REQUESTED" and st["leftovers"] == []


def test_supervisor_sigterm_is_orderly(tmp_path):
    p = _sup(tmp_path, ["--max-s", "60"], "sleep 30")
    time.sleep(1.5)
    p.terminate()
    p.wait(20)
    st = json.loads((tmp_path / "st.json").read_text())
    assert st["end_reason"] == "STOP_REQUESTED" and st["leftovers"] == []


def test_supervisor_no_resource_abort_keeps_fastlio_child(tmp_path):
    p = _sup(tmp_path, ["--max-s", "60", "--abort-temp", "0", "--no-resource-abort"], "sleep 2")
    p.wait(20)
    st = json.loads((tmp_path / "st.json").read_text())
    assert st["end_reason"].startswith("CHILD_EXITED")
    assert any(json.loads(line)["kind"] == "RESOURCE_ALERT_NO_ABORT" for line in open(tmp_path / "s.jsonl"))


# ---------------------------------------------------------------- observe guard (before any heavy import)
@pytest.mark.parametrize("env,msg", [(dict(LIVE1_RUN_S="900", LIVE1_RUN_S_APPROVED_MAX="300", SCI_E5_OBSERVE_MODE="shadow"), "exceeds approved"),
                                     (dict(LIVE1_RUN_S="60", LIVE1_RUN_S_APPROVED_MAX="300", SCI_E5_OBSERVE_MODE="walking"), "shadow only"),
                                     (dict(LIVE1_RUN_S="60", LIVE1_RUN_S_APPROVED_MAX="300", SCI_E5_OBSERVE_MODE="shadow",
                                           SCI_PANTILT_APPROVED="1"), "shadow only")])
def test_observe_node_refuses_before_loading_anything(env, msg):
    e = dict(os.environ, SCI_WS=str(CAND), **env)
    e.pop("SCI_PANTILT_APPROVED", None) if "SCI_PANTILT_APPROVED" not in env else None
    r = subprocess.run([sys.executable, "-I", str(CAND / "live1" / "observe_node.py")], env=e, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and msg in r.stderr and "torch" not in r.stderr


def test_observe_node_refuses_foreign_sci_ws():
    e = dict(os.environ, SCI_WS=str(Path.home() / "sci_ws"), LIVE1_RUN_S="60", LIVE1_RUN_S_APPROVED_MAX="300", SCI_E5_OBSERVE_MODE="shadow")
    r = subprocess.run([sys.executable, "-I", str(CAND / "live1" / "observe_node.py")], env=e, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "is not this LIVE1 candidate" in r.stderr


def test_supervisor_ignores_other_supervisors_protect_arguments(tmp_path):
    """G1 field bug 06:56: a 2nd supervisor's '--protect realsense2_camera_node' argv token was counted as a protected
    process -> the FAST-LIO supervisor saw PROTECTED_PROCESS_CHANGED and stopped FAST-LIO."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sup", CAND / "live1" / "live1_supervisor.py")
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    other = subprocess.Popen([sys.executable, "-I", str(CAND / "live1" / "live1_supervisor.py"), "--log", str(tmp_path / "o.jsonl"),
                              "--max-s", "20", "--start-temp", "200", "--start-mb", "0", "--protect", "zz_protected_name_xyz",
                              "--", "sleep", "20"], start_new_session=True)
    try:
        time.sleep(1.0)
        assert sup.find_pids("zz_protected_name_xyz") == []
    finally:
        other.terminate()
        other.wait(20)


# ---------------------------------------------------------------- OPERATOR_FREE_DRIVE shared hub (RGB lifetime != SigLIP lifetime)
def _img_msg(st, val, w=8, h=6, enc="bgr8"):
    N = types.SimpleNamespace
    ch, bpc = (3, 1) if enc == "bgr8" else (1, 2)
    data = (np.full((h, w, ch), val, np.uint8).tobytes() if enc == "bgr8" else np.full((h, w), val, "<u2").tobytes())
    return N(header=N(stamp=N(sec=int(st), nanosec=int(round((st % 1) * 1e9))), frame_id="cam"), encoding=enc, width=w, height=h,
             step=w * ch * bpc, is_bigendian=0, data=data)


def test_shared_hub_roundtrip_frames_depth_and_odom(tmp_path):
    import shared_hub as SH
    import ros_readonly as RR
    src = RR.RosReadonlySource.__new__(RR.RosReadonlySource)
    src._init_state(contract=None, wall=lambda: 1000.0, visual_identity_pose=True)
    src.visual_only, src.subscribed, src.refused = False, ["/camera/color/image_raw", "/Odometry"], {}
    wr = SH.HubWriter(tmp_path / "hub", src, jpeg_path=tmp_path / "hub" / "live.jpg")
    for i in range(3):
        src._on_color(_img_msg(999.0 + 0.1 * i, 10 * (i + 1)))
        src._on_depth(_img_msg(999.0 + 0.1 * i, 500 + i, enc="16UC1"))
        src._on_odom(_odom(999.0 + 0.1 * i, x=0.1 * i))
        wr.step()
        wr.write_index()
    rd = SH.SharedHubSource(tmp_path / "hub")
    fr = rd.poll()
    assert [f[1] for f in fr] == [2, 3] and fr[-1][2].shape == (6, 8, 3) and int(fr[-1][2][0, 0, 0]) == 30
    assert rd.poll() == []                                                      # nothing new -> nothing returned
    d, dst = rd.depth(999.2)
    assert dst == pytest.approx(999.2) and int(d[0, 0]) == 502
    assert rd.odom_hist == [] and rd.pose(999.2) == ((0.0, 0.0, 0.0), 999.2)    # no contract: flagged identity only
    assert rd.health()["source"] == "shared_hub" and rd.health()["pose_mode"] == "IDENTITY_FLAGGED_CONTRACT_MISSING"
    assert (tmp_path / "hub" / "live.jpg").exists()
    rd2 = SH.SharedHubSource(tmp_path / "hub")
    rd2.contract, rd2.geometry_status = MIRROR, "CONTRACT_PRESENT"
    rd2._refresh()
    assert len(rd2.odom_hist) == 3 and rd2.odom_hist[-1][1:] == pytest.approx(rd2.pose(999.2)[0])


def test_shared_hub_reader_refuses_stale_index(tmp_path):
    import shared_hub as SH
    (tmp_path / "hub").mkdir()
    (tmp_path / "hub" / "index.json").write_text(json.dumps(dict(wall=time.time() - 30, frames=[dict(seq=1, slot=0, w=2, h=2, stamp=1.0, rx_mono_ns=0)])))
    rd = SH.SharedHubSource(tmp_path / "hub")
    assert rd.poll() == [] and rd.hub_stale == 1
