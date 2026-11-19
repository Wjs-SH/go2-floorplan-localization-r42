"""SEM1 contract tests (ROBOT_CPU_FAKE / LOCAL_CPU_FUNCTION): P01 geometry gate, P02 payload fingerprint + propagation,
P03 UNCALIBRATED STOP block through the REAL event_busy, ros_readonly allowlist, shared_raw refusal, WOULD_STOP
counterfactual separation."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_sensor_bridge", WS / "tests" / "e5rt2", WS / "tests" / "r1"):
    sys.path.insert(0, str(p))
from test_e5rt2 import F, N, OB, RC, TE, env, gnode, write_gate, _confirmed_setup, _stop  # noqa: E402,F401
from e5rt import observation as OBS  # noqa: E402
from e5rt.would_stop import WouldStopEvaluator  # noqa: E402


def _good_depth():
    return OBS.depth_profile(np.full((480, 640), 2000, np.uint16), [300, 220, 340, 260], ref_z=2.02)


# ---------------------------------------------------------------- P01
def test_P01_high_mad_and_nonfinite_and_plane_residual_block_confirmation():
    d = dict(_good_depth())
    assert OBS.geometry_confirmation(d, 2, True)[0]                                          # normal plane passes
    hm = dict(d, mad_m=0.5, high_mad=True, uncertainty="HIGH", risks=["HIGH_MAD:0.50m"])
    ok, why = OBS.geometry_confirmation(hm, 2, True)
    assert not ok and "HIGH_MAD" in why                                                       # review probe P01 (was confirmed)
    noisy = np.full((480, 640), 2000, np.uint16)
    noisy[220:260, 300:340] = (2000 + np.random.RandomState(0).randint(-400, 400, (40, 40))).astype(np.uint16)
    pn = OBS.depth_profile(noisy, [300, 220, 340, 260], ref_z=2.0)
    assert pn["high_mad"] and pn["uncertainty"] == "HIGH" and not OBS.geometry_confirmation(pn, 3, True)[0]
    nf = dict(d, median_m=float("nan"))
    assert "NONFINITE_DEPTH_STATS" in OBS.geometry_confirmation(nf, 2, True)[1]
    ok, why = OBS.geometry_confirmation(d, 2, True, plane=dict(rms_m=0.08, valid_in=0.9))
    assert not ok and any(w.startswith("PLANE_RESIDUAL_HIGH") for w in why)
    ok, why = OBS.geometry_confirmation(d, 2, True, plane=dict(rms_m=0.005, valid_in=0.2))
    assert not ok and any(w.startswith("LOW_ROI_SUPPORT") for w in why)
    assert OBS.geometry_confirmation(d, 2, True, plane=dict(rms_m=0.005, valid_in=0.9))[0]
    assert d["ref_independent"] is False and d["ref_source"].startswith("same_aligned_depth")


# ---------------------------------------------------------------- P02
def test_P02_entity_id_stable_payload_changes_with_geometry_and_revision_increments():
    stop = dict(_stop(), box_frame=dict(frame_id=5, header_stamp=100.0, box=[300, 220, 340, 260]))
    dep = dict(dimg=np.full((480, 640), 2000, np.uint16), depth_stamp=100.0, pose=[0, 0, 0], pose_stamp=100.0)
    a = OBS.build_observation("r", "g", "k", stop, None, None, dep, n_verify=2, plane_ok=True, relative_target=dict(range=2.0),
                              calibration_version="A")
    b = OBS.build_observation("r", "g", "k", stop, None, None, dep, n_verify=2, plane_ok=True, relative_target=dict(range=5.0),
                              calibration_version="B", prev=a)
    c = OBS.build_observation("r", "g", "k", stop, None, None, dep, n_verify=2, plane_ok=True, relative_target=dict(range=5.0),
                              calibration_version="B", prev=b)
    assert a.observation_id == b.observation_id == c.observation_id                          # same entity
    assert a.payload_sha256 != b.payload_sha256 and b.payload_sha256 == c.payload_sha256     # content fingerprint
    assert (a.payload_revision, b.payload_revision, c.payload_revision) == (1, 2, 2)


def test_P02_payload_propagates_to_aim_reader_store_and_mismatch_refused(env, monkeypatch):
    import r1_jobs as J
    import test_r1_reader as TR
    write_gate(env, monkeypatch, estop=False)
    con = F.FakeConsole()
    node, clk = gnode(env, con)
    _confirmed_setup(node)
    node.start_event(_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None, latest_frame=dict(frame_id=9, header_stamp=101.0))
    TE.drive(node, F.FakeSource(), clk, 6.0, until=lambda n: n.fsm is not None and not n.fsm.active and not n.device.busy)
    rec = node.store.get(node.fsm.key)
    ps = rec["payload_sha256"]
    assert ps and rec["aim_request"]["payload_sha256"] == ps and rec["observation"]["payload_sha256"] == ps
    job = json.loads(next((node.run / "jobs" / "pending").glob("*.json")).read_text())
    assert job["payload_sha256"] == ps and job["payload_revision"] == rec["payload_revision"]
    core = J.ReaderCore(node.run, dict(TR.CFG), ocr_fn=TR.FakeOcr([("418", 0.95)]), engine_info=TR.ENGINE, vlm_fn=None,
                        admit_fn=lambda n: (False, "fake"), clock=TR.Clock(), wall=lambda: 1.7e9, boot_id="B")
    for _ in range(4):
        core.step()
    node.event_step(F.FakeSource(), None)
    rec = node.store.get(node.fsm.key)
    assert rec["reading"]["payload_sha256"] == ps
    ev = [e for e in node.status["events"] if e["event_key"] == node.fsm.key][0]
    assert ev["payload_sha256"] == ps                                                         # UI status carries it
    assert node.store.attach_result(dict(rec["reading"], payload_sha256="other", result_revision=99)) == "PAYLOAD_MISMATCH"


# ---------------------------------------------------------------- P03 (existing block preserved, REAL event_busy)
def test_P03_uncalibrated_blocks_stop_via_real_event_busy_and_calibrated_legacy_still_proposes(env, monkeypatch):
    monkeypatch.setenv("SCI_E5RT2_THRESHOLD_MODE", "calibrated")
    n1, _ = gnode(env, F.FakeConsole(), run="p03_uncal")
    assert n1.threshold_mode.state == "UNCALIBRATED" and n1.event_busy() is True
    assert N.Node.event_busy(n1) is True                                                      # the SCI base method itself
    monkeypatch.setenv("SCI_E5RT2_THRESHOLD_MODE", "legacy_p90")
    write_gate(env, monkeypatch)
    n2, _ = gnode(env, F.FakeConsole(), run="p03_legacy")
    assert n2.threshold_mode.state == "legacy_p90" and N.Node.event_busy(n2) is False


# ---------------------------------------------------------------- F01/F02 sources
def test_F01_ros_readonly_allowlist_refuses_other_topics():
    import ros_readonly as RR
    for t in ("/lidar_points", "/utlidar/robot_odom", "/camera/imu", "/cmd_vel"):
        with pytest.raises(PermissionError):
            RR.check_allowed(t)
    assert RR.check_allowed("/Odometry") == "nav_msgs/msg/Odometry"
    src = Path(RR.__file__).read_text()
    assert "create_publisher" not in src and "create_client" not in src and "VideoCapture" not in src


def test_F02_shared_raw_without_producer_refuses_and_never_falls_back_to_ros():
    import argparse
    with pytest.raises(OB.SharedRawUnavailable):
        OB._source(argparse.Namespace(source="shared_raw", odom_topic="/Odometry"))


def test_F01_shadow_launcher_requires_shadow_and_run_limit():
    src = (WS / "tools" / "sem1_shadow_node.py").read_text()
    assert 'SCI_E5_OBSERVE_MODE") == "shadow"' in src and "run_s <= 40.0" in src and "SCI_PANTILT_APPROVED" in src


# ---------------------------------------------------------------- F04 counterfactual evaluator
def test_F04_would_stop_is_counterfactual_memory_only(tmp_path):
    w = WouldStopEvaluator(path=str(tmp_path / "ws.jsonl"), dedup_s=30.0, clock=lambda: 100.0)
    summ = [dict(spot=3, state="VERIFIED", why=[], pred={"range": 2.0}), dict(spot=4, state="DEFER_GEOMETRY_UNAVAILABLE", why=["no_odom"])]
    out = w.on_tick(1.0, summ)
    assert [o["kind"] for o in out] == ["WOULD_STOP", "WOULD_DEFER"]
    assert w.on_tick(1.1, summ) == []                                                          # deduped, no count inflation
    assert summ[0]["state"] == "VERIFIED"                                                      # P2 state untouched
    assert w.counts == dict(ticks=2, would_stop=1, would_defer=1)


def test_F04_shadow_node_records_would_defer_with_real_p2_tick(env, monkeypatch):
    monkeypatch.setattr(OB, "OBSERVE_MODE", "shadow")
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    node, _ = gnode(env, F.FakeConsole(), run="shadow_ws", stage="P2")
    assert node.would_stop is not None and node.event_busy()                                   # shadow: no request
    node.p2.tracks[3] = dict(spot=3, state="VERIFIED", n_verify=2, xy=np.array([2.0, -0.2]), n_xy=np.array([-1.0, 0.0]), up=0.1,
                             h_m=0.2, last_t=0.0, box=[0, 0, 1, 1], tile=[0, 0, 1, 1], z_last=2.0, sim=0.2, t_verified=0.0)
    src = F.FakeSource()                                                                        # no odometry: geometry deferred
    src.pose = lambda stamp, max_gap=0.3: ((0.0, 0.0, 0.0), stamp)
    node.p2_tick(src, 100.0, 1.0)
    st = node.would_stop.status()
    assert st["counts"]["would_defer"] == 1 and st["counts"]["would_stop"] == 0 and node.fsm is None
    assert not node.proposals.take()


# ---------------------------------------------------------------- D01/D02 value-ordered budget (runtime wrapper)
def test_D02_value_budget_reserves_exploration_and_prioritises_unconfirmed_near_threshold():
    import types
    from e5rt.budget import BudgetedSiglip, value_plan, n_tiles
    regs = [[0, 0, 300, 200], [300, 0, 640, 200], [10, 250, 60, 300], [500, 300, 550, 350], [100, 300, 160, 350]]
    # FOLLOW-UP: keys renamed to the shared contract (semantic_confirmed / observed); reserve picks the cheapest NEW spot
    info = [dict(new=False, sims=[0.20, 0.21], waited_s=0.1, semantic_confirmed=True),      # already high + confirmed
            dict(new=False, sims=[0.09, 0.091], waited_s=2.0, semantic_confirmed=False),    # near threshold, needs confirmation
            None, dict(new=True), dict(new=False, sims=[0.05], waited_s=5.0, semantic_confirmed=False)]
    send, deferred, inf = value_plan(regs, 12, info, thr=0.092)
    assert inf["binding"] and inf["crops_planned_upper"] <= 12
    assert any(i in send for i in (2, 3))                  # exploration reserve: a NEW spot is always served
    assert 1 in send                                       # the unconfirmed near-threshold spot is served
    assert 0 not in send                                   # the confirmed high scorer does not monopolise the budget
    class W:
        def score(self, img, rr):
            n = len(rr); self.last_n_tiles = sum(n_tiles(b) for b in rr)
            self.last_adv, self.last_board, self.last_ceil = [0.0] * n, [-1.0] * n, [-1.0] * n
            self.last_tiles, self.last_emb, self.last_ctx_plate, self.last_neg, self.last_ctx = [list(r) for r in rr], [b"e"] * n, [0.1] * n, {}, {}
            return np.full(n, 0.1), 0.0
    L = types.SimpleNamespace(spots=[dict(sims=[0.2, 0.21]), dict(sims=[0.09, 0.091])], seen=[0.05] * 60 + [0.2] * 6, fired={0})
    b = BudgetedSiglip(W(), legacy_events=L, max_crops=16, policy="value")   # new spots (6) + near-threshold spot (9) fit
    spot_of = {0: 0, 1: 1}
    b.set_context(region_world=lambda bx, d, vp, pose: ([float(bx[0]), 0.0, 1.0], 1.0), dimg=None, vp=None, pose=None,
                  match=lambda spots, xyz, r, v, taken: spot_of.get(0 if xyz[0] == 0 else (1 if xyz[0] == 300 else 99)),
                  P=dict(radius=0.7, quant=0.9, warm=50))
    b.score(np.zeros((480, 640, 3), np.uint8), regs)
    st = b.last_score_status
    assert st[0] == "DEFERRED_BUDGET" and st[1] == "FRESH" and b.last_budget["policy"] == "value"
    assert L.no_evidence_keys == {tuple(float(v) for v in regs[i]) for i, s in enumerate(st) if s != "FRESH"}


# ---------------------------------------------------------------- B02 / B05
def test_B02_context_preview_or_reencoded_image_is_not_encoder_input(tmp_path):
    import cv2, hashlib
    from e5rt.data_audit import verify_encoder_input
    rgb = (np.random.RandomState(1).rand(224, 224, 3) * 255).astype(np.uint8)
    meta = dict(rgb_uint8_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    png = tmp_path / "tile00.png"
    cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert verify_encoder_input(png, meta) == (True, "EXACT_ENCODER_INPUT")
    jpg = tmp_path / "preview.jpg"
    cv2.imwrite(str(jpg), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert verify_encoder_input(jpg, meta)[0] is False
    big = tmp_path / "ctx.png"
    cv2.imwrite(str(big), np.zeros((300, 400, 3), np.uint8))
    assert verify_encoder_input(big, meta)[1].startswith("SHAPE_")
    tweaked = rgb.copy(); tweaked[0, 0, 0] ^= 1
    p2 = tmp_path / "t.png"; cv2.imwrite(str(p2), cv2.cvtColor(tweaked, cv2.COLOR_RGB2BGR))
    assert verify_encoder_input(p2, meta)[1] == "SHA_MISMATCH_WITH_DUMP_RECORD"


def test_B05_same_physical_target_in_two_splits_is_a_leak():
    from e5rt.data_audit import split_leaks
    items = [dict(physical_target_id="S1F-101", split="DEV"), dict(physical_target_id="S1F-101", split="FINAL"),
             dict(physical_target_id="S6F-601", split="DEV"), dict(physical_target_id=None, split="FINAL")]
    assert split_leaks(items) == {"S1F-101": ["DEV", "FINAL"]}
