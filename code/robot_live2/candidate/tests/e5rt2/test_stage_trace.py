"""E5RT2 A28 full stage trace on REAL candidate methods (ROBOT_CPU_FAKE):
scene result -> semantic owner (handle_scene: P2 VERIFIED world track) -> current track / P2 STOP -> event record
-> planner request (aim-time ObservationSnapshot) -> fake console capture -> reader job -> reader result
-> original event (store) -> UI status.  Every stage is asserted through the record it leaves behind."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_e5rt2 import F, J, N, RC, TE, TR, env, gnode, tex, write_gate  # noqa: E402,F401

CS, RS = RC.CaptureStatus, RC.ReleaseStatus


def _region(fid_unused=None):
    return dict(box=[380.0, 200.0, 430.0, 240.0], tile=[380.0, 200.0, 430.0, 240.0], sim=0.15, msim=0.15, like=True, spot=11, nobs=3,
                nscene=3, xyz=[2.5, -0.35, 0.1], z=2.5, hgt=1.3, board=-1.0, adv=0.03, neg={}, ctx={}, ctx_plate=None,
                score_status="FRESH", fired=False)


def test_A28_stage_trace_scene_to_reader_to_event_and_ui(env, monkeypatch):
    write_gate(env, monkeypatch, estop=False, operator_abort=False)
    con = F.FakeConsole()
    node, clk = gnode(env, con, stage="P2")
    img = tex(3)
    depth = np.full((480, 640), 2500, np.uint16)
    src = F.FakeSource()
    src.odom_hist = [(100.0 + 0.1 * i, 0.0, 0.0, 0.0) for i in range(40)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for fid in range(1, 9):
        st = 100.0 + 0.1 * fid
        node.note_frame(src, st, fid)
        node.frontend.buf.items.append(dict(id=fid, tb=0.1 * fid, th=st, bgr=img, gray=gray))
        node.track_frame(fid, st, img)

    def scene(t, fid):
        return dict(t=t, base=dict(id=fid, bgr=img, tb=0.1 * fid), regs=[], vp=None, thr=0.10, legacy_ev=None,
                    timing=dict(scene_ms=1.0), regions=[_region()])
    # stage 1-3: scene result -> P2 owner -> VERIFIED world track -> STOP proposal
    node.handle_scene(src, scene(1.0, 4), 1.0, 100.4, depth, (0.0, 0.0, 0.0), img, 1, 0, origin=100.0, depth_stamp=100.4, pose_stamp=100.4)
    tr = node.p2.tracks[11]
    assert tr["n_verify"] == 1 and tr["box_frame"]["frame_id"] == 4
    stops = [json.loads(x) for x in (node.run / "stops.jsonl").read_text().splitlines()]
    assert stops and stops[-1]["stage"] == "P2" and stops[-1]["spot"] == 11, stops
    assert node.rt_tracker.tracks[11]["status"] == "LIVE" and node.rt_tracker.tracks[11]["frame_id"] == 8   # caught up to newest
    # stage 4: event record (single FSM owner)
    node.event_step(src, None)
    assert node.fsm is not None and node.fsm.stop["spot"] == 11
    key = node.fsm.key
    rec = node.store.get(key)
    stop_oid = rec["observation_stop_id"]
    assert rec["observation"]["source_frame_id"] == 4 and stop_oid
    # a second scene of the same sign while the event waits for the base stop (more evidence, no new STOP)
    node.handle_scene(src, scene(2.0, 8), 2.0, 100.8, depth, (0.0, 0.0, 0.0), img, 2, 0, origin=100.0, depth_stamp=100.8, pose_stamp=100.8)
    assert node.p2.tracks[11]["n_verify"] == 2 and node.p2.tracks[11]["box_frame"]["frame_id"] == 8
    # stage 5-6: planner request uses the aim-time hand-off; fake console capture of THIS request
    TE.drive(node, src, clk, 6.0, until=lambda n: not n.fsm.active and not n.device.busy)
    rec = node.store.get(key)
    oid = rec["observation_id"]
    ob = rec["observation"]
    assert oid != stop_oid and ob["source_frame_id"] == 8 and ob["geometry_confirmed"], ob["geometry_reasons"]
    assert ob["relative_target"] and ob["relative_target"]["source"].startswith("P2.view_from")
    assert rec["aim_request"]["observation_id"] == oid and rec["capture_status"] == CS.CAPTURE_QUALITY_OK
    aim_reqs = [q["req"] for p, q in con.calls if p == "/aim"]
    assert aim_reqs == [rec["request_id"]]
    # stage 7-8: reader job -> reader result
    job = json.loads(next((node.run / "jobs" / "pending").glob("*.json")).read_text())
    assert (job["event_key"], job["capture_id"], job["request_id"], job["observation_id"]) == (key, rec["capture_id"], rec["request_id"], oid)
    core = J.ReaderCore(node.run, dict(TR.CFG), ocr_fn=TR.FakeOcr([("418", 0.95)]), engine_info=TR.ENGINE, vlm_fn=None,
                        admit_fn=lambda n: (False, "fake"), clock=TR.Clock(), wall=lambda: 1.7e9, boot_id="B")
    for _ in range(4):
        core.step()
    done = json.loads((node.run / "jobs" / "done" / f"{J.safe(key)}.json").read_text())
    assert done["observation_id"] == oid and done["terminal"]
    # stage 9-10: original event + UI status
    node.event_step(src, None)
    rec = node.store.get(key)
    assert rec["reading"]["observation_id"] == oid and rec["release"] == RS.RELEASE_READY_CAPTURE_OK
    st = json.loads((N.SHM / "status.json").read_text())
    ev = [e for e in st["events"] if e["event_key"] == key][0]
    assert ev["capture_id"] == rec["capture_id"] and ev["reading"]["observation_id"] == oid
    fsm = [json.loads(x) for x in (node.run / "fsm.jsonl").read_text().splitlines()]
    seq = [(r["state"], r["reason"]) for r in fsm]
    for need in (("STOP_REQUEST", "policy_pass"), ("OBSERVATION", "handoff_built"), ("OBSERVATION", "handoff_rebuilt_at_aim"),
                 ("CAPTURE_QUALITY_OK", "quality_ok")):
        assert need in seq, (need, seq)
    trace = dict(scene_source_frames=[4, 8], p2_track=11, stop_observation_id=stop_oid, aim_observation_id=oid, event_key=key,
                 request_id=rec["request_id"], capture_id=rec["capture_id"], reader_done=done["work_key"], ui_event=ev["event_key"])
    out = Path(os.environ["RT_ROOT"]) / "test_artifacts" / "stage_trace_A28.json"
    out.write_text(json.dumps(dict(trace=trace, fsm=seq), indent=1))
