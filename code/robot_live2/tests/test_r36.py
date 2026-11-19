"""R36 tests. Real code; REAL = real field records (no device, no robot). SYNTHETIC cases say so."""
import json
import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
sys.path.insert(0, str(HERE))
import e6_core as G  # noqa: E402
import e6_precision_bridge as B  # noqa: E402
import e6_tracker as TR  # noqa: E402
import e7_long as EL  # noqa: E402
import e8_pair as EP  # noqa: E402
import obs_flow as OF  # noqa: E402
import status_view as SVW  # noqa: E402
import visual_stop as VS  # noqa: E402
import vlm_ondemand as VO  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
ENV = dict(pan=(60.0, 135.0), tilt=(135.0, 167.189), source="operator envelope 2026-10-27")
K = json.loads((OPR / "experiments/X20261102_025326/archive/hub_index.json/index.json").read_text())["caminfo"]
K = K.get("k") or K.get("K")


def obs_of(run):
    out = {}
    for ln in open(OPR / "experiments" / run / "state/e6_observations.jsonl"):
        d = json.loads(ln)
        if d.get("state") == "STOPPED_TARGET_READY":
            out[d["observation_id"][-4:]] = d
    return out


def region(box, z, sim=0.13, msim=0.13, spot=1, xyz=(4.0, 1.0, 1.3)):
    return dict(box=list(box), z=z, sim=sim, msim=msim, spot=spot, xyz=list(xyz), dec=dict(P2=dict(state="VERIFIED")))


# ---------------- E5 gates (boxes from the REAL labelled 11-02 stops)
def vs(tmp_path=None, **k):
    return VS.VisualStop(tmp_path, K_fn=lambda: K, envelope=ENV, **k)


def feed1(v, r, t=1000.0):
    line = dict(frame_id=1, odom_raw=dict(x=0.0, y=0.0, yaw=0.0, status="INTERPOLATED"), depth_dt_s=0.01)
    return v.feed(dict(t=1.0, regions=[r]), t, line, now=t)


def test_vending_machine_cabinet_and_lock_do_not_stop_the_robot():
    assert feed1(vs(), region([428, 0, 538, 195], 3.279, sim=0.096, msim=0.096)) is None        # O001 vending machine
    assert feed1(vs(), region([369, 100, 463, 153], 4.05, sim=0.095, msim=0.081)) is None        # O002 cabinet
    assert feed1(vs(), region([508, 180, 556, 273], 4.216, sim=0.099, msim=0.099)) is None       # O006 door handle


def test_a_real_plate_still_stops_the_robot():
    o = feed1(vs(), region([105, 106, 135, 152], 4.842, sim=0.134, msim=0.133))                  # O003, the plate that read 405
    assert o and o["kind"] == "VISUAL_VERIFIED_STOP"
    o = feed1(vs(), region([44, 16, 104, 89], 4.842, sim=0.098, msim=0.117))                     # 10-31 hallway sign: below the sim floor, kept by the mean
    assert o


def test_low_object_is_refused_by_height_only_with_a_known_camera():
    r = region([508, 180, 556, 273], 4.216, sim=0.15, msim=0.15)
    v = vs()
    assert feed1(v, dict(r)) is None and v.n_gate.get("SUPPRESSED_BELOW_PLATE_HEIGHT") == 1
    v2 = VS.VisualStop(None, K_fn=lambda: None)
    assert feed1(v2, dict(r))                                                                     # no K: the gate does not guess


def test_target_too_close_to_aim_is_not_requested():
    v = vs()
    assert feed1(v, region([51, 31, 86, 93], 3.279, sim=0.125, msim=0.125)) is None              # REAL O004: refused at the stop in the field (tilt 137)
    assert v.n_gate.get("SUPPRESSED_NOT_AIMABLE") == 1


def test_strongest_region_of_a_scene_is_chosen(tmp_path):
    a, b = region([105, 106, 135, 152], 4.842, sim=0.112, msim=0.112, spot=2), region([177, 72, 204, 124], 4.024, sim=0.134, msim=0.134, spot=3)
    v = vs()
    o = v.feed(dict(t=1.0, regions=[a, b]), 1000.0, dict(frame_id=1, odom_raw=dict(x=0.0, y=0.0, yaw=0.0), depth_dt_s=0.0), now=1000.0)
    assert o["target_track_id"] == 3


def test_already_photographed_plate_is_not_requested_again(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    st = tmp_path / "state"; st.mkdir()
    v = VS.VisualStop(run, K_fn=lambda: K, envelope=ENV, state_dir=st)
    box = [105, 106, 135, 152]
    o = feed1(v, region(box, 4.842, spot=1), 1000.0)
    assert o
    (st / "e6_precision_results.jsonl").write_text(json.dumps(dict(observation_id=o["observation_id"], status="CAPTURE_STORED")) + "\n")
    assert feed1(v, region(box, 4.842, spot=2), 1200.0) is None                                   # 200 s later, new spot id, same place
    assert v.n_gate.get("SUPPRESSED_ALREADY_PHOTOGRAPHED") == 1
    assert "SUPPRESSED_ALREADY_PHOTOGRAPHED" in (run / "live2_stop_suppressed.jsonl").read_text()


# ---------------- aim: same object, plate height (REAL 11-02 records)
def test_redetection_refuses_another_object_REAL():
    O = obs_of("X20261102_023712")["O007"]
    rd = O["redetection"]
    g = dict(box=rd["box"], z=rd["depth_m"], sim=rd["sim"])
    pc = O["pantilt_command_prediction"]
    why = TR.same_object(g, dict(z=pc["predicted_depth_m"], up=O["target_pose_in_odom"]["up_m"], K=K))
    assert why and ("sim" in why or "depth" in why)                                                # sim 0.047, 1.6 m where the plate was 3.2 m away


def test_large_region_keeps_the_predicted_height_REAL():
    O = obs_of("X20261102_023712")["O008"]
    rd = O["redetection"]
    pc = O["pantilt_command_prediction"]
    tr = TR.E6Tracker.__new__(TR.E6Tracker)
    st = dict(scene=dict(stamp=1000.0, regions=[dict(box=rd["box"], z=rd["depth_m"], sim=rd["sim"])]))
    p = Path(os.environ.get("TMPDIR", "/tmp")) / f"r36_e5_{os.getpid()}.json"
    p.write_text(json.dumps(st)); tr.e5_status = p
    r = tr.redetect(pc["predicted_pixel"][0], pc["predicted_pixel"][1], 1000.0, expect=dict(z=pc["predicted_depth_m"], up=O["target_pose_in_odom"]["up_m"], K=K))
    p.unlink()
    assert r["status"] == "REDETECTED" and r["pixel_basis"] == "PREDICTION_IN_LOOSE_BOX"
    assert r["pixel"][1] < (rd["box"][1] + rd["box"][3]) / 2 - 20                                  # aimed above the door lock, at the plate's height


def test_tall_box_aims_at_plate_height():
    box = [474, 111, 534, 271]                                                                     # REAL O007 loc2: plate at the top, lock + handle below
    _, _, up = G.target_in_body(box, 4.731, K)
    up2, basis = TR.aim_up(box, 4.731, K, up)
    assert basis == "PLATE_HEIGHT_IN_TALL_BOX" and up < 1.4 and abs(up2 - TR.PLATE_UP_M) < 1e-6
    tight = [105, 106, 135, 152]
    _, _, u3 = G.target_in_body(tight, 4.842, K)
    assert TR.aim_up(tight, 4.842, K, u3) == (u3, "BOX_CENTRE") or TR.aim_up(tight, 4.842, K, u3)[1] == "PLATE_HEIGHT_IN_TALL_BOX"


@pytest.mark.parametrize("oid", ["O004", "O006", "O007"])
def test_refused_targets_get_a_way_back_inside_the_envelope_REAL(oid):
    O = obs_of("X20261102_025326")[oid]
    g = TR.reposition_guidance(O, K, ENV)
    assert g["ok"] and g["moves"] and "m" in g["text"] or "회전" in g["text"]
    f0, l0 = O["target_pose_at_stop"]["fwd"], O["target_pose_at_stop"]["left"]
    up = O["target_pose_in_odom"]["up_m"]
    for m in g["moves"]:
        if m["kind"] == "BACK":
            pc = G.calibrated_aim(*G.body_to_pixel(f0 + m["m"], l0, up, K))
            assert B.path_check(pc["pan_cmd"], pc["tilt_cmd"], ENV)["ok"]                          # no limit changed: the move really brings it inside


def test_repredict_uses_the_new_pose_and_keeps_the_target_REAL():
    O = obs_of("X20261102_025326")["O006"]
    p = O["odom_pose_at_stop"]
    n = TR.repredict(O, [(p["t"], p["x"], p["y"], p["yaw"])], K)
    assert n["target_pose_in_odom"] == O["target_pose_in_odom"] and n["redetection"]["status"].startswith("NOT_ATTEMPTED")
    assert abs(n["target_pose_at_stop"]["fwd"] - O["target_pose_at_stop"]["fwd"]) < 0.02


# ---------------- E6 flow after the photo
def flow_pair(state):
    rec = dict(status="CAPTURE_STORED", home_confirmed=True, wall=100.0)
    o = dict(observation_id="O1", state="STOPPED_TARGET_READY", stopped_wall=90.0, created_wall=80.0)
    pair = dict(state=state, processed=["O1"], resume_ok=True, since=95.0)
    return OF.flow([o], {"O1": rec}, {}, None, True, True, False, 101.0, pair=pair, manual=dict(observation_id="O1", phase="HOME_CONFIRMED", phase_since=99.0))


@pytest.mark.parametrize("st", ["PAIR_READING", "PAIR_CONSISTENCY_CHECK", "REGION_CANDIDATE_READY", "E7_VERIFYING", "OBS1_WAITING_RESUME", "MORE_OBSERVATION_REQUIRED"])
def test_reading_and_e7_never_hold_the_robot(st):
    assert flow_pair(st)["state"] == "RESUME_READY"


def test_faults_still_hold_the_robot():
    assert flow_pair("HOLD_ERROR")["state"] == "HOLD_ERROR"
    o = dict(observation_id="O1", state="STOPPED_TARGET_READY", stopped_wall=90.0, created_wall=80.0)
    f = OF.flow([o], {"O1": dict(status="CAPTURE_STORED", home_confirmed=False, wall=100.0)}, {}, None, True, True, False, 101.0, pair=dict(state="PAIR_READING", processed=["O1"]))
    assert f["state"] == "HOLD_ERROR"                                                              # HOME not confirmed


def test_home_not_finished_still_holds():
    o = dict(observation_id="O1", state="STOPPED_TARGET_READY", stopped_wall=90.0, created_wall=80.0)
    f = OF.flow([o], {}, {}, None, True, True, False, 101.0, pair=dict(state="PAIR_READING", processed=[]), manual=dict(observation_id="O1", phase="RETURNING_HOME", phase_since=99.0))
    assert f["state"] == "RETURNING_HOME"


def test_siglip_stays_active_in_every_reading_state():
    pm = EP.PairManager.__new__(EP.PairManager)
    pm.clock = lambda: 100.0
    pm.since = 0.0
    for s in ("PAIR_READING", "PAIR_CONSISTENCY_CHECK", "REGION_CANDIDATE_READY", "E7_VERIFYING"):
        pm.status = s
        assert pm.siglip_override(dict(mode="IDLE_OK", why="x"))["mode"] == "ACTIVE" and pm.resume_ok()
    pm.status = "OBS2_E6_RUNNING"
    assert not pm.resume_ok()


def test_reposition_is_a_manual_flow_state():
    assert OF.MANUAL_FLOW["REPOSITION_NEEDED"] == "REPOSITION_NEEDED"


# ---------------- whole-path comparison has no authority
def fl(*a):
    return [dict(floor=f, cost=c, exact_done=x, stage="exact" if x else "coarse") for f, c, x in a]


def test_long_rival_without_exact_search_never_confirms():
    h = dict(t_end=1, path_m=50, floors=fl((4, 0.03, True), (8, 0.15, False)), floors_exact=[4], n_refresh=5)
    assert EL.classify([h, dict(h, path_m=80)], 2)["status"] == "LEADER_UNRESOLVED"


def test_long_close_exact_rival_is_ambiguous_REAL_numbers():
    h = dict(t_end=1, path_m=91, floors=fl((4, 0.025, True), (8, 0.032, True), (9, 0.081, True)), floors_exact=[4, 8, 9])   # 11-01 loc1 +270 s
    c = EL.classify([h], 3)
    assert c["status"] == "AMBIGUOUS_FLOORS" and c["close_floors"] == [8]


def test_long_would_confirm_needs_two_evaluations_and_new_path():
    a = dict(t_end=1, path_m=50, floors=fl((4, 0.03, True), (8, 0.15, True)), floors_exact=[4, 8], n_refresh=5)
    assert EL.classify([a], 2)["status"] == "LEADER_UNRESOLVED"
    b = dict(a, t_end=2, path_m=52)
    assert EL.classify([a, b], 2)["status"] == "LEADER_UNRESOLVED"
    c = dict(a, t_end=3, path_m=60)
    assert EL.classify([a, c], 2)["status"] == "WOULD_CONFIRM"


def test_long_module_cannot_write_a_fix():
    src = (HERE.parent / "candidate/live2/e7_long.py").read_text()
    assert "e7_state.json" not in src and "GLOBAL_CONFIRMED" not in src


# ---------------- VLM admission (SYNTHETIC limits, real class; the server itself: VLM_MEASUREMENT.md)
def test_vlm_refuses_low_memory_and_low_fastlio_rate(tmp_path, monkeypatch):
    (tmp_path / "lio_state.json").write_text(json.dumps(dict(odom=dict(hz_10s=10.0))))
    v = VO.OnDemandVLM(tmp_path)
    monkeypatch.setattr(VO, "max_temp_c", lambda: 60.0)
    monkeypatch.setattr(VO, "mem_available_mb", lambda: 5000)
    ok, why = v.admit()
    assert not ok and "MemAvailable" in why
    monkeypatch.setattr(VO, "mem_available_mb", lambda: 7000)
    assert v.admit()[0]
    (tmp_path / "lio_state.json").write_text(json.dumps(dict(odom=dict(hz_10s=7.5))))
    ok, why = v.admit()
    assert not ok and "FAST-LIO" in why
    (tmp_path / "lio_state.json").write_text(json.dumps(dict(odom=dict(hz_10s=10.0))))
    monkeypatch.setattr(VO, "max_temp_c", lambda: 79.0)
    ok, why = v.admit()
    assert not ok and "temperature" in why
    assert json.loads((tmp_path / "vlm_state.json").read_text())["state"] == "OFF"


def test_vlm_alone_is_not_a_reading():
    import e6_semantic as S
    alts, info = S.label_alternatives([], dict(sign_type="room_number_plate", room_number="202"))     # REAL: what the VLM invented for the blurry O002
    from e3_tracker import SEM_DECISIVE_P
    assert info["fusion"]["status"] == "VLM_ONLY_UNCONFIRMED" and max((p for _, p in alts), default=0) < SEM_DECISIVE_P


def test_same_image_ocr_and_vlm_are_one_observation():
    import e6_semantic as S
    alts, info = S.label_alternatives([("405", 0.99)], dict(sign_type="room_number_plate", room_number="405"))
    assert info["fusion"]["status"] == "AGREE" and [a[0] for a in alts] == ["405"]                     # one label of one observation; independence is per plate position


def test_vlm_crop_uses_the_operator_box():
    import numpy as np
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert EP._crop_for_vlm(img, None) is img
    c = EP._crop_for_vlm(img, [628, 115, 800, 354])
    assert 0 < c.shape[1] < 1920 and 0 < c.shape[0] < 1080


# ---------------- status chips
def test_status_chips_show_mode_and_no_fix_claim(tmp_path):
    (tmp_path / "e7_long.json").write_text(json.dumps(dict(status="AMBIGUOUS_FLOORS", leader_floor=4, close_floors=[7, 8], next_gap_m=0.003, path_m=64, floors_exact=9, floors_total=9)))
    c = SVW.long_chip(tmp_path)
    assert "4F" in c["text"] and "[7, 8]" in c["text"] and "FIX" not in c["text"]


# ---------------- reposition, real pipeline with the fake aim console (no device). Target and pose: REAL O006 of X20261102_025326, refused in the field (lash point 56.9 < 60)
def real_o006():
    o = obs_of("X20261102_025326")["O006"]
    return dict(o, observation_id="O_m1")


def moved_back(o, d):
    p = o["odom_pose_at_stop"]
    u, v, h = G.raw_to_planar(p["x"], p["y"], p["yaw"])
    import math
    x, y, yaw = G.planar_to_raw(u - d * math.cos(h), v - d * math.sin(h), h)
    return [(p["t"] + 5, x, y, yaw)]


def test_refused_aim_waits_for_the_operator_then_aims_from_the_new_stop(tmp_path):
    import test_e8_manual_focus_fake as F
    o = real_o006()
    g = TR.reposition_guidance(o, K, ENV)
    assert g["ok"]
    c = F.FakeConsole(F.render())
    d = next(m["m"] for m in g["moves"] if m["kind"] == "BACK")
    script = [dict(action="reposition_done")] + F.first(dict(action="capture"))
    p, ch, _ = F.build(tmp_path, c, script=script)
    p.envelope = ENV
    p.reposition = dict(K=lambda: K, samples=lambda: moved_back(o, d))
    rec = p.run(o, 1)
    ph = F.phases(ch)
    assert "REPOSITION_NEEDED" in ph and ph.index("REPOSITION_NEEDED") < ph.index("WAIT_MANUAL_FOCUS")
    assert c.kinds().count("aim") == 1                                     # the refused aim never reached the head
    assert rec["status"] == "CAPTURE_STORED" and rec["repositions"] == 1 and rec["path"]["ok"]
    assert rec["pan_command"] > 60 + 5


def test_refused_aim_skip_moves_nothing(tmp_path):
    import test_e8_manual_focus_fake as F
    o = real_o006()
    c = F.FakeConsole(F.render())
    p, ch, _ = F.build(tmp_path, c, script=[dict(action="reposition_skip")])
    p.envelope = ENV
    p.reposition = dict(K=lambda: K, samples=lambda: moved_back(o, 0.0))
    rec = p.run(o, 1)
    assert rec["status"] == "OUTSIDE_VERIFIED_ENVELOPE" and "aim" not in c.kinds() and "home" not in c.kinds()


def test_without_the_reposition_channel_the_old_refusal_stands(tmp_path):
    import test_e8_manual_focus_fake as F
    c = F.FakeConsole(F.render())
    p, ch, _ = F.build(tmp_path, c, script=[])
    p.envelope = ENV
    rec = p.run(real_o006(), 1)
    assert rec["status"] == "OUTSIDE_VERIFIED_ENVELOPE" and "aim" not in c.kinds()


# ---------------- loc mode through the real console Controller.a_start (module starts recorded, no process started)
class FakeExp:
    def __init__(self):
        import threading
        self.d, self.lock, self.events, self.active = {}, threading.Lock(), [], True

    def event(self, kind, **k):
        self.events.append(dict(k, kind=kind))

    def mod(self, *a, **k):
        pass


def controller(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE.parent / "candidate/live1"))
    import console as CO
    c = CO.Controller.__new__(CO.Controller)
    c.exp, c.guard, c.started = FakeExp(), types.SimpleNamespace(reset=lambda: None), []
    c.archive_state = lambda d: 0
    c.set_pause = lambda *a: None
    c.start_module = lambda job, m, b: c.started.append(m) or True
    monkeypatch.setattr(CO, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(CO, "EVAL", tmp_path)
    monkeypatch.setattr(CO, "PENDING_START", tmp_path / "none.json")
    monkeypatch.setattr(CO, "approval", lambda: dict())
    monkeypatch.setattr(CO, "write_pantilt_consent", lambda *a: None)
    monkeypatch.setattr(CO, "sid_hash", lambda s: "h")
    return c, CO


def test_r39_one_mode_starts_everything(tmp_path, monkeypatch):
    c, CO = controller(tmp_path, monkeypatch)
    r = c.a_start(None, "kci", False, "r", "s", "body", "MANUAL_SETTLE", None, "LOC1_STRUCTURE_ONLY")          # a zone name from an old client changes nothing
    assert r["ok"] and {"observe", "e6", "e2", "e34"} <= set(c.started) and c.exp.d["loc_mode"] == CO.AUTO_MODE


# ---------------- stop rule on real-shaped odometry
def test_standing_jitter_is_stopped_and_walking_is_moving():
    import random
    rng = random.Random(3)
    jit = [(100.0 + i * 0.1, 1.0 + rng.gauss(0, 0.004), 2.0 + rng.gauss(0, 0.004), 0.5) for i in range(16)]   # 4 mm jitter at 10 Hz: the old rule says about 0.05 m/s
    old = sum(((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 for a, b in zip(jit, jit[1:])) / 1.5
    assert old >= 0.05 and G.stop_state(jit, jit[-1][0])["stopped"]
    walk = [(100.0 + i * 0.1, 1.0 + i * 0.008, 2.0, 0.5) for i in range(16)]                                  # 0.08 m/s
    assert not G.stop_state(walk, walk[-1][0])["stopped"]
    turn = [(100.0 + i * 0.1, 1.0, 2.0, 0.5 + i * 0.01) for i in range(16)]                                   # in-place turn 5.7 deg/s
    assert not G.stop_state(turn, turn[-1][0])["stopped"]


def test_long_worker_cores(tmp_path):
    (tmp_path / "ctl").mkdir(); st = tmp_path / "state"; st.mkdir()
    assert EL.LongStructure(st, None).cores() == "0,1,6,7"


# ================ R37
def test_r37_reposition_keeps_the_channel_alive(tmp_path):
    """REAL bug in R36: e6_manual.json was not rewritten while waiting: console refused [재조준] after 5 s, the panel vanished after 30 s."""
    import test_e8_manual_focus_fake as F
    o = real_o006()
    c = F.FakeConsole(F.render())
    p, ch, _ = F.build(tmp_path, c, script=[None] * 8 + [dict(action="reposition_skip")])
    p.envelope = ENV
    p.reposition = dict(K=lambda: K, samples=lambda: moved_back(o, 0.0))
    walls = []
    orig = ch.update
    ch.update = lambda **k: (walls.append(1), orig(**k))[1]
    p.run(o, 1)
    assert len(walls) >= 8


def test_r37_reposition_looks_again_and_refuses_another_object(tmp_path):
    O = obs_of("X20261102_025326")["O006"]
    p = O["odom_pose_at_stop"]
    t = p["t"] + 5
    n0 = TR.repredict(O, [(t, p["x"], p["y"], p["yaw"])], K)
    assert n0["redetection"]["status"] == "NOT_ATTEMPTED_AFTER_REPOSITION"
    u, v = n0["pantilt_command_prediction"]["predicted_pixel"]; z = n0["pantilt_command_prediction"]["predicted_depth_m"]
    box = [u - 15, v - 20, u + 15, v + 20]
    f = tmp_path / "status.json"
    f.write_text(json.dumps(dict(scene=dict(stamp=t, regions=[dict(box=box, z=z * 0.3, sim=0.13)]))))
    n = TR.repredict(O, [(t, p["x"], p["y"], p["yaw"])], K, e5_status=f)
    assert n["redetection"]["status"] == "NOT_REDETECTED" and n["redetection"]["refused"]            # another object at 30 % of the distance
    f.write_text(json.dumps(dict(scene=dict(stamp=t, regions=[dict(box=box, z=z, sim=0.13)]))))
    n = TR.repredict(O, [(t, p["x"], p["y"], p["yaw"])], K, e5_status=f)
    assert n["redetection"]["status"] == "REDETECTED"                                                # the same plate, seen again now


class PMQ:
    pass


def _pm(tmp_path, vlm, admit=None, reader=None):
    from test_e8_pair_fake import Hyp, floors, manager, OCR
    pm, tr, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    pm.vlm, pm.vlm_admit = vlm, admit
    return pm


def _wait(f, s=6.0):
    import time as _t
    t = _t.time()
    while _t.time() - t < s:
        if f():
            return True
        _t.sleep(0.05)
    return False


def test_r37_vlm_reviews_even_when_ocr_read_the_number_and_nothing_waits(tmp_path):
    from test_e8_pair_fake import stored, T401
    import threading, time as _t
    gate = threading.Event()
    def slow(img):
        gate.wait(5)
        return dict(parsed=dict(sign_type="room_number_plate", room_number="401"), model="fake")
    pm = _pm(tmp_path, slow, lambda: (True, "ok"))
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    assert _wait(lambda: pm.group and pm.group[0].get("read"))                 # OCR done while the VLM is still busy
    assert pm.group[0]["vlm"] is None and pm.group[0]["reading"]["vlm"]["state"] in ("QUEUED", "ASKING")
    assert pm.last_eval is not None or _wait(lambda: pm.last_eval is not None)
    gate.set()
    assert _wait(lambda: (pm.group[0].get("vlm") or {}).get("room_number") == "401")
    assert pm.group[0]["reading"]["vlm"]["state"] == "ASKED"
    assert _wait(lambda: "AGREE" in str((pm.last_eval or {}).get("per_observation", [{}])[0].get("fusion_status")))


def test_r37_vlm_refusal_is_named(tmp_path):
    from test_e8_pair_fake import stored, T401
    import e8_pair as E
    E.VLM_TRIES, E.VLM_RETRY_S = 1, 0.01
    pm = _pm(tmp_path, lambda img: pytest.fail("must not run"), lambda: (False, "MemAvailable 3000 MB < 6200 MB"))
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    assert _wait(lambda: pm.group and (pm.group[0].get("reading") or {}).get("vlm", {}).get("state") == "NOT_ADMITTED_RESOURCE")
    assert "MemAvailable" in pm.group[0]["reading"]["vlm"]["why"]
    E.VLM_TRIES, E.VLM_RETRY_S = 4, 20.0


def test_r37_vlm_load_timeout_is_a_named_error(tmp_path):
    from test_e8_pair_fake import stored, T401
    def boom(img):
        raise RuntimeError("no health within 120 s")
    pm = _pm(tmp_path, boom, lambda: (True, "ok"))
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    assert _wait(lambda: pm.group and (pm.group[0].get("reading") or {}).get("vlm", {}).get("state") == "ERROR")
    assert "no health" in pm.group[0]["reading"]["vlm"]["error"] and pm.group[0].get("read") is True


# ---- rule
def shadow_rec(**k):
    return dict(dict(verdict="WOULD_FIX", room_readings=[dict(label="405"), dict(label="402")], vlm_agree=["O4", "O6"], floor_markers=[], floors_explaining=[4]), **k)


def test_r38_fix_rule_only_from_would_fix():
    import e7_semantic_shadow as SH
    assert SH.fix_allowed(shadow_rec(floor_evidence=[dict(kind="DEPARTMENT_ON_2_PLATES")]))[0]
    assert not SH.fix_allowed(shadow_rec(verdict="WOULD_AMBIGUOUS"))[0]
    assert not SH.fix_allowed(None)[0]


def test_r38_plate_text_against_the_map():
    import plate_text as PT, e6_semantic as S, re
    sf = S.SemanticFloors()
    n = PT.person_names(sf.maps)
    assert n["가나다"] == (4, "room_405") and n["사아자"] == (4, "room_402") and n["라마바"] == (5, "room_505")
    for rx, floors in PT.DEPT:                                  # a word never appears in the map outside the floors it claims (the map's own OCR noise hides some, it adds none)
        seen = {f for f, m in sf.maps.items() for s in m["spaces"] if s.get("room_number") and rx.search(str(s.get("name") or ""))}
        assert seen <= set(floors), (rx.pattern, seen, floors)


def _plate(oid, ocr, vlm_text=None, crop=True):
    return dict(observation_id=oid, ocr_lines=[[ocr, .98]], vlm=dict(text=vlm_text, room_number=ocr) if vlm_text else None, reading=dict(vlm=dict(crop=crop)))


def test_r38_digits_contradicted_by_their_own_plate():
    import plate_text as PT, e6_semantic as S
    m = S.SemanticFloors().maps
    j = PT.judge([(_plate("A", "505", "로봇공학과 가나다"), 5), (_plate("B", "502", "로봇공학과"), 5)], m)
    assert j["conflicts"] and not j["ok"]
    j = PT.judge([(_plate("A", "405", "로봇공학과 가나다"), 4), (_plate("B", "402", "로봇공학과"), 4)], m)
    assert j["ok"] and not j["conflicts"] and set(j["dept_plates"]) == {"A", "B"} and j["name_plates"] == ["A"]
    j = PT.judge([(_plate("A", "405", "로봇공학과"), 4), (_plate("B", "402"), 4)], m)
    assert not j["ok"]                                          # one department, no name: not enough
    j = PT.judge([(_plate("A", "405", "로봇공학과 가나다", crop=False), 4), (_plate("B", "402", "전기공학과", crop=False), 4)], m)
    assert not j["ok"] and not j["conflicts"]                   # whole-frame VLM text counts for nothing, either way
    j = PT.judge([(_plate("A", "705", "컴퓨터공학과"), 7), (_plate("B", "702", "컴퓨터공학과"), 7)], m)
    assert not j["ok"] and not j["conflicts"]                   # a word on several floors supports none


def test_r37_policy_default_shadow(tmp_path):
    import e7_policy as P
    assert P.semantic_policy(tmp_path / "none.json", {})["mode"] == "SHADOW"
    f = tmp_path / "p.json"
    f.write_text(json.dumps(dict(semantic_fix="ENABLED")))
    assert P.semantic_policy(f, {})["mode"] == "SHADOW"                         # no report
    f.write_text(json.dumps(dict(semantic_fix="ENABLED", verification_report="x")))
    assert P.semantic_policy(f, {})["mode"] == "ENABLED"
    assert P.semantic_policy(f, dict(LIVE2_SEMANTIC_FIX="0"))["mode"] == "SHADOW"


# ---- estimate view
def _pim(rec, x, y, yaw):
    return rec["pos"][0] + x, rec["pos"][1] + y, yaw


def test_r37_estimate_from_first_reading_moves_with_odometry_and_is_robot_not_plate():
    import semantic_view as V
    ev = dict(per_observation=[dict(observation_id="O1", usable=True, reading_kind="ROOM", labels=[["405", 0.8]], floor_likelihood={"4": 0.86, "5": 0.02}, fusion_status="OCR_ONLY")], verdict="PROVISIONAL")
    hy = [dict(id="A", floor=4, rec=dict(pos=[100.0, 200.0], ppm=25.0), sem_verdict="SEMANTIC_CONSISTENT", score=0.3, n_kf=2, provenance="SEMANTIC_ANCHORED"),
          dict(id="B", floor=2, rec=dict(pos=[0.0, 0.0], ppm=25.0), sem_verdict="FLOOR_MISMATCH", score=0.1, n_kf=9)]
    d = V.build(None, ev, hy, (1000.0, 1.0, 2.0, 0.0), _pim, now=1001.5)
    assert d["status"] == "PROVISIONAL" and d["best"]["floor"] == 4 and d["best"]["pose_px"] == [101.0, 202.0] and d["age_s"] == 1.5
    d2 = V.build(None, ev, hy, (1010.0, 5.0, 2.0, 0.0), _pim, now=1010.2)
    assert d2["best"]["pose_px"] == [105.0, 202.0]
    assert V.build(None, ev, hy[1:], (1000.0, 0, 0, 0), _pim, now=1000.0)["status"] == "WAITING_CANDIDATE"
    assert V.build(None, dict(per_observation=[]), hy, (1000.0, 0, 0, 0), _pim, now=1000.0)["status"] == "NO_READING"


def test_r37_estimate_reports_separated_candidates_instead_of_averaging():
    import semantic_view as V
    ev = dict(per_observation=[dict(observation_id="O1", usable=True, reading_kind="ROOM", labels=[["405", 0.8]], floor_likelihood={"4": 0.86})], verdict="PROVISIONAL")
    hy = [dict(id="A", floor=4, rec=dict(pos=[100.0, 200.0], ppm=25.0), sem_verdict="SEMANTIC_CONSISTENT"), dict(id="B", floor=4, rec=dict(pos=[800.0, 200.0], ppm=25.0), sem_verdict="SEMANTIC_CONSISTENT")]
    d = V.build(None, ev, hy, (1.0, 0, 0, 0), _pim, now=2.0)
    assert d["best"]["id"] == "A" and [a["id"] for a in d["alternatives"]] == ["B"]


# ---- final verification
class _L:
    def __init__(s):
        s.d, s.wanted, s.r = dict(status="LEADER_UNRESOLVED"), 0, None
    def request_now(s):
        s.wanted += 1
    def latest_after(s, w):
        return s.r


class _Link:
    def __init__(s, mode="SHADOW"):
        s.struct_policy = dict(mode=mode)
        s.tr = types.SimpleNamespace(store=types.SimpleNamespace(rows=[(0, 1.0, 0.0, 0.0, 0.0), (1, 2.0, 40.0, 0.0, 0.0)]))
        s.fixed = None
    def write_structure_long_fix(s, r, why):
        s.fixed = why
        return True, why


FL = lambda gap=0.08, conv=True: dict(floors=[dict(floor=4, cost=0.04, inlier=0.6, converged=conv, crossing=0.1), dict(floor=7, cost=0.04 + gap, inlier=0.5, converged=True, crossing=0.1)], path_m=80.0, t_end=2.0)


def fv(tmp_path, link=None, stopped=True):
    import final_verify as FV
    t = [1000.0]
    L, K_ = _L(), link or _Link()
    f = FV.FinalVerify(tmp_path, K_, L, lambda: dict(stopped=stopped), clock=lambda: t[0])
    f.policy_mode = lambda: K_.struct_policy["mode"]
    return f, L, K_, t


def test_r37_no_stop_request_without_evidence_even_after_240_s(tmp_path):
    f, L, _, t = fv(tmp_path)
    for s in ("LEADER_UNRESOLVED", "AMBIGUOUS_FLOORS", "WAITING_MOTION"):
        L.d = dict(status=s, leader_floor=4)
        t[0] += 300
        assert f.tick(t[0])["status"] == "WATCHING"


def test_r37_stop_request_then_shadow_when_policy_is_off(tmp_path):
    f, L, K_, t = fv(tmp_path)
    L.d = dict(status="WOULD_CONFIRM", leader_floor=4, next_gap_m=0.08)
    assert f.tick(t[0])["status"] == "FINAL_VERIFICATION_STOP_REQUESTED" and f.d["why"] == "최종 위치 검증을 위해 정지해 주세요."
    ev = f.d["event_id"]
    t[0] += 1; f.tick(t[0])
    t[0] += 4; assert f.tick(t[0])["status"] == "VERIFYING" and L.wanted == 1
    L.r = FL(); t[0] += 1
    assert f.tick(t[0])["status"] == "SHADOW_WOULD_CONFIRM" and K_.fixed is None
    t[0] += 1; f.tick(t[0])
    assert f.d["event_id"] == ev                                                # no second request without new evidence


def test_r37_enabled_policy_writes_the_fix(tmp_path):
    f, L, K_, t = fv(tmp_path, _Link("ENABLED"))
    L.d = dict(status="WOULD_CONFIRM", leader_floor=4)
    f.tick(t[0]); t[0] += 1; f.tick(t[0]); t[0] += 4; f.tick(t[0]); L.r = FL(); t[0] += 1
    assert f.tick(t[0])["status"] == "VERIFIED_FIX" and K_.fixed


@pytest.mark.parametrize("res,why", [(FL(0.01), "within"), (FL(0.08, False), "bound"), (dict(floors=[dict(floor=7, cost=0.03, inlier=0.6, converged=True), dict(floor=4, cost=0.1, inlier=0.6, converged=True)]), "7F first")])
def test_r37_failed_verification_writes_nothing_and_stays_quiet(tmp_path, res, why):
    f, L, K_, t = fv(tmp_path, _Link("ENABLED"))
    L.d = dict(status="WOULD_CONFIRM", leader_floor=4)
    f.tick(t[0]); ev = f.d["event_id"]; t[0] += 1; f.tick(t[0]); t[0] += 4; f.tick(t[0]); L.r = res; t[0] += 1
    assert f.tick(t[0])["status"] == "VERIFY_FAILED" and why in f.d["why"] and K_.fixed is None
    t[0] += 10; f.tick(t[0])
    assert f.d["event_id"] == ev and f.d["status"] == "VERIFY_FAILED"            # quiet during the cooldown


def test_r37_stop_request_expires(tmp_path):
    f, L, K_, t = fv(tmp_path, stopped=False)
    L.d = dict(status="WOULD_CONFIRM", leader_floor=4)
    f.tick(t[0]); t[0] += 50
    assert f.tick(t[0])["status"] == "STOP_EXPIRED"


def test_r37_e7_long_needs_repeat_searches():
    h = dict(t_end=1, path_m=50, floors=fl((4, 0.03, True), (8, 0.15, True)), floors_exact=[4, 8], n_refresh=1)
    assert "repeat" in EL.classify([h, dict(h, path_m=70)], 2)["why"]
    h2 = dict(h, n_refresh=2)
    assert EL.classify([h2, dict(h2, path_m=70)], 2)["status"] == "WOULD_CONFIRM"


def test_r37_request_carries_the_depth_age_for_the_pre_stop_guidance():
    o = feed1(vs(), region([105, 106, 135, 152], 4.842, sim=0.134, msim=0.133))
    assert o["depth_dt_s"] == 0.01
    pa = TR.pre_stop_aim(G.raw_to_planar(0.0, 0.0, 0.0), 4.0, 1.0, 1.7, K, o["depth_dt_s"], o["depth_m"], ENV)
    assert pa["status"] != "AIM_TARGET_UNCERTAIN"
