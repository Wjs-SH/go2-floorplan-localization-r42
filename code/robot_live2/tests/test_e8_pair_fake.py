"""(copied 2026-10-30 into the final staging; two link tests now give the fake tracker an E2 list, see below)
E8 dual semantic observation (e8_pair + e6e7_link E8_PAIR policy + obs_flow pair guidance + pipeline deferred reading).
OFFLINE: synthetic semantic floor maps (same schema as kci_handoff), fake E3 tracker / OCR reader / E7. The student semantic
code (classify_text / fuse / norm_label) is the real one. Nothing here is hardware or field evidence."""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "candidate" / "live2"))
sys.path.insert(0, str(W / "candidate" / "live1"))
import e6_semantic as S  # noqa: E402
import e8_pair as EP  # noqa: E402
import obs_flow as OF  # noqa: E402

RES = 0.05                                     # m/px; ppm 20


def room(rid, num, x0, y0, x1, y1):
    return dict(id=rid, type="room", room_number=num, polygon_px=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def floor_map(f):
    """Corridor y 400-440 px; room f01 x 100-200, f05 x 700-800, f09 x 140-260 (above the corridor, touching it)."""
    return dict(map_meta=dict(resolution_m_per_px=RES), corridors=[dict(id=f"c{f}", polygon_px=[[0, 400], [2000, 400], [2000, 440], [0, 440]])],
                spaces=[room(f"room_{f}01", f"{f}01", 100, 300, 200, 400), room(f"room_{f}05", f"{f}05", 700, 300, 800, 400),
                        room(f"room_{f}09", f"{f}09", 1400, 300, 1500, 400)])


def floors(*fs):
    sf = S.SemanticFloors.__new__(S.SemanticFloors)
    sf.maps = {f: floor_map(f) for f in fs}
    return sf


class Hyp:
    def __init__(self, hid, floor, pos=(150.0, 400.0)):
        self.id, self.floor, self.light_dir, self.sem_verdict, self.sem = hid, floor, f"/L{floor}", None, []
        self.rec = dict(phi=0.0, pos=list(pos), ppm=1.0 / RES, anchor=[0.0, 0.0])
        self.status = "ACTIVE"

    def rank_key(self):
        return 0


class Tracker:
    def __init__(self, hyps, session="S1"):
        self.hyps, self.sess, self.applied = hyps, [session], []
        self.store = types.SimpleNamespace(refresh=lambda: None, rows=[], session=lambda: {"odom_session_id": self.sess[0]})

    def alive(self):
        return self.hyps

    def light(self, d):
        return d

    def apply_semantic(self, res, oid, src):
        self.applied.append((oid, src, res))
        for h in self.hyps:
            h.sem_verdict = (res["per_hypothesis"].get(h.id) or {}).get("verdict")
        return 1


def make_img(tmp, oid, k):
    import cv2
    img = np.full((108, 192, 3), 100, np.uint8)
    img[0, 0] = (k, k, k)
    p = tmp / "e6_captures" / f"{oid}.jpg"
    p.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def stored(tmp, oid, k, target, stop_xy=(0.0, 0.0), t=100.0, wall=1000.0, ok=True, status="CAPTURE_STORED"):
    p = make_img(tmp, oid, k)
    import hashlib
    return dict(kind="LIVE2_E6_PRECISION", observation_id=oid, capture_id=f"cap_{oid}", status=status, source="AUTO_E5",
                capture_file=str(p), image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), resolution=[1920, 1080],
                image_quality=dict(ok=ok, roi_laplacian_var=60.0 if ok else 6.0), target_odom_link_xy=list(target),
                odom_pose_at_stop=dict(t=t, x=stop_xy[0], y=stop_xy[1], yaw=0.0), stopped_wall=wall, wall=wall + 20,
                experiment_id="X_TEST", ocr_lines=[], reading="DEFERRED_TO_E8_PAIR")


class Reader:
    """Fake OCR: the capture's pixel (0,0) value selects the text; counts calls (OCR must wait for the 2nd capture)."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, img):
        k = int(img[0, 0, 0] + 0.5)
        k = min(self.table, key=lambda x: abs(x - k))
        self.calls.append(k)
        return [list(x) for x in self.table[k]], dict(engine="FAKE_OCR")


def manager(tmp, hyps, sf, table, lio=None, vlm=None, vlm_admit=None):
    tr = Tracker(hyps)
    e7_req = []
    pm = EP.PairManager(tmp, tr, sf, lambda cons, r: e7_req.append(([h.id for h in cons], r)), reader=Reader(table),
                        vlm=vlm, vlm_admit=vlm_admit, lio_series=lio or [])
    return pm, tr, e7_req


def wait(pm):
    t0 = time.time()
    while pm._running and time.time() - t0 < 20:
        time.sleep(0.01)


T401, T405 = (0.0, 0.0), (30.0, 0.0)          # plate positions (odom_link m) that land on rooms 401 / 405 under H4
OCR = {10: [("401", 0.99)], 20: [("405", 0.99)], 30: [("409", 0.99)], 40: [], 50: [("501", 0.99)], 60: [("401", 0.99)]}


def test_01_05_first_capture_is_stored_read_in_background_provisional_and_resume_is_offered(tmp_path):
    """2026-10-26: the 1st capture is read at once (background) -> PROVISIONAL place; drive-on is not held for it."""
    pm, tr, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    snap = pm.on_precision(stored(tmp_path, "O1", 10, T401))
    assert snap["state"] == "OBS1_WAITING_RESUME" and snap["resume_ok"]
    obs = [dict(observation_id="O1", state="STOPPED_TARGET_READY", created_wall=990, stopped_wall=1000)]
    prec = {"O1": dict(status="CAPTURE_STORED", wall=1020)}
    f = OF.flow(obs, prec, {}, None, True, True, False, 1030, pair=snap)
    assert f["state"] == "RESUME_READY"
    wait(pm)
    assert pm.reader.calls == [10] and pm.status == "OBS1_WAITING_RESUME" and pm.resume_ok() and not e7
    ev = pm.last_eval
    assert ev["kind"] == "LIVE2_E8_PROVISIONAL" and ev["verdict"] == "PROVISIONAL"
    r = ev["per_observation"][0]
    assert r["observation_id"] == "O1" and r["labels"][0][0] == "401" and r["provisional"]["floor"] == 4
    assert not tr.applied                                                                              # no E3 update, no fix
    snap = json.loads((tmp_path / "e8_pair_state.json").read_text())
    assert snap["evaluation"]["verdict"] == "PROVISIONAL" and snap["observations"][0]["read"]


def test_02_17_resume_is_never_offered_on_fault_unknown_hold_or_stale_odometry(tmp_path):
    pm, _, _ = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    snap = pm.on_precision(stored(tmp_path, "O1", 10, T401))
    obs = [dict(observation_id="O1", state="STOPPED_TARGET_READY", created_wall=990, stopped_wall=1000)]
    prec = {"O1": dict(status="CAPTURE_STORED", wall=1020)}
    assert OF.flow(obs, prec, {}, None, True, False, False, 1030, pair=snap)["state"] == "HOLD_ERROR"     # odometry stale
    assert OF.flow(obs, prec, {}, None, True, True, True, 1030, pair=snap)["state"] == "HOLD_ERROR"      # safety pending
    assert OF.flow(obs, prec, {}, None, True, True, False, 1030, e6_blocked="FOCUS_x", pair=snap)["state"] == "HOLD_ERROR"
    for st in ("DEVICE_ERROR_BLOCKED", "FOCUS_FAULT_BLOCKED"):                                            # device UNKNOWN / fault
        pm2, _, _ = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
        s2 = pm2.on_precision(dict(observation_id="O9", status=st, why="DEVICE UNKNOWN"))
        assert s2["state"] == "HOLD_ERROR" and not s2["resume_ok"]
        f = OF.flow([dict(obs[0], observation_id="O9")], {"O9": dict(status=st, wall=1020)}, {}, None, True, True, False, 1030, pair=s2)
        assert f["state"] == "HOLD_ERROR"
    unseen = dict(snap, processed=[])                                                                    # pair has not handled it yet
    assert OF.flow(obs, prec, {}, None, True, True, False, 1030, pair=unseen)["state"] == "E6_OBSERVING"


def test_03_06_images_metadata_and_ids_are_kept_per_observation(tmp_path):
    pm, _, _ = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    r1 = stored(tmp_path, "O1", 10, T401)
    pm.on_precision(r1)
    pm.on_precision(stored(tmp_path, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    wait(pm)
    rows = [json.loads(x) for x in (tmp_path / "e8_observations.jsonl").read_text().splitlines()]
    o1 = [r for r in rows if r["observation_id"] == "O1"]
    assert o1[0]["image_sha256"] == r1["image_sha256"] and o1[0]["capture_id"] == "cap_O1" and o1[0]["session"] == "S1"
    assert Path(o1[0]["capture_file"]).exists() and EP.sha256_file(o1[0]["capture_file"]) == r1["image_sha256"]
    ev = pm.last_eval
    by = {r["observation_id"]: r for r in ev["per_observation"]}
    assert by["O1"]["ocr_lines"] == [["401", 0.99]] and by["O2"]["ocr_lines"] == [["405", 0.99]]
    assert by["O1"]["capture_id"] == "cap_O1" and by["O2"]["capture_id"] == "cap_O2"
    reads = [r for r in rows if r["event"] == "READ"]
    assert {r["observation_id"]: r["reading"]["capture_id"] for r in reads} == {"O1": "cap_O1", "O2": "cap_O2"}


def test_04_16_siglip_searches_between_observations_and_yields_the_gpu_while_reading(tmp_path):
    pm, _, _ = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    base = dict(mode="IDLE_OK", why="E4 no need", event=None)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    assert pm.siglip_override(base)["mode"] == "ACTIVE" and "E8_PAIR_NEXT_OBSERVATION" in pm.siglip_override(base)["why"]
    pm.status = "PAIR_READING"
    assert pm.siglip_override(dict(base, mode="ACTIVE"))["mode"] == "ACTIVE"     # R36: the robot may be driving while the photo is read: SigLIP keeps looking
    pm.status = "MORE_OBSERVATION_REQUIRED"
    assert pm.siglip_override(base)["mode"] == "ACTIVE"                                                 # SigLIP restored
    pm.status = "GLOBAL_FIX"
    assert pm.siglip_override(base) == base                                                             # back to E4's own rule


def test_05_08_13_consistent_pair_reads_both_after_the_second_capture_then_requests_e7(tmp_path):
    pm, tr, e7 = manager(tmp_path, [Hyp("H4", 4), Hyp("H5", 5)], floors(4, 5), OCR)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    wait(pm)
    assert sorted(pm.reader.calls) == [10, 20]
    ev = pm.last_eval
    assert ev["verdict"] == "REGION_CANDIDATE_READY" and [c["id"] for c in ev["candidates"]] == ["H4"], ev["why"]
    pr = ev["pairs"][0]
    assert pr["odometry"]["net_displacement_m"] == 28.0 and pr["target_separation_m"] == 30.0
    assert pr["floors"][4]["metric_ok"] and pr["floors"][4]["topology"] == "CONNECTED"
    assert pm.status == "E7_VERIFYING" and e7 and e7[0][0] == ["H4"]
    assert e7[0][1]["source"] == "E8_PAIR" and e7[0][1]["pair_observation_ids"] == ["O1", "O2"]
    assert tr.applied and tr.applied[-1][1] == "E8_PAIR"                                                # E3 updated by the PAIR


def test_07_same_floor_but_wrong_spatial_relation_is_not_consistent(tmp_path):
    pm, _, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 20, (6.0, 0.0), stop_xy=(5.0, 0.0), t=130.0, wall=1100.0))   # 405 read only 6 m away
    wait(pm)
    ev = pm.last_eval
    assert ev["verdict"] == "MORE_OBSERVATION_REQUIRED" and not e7
    assert ev["pairs"][0]["floors"][4]["metric_ok"] is False and ev["pairs"][0]["floors"][4]["translation_residual_m"] > 0


def test_09_session_change_and_low_rate_odometry_block_the_pair(tmp_path):
    pm, tr, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    tr.sess[0] = "S2"                                                                                   # FAST-LIO restarted
    snap = pm.on_precision(stored(tmp_path, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    assert snap["state"] == "HOLD_ERROR" and "session" in snap["why"] and not e7 and not snap["resume_ok"]
    lio = [dict(wall=1050.0, hz=5.8)]
    pm, _, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR, lio=lio)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    wait(pm)
    assert pm.last_eval["verdict"] == "MORE_OBSERVATION_REQUIRED" and "8 Hz" in pm.last_eval["why"] and not e7


def test_10_the_same_plate_twice_is_not_two_observations(tmp_path):
    pm, _, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 60, (0.6, 0.2), stop_xy=(1.5, 0.0), t=110.0, wall=1040.0))  # 401 again, 0.6 m away
    wait(pm)
    ev = pm.last_eval
    assert ev["verdict"] == "MORE_OBSERVATION_REQUIRED" and not e7
    assert "SAME_PLATE_AS O1" in {r["observation_id"]: r for r in ev["per_observation"]}["O2"]["why_not"]


def test_11_ocr_failure_and_vlm_only_guess_are_no_evidence(tmp_path):
    vlm = lambda img: dict(parsed=dict(sign_type="room_number", room_number="405"), sec=1.0)        # noqa: E731
    pm, _, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR, vlm=vlm, vlm_admit=lambda: (True, "test"))
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 40, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))       # OCR reads nothing
    wait(pm)
    by = {r["observation_id"]: r for r in pm.last_eval["per_observation"]}
    assert by["O2"]["usable"] is False and "NO_DECISIVE_LABEL" in by["O2"]["why_not"]
    assert pm.last_eval["verdict"] == "MORE_OBSERVATION_REQUIRED" and not e7
    o2 = [g for g in pm.group if g["observation_id"] == "O2"][0]
    assert o2["reading"]["vlm"]["state"] == "ASKED"
    o1 = [g for g in pm.group if g["observation_id"] == "O1"][0]
    assert o1["reading"]["vlm"]["state"] == "ASKED"                                                     # R37: every stored photo is reviewed; here OCR 401 vs VLM 405 = conflict, not evidence
    assert by["O1"]["usable"] is False


def test_12_candidate_missing_when_the_map_fits_but_no_live_hypothesis_does(tmp_path):
    pm, _, e7 = manager(tmp_path, [Hyp("H5", 5)], floors(4, 5), OCR)                                    # only a 5F hypothesis alive
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    wait(pm)
    assert pm.status == "CANDIDATE_MISSING" and "no hypothesis is invented" in pm.why and not e7
    assert pm.last_eval["floors_explaining"] == [4] and pm.last_eval["candidates"] == []


def test_14_e7_ambiguous_keeps_the_group_open_and_never_fixes(tmp_path):
    pm, _, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    wait(pm)
    pm.on_e7(dict(observation_id=pm.gid, global_decision="AMBIGUOUS", why="E7 gap 0.1 m"))
    assert pm.status == "MORE_OBSERVATION_REQUIRED" and pm.snapshot()["resume_ok"]
    pm.on_e7(dict(observation_id=pm.gid, global_decision="GLOBAL_CONFIRMED", why="unique"))
    assert pm.status == "GLOBAL_FIX" and pm.snapshot()["last_closed"]["e7"]["used_observation_ids"] == ["O1", "O2"]


def test_14b_link_single_observation_e7_is_provisional_and_pair_fix_obeys_fastlio(tmp_path, monkeypatch):
    import e6e7_link as L
    st = tmp_path / "state"
    st.mkdir()
    hs = [Hyp("H4", 4)]
    tr = Tracker(hs)
    tr.store.rows = [(0, 10.0)]
    # 2026-10-30 (E7 rival completion, fail closed): E7 needs this session's E2 list; here it has no other-floor candidate
    (tmp_path / "e2").mkdir(exist_ok=True)
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(dict(odom_session="S_TEST", candidates=[], per_floor={})))
    tr.e2_dir, tr.session = str(tmp_path / "e2"), "S_TEST"
    monkeypatch.setattr(L.E7, "evaluate", lambda *a, **k: dict(results=[1], best="H4", final_decision="GLOBAL_CONFIRMED"))
    link = L.E6E7Link(st, tr, sem_floors=floors(4), policy="SINGLE_DIAGNOSTIC")
    link._run([dict(candidate_id="H4", floor=4, rec=dict(hs[0].rec, light="/L4"))], dict(observation_id="O1", source="AUTO_E5"), "H4")
    e7 = json.loads((st / "e7_state.json").read_text())
    assert e7["global_decision"] == "PROVISIONAL_SINGLE" and e7["fix_blocked"]["e7_decision"] == "GLOBAL_CONFIRMED"
    link._run([dict(candidate_id="H4", floor=4, rec=dict(hs[0].rec, light="/L4"))],
              dict(observation_id="PAIR_1", source="E8_PAIR", pair_observation_ids=["O1", "O2"]), "H4")
    e7 = json.loads((st / "e7_state.json").read_text())     # R42: E7's own confirm on a pair without the R38 floor evidence / 2 windows is not a FIX
    assert e7["global_decision"] == "MORE_OBSERVATION_REQUIRED" and e7["fix_blocked"]["e7_decision"] == "GLOBAL_CONFIRMED"
    link.last = None      # R41: a standing FIX refuses later looks (test_r41); this test looks at the FAST-LIO gate of a new one
    (st / "lio_series.jsonl").write_text(json.dumps(dict(wall=time.time() - 5, hz=5.5)) + "\n")
    link._run([dict(candidate_id="H4", floor=4, rec=dict(hs[0].rec, light="/L4"))],
              dict(observation_id="PAIR_2", source="E8_PAIR", pair_observation_ids=["O1", "O2"]), "H4")
    e7 = json.loads((st / "e7_state.json").read_text())
    assert e7["global_decision"] == "MORE_OBSERVATION_REQUIRED" and "8 Hz" in e7["fix_blocked"]["why"]
    assert e7["used_observation_ids"] == ["O1", "O2"]


def test_15_third_observation_re_evaluates_all_without_majority_vote(tmp_path):
    pm, _, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4, 5), OCR)
    pm.on_precision(stored(tmp_path, "O1", 10, T401))
    pm.on_precision(stored(tmp_path, "O2", 50, (12.0, 0.0), stop_xy=(11.0, 0.0), t=120.0, wall=1060.0))  # reads 501: contradicts
    wait(pm)
    assert pm.status == "MORE_OBSERVATION_REQUIRED" and not e7
    pm.on_precision(stored(tmp_path, "O3", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))         # agrees with O1
    wait(pm)
    ev = pm.last_eval
    assert ev["observation_ids"] == ["O1", "O2", "O3"] and pm.reader.calls.count(10) == 1                # O1 not re-read
    assert ev["verdict"] != "REGION_CANDIDATE_READY" and not e7                                          # 2 of 3 is not a fix
    pm2, _, e7b = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)                                      # unusable 2nd, good 3rd
    pm2.on_precision(stored(tmp_path, "P1", 10, T401))
    pm2.on_precision(stored(tmp_path, "P2", 40, (12.0, 0.0), stop_xy=(11.0, 0.0), t=120.0, wall=1060.0))
    wait(pm2)
    pm2.on_precision(stored(tmp_path, "P3", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0))
    wait(pm2)
    assert pm2.last_eval["verdict"] == "REGION_CANDIDATE_READY" and e7b[0][1]["pair_observation_ids"] == ["P1", "P2", "P3"]
    assert pm2.last_eval["n_evidence"] == 2


def test_blurry_first_capture_is_not_counted_and_not_retried(tmp_path):
    pm, _, _ = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    snap = pm.on_precision(stored(tmp_path, "O1", 10, T401, ok=False, status="CAPTURED_BLURRY_REOBSERVE"))
    assert snap["state"] == "PAIR_IDLE" and not pm.group and "not counted" in snap["why"] and snap["resume_ok"]


def test_18_gt_is_never_an_input_of_the_pair_path():
    src = (W / "candidate" / "live2" / "e8_pair.py").read_text()
    for bad in ("gt.jsonl", "evaluation/", "EVAL /", "\"evaluation\"", "ground_truth", "start_gt"):
        assert bad not in src
    import inspect
    assert "gt" not in inspect.signature(EP.evaluate_group).parameters


def test_pipeline_deferred_reading_stores_capture_without_ocr(tmp_path):
    import cv2
    import e6_pipeline as P

    def render(sharp=True):
        img = np.full((1080, 1920, 3), 120, np.uint8)
        cv2.rectangle(img, (300, 80), (1100, 580), (235, 235, 235), -1)
        cv2.putText(img, "418", (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 10, (25, 25, 25), 28)
        return img

    class Con:
        def __init__(self):
            self.jpg, self.calls = cv2.imencode(".jpg", render())[1].tobytes(), []

        def run(self, kind, params, *a, **k):
            self.calls.append(kind)
            return dict(status="COMPLETED", device_state="IDLE", capture=dict(capture_id="c1")) if kind == "aim" else \
                dict(status="COMPLETED", device_state="IDLE")

        def fetch_capture(self, cid):
            return self.jpg

    def no_ocr(img):
        raise AssertionError("OCR must wait for the second observation")
    c = Con()
    p = P.E6Pipeline(c, lambda kind, **kw: (True, None), lambda: dict(stopped=True), tmp_path / "caps", ocr=no_ocr,
                     envelope=dict(pan=(90.0, 112.0), tilt=(150.0, 166.0), source="T"), log=lambda *_: None, defer_reading=True)
    obs = dict(observation_id="O1", source="AUTO_E5", target_pose_in_odom=dict(odom_link_xy=[1.0, 0.5]), stopped_wall=1000.0,
               odom_pose_at_stop=dict(t=5.0, x=1.0, y=2.0, yaw=0.1),
               redetection=dict(status="REDETECTED", pixel=[175, 217], depth_m=1.36, pantilt_command=dict(inside_limits=True)))
    rec = p.run(obs, 1)
    assert rec["status"] == "CAPTURE_STORED" and c.calls == ["aim", "home"] and rec["ocr_lines"] == []
    assert rec["image_sha256"] == EP.sha256_file(rec["capture_file"]) and rec["odom_pose_at_stop"]["x"] == 1.0
    assert ("OCR", "DEFERRED_TO_E8_PAIR") in [(t["state"], t.get("outcome")) for t in rec["e6_trace"]]


def test_link_pair_policy_end_to_end_offline(tmp_path, monkeypatch):
    """precision records (stored) -> link E8_PAIR -> pair reading -> consistency -> E7 (stub) -> GLOBAL_FIX; flow + files."""
    import e6e7_link as L
    st = tmp_path
    hs = [Hyp("H4", 4)]
    tr = Tracker(hs)
    tr.store.rows = [(0, 10.0)]
    # 2026-10-30 (E7 rival completion, fail closed): E7 needs this session's E2 list; here it has no other-floor candidate
    (tmp_path / "e2").mkdir(exist_ok=True)
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(dict(odom_session="S_TEST", candidates=[], per_floor={})))
    tr.e2_dir, tr.session = str(tmp_path / "e2"), "S_TEST"
    monkeypatch.setattr(L.E7, "evaluate", lambda *a, **k: dict(results=[1], best="H4", final_decision="GLOBAL_CONFIRMED"))
    link = L.E6E7Link(st, tr, sem_floors=floors(4), policy="E8_PAIR", reader=Reader(OCR))
    r1 = stored(st, "O1", 10, T401)
    with open(st / "e6_precision_results.jsonl", "a") as f:
        f.write(json.dumps(r1) + "\n")
    link.tick()
    assert json.loads((st / "e7_observation.json").read_text())["outcome"] == "PAIR_OBS1_WAITING_RESUME"
    with open(st / "e6_precision_results.jsonl", "a") as f:
        f.write(json.dumps(stored(st, "O2", 20, T405, stop_xy=(28.0, 0.0), t=130.0, wall=1100.0)) + "\n")
    link.tick()
    wait(link.pair)
    for _ in range(50):
        if link.job and link.job.is_alive():
            link.job.join(timeout=1)
        if link.pair.status == "GLOBAL_FIX":
            break
        time.sleep(0.1)
    snap = json.loads((st / "e8_pair_state.json").read_text())
    e7 = json.loads((st / "e7_state.json").read_text())     # R42: rooms 401 + 405 from digits only, one window: E7's stubbed confirm is held by the R38/R41 rule
    assert snap["state"] != "GLOBAL_FIX" and e7["global_decision"] == "MORE_OBSERVATION_REQUIRED" and e7["fix_blocked"]["e7_decision"] == "GLOBAL_CONFIRMED"
    assert e7["used_observation_ids"] == ["O1", "O2"]
    kinds = [json.loads(x)["kind"] for x in (st / "e8_pairs.jsonl").read_text().splitlines()]
    assert [k for k in kinds if k != "LIVE2_E8_PROVISIONAL"] == ["LIVE2_E8_EVALUATION", "LIVE2_E8_E7_RESULT"]
    assert kinds.count("LIVE2_E8_PROVISIONAL") <= 1 and kinds[-1] == "LIVE2_E8_E7_RESULT"         # O1 read at once (2026-10-26)
