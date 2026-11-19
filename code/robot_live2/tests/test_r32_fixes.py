"""STAGING (2026-11-01) r32 fixes, reproduced from the field run X20261031_231940 (loc2):
  F1 structure path left 'RUNNING' for the rest of the run when E4 left CONFIDENT_GEOMETRY while the retro E7 job ran (+547.8 s
     CONFIDENT 1.3 s on H159_F3, job 114 s) -> it must end NOT_CONFIRMED at the first tick after the job.
  F2 one decisive reading ('402', 4F p 0.86) with every 4F hypothesis 15-27 m away and no second plate -> the pair rule never asked
     for the regional E2 search -> now ONE regional request per such reading (candidate generation only).
  F3 e2_worker: region radius recomputed at fit time from the odometry path since the capture; a region fit can run inside a pass.
Synthetic helpers of test_e8_pair_fake for F2; F1 / F3 use the real classes with stubs. Nothing here is field evidence."""
import csv
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STG))
import test_e8_pair_fake as T  # noqa: E402


# ---------------- F1
def test_structure_attempt_is_ended_after_the_job_when_e4_left_confident(tmp_path):
    import e6e7_link as L
    lk = L.E6E7Link.__new__(L.E6E7Link)
    lk.state, lk.last = tmp_path, None
    lk.struct_policy = dict(mode="SHADOW", enabled=False)
    lk.tr = types.SimpleNamespace(store=types.SimpleNamespace(rows=[(0, 1.0, 0, 0, 0)], session=lambda: {"odom_session_id": "S"}))
    sa = lk.struct = dict(kind="LIVE2_STRUCTURE_PATH", leader="H159_F3", floor=3, status="RUNNING", mode="SHADOW", phase="RETRO_VERIFY",
                          passes=[], t_confident=1.0, started_wall=0.0, epoch="S")
    ev = threading.Event()
    lk.job = threading.Thread(target=ev.wait, daemon=True); lk.job.start()
    amb = dict(state="AMBIGUOUS", metrics=dict(best_id="H159_F3"))
    assert lk.structure_tick(amb, 10.0) is None and lk.struct is sa and sa["status"] == "RUNNING"   # job alive: kept, not dropped
    sa["phase"] = "VERIFY_WHILE_MOVING"                                                           # what the retro job writes
    ev.set(); lk.job.join(2)
    lk.structure_tick(amb, 11.0)
    d = json.loads((tmp_path / "structure_path.json").read_text())
    assert d["status"] == "NOT_CONFIRMED" and "E4 left CONFIDENT_GEOMETRY" in d["why"]
    conf = dict(state="CONFIDENT_GEOMETRY", metrics=dict(best_id="H159_F3", best_floor=3, gap=0.2))
    assert lk.structure_tick(conf, 12.0)["status"] == "NOT_CONFIRMED"                            # same leader: no retry loop


# ---------------- F2
def link_with_one_reading(tmp_path, hyps, ocr_key=10):
    import e6e7_link as L
    st = tmp_path / "state"; st.mkdir()
    tr = T.Tracker(hyps)
    tr.store.rows = [(0, 10.0, 0.0, 0.0, 0.0)]
    (tmp_path / "e2").mkdir()
    tr.e2_dir, tr.session = str(tmp_path / "e2"), "S1"
    link = L.E6E7Link(st, tr, sem_floors=T.floors(4, 5), policy="E8_PAIR", reader=T.Reader(T.OCR))
    with open(st / "e6_precision_results.jsonl", "a") as f:
        f.write(json.dumps(T.stored(st, "O1", ocr_key, T.T401)) + "\n")
    link.tick(); T.wait(link.pair); link.tick()
    return link, st, tmp_path / "e2" / "e2_region_request.json"


def test_one_decisive_reading_without_a_fitting_hypothesis_requests_a_region(tmp_path):
    link, st, reqf = link_with_one_reading(tmp_path, [T.Hyp("H4X", 4, pos=(1450.0, 400.0)), T.Hyp("H5", 5)])
    assert link.pair.status not in ("GLOBAL_FIX", "E7_VERIFYING")                                  # one reading is never a fix
    req = json.loads(reqf.read_text())
    assert req["request_id"].startswith("OBS_") and req["region"]["floor"] == 4 and req["region"]["rooms"] == ["room_401"]
    assert req["region"]["basis"] == "SINGLE_DECISIVE_READING" and req["region"]["px_per_m_hi"] > 0 and req["region"]["radius_base_px"] > 0
    rec = json.loads((st / "semantic_recovery_single.json").read_text())
    assert rec["status"] == "REGION_REQUESTED"
    n0 = len((st / "semantic_recovery_single.jsonl").read_text().splitlines())
    link.tick(); link.tick()
    assert len((st / "semantic_recovery_single.jsonl").read_text().splitlines()) == n0              # once per observation


def test_no_request_when_a_hypothesis_already_fits(tmp_path):
    link, st, reqf = link_with_one_reading(tmp_path, [T.Hyp("H4", 4), T.Hyp("H5", 5)])
    assert not reqf.exists() and json.loads((st / "semantic_recovery_single.json").read_text())["status"] == "NOT_NEEDED"


def test_no_request_for_an_unreadable_capture(tmp_path):
    link, st, reqf = link_with_one_reading(tmp_path, [T.Hyp("H4X", 4, pos=(1450.0, 400.0))], ocr_key=40)   # OCR returns nothing
    assert not reqf.exists() and not (st / "semantic_recovery_single.json").exists()


# ---------------- F3
def worker(tmp_path, rows):
    spec = importlib.util.spec_from_file_location("e2w_r32", STG / "e2_worker.py")
    W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
    store = tmp_path / "store"; store.mkdir()
    with open(store / "odom_link.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "t_sec", "fastlio_x", "fastlio_y", "fastlio_yaw_rad", "delta_s_m", "delta_yaw_deg"])
        for i, (t, x, y) in enumerate(rows):
            w.writerow([f"S{i}", t, x, y, 0, 0, 0])
    c = W.Coordinator.__new__(W.Coordinator)
    c.a = types.SimpleNamespace(min_window_path_m=2.0)
    c.out, c.W, c.pass_id, c.store = tmp_path, 60.0, 1, str(store)
    c.state = dict(candidates=[], candidates_pass=[], floors_done=[1], per_floor={"4": dict(light="/L4")})
    c.span = lambda: (0.0, rows[-1][0])
    c.session = lambda: {"odom_session_id": "S1"}
    c.hold_reason = lambda: None
    c.window_path_m = lambda t1, W=None: 8.0
    c.ledger = W.RL.Ledger(tmp_path / "led.json"); c.ledger.load(dict(run_id="R", odom_session="S1", maps="m", calib="c"))
    c.region_added, c._resume, c.logs = [], None, []
    real_log = c.log
    c.log = lambda kind, **k: (c.logs.append((kind, k)), real_log(kind, **k))[1]   # the REAL log() runs (R34: never a stub)
    return W, c


def test_region_radius_is_recomputed_at_fit_time(tmp_path):
    rows = [(100.0 + i, float(i), 0.0) for i in range(21)]                                         # 20 m walked after t=100
    W, c = worker(tmp_path, rows)
    rq = dict(request_id="OBS_O1", odom_session="S1", region=dict(floor=4, center_px=[750.0, 350.0], radius_px=430.0, radius_base_px=430.0,
                                                                  px_per_m_hi=27.6, capture_wall=100.0))
    (tmp_path / "e2_region_request.json").write_text(json.dumps(rq))
    assert c.region_pending()["request_id"] == "OBS_O1"
    rec, msg = c.region_start(c.region_pending())
    assert msg[3] == "exact" and abs(msg[4]["radius_px"] - (430.0 + (20.0 + 1.0) * 27.6)) < 1.0 and rec["walked_since_capture_at_fit_m"] == 20.0 and rec["net_since_capture_m"] == 20.0
    assert c.region_pending() is None                                                              # taken
    c.region_finish(rec, "floor_done", 4, dict(candidates=[dict(floor=4, merit=0.2, twoway=0.5, px=760.0)], sec=40.0, light="/L4", H=1, map_version="m"))
    assert rec["status"] == "DONE" and c.state["candidates_pass"][0]["regional_request"] == "OBS_O1" and len(c.region_added) == 1


def test_region_fit_failure_at_rest_is_retried(tmp_path):
    W, c = worker(tmp_path, [(100.0, 0.0, 0.0), (101.0, 0.0, 0.0)])
    rq = dict(request_id="OBS_O2", odom_session="S1", region=dict(floor=4, center_px=[1.0, 1.0], radius_px=400.0))
    c.ledger.start(c.ledger.ingest(rq))
    rec = dict(request_id="OBS_O2", attempt=1)
    c.region_finish(rec, "floor_failed", 4, dict(error="CalibrationUnavailable: Not enough overlapping scans"))
    assert rec["status"] == "RETRY_WAIT" and c.ledger.get("OBS_O2")["state"] == "RETRY_WAIT"


def test_region_retry_is_bounded_and_respects_the_fastlio_hold(tmp_path):
    W, c = worker(tmp_path, [(100.0 + i, float(i), 0.0) for i in range(31)])
    rq = dict(request_id="OBS_O9", odom_session="S1", region=dict(floor=4, center_px=[1.0, 1.0], radius_px=400.0))
    (tmp_path / "e2_region_request.json").write_text(json.dumps(rq))
    c.hold_reason = lambda: "HELD_FASTLIO odom 7.6 Hz < 8.0 Hz (FAST-LIO keeps priority)"
    rec, msg = c.region_start(c.region_pending())
    assert msg is None and rec["status"] == "RETRY_WAIT" and "HELD_FASTLIO" in rec["why_failed"]      # no fit while FAST-LIO is slow
    e = c.ledger.get("OBS_O9")
    assert c.region_pending() is None and e["attempts"] == 0                                          # waits, and waiting costs no attempt
    e["not_before"], e["first_seen_wall"] = 0.0, 1.0                                                  # far past the time limit
    rec, msg = c.region_start(c.region_pending())
    assert msg is None and rec["status"] == "FAILED" and "NEEDS_NEW_OBSERVATION" in rec["why_failed"] and c.region_pending() is None   # bounded, then never again


def test_low_quality_override_counts_only_with_a_strong_read():
    import e8_pair as EP
    import e6_semantic as S
    sf = T.floors(4)
    base = dict(observation_id="O1", target_odom_link_xy=[0.0, 0.0], session="S", image_quality=dict(ok=True, operator_override_low_sharpness=True,
                                                                                                    flags=["MANUAL_ACCEPTED", "LOW_SHARPNESS"]))
    hy = [dict(id="H4", floor=4, rec=T.Hyp("H4", 4).rec)]
    weak = EP.evaluate_group([dict(base, ocr_lines=[["401", 0.86]])], hy, sf)["per_observation"][0]
    strong = EP.evaluate_group([dict(base, ocr_lines=[["401", 0.975]])], hy, sf)["per_observation"][0]
    normal = EP.evaluate_group([dict(base, ocr_lines=[["401", 0.86]], image_quality=dict(ok=True, flags=["MANUAL_ACCEPTED"]))], hy, sf)["per_observation"][0]
    assert weak["decisive"] and not weak["usable"] and weak["why_not"].startswith("LOW_QUALITY_OVERRIDE_WEAK_READ")
    assert strong["usable"] and normal["usable"]                     # the field '402' 0.975 case and a clean photo are unchanged


def test_ocr_threads_follow_the_caller_affinity_and_operator_roi_is_used(monkeypatch):
    import os
    import numpy as np
    import e6_ocr as O
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {5})
    assert O.ocr_threads() == 1                                       # e34 on core 5 -> onnxruntime keeps to core 5
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    assert O.ocr_threads() == O.MAX_THREADS
    seen = {}

    def fake_many(imgs, timeout=120):
        seen["shapes"], seen["timeout"] = [i.shape for i in imgs], timeout
        return [[["402", 0.9]], []], dict(engine="FAKE", load_s=0.5, per_image_s=[0.1, 0.2], ocr_s=0.8, threads=1)
    monkeypatch.setattr(O, "read_many", fake_many)
    img = np.zeros((1080, 1920, 3), np.uint8)
    lines, meta = O.sign_read(img, roi=(420, 505, 544, 614))
    assert meta["roi_source"] == "OPERATOR_SELECTED" and seen["shapes"][0] == (int(614 * 1.5) - int(505 * 1.5), int(544 * 1.5) - int(420 * 1.5), 3)
    assert len(seen["shapes"]) == 2 and seen["timeout"] == 240 and lines == [["402", 0.9]]   # ONE call (one model load) for ROI + full
