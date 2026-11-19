"""R1 W-A: trial generation / stale floor-pose / beta calibration / evidence persistence / map contract / eval labels.
T01 T02 T32 T33 T34 T35 T36. Pure Python, fake clocks/publishers/estimators; every write goes to pytest tmp (inside R1).
The tree under test is taken from env SCI_WS (baseline or candidate)."""
import ast
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
RT = WS / "external" / "sci_live" / "realtime"
LOCZ = WS / "external" / "sci_live" / "floor_id" / "localizer"
E57 = WS / "src" / "sci_semantic_localization"
for p in (str(RT), str(RT / "tools"), str(LOCZ), str(E57), str(WS / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import r1_trial as T  # noqa: E402
import loc_worker as LW  # noqa: E402
import live_viewer as LV  # noqa: E402
import live_store as LS  # noqa: E402
import sensor_frame as SF  # noqa: E402
from e5_e7 import map_contract as MC  # noqa: E402
from e5_e7 import geometry as G  # noqa: E402
from e5_e7 import e6_region as E6  # noqa: E402
from e5_e7 import e7_rematch as E7  # noqa: E402


# ----------------------------------------------------------------------------------------------- helpers
def mk_root(tmp_path):
    root = tmp_path / "rt_live"
    (root / "out").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    rows = ["sample_id,t_sec,fastlio_x,fastlio_y,fastlio_yaw_rad,delta_s_m,delta_yaw_deg"]
    for i in range(40):
        rows.append(f"LIVE_{i:06d},{1000 + 3 * i:.6f},{0.5 * i:.6f},0.0,0.0,0.5,0.0")
    (root / "data" / "odom_link.csv").write_text("\n".join(rows) + "\n")
    return root


def mk_state(tmp_path, monkeypatch):
    root = mk_root(tmp_path)
    monkeypatch.setenv("RT_PERSIST_DIR", str(tmp_path / "persist"))
    monkeypatch.setenv("SCI_SHM", str(tmp_path / "shm"))
    maps = tmp_path / "maps"
    (maps / "4F").mkdir(parents=True)
    S = LV.State(root, maps, 60.0, 300, trigger=str(tmp_path / "trigger"), archive=None)
    return S, root


def floor_rec(verdict, signal, t_now=1100.0):
    c = lambda m, tw: dict(map=m, t=t_now, t_now=t_now, twoway=tw, ppm=20.0, phi=0.0, beta=3.9, mirror=1, pos=[100.0, 100.0],
                           anchor=[0.0, 0.0], wall_cost_m=0.05, scene=None, sec=1.0, X_cross_m=0.0, wall_pass=0)
    return dict(t_now=t_now, verdict=verdict, ranking=[verdict, 4 if verdict != 4 else 3], vote=verdict, vote_counts={},
                vote_n=1, cands=[c(verdict, 0.9), c(4 if verdict != 4 else 3, 0.8)], pass_sec=141.6, pass_index=0,
                wall_time=time.time(), procs=3, signal=signal, status="OK")


def write_tagged(out, name, rec, token):
    (Path(out) / name).write_text(json.dumps(T.tag(rec, token)))


# ============================================================================================== T01
def test_T01_new_trial_failure_never_shows_previous_result(tmp_path, monkeypatch):
    S, root = mk_state(tmp_path, monkeypatch)
    out = root / "out"
    S.start()
    tokA = T.token_now(out)
    write_tagged(out, "floor.json", floor_rec(3, "1100.000"), tokA)
    write_tagged(out, "pose_fit.json", dict(floor_rec(3, "1100.000")["cands"][0], source="floor"), tokA)
    S.stop(); S.exp.update(state="done", signal="1100.000", t_now=1100.0)
    # new trial B
    S.exp = dict(state="idle")
    S.start()
    assert not (out / "floor.json").exists(), "previous trial result must be moved out of the current slot"
    # a late writer from trial A drops its files back (other process)
    write_tagged(out, "floor.json", floor_rec(3, "1100.000"), tokA)
    write_tagged(out, "pose_fit.json", dict(floor_rec(3, "1100.000")["cands"][0], source="pose"), tokA)
    S.stop()
    S.exp.update(state="computing", signal="1200.000")
    tokB = T.token_now(out)
    # trial B: all floors failed -> worker writes a FAILED terminal record for B
    errs = [dict(floor=m, error_type="CalibrationUnavailable", reason_code="CALIBRATION_UNAVAILABLE",
                 error="Not enough overlapping scans with translation to calibrate", stage="fit") for m in range(1, 10)]
    assert LW.finalize_failed(out, tmp_path / "persist", tokB, "1200.000", 1200.0, errs, "all_floors_failed")
    S.tick()
    assert S.exp["state"] == "done" and S.exp["verdict"] is None
    assert "걸은 거리가 너무 짧음" not in S.exp["error"] and "CALIBRATION_UNAVAILABLE" in S.exp["error"]
    snap = S.snapshot(None, "result")
    assert snap["floor"] is None and snap["map"] is None
    assert snap["stale"]["floor"]["status"] == "FAILED"
    # pose file from trial A is STALE, never current
    assert T.classify(json.loads((out / "pose_fit.json").read_text()), T.read_pointer(out))[0] == "STALE_OTHER_TRIAL"
    assert S.pose(None).get("map") is None
    # final ROS publish guard (fake publisher)
    sent = []
    ok, why = T.guarded_publish(out, json.loads((out / "pose_fit.json").read_text()), lambda: sent.append(1))
    assert not ok and why == "STALE_OTHER_TRIAL" and sent == []
    # pose worker does not refit the failed / other-trial floor
    assert LW.pose_step(out, tmp_path / "persist", lambda m: pytest.fail("must not fit"), lambda *_: None).startswith("SKIPPED")
    # E2 top-1 is never labelled FIX_CONFIRMED
    assert "NOT_CONFIRMED" in snap["fix_status"]


def test_T01_static_rt_live_node_publish_paths_are_guarded():
    """rt_live_node imports rclpy (blocked in R1 tests) -> verify the call chain statically (STATIC evidence)."""
    src = (RT / "rt_live_node.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    pp = fns["_publish_pose"]
    assert "T.guarded_publish(self.out, fit, lambda: self.pub_pose.publish(msg)" in pp
    assert pp.count("self.pub_pose.publish(") == 1                       # only inside the guard lambda
    assert "T.usable_now(fit, T.read_pointer(self.out))" in pp
    st = fns["on_status"]
    assert "T.classify(fl, ptr)" in st and "T.usable_now(fit, ptr)" in st
    lw = (RT / "loc_worker.py").read_text()
    assert "finalize_failed(out, persist, token" in lw and "finalize_ok(out, persist, token" in lw
    assert "res = pose_step(out, persist, fit_fn, C.log)" in lw


# ============================================================================================== T02
def test_T02_late_previous_generation_never_overwrites(tmp_path, monkeypatch):
    out, persist = tmp_path / "out", tmp_path / "persist"
    out.mkdir()
    tp = T.TrialPointer(out, persist)
    tp.start("exp_A")
    tokA = T.token_now(out)
    tp.start("exp_B")                                   # generation changed while A's round was computing
    tokB = T.token_now(out)
    assert tokA != tokB
    res = floor_rec(3, "1.0")
    win = {c["map"]: c for c in res["cands"]}
    assert LW.finalize_ok(out, persist, tokA, res, win, 3, 3) is False
    assert not (out / "floor.json").exists() and not (out / "pose_fit.json").exists()
    disc = [json.loads(l) for l in (persist / "discarded.jsonl").read_text().splitlines()]
    assert disc and disc[-1]["reason"] == "GENERATION_MISMATCH"
    assert list((persist / "trials" / tokA["trial_id"]).glob("floor_*.json")), "immutable per-trial copy kept"
    # current generation writes normally
    assert LW.finalize_ok(out, persist, tokB, floor_rec(4, "2.0"), {c["map"]: c for c in floor_rec(4, "2.0")["cands"]}, 4, 4)
    assert json.loads((out / "floor.json").read_text())["trial_id"] == tokB["trial_id"]


def test_T02_race_generation_changes_between_check_and_write(tmp_path, monkeypatch):
    """Cross-process race: the writer's check passed, then the viewer bumped the generation, then the write landed.
    The file carries the old tag, so every reader rejects it as current."""
    out, persist = tmp_path / "out", tmp_path / "persist"
    out.mkdir()
    tp = T.TrialPointer(out, persist)
    tp.start("exp_A")
    tokA = T.token_now(out)
    real = T.still_current

    def check_then_bump(o, tok):
        r = real(o, tok)
        tp.start("exp_B")                               # change lands right after the check
        return r
    monkeypatch.setattr(T, "still_current", check_then_bump)
    assert T.write_result(out, "floor.json", floor_rec(3, "1.0"), tokA, persist, "floor") is True   # landed late
    monkeypatch.setattr(T, "still_current", real)
    late = json.loads((out / "floor.json").read_text())
    assert T.classify(late, T.read_pointer(out))[0] == "STALE_OTHER_TRIAL"
    assert not T.usable_now(late, T.read_pointer(out))
    assert LW.pose_step(out, persist, lambda m: pytest.fail("must not fit stale floor"), lambda *_: None) == "SKIPPED_STALE_OTHER_TRIAL"


def test_T02_pose_target_changes_during_fit_is_discarded(tmp_path):
    out, persist = tmp_path / "out", tmp_path / "persist"
    out.mkdir()
    tp = T.TrialPointer(out, persist)
    tp.start("exp_A")
    tok = T.token_now(out)
    T.write_result(out, "floor.json", floor_rec(3, "1.0"), tok, persist, "floor")

    def slow_fit(m):
        tp.start("exp_B")                               # new trial starts while the pose fit is running
        return dict(floor_rec(3, "1.0")["cands"][0], t=5000.0)
    assert LW.pose_step(out, persist, slow_fit, lambda *_: None) == "DISCARDED"
    assert not (out / "pose_fit.json").exists()


# ============================================================================================== T32
def stationary_frames(n=20):
    rng = np.random.RandomState(0)
    pts = rng.uniform(-5, 5, size=(300, 2))
    return [dict(t=1000.0 + 1.5 * i, pts=pts, xy=np.array([2.0, 3.0]), yaw=0.1) for i in range(n)]


def test_T32_no_moving_pairs_is_calibration_unavailable_no_228(tmp_path):
    with pytest.raises(SF.CalibrationUnavailable) as ei:
        SF.estimate_beta(stationary_frames())
    assert isinstance(ei.value, ValueError) and ei.value.result["status"] == "CALIBRATION_UNAVAILABLE"
    assert ei.value.result["beta"] is None and ei.value.result["provenance"]["n_moving_pairs"] == 0
    r = SF.estimate_beta_typed(stationary_frames())
    assert r["status"] == "CALIBRATION_UNAVAILABLE" and r["beta"] is None and r["verified"] is False
    # through fit_window_v2's beta step with a fake localiser: no cache -> explicit failure, beta untouched
    import metric_fit_v2 as V

    class FakeZ:
        BETA, MIRROR, beta_quality = None, 1, None

        def fit_beta(self, a, b):
            SF.estimate_beta(stationary_frames())
    z = FakeZ()
    with pytest.raises(SF.CalibrationUnavailable):
        V._window_beta(z, 0.0, 60.0, SF.BetaProvider(), dict(sensor="xt16"))
    assert z.BETA is None
    src = (LOCZ / "metric_fit_v2.py").read_text() + (LOCZ / "sensor_frame.py").read_text() + (RT / "loc_worker.py").read_text()
    assert "228" not in src.replace("228°", "").replace("228° 등", "")


# ============================================================================================== T33
def test_T33_verified_beta_reused_only_under_same_contract(tmp_path):
    clock = [0.0]
    contract = dict(sensor="hesai_xt16", calibration_id="cal-1", odom_session_id="s1", building="kci", window_s=60.0)
    ver = dict(beta=3.97, mirror=1, verified=True, contract=contract, t_mono=0.0, provenance=dict(method="bench"))
    calls = []

    def est():
        calls.append(1)
        return dict(status="CALIBRATION_UNAVAILABLE", beta=None, beta_null_reason="no motion")
    bp = SF.BetaProvider(verified=ver, max_age_s=100.0, clock=lambda: clock[0])
    import metric_fit_v2 as V

    class FakeZ:
        BETA, MIRROR, beta_quality = None, 1, None

        def fit_beta(self, a, b):
            raise AssertionError("verified cache must be used")
    z = FakeZ()
    for _ in range(2):                                   # survives fit_window_v2's try/finally restore (provider is outside Z)
        src = V._window_beta(z, 0.0, 60.0, bp, contract)
        assert src == dict(source="VERIFIED_CACHE", verified=True) and z.BETA == 3.97
        z.BETA = None                                    # what the finally block does
    # odom session changed -> invalidated, estimator tried, failure explicit
    r = bp.get(dict(contract, odom_session_id="s2"), est)
    assert r["status"] == "CALIBRATION_UNAVAILABLE" and calls == [1] and bp.verified is None
    assert any(h.get("why") == "CONTRACT_MISMATCH:odom_session_id" for h in bp.history)
    # stale
    bp2 = SF.BetaProvider(verified=dict(ver), max_age_s=10.0, clock=lambda: clock[0])
    clock[0] = 11.0
    assert bp2.get(contract, est)["status"] == "CALIBRATION_UNAVAILABLE" and bp2.verified is None
    # runtime estimate is used for that window only, never promoted to cache
    bp3 = SF.BetaProvider()
    r3 = bp3.get(contract, lambda: dict(status="OK", beta=3.9, quality=dict(mirror=-1), verified=False))
    assert r3["source"] == "RUNTIME_ESTIMATE" and bp3.verified is None and r3["mirror"] == -1
    # an unverified calibration file is refused
    f = tmp_path / "cal.json"
    f.write_text(json.dumps(dict(beta=3.9, verified=False)))
    with pytest.raises(ValueError):
        SF.BetaProvider.from_file(str(f))


# ============================================================================================== T34
def test_T34_evidence_survives_stop_and_eviction_and_io_failure_is_explicit(tmp_path, monkeypatch):
    arch = LS.RunEvidenceArchive(tmp_path / "evidence", max_bytes=10 ** 9)
    st = LS.LiveStore(tmp_path / "store", keep_s=60.0, reset=True, archive=arch)
    LS.pin(tmp_path / "store", "floor_round", 30.0, 45.0, ttl_s=1e6)
    for i in range(80):                                  # 240 s of samples, the last 120 s stationary (same x)
        x = min(i, 40) * 0.5
        assert st.add(3.0 * i, x, 0.0, 0.0, np.zeros((20, 2)), np.zeros(20), np.zeros((3, 2))) is not None
        assert st.archive_status == "OK"
    kept = [r["t_sec"] for r in st.rows]
    assert 0.0 in kept and 30.0 in kept and 42.0 in kept           # first sample + pinned window kept past keep_s
    assert 15.0 not in kept                                         # unpinned old samples still evicted (bounded)
    full = (tmp_path / "evidence" / "odom_link_full.csv").read_text().splitlines()
    assert len(full) == 80, "every sample persisted despite the 60 s live window / stationary tail"
    LS.unpin(tmp_path / "store", "floor_round")
    assert LS.pinned_ranges(tmp_path / "store") == []
    # disk full -> IO_FAILURE (no fake success)
    import shutil
    def enospc(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(shutil, "copyfile", enospc)
    st.add(3.0 * 80, 20.0, 0.0, 0.0, np.zeros((20, 2)), np.zeros(20), np.zeros((3, 2)))
    assert st.archive_status == "IO_FAILURE"
    man = [json.loads(l) for l in (tmp_path / "evidence" / "manifest.jsonl").read_text().splitlines()]
    assert man[-1]["enospc"] is True
    assert len((tmp_path / "evidence" / "odom_link_full.csv").read_text().splitlines()) == 80
    # cap -> RESOURCE_LIMITED
    a2 = LS.RunEvidenceArchive(tmp_path / "ev2", max_bytes=10)
    s2 = LS.LiveStore(tmp_path / "store2", keep_s=60.0, reset=True, archive=a2)
    monkeypatch.undo()
    s2.add(0.0, 0, 0, 0, np.zeros((20, 2)), np.zeros(20), np.zeros((3, 2)))
    assert s2.archive_status == "RESOURCE_LIMITED"


# ============================================================================================== T35
def png(path, h, w, fill=0):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w), 255, np.uint8)
    img[:fill, :fill] = 0
    cv2.imwrite(str(path), img)
    return path


def test_T35_map_contract_hash_shape_frame_scale_are_separate_and_e7_refuses(tmp_path, monkeypatch):
    a = MC.entry("E2_live", png(tmp_path / "a" / "base_map.png", 50, 80, 5))
    b = MC.entry("E6", png(tmp_path / "b" / "base_map.png", 50, 80, 7))
    c = MC.entry("E7", png(tmp_path / "c" / "base_map.png", 50, 80, 5))
    assert a["shape"] == b["shape"] == [50, 80] and a["sha256"] != b["sha256"]
    assert a["scale_provenance"]["status"] == "UNKNOWN" and a["origin_axis"]["status"] == "ASSUMED_PIXEL_FRAME"
    ab = MC.compare(a, b)
    assert ab["shape_equal"] and not ab["content_identical"] and ab["coordinate_frame"] == "UNPROVEN"
    assert MC.compare(a, c)["coordinate_frame"] == "PROVEN"
    proof = [dict(sha256_a=a["sha256"], sha256_b=b["sha256"], verified=True, method="manual overlay")]
    assert MC.compare(a, b, proof)["coordinate_frame"] == "PROVEN"
    pf = MC.e7_live_preflight([ab], calibration=dict(status="CALIBRATION_UNAVAILABLE"))
    assert pf["status"] == "NOT_WIRED" and set(pf["blockers"]) == {"NOT_WIRED", "MAP_CONTRACT_UNPROVEN", "CALIBRATION_UNAVAILABLE"}
    assert MC.e7_live_preflight([MC.compare(a, c)], dict(status="OK"), wired=True)["status"] == "ALLOWED"
    monkeypatch.setattr(E7, "build_window", lambda *a, **k: pytest.fail("E7 must not compute when preflight refuses"))
    e6 = dict(packet_id="p1", observation=dict(bearing_sigma_deg=3.0), params=dict(ppm_lo=15, ppm_hi=27), floor_hypotheses=[])
    r = E7.run_e7(e6, "kci", tmp_path, 10.0, tmp_path / "e7out", live_preflight=pf)
    assert r["status"] == "NOT_WIRED" and r["final"] is None
    # maps on disk are untouched
    assert MC.entry("E2_live", tmp_path / "a" / "base_map.png")["sha256"] == a["sha256"]


def test_T35_historical_kci_4F_maps_unproven_but_same_shape():
    """HISTORICAL (read-only): the live 9th-door 4F map vs the E6 kci_handoff 4F map (audit 08)."""
    live = Path("/home/unitree/sci_ws/external/sci_live/floor_id/maps/kci_maps_perfloor/4F/maps/base_map.png")
    e6 = Path("/home/unitree/sci_ws/maps/kci_handoff/4F/base_map.png")
    if not (live.exists() and e6.exists()):
        pytest.skip("historical maps not present")
    cmp_ = MC.compare(MC.entry("E2_live", live), MC.entry("E6", e6))
    assert cmp_["shape_equal"] and not cmp_["content_identical"] and cmp_["coordinate_frame"] == "UNPROVEN"


@pytest.mark.parametrize("z", [0.0, -1.0, float("nan"), float("inf"), None])
def test_T35_invalid_depth_is_never_clamped(z, tmp_path):
    intr = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    with pytest.raises(G.InvalidDepth):
        G.rs_pixel_to_body(320, 240, z, intr)
    pkt = dict(packet_id="x", observation=dict(bearing_body_rad=0.1, range_m=z, bearing_sigma_rad=0.05, range_sigma_m=0.1),
               semantic=dict(label_alternatives=[dict(label="418", p=1.0)]))
    r = E6.run_e6(pkt, ["4F"], tmp_path / "e6")
    assert r["status"] == "INPUT_INVALID" and r["floor_hypotheses"] == []


def test_T35_z_depth_to_horizontal_range_once():
    intr = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    p = G.rs_pixel_to_body(320 + 300, 240, 2.0, intr)          # z = 2 m, 1 m to the right
    b, r = G.body_bearing_range(p)
    assert abs(r - math.hypot(2.0, 1.0)) < 1e-9 and abs(b + math.atan2(1.0, 2.0)) < 1e-9


# ============================================================================================== T36
def test_T36_label_types_need_their_own_evidence_and_legacy_is_excluded(tmp_path, monkeypatch):
    S, root = mk_state(tmp_path, monkeypatch)
    run = tmp_path / "exp" / "kci_e5_x"
    (run / "trials" / "ev001" / "runner").mkdir(parents=True)
    (run / "trials" / "ev001" / "runner" / "stop_frame.jpg").write_bytes(b"jpg")
    shm = tmp_path / "shm"
    shm.mkdir()
    (shm / "status.json").write_text(json.dumps(dict(run=str(run), events=[dict(id=1, state="WAIT_STABLE")])))
    legacy = S._labels_path()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps(dict(run="kci_e5_x", ev=1, label="O", state="WAIT_STABLE")) + "\n")
    h0 = hashlib.sha256(legacy.read_bytes()).hexdigest()
    assert S.e5_labels() == {}, "invalid 10-15 style O/X must not count"
    r = S.e5_label(dict(ev=1, label="O"))                                  # old UI button -> sign_true, allowed before OCR
    assert r["ok"] and r["label_type"] == "sign_true" and r["revision"] == 1
    r2 = S.e5_label(dict(ev=1, label_type="sign_true", value=False))
    assert r2["revision"] == 2 and S.e5_labels() == {"kci_e5_x#1": "X"}
    assert "error" in S.e5_label(dict(ev=1, label_type="text_correct", value=True))      # no reading yet
    assert "error" in S.e5_label(dict(ev=1, label_type="aim_ok", value=True))            # no pan/tilt capture
    assert "error" in S.e5_label(dict(ev=1, label_type="final_pose_correct", value=True))  # no FIX (E7 not wired)
    (shm / "status.json").write_text(json.dumps(dict(run=str(run), events=[dict(
        id=1, state="READ_ASYNC", capture_status="CAPTURE_QUALITY_FAIL", pantilt_ok=True, capture="c.jpg",
        reading=dict(status="NO_TEXT_DETECTED", input_sha256="ab"))])))
    assert S.e5_label(dict(ev=1, label_type="text_correct", value=False))["ok"]
    assert S.e5_label(dict(ev=1, label_type="focus_ok", value=False))["ok"]
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == h0


def test_T36_old_gt_click_does_not_produce_current_error(tmp_path, monkeypatch):
    """Replays the 10-15 case: trial B is running, floor.json/pose_fit from trial A remain; a click must not be paired."""
    S, root = mk_state(tmp_path, monkeypatch)
    out = root / "out"
    S.start()
    tokA = T.token_now(out)
    S.exp = dict(state="idle")
    S.start()
    write_tagged(out, "floor.json", floor_rec(4, "1100.000"), tokA)
    write_tagged(out, "pose_fit.json", dict(floor_rec(4, "1100.000")["cands"][0], source="pose"), tokA)
    (out / "live_odom.json").write_text(json.dumps(dict(t=1200.0, odom_link=[3.0, 0.0, 0.0])))
    m = S.mark("P1", 4, 120.0, 100.0, None, role="end")
    assert "err_m" not in m and m["err_null_reason"] == "NO_CURRENT_TRIAL_ESTIMATE"
    m0 = S.mark("P2", 4, 120.0, 100.0, None)                              # no role -> no distance error
    assert m0["err_null_reason"] == "NO_ROLE"
    tokB = T.token_now(out)
    write_tagged(out, "pose_fit.json", dict(floor_rec(4, "1200.000")["cands"][0], source="pose"), tokB)
    m2 = S.mark("P3", 4, 120.0, 100.0, None, role="checkpoint")
    assert "err_m" in m2 and m2["estimate_id"].startswith(tokB["trial_id"]) and m2["reference_time"] == 1200.0
