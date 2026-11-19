"""R1 W-A BEFORE: reproduce the audited defects through APIs that exist in BOTH trees (env SCI_WS selects the tree).
Each test asserts the CORRECT behaviour, so on the baseline tree it is expected to FAIL (defect reproduced) and on the
candidate tree to PASS. Evidence type: SIMULATED inputs shaped after the 10-15 logs (HISTORICAL_REPLAY where noted)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
RT = WS / "external" / "sci_live" / "realtime"
LOCZ = WS / "external" / "sci_live" / "floor_id" / "localizer"
for p in (str(RT), str(RT / "tools"), str(LOCZ), str(WS / "src" / "sci_semantic_localization"), str(WS / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import live_viewer as LV  # noqa: E402
import live_store as LS  # noqa: E402
import sensor_frame as SF  # noqa: E402
from e5_e7 import geometry as G  # noqa: E402


def mk(tmp_path, monkeypatch):
    root = tmp_path / "rt"
    (root / "out").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    rows = ["sample_id,t_sec,fastlio_x,fastlio_y,fastlio_yaw_rad,delta_s_m,delta_yaw_deg"]
    rows += [f"LIVE_{i:06d},{1000 + 3 * i:.6f},{0.5 * i:.6f},0.0,0.0,0.5,0.0" for i in range(40)]
    (root / "data" / "odom_link.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setenv("RT_PERSIST_DIR", str(tmp_path / "persist"))
    monkeypatch.setenv("SCI_SHM", str(tmp_path / "shm"))
    (tmp_path / "maps" / "1F").mkdir(parents=True)
    return LV.State(root, tmp_path / "maps", 60.0, 300, trigger=str(tmp_path / "trig"), archive=None), root


def prev_floor(signal):
    c = dict(map=3, t=1060.0, t_now=1060.0, twoway=0.9, ppm=20.0, phi=0.0, beta=3.9, mirror=1, pos=[100.0, 100.0],
             anchor=[0.0, 0.0], wall_cost_m=0.05, scene=None, sec=1.0)
    return dict(t_now=1060.0, verdict=3, ranking=[3, 4], vote=3, cands=[c, dict(c, map=4, twoway=0.81)], pass_sec=141.6,
                pass_index=0, signal=signal)


def test_B_T01_failed_trial_shows_previous_floor(tmp_path, monkeypatch):
    """exp_222233: all floors failed; the 21:21 floor.json (other signal) stayed and was drawn as the result."""
    S, root = mk(tmp_path, monkeypatch)
    (root / "out" / "floor.json").write_text(json.dumps(prev_floor("1080.000")))
    (root / "out" / "pose_fit.json").write_text(json.dumps(dict(prev_floor("1080.000")["cands"][0], source="pose")))
    S.exp = dict(state="done", start=1090.0, end=1110.0, signal="1110.000", verdict=None, ranking=[], t_now=1110.0,
                 error="모든 층 정합 실패")
    snap = S.snapshot(None, "result")
    assert snap["floor"] is None, f"previous trial floor shown as current: {snap['floor']}"
    assert snap["map"] is None


def test_B_T01_fixed_failure_text(tmp_path, monkeypatch):
    S, root = mk(tmp_path, monkeypatch)
    sig = time.time() - 5
    lines = [f"{time.strftime('%H:%M:%S')} [floor] {m}층 정합 실패: ValueError: Not enough overlapping scans with translation to calibrate"
             for m in range(1, 2)]
    (root / "out" / "floor.stdout").write_text("\n".join(lines) + "\n")
    why = S._floor_failed(dict(signal=f"{sig:.3f}"))
    assert why is not None and "걸은 거리가 너무 짧음" not in why, why


def test_B_T02_pose_role_has_no_trial_check():
    """STATIC: run_pose follows floor.json vote with no trial/generation check (audit 01/09: 3F refit until 22:39)."""
    src = (RT / "loc_worker.py").read_text()
    body = src[src.index("def run_pose"):]
    assert "pose_step" in body or "generation" in body, "pose role refits whatever floor.json says"


def test_B_T32_beta_failure_is_untyped(tmp_path):
    rng = np.random.RandomState(0)
    pts = rng.uniform(-5, 5, size=(300, 2))
    frames = [dict(t=1000.0 + 1.5 * i, pts=pts, xy=np.array([2.0, 3.0]), yaw=0.1) for i in range(20)]
    with pytest.raises(ValueError) as ei:
        SF.estimate_beta(frames)
    assert getattr(ei.value, "result", None) and ei.value.result["status"] == "CALIBRATION_UNAVAILABLE"


def test_B_T34_inflight_window_evicted(tmp_path):
    """keep_s eviction removes samples an in-flight floor round still reads (exp_222233: first 94 s already gone)."""
    st = LS.LiveStore(tmp_path / "store", keep_s=60.0, reset=True)
    if hasattr(LS, "pin"):
        LS.pin(tmp_path / "store", "floor_round", 30.0, 45.0, ttl_s=1e6)
    for i in range(80):
        st.add(3.0 * i, i * 0.5, 0.0, 0.0, np.zeros((20, 2)), np.zeros(20), np.zeros((3, 2)))
    assert 33.0 in [r["t_sec"] for r in st.rows]


@pytest.mark.parametrize("z", [0.0, float("nan")])
def test_B_T35_invalid_depth_accepted(z):
    intr = dict(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    with pytest.raises(ValueError):
        G.rs_pixel_to_body(320, 240, z, intr)


def test_B_T36_late_click_paired_with_window_end_pose(tmp_path, monkeypatch):
    """HISTORICAL_REPLAY of checkpoints.jsonl line 5: a click 61 min after the result was paired with the window-end pose."""
    S, root = mk(tmp_path, monkeypatch)
    fl = prev_floor("1110.000")
    (root / "out" / "floor.json").write_text(json.dumps(fl))
    (root / "out" / "live_odom.json").write_text(json.dumps(dict(t=4800.0, odom_link=[30.0, 0.0, 0.0])))
    S.exp = dict(state="done", start=1000.0, end=1110.0, signal="1110.000", verdict=3, ranking=[3, 4], t_now=1060.0)
    m = S.mark("P5", 3, 50.0, 50.0, None)
    assert "err_m" not in m, f"distance error produced from a different-time pose: {m}"


def test_B_T36_invalid_legacy_ox_counted(tmp_path, monkeypatch):
    S, root = mk(tmp_path, monkeypatch)
    p = S._labels_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(run="kci_e5_20261015_220127", ev=1, label="O", state="WAIT_STABLE", reading=None)) + "\n")
    assert S.e5_labels() == {}, "10-15 invalid O/X (pressed before reading) counted in accuracy"
