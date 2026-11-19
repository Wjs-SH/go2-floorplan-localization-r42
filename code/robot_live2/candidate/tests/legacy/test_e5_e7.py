"""Unit tests for the E5/E6/E7 research modules (synthetic maps with known truth + real-map frame checks).

Synthetic truth is used ONLY inside tests to verify the code; the algorithms never read GT.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "sci_semantic_localization"))
from e5_e7 import geometry as G          # noqa: E402
from e5_e7 import e6_region as E6        # noqa: E402
from e5_e7 import e7_rematch as E7       # noqa: E402
from e5_e7 import e5_packet as E5        # noqa: E402

INTR = dict(fx=606.7977, fy=606.7451, cx=310.4879, cy=240.5169)


# ------------------------------------------------------------------ geometry
def test_convention_matches_floor_id():
    rng = np.random.RandomState(0)
    for _ in range(50):
        phi, ppm = rng.uniform(-math.pi, math.pi), rng.uniform(10, 40)
        b, pos = rng.uniform(-5, 5, 2), rng.uniform(0, 500, 2)
        assert np.allclose(G.body_to_map_px(pos, phi, b, ppm), pos + ppm * G.floor_id_A(phi) @ b)
        q = G.body_to_map_px(pos, phi, b, ppm)
        assert abs(G.wrap(G.map_px_heading(pos, q) - (phi + math.atan2(b[1], b[0])))) < 1e-9


def test_rs_pixel_bearing():
    p = G.rs_pixel_to_body(INTR["cx"], INTR["cy"], 3.0, INTR)          # optical centre -> straight ahead
    assert np.allclose(p, [3.0, 0.0, 0.0])
    p = G.rs_pixel_to_body(INTR["cx"] - INTR["fx"], INTR["cy"], 2.0, INTR)  # 1 focal length left -> +45 deg
    beta, r = G.body_bearing_range(p)
    assert abs(math.degrees(beta) - 45.0) < 1e-6 and abs(r - 2.0 * math.sqrt(2)) < 1e-9
    p = G.rs_pixel_to_body(INTR["cx"], INTR["cy"], 2.0, INTR, dict(yaw=10, pitch=0, roll=0))
    assert abs(math.degrees(G.body_bearing_range(p)[0]) - 10.0) < 1e-9


def test_servo_inverse():
    servo = dict(pan_zero_deg=95.35, pan_deg_per_left_deg=1.13, tilt_zero_deg=157.19, tilt_deg_per_up_deg=-0.87)
    d = G.servo_to_rs_direction(95.35 + 1.13 * 30, 157.19, servo)
    assert abs(math.degrees(math.atan2(d[1], d[0])) - 30) < 1e-9


# ------------------------------------------------------------------ synthetic floor
def make_floor(tmp: Path, name="9F"):
    """Thin-walled corridor (y 150..200) below one room (y 60..140) separated by a 10-px wall; a pillar breaks
    symmetry; outside the building = unknown grey (128), like real floor plans (walls < 50, free > 200)."""
    img = np.full((300, 400), 128, np.uint8)
    img[146:204, 16:384] = 0            # corridor outer wall (4 px)
    img[56:150, 146:204] = 0            # room outer wall; room/corridor shared wall y 140..150
    img[150:200, 20:380] = 255          # corridor
    img[60:140, 150:200] = 255          # room 901
    img[150:165, 330:345] = 0           # pillar (asymmetry)
    d = tmp / name
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "base_map.png"), img)
    sem = dict(schema_version="test", map_meta=dict(frame_id="map", image_size=dict(width=400, height=300),
                                                     resolution_m_per_px=0.05),
               spaces=[dict(id="room_901", type="room", room_number="901",
                            polygon_px=[[150, 60], [200, 60], [200, 140], [150, 140]])])
    (d / f"{name}_semantic_map.json").write_text(json.dumps(sem))
    return img


TRUE = dict(x=175.0, y=190.0, psi=0.0, ppm=20.0)   # facing map +x (east); sign is to the robot's left


def packet_for_truth(label="901"):
    s = np.array([175.0, 145.0])                                   # point on the room's corridor wall
    d_px = np.hypot(*(s - [TRUE["x"], TRUE["y"]]))
    r = d_px / TRUE["ppm"]
    beta = G.map_px_heading([TRUE["x"], TRUE["y"]], s) - TRUE["psi"]
    return dict(packet_id="synthetic", semantic=dict(label_alternatives=[dict(label=label, p=0.9)]),
                observation=dict(bearing_body_rad=beta, range_m=r, bearing_sigma_rad=math.radians(4),
                                 range_sigma_m=G.range_sigma(r)))


PARAMS = dict(ppm_lo=18.0, ppm_hi=22.0)


def test_e6_region_contains_truth(tmp_path):
    make_floor(tmp_path / "maps")
    res = E6.run_e6(packet_for_truth(), ["9F"], tmp_path / "out", building="synthetic", params=PARAMS,
                    maps_root=tmp_path / "maps")
    assert res["status"] == "OK" and res["gt_used"] is False
    h = res["floor_hypotheses"][0]
    mask = cv2.imread(h["region_mask_png"], 0) > 0
    assert mask[int(TRUE["y"]), int(TRUE["x"])], "E6 region must contain the true position"
    hp = np.load(h["heading_npy"])
    assert abs(G.wrap(math.radians(hp[int(TRUE["y"]), int(TRUE["x"])]) - TRUE["psi"])) < math.radians(20)
    assert not mask[100, 175], "inside the room (behind the wall) must be excluded by line of sight"


def test_e6_wrong_label_gives_no_hypothesis(tmp_path):
    make_floor(tmp_path / "maps")
    res = E6.run_e6(packet_for_truth("402"), ["9F"], tmp_path / "out", building="synthetic", params=PARAMS,
                    maps_root=tmp_path / "maps")
    assert res["status"] == "NO_HYPOTHESIS"


def test_norm_label():
    assert E6.norm_label("401호") == "401" and E6.norm_label(" Room 401 ") == "401" and E6.norm_label("b102") == "B102"
    assert E6.norm_label("none") == ""


# ------------------------------------------------------------------ E7 core on the synthetic floor
def synthetic_window(img, phi):
    wall, free = img < 50, img > 200
    seen = free.copy()
    seen[:146, :] = False                                   # robot in the corridor cannot see inside the room
    edge = wall & (cv2.dilate(seen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    ys, xs = np.nonzero(edge)
    w = np.c_[xs, ys].astype(float)
    pos = np.array([TRUE["x"], TRUE["y"]])
    w = w[np.hypot(*(w - pos).T) < 12 * TRUE["ppm"]]
    cloud = (G.floor_id_A(phi) @ (w - pos).T).T / TRUE["ppm"]          # A is an involution
    # same construction as floor_id metric_fit_v2._maps: distance to wall + excess depth inside thick walls
    from scipy.ndimage import distance_transform_edt
    DF = distance_transform_edt(~wall)
    depth = distance_transform_edt(wall)
    h = float(np.percentile(depth[wall], 90))
    F = DF + np.maximum(depth - h, 0.0)
    return F, free, wall, cloud


def test_e7_recovers_pose_inside_region_and_passes_gate(tmp_path):
    img = make_floor(tmp_path / "maps")
    res = E6.run_e6(packet_for_truth(), ["9F"], tmp_path / "out", building="synthetic", params=PARAMS,
                    maps_root=tmp_path / "maps")
    h = res["floor_hypotheses"][0]
    region = cv2.imread(h["region_mask_png"], 0) > 0
    hp = np.load(h["heading_npy"])
    F, free, wall, cloud = synthetic_window(img, TRUE["psi"])
    cands, diag = E7.refine_in_region(F, free, wall, region, cloud, np.zeros((1, 2)),
                                      [TRUE["psi"], TRUE["psi"] + math.pi], 18.0, 22.0, max_seeds=40)
    assert cands, diag
    b = cands[0]
    assert math.hypot(b["x"][0] - TRUE["x"], b["x"][1] - TRUE["y"]) < 4.0
    assert abs(b["ppm"] - TRUE["ppm"]) < 1.0

    def e6h(px, py):
        v = hp[int(round(py)), int(round(px))]
        return float(v) if np.isfinite(v) else None

    g = E7.gate(cands, lambda x: x[2], e6h, math.radians(4))
    assert g["passed"], json.dumps(g, default=float)


def test_e7_wrong_region_fails_honestly(tmp_path):
    """Spec: if E6 hands a wrong region, E7 must fail or return the wrong result honestly — it may never reach
    the truth outside the region (no repair). Whether the gate passes is recorded, not forced: in a self-similar
    corridor a wrong region can still fit (documented limitation, E7 report)."""
    img = make_floor(tmp_path / "maps")
    F, free, wall, cloud = synthetic_window(img, TRUE["psi"])
    region = np.zeros_like(free)
    region[150:200, 40:90] = True                                        # far west end of the corridor
    cands, _ = E7.refine_in_region(F, free, wall, region, cloud, np.zeros((1, 2)),
                                   [TRUE["psi"], TRUE["psi"] + math.pi], 18.0, 22.0, max_seeds=40)
    g = E7.gate(cands, lambda x: x[2], lambda px, py: 0.0, math.radians(4))
    print("WRONG_REGION_GATE", json.dumps(dict(passed=g["passed"], reasons=g["reasons"],
                                                best=cands[0]["x"] if cands else None), default=float))
    tol = cv2.dilate(region.astype(np.uint8), np.ones((45, 45), np.uint8)).astype(bool)
    for c in cands:                                                      # never leaves the (tolerated) region
        assert tol[int(round(c["x"][1])), int(round(c["x"][0]))]
        assert math.hypot(c["x"][0] - TRUE["x"], c["x"][1] - TRUE["y"]) > 40   # cannot reach the truth


# ------------------------------------------------------------------ E5 packet builder
def test_e5_packet_builder(tmp_path):
    tr = tmp_path / "trial_x"
    (tr / "runner").mkdir(parents=True)
    (tr / "console").mkdir()
    (tr / "reading").mkdir()
    box = [INTR["cx"] - INTR["fx"] - 10, 200, INTR["cx"] - INTR["fx"] + 10, 220]
    ev = dict(t=12.0, stamp=1.0, picks=[box], z=[2.0], xyz=[1, 2, 1.5], pose=[0, 0, 0], image="events/e.jpg",
              retarget=dict(t=13.0, pose=[0.1, 0, 0], box=box, z=2.0, in_view=True))
    (tr / "runner" / "events.jsonl").write_text(json.dumps(ev) + "\n")
    (tr / "console" / "aim_log.jsonl").write_text(json.dumps(dict(time="x", u=1, v=2, pan=130.0, tilt=157.0,
                                                                  clipped=False, aim_sec=1.0, focus={}, total_sec=2.0,
                                                                  image="out/aim/a.jpg")) + "\n")
    (tr / "reading" / "e5_reading.json").write_text(json.dumps(dict(engine="t", raw_answer="401",
                                                                     label_alternatives=[dict(label="401", p=0.8)],
                                                                     sec=0.5, t_request=1, t_answer=2)))
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps(dict(realsense_color=INTR, realsense_mount_deg=dict(yaw=0, pitch=0, roll=0),
                                     servo=dict(pan_zero_deg=95.35, pan_deg_per_left_deg=1.13,
                                                tilt_zero_deg=157.19, tilt_deg_per_up_deg=-0.87))))
    pk = E5.build_packet(tr, calib)
    assert pk["schema"] == E5.SCHEMA and pk["gt_used"] is False
    beta = math.degrees(pk["observation"]["bearing_body_rad"])
    v = 0.5 * (box[1] + box[3])
    assert abs(beta - 45.0) < 0.5, beta
    assert pk["semantic"]["final_label"] == "401" and pk["failure_reason"] is None
    assert pk["siglip"]["note"]  # not wrapped -> explicitly flagged


# ------------------------------------------------------------------ real maps: frame provenance (read-only)
@pytest.mark.skipif(not (E6.EVENT_MAPS / "4F" / "base_map.png").exists(), reason="event maps not present")
@pytest.mark.parametrize("building,floor", [("kci", f"{n}F") for n in range(1, 10)]
                         + [("gong", "G1F"), ("gong", "G2F"), ("sw", "S1F"), ("sw", "S6F")])
def test_kci_event_vs_floor_id_frame(building, floor):
    r = E6.frame_equivalence(building, floor)
    print(floor, json.dumps(r))
    assert r["status"] in ("IDENTICAL_FILE", "SAME_SIZE_DIFFERENT_CONTENT", "DIFFERENT_SIZE", "UNVERIFIED_MISSING_FILE")
