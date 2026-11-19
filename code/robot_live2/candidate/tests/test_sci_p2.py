"""Unit tests for the Phase-2 P2 stop-timing policy and the pan/tilt planner additions (synthetic geometry)."""
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
for p in ("sci_visual_observation", "sci_pantilt", "sci_text_reading"):
    sys.path.insert(0, str(SRC / p))

import policy as P  # noqa: E402
import timing as T  # noqa: E402
from features import Geometry  # noqa: E402

GEOM = Geometry()
CFG_A = str(Path(__file__).resolve().parents[1] / "config" / "e5_policy.json")   # production gates
SIGN = np.array([6.0, 1.4])          # left wall sign (odom frame), wall normal points into the corridor (-y)
UP = 0.9                              # sign height above the camera (m)


def region_at(robot_x, spot=7, sim=0.12):
    """Region + features of the sign seen from robot pose (robot_x, 0, yaw 0), consistent with legacy conventions."""
    fwd, left = SIGN[0] - robot_x, SIGN[1]
    z = fwd
    u = GEOM.cx - GEOM.fx * left / z
    v = GEOM.cy - GEOM.fy * UP / z
    box = [u - 15, v - 10, u + 15, v + 10]
    basis = T.cam_basis(None)
    n_c = (-1.0) * basis[1]           # world normal (0,-1) = body "right" = -left
    ray = fwd * basis[0] + left * basis[1]
    obl = math.degrees(math.acos(abs(float(n_c @ ray) / np.linalg.norm(ray))))
    r = dict(box=box, tile=box, sim=sim, msim=sim, z=z, xyz=[SIGN[0], SIGN[1], UP], spot=spot, nobs=3, nscene=3, board=-0.05,
             adv=0.02, hgt=1.5, fired=False, neg={c: 0.0 for c in ("poster", "board", "glare", "light", "elevator", "directory", "other")},
             ctx={c: 0.0 for c in ("poster", "board", "directory", "elevator", "glare")}, ctx_plate=0.05)
    f = dict(glare=dict(sat=0.0, spec=0.0, contrast=100, sharp=200), plane=dict(ok=True, normal=list(n_c), obliquity_deg=obl, valid_in=1.0),
             edge=50, speed=0.4, bearing=0.0, up=None, pantilt=dict(ok=True), text_px=100, z_tile=z, obliquity_pred=obl)
    return r, f


def run(xs, cfg_p2=None, e4=None):
    pol = T.P2Policy(P.load_cfg(CFG_A), cfg_p2 or T.load_p2(None), GEOM)
    stops = []
    for k, x in enumerate(xs):
        if x <= SIGN[0] - 1.6:        # the sign is in the RealSense view only while far enough ahead
            r, f = region_at(x)
            _, st, _ = pol.step(float(k) * 0.25, (x, 0.0, 0.0), 0.4, [r], [f], thr=0.08, e4=e4)
        else:
            st, _ = pol.tick(float(k) * 0.25, (x, 0.0, 0.0), 0.4, e4)
        if st:
            stops.append((x, st))
    return pol, stops


def test_p2_waits_for_readable_view_then_stops_once():
    xs = np.arange(0.0, 6.5, 0.1)
    pol, stops = run(xs)
    assert len(stops) == 1
    x, st = stops[0]
    assert x > 0.5                                   # not at first sight (oblique, far)
    assert st["predicted"]["obliquity"] <= 55.0 - 5.0 + 1e-6
    assert st["predicted"]["reachable"]


def test_p2_never_stops_without_verification():
    pol = T.P2Policy(P.load_cfg(CFG_A), T.load_p2(None), GEOM)
    r, f = region_at(0.0)
    r["sim"] = 0.01                                  # below SigLIP floor -> never verified
    for k in range(20):
        _, st, _ = pol.step(float(k), (0.1 * k, 0.0, 0.0), 0.4, [r], [f], thr=0.08, e4=None)
        assert st is None


def test_p2_missed_when_passed_out_of_reach():
    cfg = T.load_p2(None)
    cfg["readability"]["obliquity_max_deg"] = 5.0    # unreachable envelope -> must end as MISSED, not stop late
    pol, stops = run(np.arange(0.0, 7.5, 0.1), cfg)
    assert not stops and pol.missed


def test_p2_e4_confident_suppresses():
    _, stops = run(np.arange(0.0, 6.5, 0.1), e4=dict(available=True, level="CONFIDENT"))
    assert not stops


def test_p2_merges_near_duplicate_spots():
    pol = T.P2Policy(P.load_cfg(CFG_A), T.load_p2(None), GEOM)
    for k, x in enumerate((1.5, 1.8)):              # within the 5 m verification range
        r, f = region_at(x, spot=10 + k)            # two spot ids, same physical sign
        pol.step(float(k), (x, 0.0, 0.0), 0.4, [r], [f], thr=0.08, e4=None)
    assert len({id(t) for t in pol.tracks.values()}) == 1


def test_plan_point_off_image_and_behind():
    from planner import PanTiltPlanner
    pl = PanTiltPlanner(GEOM)
    d = pl.plan_point([-1.6, -0.3, 1.0])            # far left of the RealSense image, still ahead
    assert d["off_image"] and d["url"] and "strict=1" in d["url"]
    assert pl.plan_point([1.0, 0.0, -0.5])["reachable"] is False


def test_refine_is_bounded():
    from planner import PanTiltPlanner
    pl = PanTiltPlanner(GEOM)
    plan = pl.plan(320, 240, 3.0)
    r = pl.refine_from_capture((5000, 0), plan)
    assert abs(r["d_pan"]) <= 5.0 and abs(r["d_tilt"]) <= 5.0
    r0 = pl.refine_from_capture((0, 0), plan)
    assert abs(r0["d_pan"]) < 1e-9 and abs(r0["d_tilt"]) < 1e-9


def test_p2_yaml_loads():
    cfg = T.load_p2(str(Path(__file__).resolve().parents[1] / "config" / "e5_policy_p2.yaml"))
    assert cfg["verify"]["dmax_m"] == 5.0 and cfg["timing"]["tick_hz"] == 10.0
