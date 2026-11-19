"""STAGING (2026-10-30): semantic-recovery REGIONAL E2 on the REAL loc2 LiveStore (X20261030_102650).
The algorithm input is only (floor, room id) of a SIMULATED plate reading — there is no field E6 reading for loc2 yet. The room is
chosen by the EVALUATION side as the semantic room nearest to where the robot really was at t_now (truth-anchored transform of the
earlier probe, data/probe_loc2.json of e7_scale_audit_20261030_1600). The regional fit itself is the student's metric_fit_v2 with
its seed grid limited to the plate region (e2_core region=). Scoring = start pose implied by each candidate vs the start mark
(1013.3, 705.6) px — GT is used for scoring only. Also: a region for a WRONG room (far away) must not yield a near-truth candidate
(that would mean the region is ignored). Run: python3 -s -B test_e2_region_loc2.py OUT_DIR"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
STORE = OPR / "experiments/X20261030_102650/archive/live2_store"
GT = (1013.3, 705.6)
PROBE = Path.home() / "sci_ws/reports/e7_scale_audit_20261030_1600/data/probe_loc2.json"
SEM4 = Path.home() / "sci_ws/maps/kci_handoff/4F/4F_semantic_map.json"


def main(out):
    out = Path(out); out.mkdir(parents=True)
    sys.path.insert(0, str(STG))
    spec = importlib.util.spec_from_file_location("e2w", STG / "e2_worker.py")
    W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
    W.SCI_LIVE = OPR / "candidate" / "external" / "sci_live"          # same cache key as the operational worker (read-only cache)
    import e8_pair as EP
    a = argparse.Namespace(building="kci", store=str(STORE), out=str(out / "e2"), cache_dir=str(OPR / "runtime/e2_cache"), floors=None,
                           procs=1, exact_dt=6.0, device="cpu", min_span=70.0, min_pass_gap_s=120.0, min_window_path_m=2.0,
                           wall_pass_top=4, mem_start_mb=3500.0, mem_kill_mb=2800.0, temp_stop_c=78.0, temp_hold_c=76.0,
                           early_window_s=20.0, coarse=1)
    (out / "e2").mkdir()
    FR = W.FloorRunner(a)
    import csv
    rows = [r for r in csv.DictReader(open(STORE / "odom_link.csv"))]
    T = np.array([float(r["t_sec"]) for r in rows]); XY = np.array([[float(r["fastlio_x"]), float(r["fastlio_y"])] for r in rows])
    t0 = T[0]
    pr = json.loads(PROBE.read_text())["probe"]
    sem = json.loads(SEM4.read_text())
    # truth-anchored transform (evaluation only): pos at start = GT, phi / ppm from the probe (coarse + full correction)
    phi = math.radians(pr["coarse"]["phi"] + pr["full"]["corr"]["dphi_deg"]); ppm = pr["coarse"]["ppm"] * pr["full"]["corr"]["scale"]
    A = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])

    def truth_px(t):
        i = int(np.argmin(abs(T - t)))
        return np.asarray(GT) + ppm * (A @ (XY[i] - XY[0]))

    def room_near(p, exclude_far=False):
        best = None
        for s in sem["spaces"]:
            if s.get("type") not in ("room", "stair") or not s.get("polygon_px"):
                continue
            c = np.asarray(s["polygon_px"], float).mean(0); d = float(np.hypot(*(c - p)))
            key = -d if exclude_far else d
            if best is None or key < best[0]:
                best = (key, s, c)
        return best[1], best[2]

    def implied_start_err(c):
        rec_phi = math.radians(c["phi"]) if "phi" in c else math.radians(c["phi_deg"])
        Ac = np.array([[math.cos(rec_phi), -math.sin(rec_phi)], [math.sin(rec_phi), math.cos(rec_phi)]])
        sp = np.asarray(c["pos"], float) + c["ppm"] * (Ac @ (XY[0] - np.asarray(c["anchor"], float)))
        return float(np.hypot(*(sp - np.asarray(GT))))
    res = []
    for t_rel in (120.0, 235.0):
        t_now = float(T[T <= t0 + t_rel][-1])
        p_true = truth_px(t_now)
        for kind in ("TRUE_ROOM", "WRONG_ROOM"):
            sp, c = room_near(p_true, exclude_far=(kind == "WRONG_ROOM"))
            res_m = 1.0 / float(EP.SCALE_OVERRIDE["4"]["px_per_m"])
            half = float(np.max(np.hypot(*(np.asarray(sp["polygon_px"], float) - c).T)))
            region = dict(center_px=[float(c[0]), float(c[1])], radius_px=half + EP.REGION_RADIUS_M * EP.REGION_PPM_MARGIN / res_m)
            t1 = time.time()
            try:
                o = FR.run_floor(4, t_now, FR.W, "exact", region=region)
                cands = o["candidates"]; err = None
            except Exception as e:                                             # noqa: BLE001
                cands, err = [], f"{type(e).__name__}: {e}"
            errs = sorted((round(implied_start_err(dict(c_, pos=[c_["px"], c_["py"]] if "pos" not in c_ else c_["pos"])) / 23.96, 2),
                           i) for i, c_ in enumerate(cands)) if cands else []
            row = dict(t_rel=t_rel, kind=kind, room=sp["id"], room_dist_to_truth_px=round(float(np.hypot(*(c - p_true))), 1),
                       region_radius_px=round(region["radius_px"], 1), n=len(cands), error=err, fit_s=round(time.time() - t1, 1),
                       best_start_err_m_core_scale=errs[0][0] if errs else None,
                       start_err_m_by_rank=[e for e, _ in sorted(errs, key=lambda x: x[1])][:12],
                       ppm=[round(c_["ppm"], 2) for c_ in cands][:12], merit=[round(c_["merit"], 4) for c_ in cands][:12])
            res.append(row); print(json.dumps(row), flush=True)
            (out / "e2_region_loc2_result.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
