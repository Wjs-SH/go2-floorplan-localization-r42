#!/usr/bin/env python3
"""e7_rematch.py — E7: region-limited structural LiDAR re-matching + explicit quality gate.

NEW module (E7 did not exist; thesis deck: "미수행 · 정합 방법·게이트 기준 미정"). The method and the gate below are
fixed here, before any trial, and are not tuned on GT (GT is never read).

Method
  * Map field, scan window, odometry and sensor rotation come from the canonical E1 localiser
    (~/jsh_onboard_robot_20261001/floor_id, imported read-only): F = wall distance field with wall-depth penalty
    (metric_fit_v2._maps), cloud = window structural points in the anchor-relative odom frame, built exactly as
    metric_fit_v2.fit_window_v2 l.71-82.
  * Seeds: grid (GRID_STEP 22 px, same as floor_id) restricted to the E6 region of that floor; orientations = the
    localiser's own Manhattan modes (bootstrap) — E6's heading is NOT used to seed, only to check (gate G7).
  * Each seed: bounded least squares on min(F/ppm, 5) with soft_l1 (same residual/bounds as metric_fit_v2 l.97-119),
    ppm restricted to [ppm_lo, ppm_hi].
  * A solution must stay inside the E6 region (dilated by one grid step) — otherwise discarded (region-limited).
  * Merit (subset of metric_fit_v2 l.137 with the same weights): cost + 0.5*blocked + 1.5*T2 + 3.0*T3 + 1.0*min(X,0.3)

Quality gate (all must pass for CONFIRMED)
  G1 converged       : least_squares status > 0 and solution not on the position bound
  G2 wall_cost_m     : <= 0.10 m  (floor_id docs/USAGE.md:86 "fits"; > 0.30 m = wrong floor/pose)
  G3 inlier_10cm     : >= 0.50                         (provisional, fixed 2026-10-12 before trials)
  G4 X_cross_m       : <= 0.30 m                        (metric_fit_v2 CROSS_MAX_M)
  G5 T2 free-space   : <= 0.20                          (provisional)
  G6 uniqueness      : best is unique: no other distinct basin (>1 m or >10 deg or ppm x1.1) within merit +0.02
  G7 heading vs E6   : |heading - E6 heading at the pose| <= max(3*sigma_bearing, 15 deg)
  G8 single floor    : exactly one floor passes G1-G7
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

try:
    from . import geometry as G
except ImportError:
    import geometry as G  # type: ignore

import os as _os  # SCI: sci_ws copy of floor_id (env SCI_FLOOR_ID)
FLOOR_ID_PKG = Path(_os.environ.get("SCI_FLOOR_ID", str(Path.home() / "sci_ws" / "external" / "floor_id")))
GRID_STEP = 22
GATE = dict(wall_cost_max=0.10, wall_cost_wrong=0.30, inlier_min=0.50, cross_max=0.30, t2_max=0.20,
            unique_merit_gap=0.02, dup_m=1.0, dup_deg=10.0, dup_scale=1.1, heading_min_deg=15.0)
W_BLOCK, W_FREE, W_OUT, W_CROSS, FAR_M, OUT_M = 0.5, 1.5, 3.0, 1.0, 0.6, 0.5


def _A(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [-s, -c]])


def _sample(field, px, cval):
    return map_coordinates(field, [px[:, 1], px[:, 0]], order=1, mode="constant", cval=cval)


def _crossing_m(wall_tr, track_px, ppm):
    H, W = wall_tr.shape
    if len(track_px) < 2:
        return 0.0
    seg = np.diff(track_px, axis=0)
    n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int), 1)
    pts = np.vstack([track_px[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))] + [track_px[-1:]])
    x, y = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    bad = ~inb
    bad[inb] = wall_tr[y[inb], x[inb]]
    return float(bad.sum() / ppm)


def refine_in_region(F, ok_px, wall_tr, region, cloud, traj, phis_rad, ppm_lo, ppm_hi, out_d=None,
                     grid_step=GRID_STEP, max_seeds=150, seed=0):
    """Pure core. F: px distance field; ok_px: traversable mask; region: E6 mask (same pixel frame);
    cloud/traj: anchor-relative odom-frame points (m, already rotated by yaw+beta, mirror applied).
    Returns (candidates sorted by merit, diagnostics)."""
    H, W = F.shape
    CV = float(max(H, W))
    ys, xs = np.nonzero(region[::grid_step, ::grid_step])
    seeds = np.c_[xs * grid_step, ys * grid_step].astype(float)
    if len(seeds) == 0:
        rys, rxs = np.nonzero(region)
        if len(rxs):
            seeds = np.array([[rxs.mean(), rys.mean()]])
    if len(seeds) > max_seeds:
        seeds = seeds[np.random.RandomState(seed).choice(len(seeds), max_seeds, replace=False)]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * grid_step + 1, 2 * grid_step + 1))
    region_tol = cv2.dilate(region.astype(np.uint8), k).astype(bool)
    sample = cloud[np.linspace(0, len(cloud) - 1, min(1000, len(cloud))).astype(int)]
    ppm_seeds = sorted({ppm_lo * (ppm_hi / ppm_lo) ** f for f in (0.25, 0.75)})

    def project(x, pts):
        return pts @ _A(x[2]).T * np.exp(x[3]) + x[:2]

    def dist(x, pts):
        return _sample(F, project(x, pts), CV) / np.exp(x[3])

    def residual(x):
        return np.minimum(dist(x, sample), 5.0)

    shift = 1.5 * grid_step
    cands, n_ls, n_out = [], 0, 0
    for s in seeds:
        for phi in phis_rad:
            for p0 in ppm_seeds:
                x0 = np.r_[s, phi, math.log(p0)]
                lo = np.r_[s - shift, phi - math.radians(8), math.log(ppm_lo)]
                hi = np.r_[s + shift, phi + math.radians(8), math.log(ppm_hi)]
                r = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.2,
                                  diff_step=[1e-3, 1e-3, 1e-4, 1e-4], max_nfev=60)
                n_ls += 1
                x = r.x
                px, py = int(round(x[0])), int(round(x[1]))
                if not (0 <= px < W and 0 <= py < H and region_tol[py, px] and ok_px[py, px]):
                    n_out += 1
                    continue
                ppm = float(np.exp(x[3]))
                err = dist(x, cloud)
                cost = float(np.mean(np.sort(err)[: max(1, int(0.7 * len(err)))]))
                tpx = project(x, traj)
                ix, iy = np.rint(tpx).astype(int).T
                inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                trav = np.zeros(len(ix), bool)
                trav[inside] = ok_px[iy[inside], ix[inside]]
                blocked = float(1 - trav.mean()) if len(trav) else 0.0
                T2 = float(np.mean(err > FAR_M))
                T3 = float(np.mean(_sample(out_d, project(x, cloud), CV) / ppm > OUT_M)) if out_d is not None else 0.0
                X = _crossing_m(wall_tr, tpx, ppm)
                merit = cost + W_BLOCK * blocked + W_FREE * T2 + W_OUT * T3 + W_CROSS * min(X, GATE["cross_max"])
                at_bound = bool(np.any(np.abs(x[:2] - s) > shift - 0.5))
                cands.append(dict(x=x.tolist(), seed=s.tolist(), ppm=ppm, wall_cost_m=cost, inlier_10cm=float(np.mean(err < 0.1)),
                                  median_m=float(np.median(err)), T2=T2, T3=T3, X_cross_m=X, blocked=blocked, merit=merit,
                                  ls_status=int(r.status), ls_nfev=int(r.nfev), at_bound=at_bound))
    cands.sort(key=lambda c: c["merit"])
    return cands, dict(n_seeds=int(len(seeds)), n_least_squares=n_ls, n_left_region=n_out, n_candidates=len(cands))


def distinct(a, b):
    xa, xb = np.asarray(a["x"]), np.asarray(b["x"])
    d_m = math.hypot(*(xa[:2] - xb[:2])) / math.exp(0.5 * (xa[3] + xb[3]))
    d_deg = abs((math.degrees(xa[2] - xb[2]) + 180) % 360 - 180)
    return d_m >= GATE["dup_m"] or d_deg >= GATE["dup_deg"] or abs(xa[3] - xb[3]) >= math.log(GATE["dup_scale"])


def gate(cands, heading_fn, e6_heading_deg_at, sigma_bearing_rad):
    """Apply G1-G7 to the best candidate. heading_fn(x)-> robot map heading (rad).
    e6_heading_deg_at(px,py) -> E6 heading (deg) or None."""
    if not cands:
        return dict(passed=False, reasons=["NO_CANDIDATE_IN_REGION"])
    b = cands[0]
    rival = next((c for c in cands[1:] if distinct(b, c)), None)
    hdg = math.degrees(heading_fn(np.asarray(b["x"])))
    e6h = e6_heading_deg_at(b["x"][0], b["x"][1])
    dh = None if e6h is None else float((hdg - e6h + 180) % 360 - 180)
    htol = max(3 * math.degrees(sigma_bearing_rad), GATE["heading_min_deg"])
    g = dict(
        G1_converged=bool(b["ls_status"] > 0 and not b["at_bound"]),
        G2_wall_cost=bool(b["wall_cost_m"] <= GATE["wall_cost_max"]),
        G3_inliers=bool(b["inlier_10cm"] >= GATE["inlier_min"]),
        G4_no_wall_crossing=bool(b["X_cross_m"] <= GATE["cross_max"]),
        G5_free_space=bool(b["T2"] <= GATE["t2_max"]),
        G6_unique=bool(rival is None or rival["merit"] - b["merit"] >= GATE["unique_merit_gap"]),
        G7_heading_vs_e6=bool(dh is not None and abs(dh) <= htol),
    )
    reasons = [k for k, v in g.items() if not v]
    if b["wall_cost_m"] > GATE["wall_cost_wrong"]:
        reasons.append("WALL_COST_INDICATES_WRONG_FLOOR_OR_POSE")
    return dict(passed=not reasons, checks=g, reasons=reasons, heading_deg=hdg % 360, e6_heading_deg=e6h,
                heading_delta_deg=dh, heading_tol_deg=htol,
                rival=None if rival is None else dict(x=rival["x"], merit=rival["merit"], wall_cost_m=rival["wall_cost_m"]))


# ---------------------------------------------------------------------------------------------- canonical adapter
def _load_floor_id():
    spec = importlib.util.spec_from_file_location("floor_id_entry", FLOOR_ID_PKG / "floor_id.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_window(building: str, floor_num: int, data_dir: Path, t_rel: float, W: float | None = None,
                 dt: float | None = None):
    """Build F/ok/wall_tr/out_d + window cloud/trajectory with the canonical floor_id localiser (read-only)."""
    fid = _load_floor_id()
    cfg = fid.load_cfg()
    fid.setup_env(cfg, building)
    import rt_localizer as RT   # noqa: E402  (path inserted by setup_env)
    import run_floor_id as R    # noqa: E402
    import metric_fit_v2 as V   # noqa: E402
    W = float(W or cfg["consts"]["window_s"])
    dt = float(dt or cfg["consts"]["dt_s"])
    Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=floor_num, ppm=R.MID,
                       data_dir=str(data_dir), bag=None)
    L = Z.L
    t_now = L.T[0] + t_rel
    start = max(L.T[0], t_now - W)
    Z.fit_beta(start, t_now)
    modes, _ = Z.bootstrap(start, boot_s=t_now - start)
    buf = Z.window_scans(t_now, W, dt)
    anchor = np.asarray(buf[-1]["xy"])
    cloud, traj = [], []
    for fr in buf:
        pts, yaw = fr["pts"], fr["yaw"]
        if len(pts) > 400:
            pts = pts[np.random.RandomState(int(fr["sample_seed_t"]) % 97).choice(len(pts), 400, replace=False)]
        delta = np.asarray(fr["xy"]) - anchor
        cloud.append(delta + (pts * [1, Z.MIRROR]) @ RT._R(yaw + Z.BETA).T)
        traj.append(delta)
    M = V._maps(Z, floor_num)
    yaw_anchor = float(buf[-1]["yaw"])
    return dict(F=M["F"], ok_px=np.asarray(Z.ok_px, bool), wall_tr=M["wall_tr"], out_d=M["out_d"],
                cloud=np.vstack(cloud), traj=np.asarray(traj), phis_rad=[math.radians(p) for p in modes],
                beta=float(Z.BETA), mirror=int(Z.MIRROR), yaw_anchor=yaw_anchor, t_now=float(t_now),
                t_rel=float(t_rel), n_frames=len(buf), map_shape=list(M["F"].shape))


CALIB_MSGS = ("Calibration needs at least two", "Not enough overlapping scans", "No geometrically consistent scan pairs")


def _is_calibration_unavailable(e: Exception) -> bool:
    """R1: sensor_frame.estimate_beta 의 '보정 불가'(이동쌍 없음 등). 새 판은 reason_code, 옛 floor_id 사본은 메시지로 구분."""
    return getattr(e, "reason_code", None) == "CALIBRATION_UNAVAILABLE" or any(m in str(e) for m in CALIB_MSGS)


def run_e7(e6: dict, building: str, data_dir: Path, t_rel: float, out_dir: Path, ppm_interval=None,
           live_preflight: dict | None = None) -> dict:
    """live_preflight (R1): when E7 is considered for LIVE data, pass map_contract.e7_live_preflight(...). Anything but
    ALLOWED (NOT_WIRED / MAP_CONTRACT_UNPROVEN / CALIBRATION_UNAVAILABLE) returns that status without computing."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    if live_preflight is not None and live_preflight.get("status") != "ALLOWED":
        res = dict(schema="e7_result/v1", status=live_preflight["status"], blockers=live_preflight.get("blockers"),
                   per_floor=[], final=None, G8_single_floor=False, sec=round(time.time() - t0, 2), gt_used=False,
                   e6_packet_id=e6.get("packet_id"), note="R1 preflight refused; E7 not computed")
        (out_dir / "e7_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
        return res
    sb = math.radians(e6["observation"]["bearing_sigma_deg"])
    ppm_lo, ppm_hi = ppm_interval or (e6["params"]["ppm_lo"], e6["params"]["ppm_hi"])
    per_floor = []
    for h in e6["floor_hypotheses"]:
        fl = h["floor"]
        eq = h["provenance"]["frame_equivalence"]["status"]
        rec = dict(floor=fl, frame_equivalence=eq)
        if eq != "IDENTICAL_FILE":
            rec.update(status="REFUSED_FRAME_UNPROVEN")
            per_floor.append(rec)
            continue
        region = cv2.imread(h["region_mask_png"], 0) > 0
        hp = np.load(h["heading_npy"])
        num = int(re.sub(r"\D", "", fl))
        t1 = time.time()
        try:
            w = build_window(building, num, data_dir, t_rel)
        except ValueError as e:
            if not _is_calibration_unavailable(e):
                raise
            rec.update(status="CALIBRATION_UNAVAILABLE", why=str(e)[:200])   # R1: β 없음 — 값을 채우지 않고 이 층은 평가 안 함
            per_floor.append(rec)
            continue
        if list(region.shape) != w["map_shape"]:
            rec.update(status="REFUSED_SHAPE_MISMATCH", shapes=[list(region.shape), w["map_shape"]])
            per_floor.append(rec)
            continue
        cands, diag = refine_in_region(w["F"], w["ok_px"], w["wall_tr"], region, w["cloud"], w["traj"],
                                       w["phis_rad"], ppm_lo, ppm_hi, out_d=w["out_d"])

        def heading(x, w=w):
            return float(G.wrap(x[2] + w["yaw_anchor"] + w["beta"]))   # pose_tools.map_pose_px convention

        def e6h(px, py, hp=hp):
            x, y = int(round(px)), int(round(py))
            if 0 <= y < hp.shape[0] and 0 <= x < hp.shape[1] and np.isfinite(hp[y, x]):
                return float(hp[y, x])
            return None

        g = gate(cands, heading, e6h, sb)
        best = cands[0] if cands else None
        rec.update(status="EVALUATED", diag=diag, gate=g, sec=round(time.time() - t1, 2),
                   window=dict(t_rel=w["t_rel"], t_now=w["t_now"], n_frames=w["n_frames"], beta=w["beta"],
                               mirror=w["mirror"], yaw_anchor=w["yaw_anchor"], modes_deg=[math.degrees(p) for p in w["phis_rad"]]),
                   best=None if best is None else dict(x_px=best["x"][0], y_px=best["x"][1], phi_deg=math.degrees(best["x"][2]),
                                                       ppm=best["ppm"], wall_cost_m=best["wall_cost_m"],
                                                       inlier_10cm=best["inlier_10cm"], median_m=best["median_m"],
                                                       T2=best["T2"], T3=best["T3"], X_cross_m=best["X_cross_m"],
                                                       blocked=best["blocked"], merit=best["merit"]),
                   top5=[dict(x=c["x"], merit=c["merit"], wall_cost_m=c["wall_cost_m"]) for c in cands[:5]])
        per_floor.append(rec)
    passing = [r for r in per_floor if r.get("gate", {}).get("passed")]
    final = None
    if len(passing) == 1:
        r = passing[0]
        b = r["best"]
        final = dict(floor=r["floor"], x_px=b["x_px"], y_px=b["y_px"], yaw_map_deg=r["gate"]["heading_deg"],
                     x_m_est=b["x_px"] / b["ppm"], y_m_est=None, ppm_est=b["ppm"],
                     note="metric coordinates use the fitted ppm (kci has no metric base_map.yaml); pixel pose is primary")
    status = "CONFIRMED" if final else ("AMBIGUOUS_MULTI_FLOOR" if len(passing) > 1 else "REJECTED")
    res = dict(schema="e7_result/v1", e6_packet_id=e6.get("packet_id"), building=building, data_dir=str(data_dir),
               t_rel=t_rel, gate_def=GATE, per_floor=per_floor, G8_single_floor=len(passing) == 1, final=final,
               status=status, sec=round(time.time() - t0, 2), gt_used=False)
    (out_dir / "e7_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


def main():
    ap = argparse.ArgumentParser(description="E7 region-limited re-matching + quality gate")
    ap.add_argument("--e6", required=True, help="e6_result.json")
    ap.add_argument("--building", default="kci")
    ap.add_argument("--data", required=True, help="driving folder (odom_link.csv, samples/, far_points.npz)")
    ap.add_argument("--t-rel", type=float, required=True, help="window end, seconds after data start")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e6 = json.loads(Path(a.e6).read_text(encoding="utf-8"))
    res = run_e7(e6, a.building, Path(a.data), a.t_rel, Path(a.out))
    print(json.dumps({k: res[k] for k in ("status", "final", "sec")}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
