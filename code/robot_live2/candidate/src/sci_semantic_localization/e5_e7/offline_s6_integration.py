#!/usr/bin/env python3
"""offline_s6_integration.py — OFFLINE E6->E7 integration on the only real driving data on the robot
(~/jsh_onboard_robot_20261001/testdata/S6 = 소융대 6F, true floor 6 per realtime/README.md:113).

THIS IS NOT AN E5 RESULT. There is no camera image in testdata/S6, so the E5 observation is SYNTHESISED from a
pseudo-reference pose = the canonical floor_id full-floor fit (metric_fit_v2.fit_window_v2) at the same time.
The pseudo-reference is NOT ground truth; it is used (a) to synthesise the observation and (b) for evaluation only.
E6 and E7 never receive the reference pose.

  consistency : plate of a room visible (line of sight, 2..6 m) from the reference pose -> E6 region must contain it,
                E7 (region-limited) should reproduce it and pass/fail the gate honestly
  decoy       : plate of a room >= 20 m away -> E7 must not CONFIRM the reference pose

Usage: python3 scripts/e5_e7/offline_s6_integration.py --t-rel 70 [--out reports/e5_e7_live/runs/...]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e5_e7 import geometry as G      # noqa: E402
from e5_e7 import e6_region as E6    # noqa: E402
from e5_e7 import e7_rematch as E7   # noqa: E402

DATA = Path.home() / "jsh_onboard_robot_20261001" / "testdata" / "S6"
SW_FLOORS = ["S1F", "S6F"]           # semantic maps present for building sw


def reference_fit(t_rel):
    fid = E7._load_floor_id()
    cfg = fid.load_cfg()
    b = fid.setup_env(cfg, "sw")
    import rt_localizer as RT   # noqa: E402
    import run_floor_id as R    # noqa: E402
    import metric_fit_v2 as V   # noqa: E402
    SC = fid.map_scales(b, R, int(b["n_maps"]))
    Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=6, ppm=R.MID, data_dir=str(DATA), bag=None)
    Z.bootstrap(Z.L.T[0], boot_s=15.0)
    t = Z.L.T[0] + t_rel
    t0 = time.time()
    r = V.fit_window_v2(Z, t, SC[6], floor=6, W=float(cfg["consts"]["window_s"]), dt=float(cfg["consts"]["dt_s"]))
    pts, yaw = Z.L._scan(t)
    heading = float(G.wrap(math.radians(r["phi"]) + yaw + r["beta"]))
    return dict(pos=[float(v) for v in r["pos"]], phi_deg=r["phi"], ppm=r["ppm"], beta=r["beta"], yaw=float(yaw),
                heading_rad=heading, wall_cost_m=r["wall_cost_m"], sec=round(time.time() - t0, 1),
                scales=SC[6], note="PSEUDO-REFERENCE (canonical full-floor fit), not GT")


def pick_plate(fm, ref, near=True):
    p = np.array(ref["pos"])
    best = None
    for sp in fm.sem.get("spaces", []):
        lab = E6.norm_label(str(sp.get("room_number") or ""))
        if not lab or not sp.get("polygon_px"):
            continue
        sup = E6.sign_support(fm, sp, E6.DEFAULTS["support_band_px"])
        ys, xs = np.nonzero(sup)
        if not len(xs):
            continue
        d = np.hypot(xs - p[0], ys - p[1]) / ref["ppm"]
        for i in np.argsort(d)[:50]:
            s = np.array([xs[i], ys[i]], float)
            if near and not (2.0 <= d[i] <= 6.0):
                continue
            if not near and d[i] < 20.0:
                continue
            n = int(np.hypot(*(s - p))) + 1
            ray = np.rint(p + np.linspace(0, 1, n, endpoint=False)[:, None] * (s - p)).astype(int)[:-4]
            if near and fm.wall[ray[:, 1], ray[:, 0]].any():
                continue
            cand = (float(d[i]), lab, s)
            if best is None or (near and cand[0] < best[0]) or (not near and cand[0] > best[0]):
                best = cand
            break
    return best


def packet(label, beta, r, kind):
    return dict(schema="e5_observation_packet/v1", packet_id=f"SYNTHETIC_S6_{kind}",
                synthetic=dict(kind=kind, source="pseudo-reference pose (canonical full-floor fit); NOT an E5 capture"),
                semantic=dict(label_alternatives=[dict(label=label, p=0.9)], final_label=label, confidence=0.9),
                observation=dict(bearing_body_rad=beta, range_m=r, bearing_sigma_rad=math.radians(4),
                                 range_sigma_m=G.range_sigma(r)), gt_used=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-rel", type=float, default=70.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"reports/e5_e7_live/runs/offline_S6_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    log = dict(t_rel=a.t_rel, data=str(DATA), started=time.time())
    ref = reference_fit(a.t_rel)
    log["reference"] = ref
    fm = E6.FloorMap("S6F")
    ppm_lo, ppm_hi = min(ref["scales"]) , max(ref["scales"])
    params = dict(ppm_lo=ppm_lo, ppm_hi=ppm_hi)
    for kind, near in (("consistency", True), ("decoy", False)):
        pl = pick_plate(fm, ref, near)
        if pl is None:
            log[kind] = dict(status="NO_SUITABLE_PLATE")
            continue
        d_m, lab, s = pl
        beta = float(G.wrap(G.map_px_heading(ref["pos"], s) - ref["heading_rad"]))
        pk = packet(lab, beta, d_m, kind)
        kdir = out / kind
        kdir.mkdir(exist_ok=True)
        (kdir / "E5_OBSERVATION_PACKET.synthetic.json").write_text(json.dumps(pk, indent=1))
        e6 = E6.run_e6(pk, SW_FLOORS, kdir / "e6", building="sw", params=params)
        rec = dict(plate=dict(label=lab, s_px=s.tolist(), dist_m=d_m), e6_status=e6["status"], e6_sec=e6["sec"])
        if e6["floor_hypotheses"]:
            h = e6["floor_hypotheses"][0]
            mask = cv2.imread(h["region_mask_png"], 0) > 0
            x, y = int(round(ref["pos"][0])), int(round(ref["pos"][1]))
            rec.update(e6_floor=h["floor"], e6_region_contains_reference=bool(mask[y, x]),
                       e6_region_px=h["region_px"], e6_n_components=len(h["components"]))
            e7 = E7.run_e7(e6, "sw", DATA, a.t_rel, kdir / "e7", ppm_interval=(ppm_lo, ppm_hi))
            rec.update(e7_status=e7["status"], e7_sec=e7["sec"])
            fl = next((f for f in e7["per_floor"] if f.get("best")), None)
            if fl:
                b = fl["best"]
                rec.update(e7_best=b, e7_gate=fl["gate"],
                           e7_vs_reference_px=math.hypot(b["x_px"] - ref["pos"][0], b["y_px"] - ref["pos"][1]),
                           e7_vs_reference_m=math.hypot(b["x_px"] - ref["pos"][0], b["y_px"] - ref["pos"][1]) / ref["ppm"],
                           e7_heading_vs_reference_deg=float((fl["gate"]["heading_deg"] - math.degrees(ref["heading_rad"]) + 180) % 360 - 180))
        log[kind] = rec
    log["finished"] = time.time()
    (out / "offline_s6_summary.json").write_text(json.dumps(log, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: log.get(k) for k in ("consistency", "decoy")}, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
