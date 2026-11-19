#!/usr/bin/env python3
"""Phase-2 policy "P2": verify while the sign is in the RealSense view, STOP when the predicted stop pose is readable.

Layers (every candidate gets a per-layer record):
  [1] candidate generation  (legacy proposals/regions, unchanged)
  [2] temporal tracking     (legacy world-anchored wall spot + A1 persistence)
  [3] SigLIP verification   (legacy relative threshold + sim floor + size/height band)
  [4] hard-negative         (A2 gates: poster/board/directory/light tile & context, saturation, specular, depth, horizontal)
  -> a spot passing [1]-[4] with range <= verify_dmax becomes a VERIFIED world track (position + wall normal, odom frame)
  [5] geometric validation  of the PREDICTED stop pose: robot keeps moving for reaction_s (+ margin_m) after the request
  [6] readability prediction at that pose: obliquity, range, estimated text height, pan/tilt reachability, tilt range
  [7] information value     E4 (CONFIDENT -> suppress; unavailable -> E4_UNAVAILABLE, independent mode); already observed
  [8] decision              best readable verified track -> STOP_REQUEST; a track that is passed without ever becoming
                            readable is closed as MISSED (logged, counts against useful-observation recall)
The reaction time / braking margin are configuration ASSUMPTIONS until measured on the robot.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

import policy as P

P2_DEFAULT = {
    "verify": {"dmax_m": 5.0, "obliquity_max_deg": 80.0, "edge_px": 6, "min_obs": 1},
    "timing": {"tick_hz": 10.0, "reaction_s": 1.5, "margin_m": 0.3, "track_timeout_s": 25.0, "min_speed_for_prediction": 0.05},
    "readability": {"obliquity_max_deg": 55.0, "range_max_m": 4.0, "range_min_m": 0.8, "text_min_px": 15.0,
                    "text_frac": 0.35, "imx_fpx_1920": 4400.0, "pan_margin_deg": 3.0, "tilt_margin_deg": 3.0,
                    "decision_margin_deg": 5.0},
    "info": {"use_e4": True},
    "dedup": {"same_m": 1.0, "same_sec": 60.0, "merge_m": 0.8},
}


def load_p2(path=None):
    cfg = {k: dict(v) for k, v in P2_DEFAULT.items()}
    if path and Path(path).exists():
        txt = Path(path).read_text()
        user = yaml.safe_load(txt) if str(path).endswith((".yaml", ".yml")) else __import__("json").loads(txt)
        for k, v in (user or {}).items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def cam_basis(up_c):
    """Camera-frame unit vectors of body forward (horizontal), left, up from the measured floor 'up' (camera optical frame)."""
    u = np.asarray(up_c, float) if up_c is not None else np.array([0.0, -math.cos(math.radians(11)), math.sin(math.radians(11))])
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    f = z - (z @ u) * u
    f /= np.linalg.norm(f)
    l = np.cross(u, f)                                # optical x right -> left = up x forward
    return f, l, u


class P2Policy:
    name = "P2"

    def __init__(self, cfg_a, cfg_p2, geom):
        self.c = cfg_p2
        self.geom = geom
        self.gates = P.StagePolicy("P2gates", cfg_a, geom, a1=True, a2=True, a3=False, a4=False)
        self.tracks = {}                              # spot -> track dict
        self.fired, self.events, self.missed = [], [], []

    # ------------------------------------------------------------------ helpers
    def world_normal(self, n_c, basis, yaw):
        f, l, u = basis
        nf, nl = float(np.dot(n_c, f)), float(np.dot(n_c, l))
        h = math.hypot(nf, nl)
        if h < 0.3:                                   # not a vertical surface
            return None
        nf, nl = nf / h, nl / h
        return np.array([math.cos(yaw) * nf - math.sin(yaw) * nl, math.sin(yaw) * nf + math.cos(yaw) * nl])

    def predict_pose(self, pose, speed):
        x, y, yaw = pose
        s = (speed or 0.0)
        adv = s * self.c["timing"]["reaction_s"] + (self.c["timing"]["margin_m"] if s >= self.c["timing"]["min_speed_for_prediction"] else 0.0)
        return (x + adv * math.cos(yaw), y + adv * math.sin(yaw), yaw), adv

    def view_from(self, tr, pose, basis):
        """Viewing geometry of track tr from robot pose: range, horizontal obliquity, bearing, servo angles, est text px."""
        x, y, yaw = pose
        dx, dy = tr["xy"][0] - x, tr["xy"][1] - y
        fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        rng = math.hypot(fwd, left)
        bearing = math.degrees(math.atan2(left, fwd))
        obl = None
        if tr["n_xy"] is not None and rng > 1e-3:
            ray = np.array([dx, dy]) / max(rng, 1e-6)
            obl = math.degrees(math.acos(min(1.0, abs(float(ray @ tr["n_xy"])))))
        f, l, u = basis
        p_cam = fwd * f + left * l + tr["up"] * u      # camera optical frame (x right, y down, z forward)
        pan = tilt = None
        reach, why = True, None
        try:
            pan, tilt = self.geom.aim.optical_to_servo(p_cam)
        except ValueError as e:
            reach, why = False, f"aim:{e}"
        rc = self.c["readability"]
        if pan is not None:
            if not (self.geom.pan_lim[0] + rc["pan_margin_deg"] <= pan <= self.geom.pan_lim[1] - rc["pan_margin_deg"]):
                reach, why = False, "pan_out_of_range"
            elif not (self.geom.tilt_lim[0] + rc["tilt_margin_deg"] <= tilt <= self.geom.tilt_lim[1] - rc["tilt_margin_deg"]):
                reach, why = False, "tilt_out_of_range"
        text = tr["h_m"] * rc["text_frac"] * rc["imx_fpx_1920"] / max(rng + 0.21, 0.3) if tr["h_m"] else None
        return dict(range=rng, fwd=fwd, left=left, bearing=bearing, obliquity=obl, pan=pan, tilt=tilt, reachable=reach,
                    reach_why=why, text_px=text, p_cam=[float(c) for c in p_cam])

    def readable(self, v, margin=0.0):
        """Readability envelope. The stop decision uses margin = decision_margin_deg (prediction error of the stop pose);
        evaluation of the achieved stop pose uses margin 0."""
        rc, why = self.c["readability"], []
        if v["obliquity"] is None:
            why.append("normal_unknown")
        elif v["obliquity"] > rc["obliquity_max_deg"] - margin:
            why.append(f"obliquity:{v['obliquity']:.0f}")
        if v["range"] > rc["range_max_m"]:
            why.append(f"too_far:{v['range']:.1f}")
        if v["range"] < rc["range_min_m"]:
            why.append(f"too_close:{v['range']:.1f}")
        if not v["reachable"]:
            why.append(f"pantilt:{v['reach_why']}")
        if v["text_px"] is not None and v["text_px"] < rc["text_min_px"]:
            why.append(f"text_small:{v['text_px']:.0f}")
        return not why, why

    def already(self, xy, t):
        d = self.c["dedup"]
        return any(t - tf <= d["same_sec"] and math.hypot(p[0] - xy[0], p[1] - xy[1]) <= d["same_m"] for tf, p in self.fired)

    # ------------------------------------------------------------------ main
    def step(self, t, pose, speed, regions, feats, thr, e4, busy=False, decide=True):
        """Returns (per-region layer records, stop dict or None, track summaries).
        E5RT2: decide=False (asynchronous scene path) updates the world-anchored tracks only; the STOP decision is left
        to tick() with the LATEST odometry (a finished scene carries the pose of its own, older, time).
        A region whose score is not a fresh SigLIP result (score_status DEFERRED_BUDGET / CACHED) is not new evidence:
        it is neither a negative nor a confirmation (state DEFERRED, no persistence vote, no track update)."""
        recs = []
        basis = cam_basis(next((f.get("up") for f in feats if f.get("up") is not None), None))
        yaw = pose[2]
        for r, f in zip(regions, feats):
            ss = r.get("score_status") or "FRESH"
            if ss != "FRESH":
                recs.append(dict(layers=dict(L2_tracking=[], L3_siglip=[ss], L4_hard_negative=[], L5_verify_geometry=[]),
                                 state="DEFERRED", spot=r["spot"], score_status=ss))
                continue
            lay = {}
            leg = self.gates.legacy_gate(r, thr)
            if r["spot"] is not None:
                self.gates.hist[r["spot"]].append(r["sim"] >= self.gates.cfg["legacy"]["sim_floor"])
            lay["L2_tracking"] = self.gates.persist_gate(r)
            lay["L3_siglip"] = [w for w in leg if w.startswith("legacy")]
            lay["L4_hard_negative"] = [w for w in leg if not w.startswith("legacy")] + self.gates.hardneg_gate(r, f)
            z = f.get("z_tile") or r["z"]
            geo = []
            if z is None or z > self.c["verify"]["dmax_m"]:
                geo.append("verify_range:%s" % (None if z is None else round(z, 1)))
            if f["edge"] < self.c["verify"]["edge_px"]:
                geo.append("cut_off")
            vo = self.c["verify"].get("obliquity_max_deg")      # in-view sanity (A3 gate); None = off
            ob = f.get("obliquity_pred") or (f["plane"].get("obliquity_deg") if f["plane"].get("ok") else None)
            if vo is not None and ob is not None and ob > vo:
                geo.append("verify_obliquity:%.0f" % ob)
            lay["L5_verify_geometry"] = geo
            verified = not (lay["L2_tracking"] or lay["L3_siglip"] or lay["L4_hard_negative"] or geo)
            if verified and r["xyz"] is not None and r["spot"] is not None:
                tr = self.tracks.get(r["spot"])
                if tr is None:                        # near-duplicate spot of an existing open track -> same sign
                    for k2, t2 in self.tracks.items():
                        if t2["state"] in ("VERIFIED", "WAIT_READABLE") and "xy" in t2 and \
                                math.hypot(t2["xy"][0] - r["xyz"][0], t2["xy"][1] - r["xyz"][1]) <= self.c["dedup"]["merge_m"]:
                            tr = t2
                            self.tracks[r["spot"]] = t2
                            break
                n_xy = self.world_normal(f["plane"]["normal"], basis, yaw) if f["plane"].get("ok") else None
                tile = r.get("tile") or r["box"]
                h_m = (tile[3] - tile[1]) * z / self.geom.fy if z else None
                if tr is None:
                    tr = self.tracks[r["spot"]] = dict(spot=r["spot"], t_verified=t, xys=[], normals=[], ups=[], h=[], state="VERIFIED",
                                                       sim=r["sim"], box=r["box"], tile=tile, z_first=z, n_verify=0)
                tr["xys"].append(r["xyz"][:2]); tr["ups"].append(r["xyz"][2]); tr["n_verify"] += 1
                if n_xy is not None:
                    tr["normals"].append(n_xy)
                if h_m:
                    tr["h"].append(h_m)
                tr["last_t"], tr["box"], tr["tile"], tr["sim"], tr["z_last"] = t, r["box"], tile, r["sim"], z
                tr["xy"] = np.mean(tr["xys"][-5:], 0)
                tr["up"] = float(np.mean(tr["ups"][-5:]))
                tr["h_m"] = float(np.median(tr["h"])) if tr["h"] else None
                if tr["normals"]:
                    m = np.sum([n if n @ tr["normals"][0] >= 0 else -n for n in tr["normals"][-5:]], 0)
                    tr["n_xy"] = m / max(np.linalg.norm(m), 1e-6)
                else:
                    tr["n_xy"] = None
            state = "VERIFIED" if verified else ("REJECT" if lay["L4_hard_negative"] else ("TRACK" if not lay["L3_siglip"] else "SEARCH"))
            recs.append(dict(layers=lay, state=state, spot=r["spot"]))
        self.last_basis = basis
        if not decide:
            return recs, None, getattr(self, "last_summary", [])
        stop, summ = self.tick(t, pose, speed, e4, busy)
        return recs, stop, summ

    def tick(self, t, pose, speed, e4=None, busy=False, basis=None, defer_why=None):
        """Fast timing layer [5]-[8] over all open verified tracks (in view or not). No image work: can run at odometry
        rate between SigLIP scenes (e5_node calls it for every camera frame).
        R1: `pose` must be the latest valid odom pose at decision time (e5_node.decision_reference), so the frame age
        is accounted once (the prediction does not start from the possibly seconds-old frame pose). `defer_why` (list)
        = geometry inputs unavailable (unsynced clocks, odom reset, stale odom) -> no STOP is issued (DEFER).
        A waiting track whose readable-and-reachable window along the heading is empty is marked
        NO_FEASIBLE_STOP_WINDOW (pan/tilt limits are never widened; the robot is never asked to rotate).
        reaction_s / margin_m stay configuration ASSUMPTIONS (unvalidated; 10-15 sample n=9 is not a model)."""
        basis = basis if basis is not None else getattr(self, "last_basis", cam_basis(None))
        ppose, adv = self.predict_pose(pose, speed)
        cands, summ = [], []
        seen_ids = set()
        for k, tr in list(self.tracks.items()):
            if id(tr) in seen_ids:
                continue
            seen_ids.add(id(tr))
            if tr["state"] not in ("VERIFIED", "WAIT_READABLE"):
                continue
            if tr["n_verify"] < self.c["verify"]["min_obs"]:
                summ.append(dict(spot=k, state="VERIFYING", n=tr["n_verify"]))
                continue
            v_now, v_pred = self.view_from(tr, pose, basis), self.view_from(tr, ppose, basis)
            ok, why = self.readable(v_pred, self.c["readability"].get("decision_margin_deg", 0.0))
            if self.already(tr["xy"], t):
                tr["state"] = "ALREADY_OBSERVED"
            elif v_pred["fwd"] < 0.2 and v_now["fwd"] < 1.0:
                tr["state"], tr["missed_why"] = "MISSED", why or ["passed"]
                self.missed.append(dict(t=t, spot=k, why=tr["missed_why"]))
            elif t - tr["last_t"] > self.c["timing"]["track_timeout_s"]:
                tr["state"] = "EXPIRED"
            elif ok:
                cands.append((k, v_pred, v_now))
            else:
                tr["state"] = "WAIT_READABLE"
                if t - tr.get("feas_t", -1e9) >= 1.0:          # bounded cost: once per second per track
                    from stop_policy import feasible_window
                    tr["feas"], tr["feas_t"] = feasible_window(self, tr, pose, basis), t
                if not tr.get("feas"):
                    why = list(why) + ["NO_FEASIBLE_STOP_WINDOW"]
            summ.append(dict(spot=k, state=tr["state"], why=why, pred=_round(v_pred), now=_round(v_now),
                             box_frame=tr.get("box_frame")))
        stop = None
        if cands:
            if self.c["info"]["use_e4"] and e4 and e4.get("available") and e4.get("level") == "CONFIDENT":
                for s in summ:
                    if s["spot"] in [c[0] for c in cands]:
                        s["state"], s["why"] = "E4_SUPPRESSED", ["e4:pose_confident"]
            elif defer_why:
                for s in summ:
                    if s["spot"] in [c[0] for c in cands]:
                        s["state"], s["why"] = "DEFER_GEOMETRY_UNAVAILABLE", list(defer_why)
            elif not busy:
                k, vp, vn = min(cands, key=lambda c: (c[1]["obliquity"] or 90.0) + 5.0 * c[1]["range"])
                tr = self.tracks[k]
                tr["state"] = "STOP_REQUEST"
                self.fired.append((t, tr["xy"].tolist()))
                stop = dict(t=t, stage="P2", spot=k, box=tr["box"], tile=tr["tile"], z=tr["z_last"], sim=tr["sim"],
                            xyz=[float(tr["xy"][0]), float(tr["xy"][1]), tr["up"]], n_xy=None if tr["n_xy"] is None else tr["n_xy"].tolist(),
                            t_verified=tr["t_verified"], predicted=_round(vp), now=_round(vn), advance_m=round(adv, 2),
                            e4=e4, msim=tr["sim"], nobs=tr["n_verify"], nscene=tr["n_verify"],
                            box_frame=tr.get("box_frame"), reaction_s=self.c["timing"]["reaction_s"],
                            reaction_s_status="UNVALIDATED_ASSUMPTION")
                self.events.append(stop)
        self.last_summary = summ
        return stop, summ


def _round(v):
    return {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items() if k != "p_cam"}


DISPLAY_P2 = {"VERIFIED": "VERIFYING", "TRACK": "CANDIDATE", "SEARCH": "SEARCHING"}


def p2_display(rec):
    """UI display state + flat reason list for a P2 per-region record."""
    reasons = [w for k in ("L2_tracking", "L3_siglip", "L4_hard_negative", "L5_verify_geometry") for w in rec["layers"][k]]
    if rec["state"] == "DEFERRED":                      # E5RT2: typed 'no fresh score' (budget / cache), not a verdict
        disp = "DEFERRED_BUDGET" if rec.get("score_status") == "DEFERRED_BUDGET" else "CACHED_NO_NEW_EVIDENCE"
    elif rec["state"] == "REJECT":
        disp = P.StagePolicy.display("REJECT", rec["layers"]["L4_hard_negative"])
    elif rec["state"] == "VERIFIED":
        disp = "VERIFIED"
    elif rec["layers"]["L5_verify_geometry"] and not rec["layers"]["L3_siglip"]:
        disp = "REJECTED_DISTANCE" if any(w.startswith("verify_range") for w in rec["layers"]["L5_verify_geometry"]) else "REJECTED_ANGLE"
    else:
        disp = DISPLAY_P2.get(rec["state"], "CANDIDATE")
    return dict(state=rec["state"], display=disp, reasons=reasons, layers=rec["layers"])
