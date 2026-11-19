#!/usr/bin/env python3
"""Offline comparison A0 / A3 / P2 on recorded runs with the SAME readability judgement for every stage.

The recorded robot did not stop; the pose recorded `reaction_s` after a stop request is used as the operator-stop pose
(robot keeps walking during the reaction time; braking distance not modelled -> ESTIMATE). Readability at that pose uses
the per-spot world geometry built from all logged observations of the spot (mean xy, wall normal from depth, sign height)
and the P2 readability envelope (config/e5_policy_p2.yaml). Precision / recall use the spot labels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WS = Path.home() / "sci_ws"
sys.path.insert(0, str(WS / "tools"))
sys.path.insert(0, str(WS / "src" / "sci_visual_observation"))
import policy as P  # noqa: E402
import rescore as RS  # noqa: E402
import timing as T  # noqa: E402
from features import Geometry  # noqa: E402
from metrics_v2 import clusters, path_length, spot_xy  # noqa: E402


def spot_geometry(sc, geom, p2):
    acc = defaultdict(lambda: dict(xy=[], up=[], n=[], h=[]))
    for s in sc:
        yaw = s["pose"][2]
        for r in s["regions"]:
            if r["spot"] is None or r["xyz"] is None:
                continue
            a = acc[r["spot"]]
            a["xy"].append(r["xyz"][:2]); a["up"].append(r["xyz"][2])
            f = r["feat"]
            basis = T.cam_basis(f.get("up"))
            if f["plane"].get("ok"):
                n = p2.world_normal(f["plane"]["normal"], basis, yaw)
                if n is not None:
                    a["n"].append(n)
            z = f.get("z_tile") or r["z"]
            tile = r.get("tile") or r["box"]
            if z:
                a["h"].append((tile[3] - tile[1]) * z / geom.fy)
    out = {}
    for k, a in acc.items():
        n = None
        if a["n"]:
            m = np.sum([v if v @ a["n"][0] >= 0 else -v for v in a["n"]], 0)
            n = m / max(np.linalg.norm(m), 1e-6)
        out[k] = dict(xy=np.mean(a["xy"], 0), up=float(np.mean(a["up"])), n_xy=n, h_m=float(np.median(a["h"])) if a["h"] else None)
    return out


def pose_at(sc, t):
    ts = np.array([s["t"] for s in sc])
    if t > ts[-1]:
        return None
    P_ = np.array([s["pose"] for s in sc])
    yaw = np.unwrap(P_[:, 2])
    return (float(np.interp(t, ts, P_[:, 0])), float(np.interp(t, ts, P_[:, 1])), float(np.interp(t, ts, yaw)))


def tf_pose_fn(ds):
    """Recorded odom->base_link pose(stamp) and speed(stamp) from the extracted bag (read-only)."""
    sys.path.insert(0, str(WS / "src" / "sci_sensor_bridge"))
    from sources import ReplaySource
    src = ReplaySource(ds, realtime=False)

    def pose(st):
        return src.pose(st)[0]

    def speed(st, win=0.5):
        a, b = pose(st - win), pose(st)
        return None if a is None or b is None else math.hypot(b[0] - a[0], b[1] - a[1]) / win
    return pose, speed


def replay_p2(sc, cfg_a, cfg_p2, geom, ds=None, tick_hz=0.0):
    """Scene updates at the logged scene times; optional timing ticks at tick_hz between scenes from recorded poses."""
    pol = T.P2Policy(cfg_a, cfg_p2, geom)
    pf = tf_pose_fn(ds) if (tick_hz and ds is not None) else None
    for i, s in enumerate(sc):
        regs = [{k: v for k, v in r.items() if k not in ("feat", "dec")} for r in s["regions"]]
        feats = [dict(r["feat"]) for r in s["regions"]]
        pol.step(s["t"], tuple(s["pose"]), s.get("speed"), regs, feats, s["thr"], s.get("e4"))
        if pf and i + 1 < len(sc):
            origin = s["stamp"] - s["t"]
            for tt in np.arange(s["t"] + 1.0 / tick_hz, sc[i + 1]["t"] - 1e-6, 1.0 / tick_hz):
                ps = pf[0](origin + tt)
                if ps is not None:
                    pol.tick(float(tt), ps, pf[1](origin + tt), s.get("e4"))
    return pol


TICK_HZ = [0.0]
PF = None


def evaluate(run, labels, cfg_a, cfg_p2, cluster_m=1.5):
    geom = Geometry()
    sc, st, ds = RS.load(run)
    name = ds.name
    by_t = {s["t"]: s for s in sc}
    pol = replay_p2(sc, cfg_a, cfg_p2, geom, ds, TICK_HZ[0])
    stages = RS.replay_policy(sc, cfg_a, geom)
    a0 = []
    for e in st:
        if e.get("stage", "A0") == "A0":
            sp = next((r["spot"] for r in by_t[e["t"]]["regions"] if np.allclose(r["box"], e["picks"][0], atol=0.51)), None)
            a0.append(dict(t=e["t"], spot=sp))
    allst = {"A0": a0, "A3": stages["A3"], "P2": pol.events}
    G = spot_geometry(sc, geom, pol)
    global PF
    PF = tf_pose_fn(ds)
    lab = {int(k.split("spot")[1]): v for k, v in labels.items() if k.startswith(name + "|")}
    xy = spot_xy(sc)
    pos = [k for k, v in lab.items() if v.startswith("POS") and k in xy]
    cl = clusters(pos, xy, cluster_m)
    nphys = len(set(cl.values()))
    L = path_length(sc)
    react = cfg_p2["timing"]["reaction_s"]
    res = dict(dataset=name, path_m=round(L, 1), physical_pos_signs=nphys, reaction_s=react, stages={})
    for stg, S in allst.items():
        tp = fp = ex = 0
        valid_tp, reached, reached_valid, detail = 0, set(), set(), []
        for s in S:
            l = lab.get(s["spot"], "UNLABELLED")
            g = G.get(s["spot"])
            ps = (PF[0](sc[0]["stamp"] - sc[0]["t"] + s["t"] + react) if PF else None) or pose_at(sc, s["t"] + react)
            v = pol.view_from(dict(g, xy=g["xy"]), ps, T.cam_basis(None)) if (g and ps) else None
            if v is not None:
                # tilt needs the measured camera basis; use the stop scene's 'up' if logged
                ts_ = min(by_t, key=lambda tt: abs(tt - s["t"]))
                ups = [r["feat"].get("up") for r in by_t[ts_]["regions"] if r["feat"].get("up") is not None]
                if ups:
                    v = pol.view_from(dict(g, xy=g["xy"]), ps, T.cam_basis(ups[0]))
            ok, why = pol.readable(v) if v else (False, ["no_pose_after_stop"])
            if l.startswith("POS"):
                tp += 1
                c = cl.get(s["spot"])
                if c is not None:
                    reached.add(c)
                if ok:
                    valid_tp += 1
                    if c is not None:
                        reached_valid.add(c)
            elif l.startswith("NEG"):
                fp += 1
            else:
                ex += 1
            detail.append(dict(t=s["t"], spot=s["spot"], label=l, readable_est=ok, why=why,
                               at_stop=None if v is None else {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()}))
        n = tp + fp
        res["stages"][stg] = dict(stops=len(S), TP=tp, FP=fp, excluded=ex, stop_precision=round(tp / n, 3) if n else None,
                                  false_stops_per_100m=round(100 * fp / L, 2), useful_observation_recall=f"{len(reached)}/{nphys}",
                                  readable_TP_est=valid_tp, invalid_stop_position_est=f"{tp - valid_tp}/{tp}",
                                  readable_stop_rate_est=round(valid_tp / n, 3) if n else None,
                                  readable_useful_recall_est=f"{len(reached_valid)}/{nphys}", detail=detail)
    res["P2_missed"] = [dict(m, label=lab.get(m["spot"], "UNLABELLED")) for m in pol.missed]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--policy", default=str(WS / "config" / "e5_policy.json"))
    ap.add_argument("--p2", default=str(WS / "config" / "e5_policy_p2.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tick-hz", type=float, default=0.0, help="P2 timing ticks between scenes (recorded poses); 0 = scene rate only")
    a = ap.parse_args()
    labels = {}
    for f in a.labels:
        labels.update(json.loads(Path(f).read_text()))
    cfg_a, cfg_p2 = P.load_cfg(a.policy), T.load_p2(a.p2)
    TICK_HZ[0] = a.tick_hz
    for kv in a.set:
        k, v = kv.split("=")
        sec, key = k.split(".")
        cfg_p2[sec][key] = json.loads(v)
    out = [evaluate(r, labels, cfg_a, cfg_p2) for r in a.run]
    if a.out:
        Path(a.out).write_text(json.dumps(dict(p2=cfg_p2, results=out), indent=1, default=float))
    for o in out:
        print(o["dataset"], f"path {o['path_m']} m, physical POS {o['physical_pos_signs']}, reaction {o['reaction_s']} s")
        for k, v in o["stages"].items():
            print(f"  {k}: stops {v['stops']} TP {v['TP']} FP {v['FP']} prec {v['stop_precision']} FP/100m {v['false_stops_per_100m']} "
                  f"UOR {v['useful_observation_recall']} readable_stop_rate(est) {v['readable_stop_rate_est']} invalidPos(est) {v['invalid_stop_position_est']} "
                  f"readableUOR(est) {v['readable_useful_recall_est']}")
        if not a.quiet:
            print("  P2 missed:", [(m["t"], m["spot"], m["label"], m["why"][:2]) for m in o["P2_missed"]])


if __name__ == "__main__":
    main()
