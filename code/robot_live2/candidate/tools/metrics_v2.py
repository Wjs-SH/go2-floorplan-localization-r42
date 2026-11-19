#!/usr/bin/env python3
"""Phase-2 E5 metrics on recorded runs (offline re-scoring, no GPU).

Definitions (per stage A0..A4):
  TP  stop on a labelled POS spot; FP stop on a labelled NEG spot; AMBIGUOUS/UNLABELLED excluded from precision
  physical sign = labelled POS spots whose mean xy lie within --cluster-m of each other (near-duplicate spots merged)
  Useful Observation Recall = physical POS signs with >= 1 TP stop / physical POS signs that are labelled
      (labels cover only spots that passed the legacy gate at least once -> recall relative to legacy-eligible signs)
  False Stops / 100 m = FP / odometry path length * 100
  Invalid Stop Position (ESTIMATE) = TP stop whose predicted viewing state is not readable:
      predicted obliquity > obl_max OR range > d_read OR estimated text height < text_min OR pan/tilt unreachable
  Readable Stop Rate (ESTIMATE) = (TP - invalid) / labelled stops   -> real value needs pan/tilt captures (UNVERIFIED)
  stop latency = stop scene time - first scene the physical sign was seen (any score) / first scene it was eligible
  duplicates = stops per physical sign (TP) beyond the first
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
from features import Geometry  # noqa: E402

READ = dict(obl_max=60.0, d_read=4.0, text_min=15.0)   # estimated readability envelope (OCR synthetic: fails from 45 deg squeeze)


def path_length(sc):
    p = np.array([s["pose"][:2] for s in sc])
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def spot_xy(sc):
    acc = defaultdict(list)
    for s in sc:
        for r in s["regions"]:
            if r["spot"] is not None and r["xyz"] is not None:
                acc[r["spot"]].append(r["xyz"][:2])
    return {k: np.mean(v, 0) for k, v in acc.items()}


def clusters(keys, xy, m):
    lab, c = {}, 0
    for k in keys:
        if k in lab:
            continue
        stack, lab[k] = [k], c
        while stack:
            a = stack.pop()
            for b in keys:
                if b not in lab and np.linalg.norm(xy[a] - xy[b]) < m:
                    lab[b] = c
                    stack.append(b)
        c += 1
    return lab


def feature_at(sc_by_t, t, spot):
    for r in sc_by_t[t]["regions"]:
        if r["spot"] == spot:
            return r
    return None


def readable(r):
    f = r["feat"]
    ob = f.get("obliquity_pred") or (f["plane"].get("obliquity_deg") if f["plane"].get("ok") else None)
    why = []
    if ob is not None and ob > READ["obl_max"]:
        why.append(f"obliquity {ob:.0f}")
    z = f.get("z_tile") or r["z"]
    if z is not None and z > READ["d_read"]:
        why.append(f"range {z:.1f}")
    if f.get("text_px") is not None and f["text_px"] < READ["text_min"]:
        why.append(f"text {f['text_px']:.0f}px")
    if not f["pantilt"]["ok"]:
        why.append("pantilt")
    return not why, why


def evaluate(run, labels, cfg, cluster_m=1.5):
    sc, st, ds = RS.load(run)
    name = ds.name
    by_t = {s["t"]: s for s in sc}
    stages = RS.replay_policy(sc, cfg, Geometry())
    a0 = []
    for e in st:
        if e.get("stage", "A0") == "A0":
            sp = next((r["spot"] for r in by_t[e["t"]]["regions"] if np.allclose(r["box"], e["picks"][0], atol=0.51)), None)
            a0.append(dict(t=e["t"], spot=sp))
    stages["A0"] = a0
    xy = spot_xy(sc)
    lab = {int(k.split("spot")[1]): v for k, v in labels.items() if k.startswith(name + "|")}
    pos = [k for k, v in lab.items() if v.startswith("POS") and k in xy]
    cl = clusters(pos, xy, cluster_m)
    n_phys = len(set(cl.values()))
    first_seen, first_elig = {}, {}
    for s in sc:
        for r in s["regions"]:
            k = r["spot"]
            if k in cl:
                c = cl[k]
                first_seen.setdefault(c, s["t"])
                if not any(w.split(":")[0] in ("legacy", "width", "height", "board") for w in r["dec"]["A0" if "A0" in r["dec"] else "A4"]["reasons"]):
                    first_elig.setdefault(c, s["t"])
    L = path_length(sc)
    out = dict(dataset=name, path_m=round(L, 1), duration_s=round(sc[-1]["t"] - sc[0]["t"], 1), physical_pos_signs=n_phys, stages={})
    for stg in ["A0", "A1", "A2", "A3", "A4"]:
        S = stages.get(stg, [])
        tp = fp = amb = 0
        reached, per_sign, lat_seen, lat_elig, invalid, inv_why = set(), defaultdict(int), [], [], 0, []
        for s in S:
            l = lab.get(s["spot"], "UNLABELLED")
            if l.startswith("POS"):
                tp += 1
                c = cl.get(s["spot"])
                if c is not None:
                    per_sign[c] += 1
                    if c not in reached:
                        lat_seen.append(s["t"] - first_seen[c])
                        if c in first_elig:
                            lat_elig.append(s["t"] - first_elig[c])
                    reached.add(c)
                r = feature_at(by_t, s["t"], s["spot"])
                if r is not None:
                    ok, why = readable(r)
                    if not ok:
                        invalid += 1
                        inv_why.append(dict(t=s["t"], spot=s["spot"], why=why))
            elif l.startswith("NEG"):
                fp += 1
            else:
                amb += 1
        n = tp + fp
        dup = sum(v - 1 for v in per_sign.values())
        q = lambda a, p: round(float(np.percentile(a, p)), 1) if a else None
        out["stages"][stg] = dict(stops=len(S), TP=tp, FP=fp, excluded=amb,
                                  stop_precision=round(tp / n, 3) if n else None,
                                  useful_observation_recall=f"{len(reached)}/{n_phys}",
                                  useful_observation_recall_v=round(len(reached) / n_phys, 3) if n_phys else None,
                                  false_stops_per_100m=round(100 * fp / L, 2) if L else None,
                                  invalid_stop_position_est=f"{invalid}/{tp}", readable_stop_rate_est=round((tp - invalid) / n, 3) if n else None,
                                  latency_from_first_seen_s=dict(mean=round(float(np.mean(lat_seen)), 1) if lat_seen else None, p95=q(lat_seen, 95)),
                                  latency_from_first_eligible_s=dict(mean=round(float(np.mean(lat_elig)), 1) if lat_elig else None, p95=q(lat_elig, 95)),
                                  duplicate_stops=dup, invalid_detail=inv_why)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--policy", default=str(WS / "config" / "e5_policy.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    labels = {}
    for f in a.labels:
        labels.update(json.loads(Path(f).read_text()))
    cfg = P.load_cfg(a.policy)
    res = [evaluate(r, labels, cfg) for r in a.run]
    if a.out:
        Path(a.out).write_text(json.dumps(dict(policy=a.policy, readability_envelope=READ, results=res), indent=1))
    for o in res:
        print(o["dataset"], "path", o["path_m"], "m,", o["duration_s"], "s, physical POS signs", o["physical_pos_signs"])
        for k, v in o["stages"].items():
            print(f"  {k}: stops {v['stops']} TP {v['TP']} FP {v['FP']} prec {v['stop_precision']} UOR {v['useful_observation_recall']} "
                  f"FP/100m {v['false_stops_per_100m']} invalidPos(est) {v['invalid_stop_position_est']} readable(est) {v['readable_stop_rate_est']} "
                  f"lat_seen {v['latency_from_first_seen_s']} lat_elig {v['latency_from_first_eligible_s']} dup {v['duplicate_stops']}")


if __name__ == "__main__":
    main()
