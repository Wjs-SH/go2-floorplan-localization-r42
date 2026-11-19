#!/usr/bin/env python3
"""Offline re-scoring of recorded E5 scenes (experiments/<run>/scenes.jsonl) with a policy config — no GPU, no camera.
The front end (legacy proposals/regions/SigLIP/spot memory) is recorded; only the SCI gates are re-run, so different
thresholds can be compared on identical inputs. Also renders spot sheets for labelling and computes spot-level metrics.

  rescore.py spots   <run> [--out DIR]             -> contact sheets of every wall spot that ever scored >= sim_floor
  rescore.py eval    <run>... --labels L.json [--policy cfg.json] [--set k=v ...]
Labels: {"<dataset>|spot<k>": CLASS} with CLASS in POS_ROOM/POS_NAME/POS_FACILITY/NEG_*. Spot ids are run-local.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "sci_visual_observation"))
import policy as P  # noqa: E402
from features import Geometry  # noqa: E402

WS = Path.home() / "sci_ws"


def load(run):
    run = Path(run)
    sc = [json.loads(l) for l in open(run / "scenes.jsonl")]
    st = [json.loads(l) for l in open(run / "stops.jsonl")] if (run / "stops.jsonl").exists() else []
    args = json.loads((run / "args.json").read_text())
    ds = Path(args["dataset"]) if Path(args["dataset"]).is_absolute() else WS / args["dataset"]
    return sc, st, ds


def frames(ds):
    rows = list(csv.DictReader(open(ds / "index.csv")))
    return rows, np.array([float(r["stamp"]) for r in rows])


def spots(a):
    sc, st, ds = load(a.run)
    rows, stamps = frames(ds)
    best = {}
    a0 = {}
    for e in st:
        if e.get("stage", "A0") == "A0":
            a0[e["t"]] = e["picks"][0]
    for s in sc:
        for r in s["regions"]:
            if r["spot"] is None:
                continue
            k = r["spot"]
            info = best.setdefault(k, dict(spot=k, n=0, max_sim=-1, legacy_pass=0, a0_stop=False))
            info["n"] += 1
            if not any(w.split(":")[0] in ("legacy", "width", "height", "board") for w in r["dec"]["A4"]["reasons"]):
                info["legacy_pass"] += 1
            if s["t"] in a0 and np.allclose(a0[s["t"]], r["box"], atol=0.51):
                info["a0_stop"] = True
            lp_obs = not any(w.split(":")[0] in ("legacy", "width", "height", "board") for w in r["dec"]["A4"]["reasons"])
            if r["sim"] > info["max_sim"] and (lp_obs or not a.eligible_only or info["legacy_pass"] == 0):
                info.update(max_sim=r["sim"], stamp=s["stamp"], t=s["t"], box=r["box"], tile=r.get("tile") or r["box"], z=r["z"])
    keep = sorted([v for v in best.values() if (v["legacy_pass"] or v["a0_stop"]) or (not a.eligible_only and v["max_sim"] >= a.min_sim)],
                  key=lambda v: v["t"])
    out = Path(a.out) / ds.name
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for v in keep:
        k = int(np.argmin(np.abs(stamps - v["stamp"])))
        img = cv2.imread(str(ds / rows[k]["rgb"]))
        x0, y0, x1, y1 = (int(round(c)) for c in v["tile"])
        a_ = img.copy()
        cv2.rectangle(a_, (x0, y0), (x1, y1), (0, 0, 255), 2)
        a_ = cv2.resize(a_, (320, 240))
        cx, cy, s_ = (x0 + x1) / 2, (y0 + y1) / 2, max(90, 2.4 * max(x1 - x0, y1 - y0))
        X0, Y0, X1, Y1 = int(max(0, cx - s_ / 2)), int(max(0, cy - s_ / 2)), int(min(640, cx + s_ / 2)), int(min(480, cy + s_ / 2))
        z = cv2.resize(img[Y0:Y1, X0:X1], (240, 240), interpolation=cv2.INTER_CUBIC)
        cell = np.hstack([a_, z])
        cv2.rectangle(cell, (0, 0), (560, 20), (0, 0, 0), -1)
        cv2.putText(cell, f"spot{v['spot']} t={v['t']:.0f} s={v['max_sim']:.3f} n={v['n']} lp={v['legacy_pass']}{' A0' if v['a0_stop'] else ''}",
                    (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    for p in range(0, len(cells), 8):
        g = cells[p:p + 8]
        while len(g) % 2:
            g.append(np.zeros_like(cells[0]))
        cv2.imwrite(str(out / f"spots_{p // 8:02d}.jpg"), np.vstack([np.hstack(g[i:i + 2]) for i in range(0, len(g), 2)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "spots_index.json").write_text(json.dumps(keep, indent=1, default=float))
    print(f"{len(keep)} spots -> {out}")


def replay_policy(sc, cfg, geom):
    stages = P.make_stages(cfg, geom)
    stops = defaultdict(list)
    for s in sc:
        regs, feats = [], []
        for r in s["regions"]:
            rr = {k: v for k, v in r.items() if k not in ("feat", "dec")}
            f = dict(r["feat"])
            f.setdefault("up", None)
            regs.append(rr)
            feats.append(f)
        for name, sp in stages.items():
            _, stop = sp.step(s["t"], regs, feats, s["thr"], s.get("e4"))
            if stop:
                stops[name].append(stop)
    return stops


def evaluate(a):
    labels = json.loads(Path(a.labels).read_text()) if a.labels else {}
    cfg = P.load_cfg(a.policy)
    for kv in a.set or []:
        k, v = kv.split("=")
        sec, key = k.split(".")
        cfg[sec][key] = json.loads(v)
    geom = Geometry()
    report = {}
    for run in a.run:
        sc, st, ds = load(run)
        name = ds.name
        stops = replay_policy(sc, cfg, geom)
        a0 = []
        by_t = {s["t"]: s for s in sc}
        for e in st:
            if e.get("stage", "A0") == "A0":
                sp = None
                for r in by_t.get(e["t"], {}).get("regions", []):
                    if np.allclose(r["box"], e["picks"][0], atol=0.51):
                        sp = r["spot"]
                a0.append(dict(t=e["t"], spot=sp))
        stops["A0"] = a0
        pos_all = {k for k, v in labels.items() if k.startswith(name + "|") and v.startswith("POS")}
        per = {}
        for stg in ["A0", "A1", "A2", "A3", "A4"]:
            S = stops.get(stg, [])
            lab = [labels.get(f"{name}|spot{s['spot']}", "UNLABELLED") for s in S]
            c = Counter(lab)
            nl = sum(v for k, v in c.items() if k not in ("UNLABELLED", "AMBIGUOUS"))
            npos = sum(v for k, v in c.items() if k.startswith("POS"))
            reached = {f"{name}|spot{s['spot']}" for s, l in zip(S, lab) if l.startswith("POS")}
            per[stg] = dict(stops=len(S), labelled=nl, pos=npos, precision=round(npos / nl, 3) if nl else None,
                            false_stops=nl - npos, fp=dict(Counter(l for l in lab if not l.startswith("POS"))),
                            excluded=c.get("UNLABELLED", 0) + c.get("AMBIGUOUS", 0),
                            pos_spots_reached=len(reached), pos_spots_total=len(pos_all),
                            spot_recall=round(len(reached) / len(pos_all), 3) if pos_all else None,
                            stop_times=[round(s["t"], 1) for s in S])
        f0 = per["A0"]["false_stops"]
        for stg in ["A1", "A2", "A3", "A4"]:
            per[stg]["false_stop_reduction_vs_A0"] = round(1 - per[stg]["false_stops"] / f0, 3) if f0 else None
        report[name] = per
    txt = json.dumps(dict(config=cfg, results=report), indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(txt)
    for name, per in report.items():
        print(name)
        for stg, v in per.items():
            print(f"  {stg}: stops={v['stops']} pos={v['pos']} prec={v['precision']} FP={v['false_stops']} {v['fp']} "
                  f"recall={v['pos_spots_reached']}/{v['pos_spots_total']}" + (f" FPred={v.get('false_stop_reduction_vs_A0')}" if stg != "A0" else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("spots"); s.add_argument("run"); s.add_argument("--out", default=str(WS / "datasets" / "labels")); s.add_argument("--min-sim", type=float, default=0.078)
    s.add_argument("--eligible-only", action="store_true", help="only spots that pass the legacy gate at least once (decision-relevant)")
    e = sp.add_parser("eval"); e.add_argument("run", nargs="+"); e.add_argument("--labels"); e.add_argument("--policy"); e.add_argument("--set", nargs="*"); e.add_argument("--out")
    a = ap.parse_args()
    {"spots": spots, "eval": evaluate}[a.cmd](a)
