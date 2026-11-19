#!/usr/bin/env python3
"""E5 stop evaluation.

  sheets  : render every stop (all stages A0..A4) of a run as numbered contact sheets for labelling
  spots   : render every legacy wall spot that ever scored >= sim_floor (for recall labelling)
  metrics : per stage precision / false stops / positive-spot recall from a labels JSON

Labels are per stop instance id "<dataset>|<t>|<x0>,<y0>" (stops) or "<dataset>|spot<k>" (spots).
Classes: POS_ROOM, POS_NAME, POS_FACILITY (location-specific signs) | NEG_POSTER, NEG_BOARD, NEG_DIRECTORY,
NEG_ELEVATOR, NEG_GLARE, NEG_LIGHT, NEG_OTHER, NEG_EXIT ; optional suffix "_UNREADABLE" for positives
judged not readable from the stop position.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

STAGES = ["A0", "A1", "A2", "A3", "A4"]


def load_run(run):
    run = Path(run)
    scenes = [json.loads(l) for l in open(run / "scenes.jsonl")]
    stops = [json.loads(l) for l in open(run / "stops.jsonl")]
    args = json.loads((run / "args.json").read_text())
    return run, scenes, stops, args


def frame_at(ds, stamp, cache={}):
    key = str(ds)
    if key not in cache:
        rows = list(csv.DictReader(open(Path(ds) / "index.csv")))
        cache[key] = (rows, np.array([float(r["stamp"]) for r in rows]))
    rows, st = cache[key]
    k = int(np.argmin(np.abs(st - stamp)))
    return cv2.imread(str(Path(ds) / rows[k]["rgb"]))


def stop_list(scenes, stops):
    """Normalise A0 (legacy) and A1..A4 stops into one list with stage, t, stamp, box, tile, spot."""
    by_t = {s["t"]: s for s in scenes}
    out = []
    for e in stops:
        st = e.get("stage", "A0")
        if st == "A0":
            box = e["picks"][0]
            sc = by_t.get(e["t"])
            spot, tile = None, box
            if sc:
                for r in sc["regions"]:
                    if np.allclose(r["box"], box, atol=0.51):
                        spot, tile = r["spot"], r.get("tile") or box
            out.append(dict(stage="A0", t=e["t"], stamp=e["stamp"], box=box, tile=tile, spot=spot, z=e["z"][0]))
        else:
            out.append(dict(stage=st, t=e["t"], stamp=e["stamp"], box=e["box"], tile=e.get("tile") or e["box"], spot=e["spot"], z=e["z"]))
    return out


def sid(ds_name, s):
    return f"{ds_name}|{s['t']:.1f}|{int(round(s['tile'][0]))},{int(round(s['tile'][1]))}"


def panel(img, box, tile, label):
    a = img.copy()
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cv2.rectangle(a, (x0, y0), (x1, y1), (0, 200, 255), 1)
    t0, t1, t2, t3 = (int(round(v)) for v in tile)
    cv2.rectangle(a, (t0, t1), (t2, t3), (0, 0, 255), 2)
    a = cv2.resize(a, (320, 240))
    cx, cy = (t0 + t2) / 2, (t1 + t3) / 2
    s = max(t2 - t0, t3 - t1) * 2.2
    s = max(s, 90)
    X0, Y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    X1, Y1 = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
    z = cv2.resize(img[Y0:Y1, X0:X1], (240, 240), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(z, (int((t0 - X0) * 240 / max(1, X1 - X0)), int((t1 - Y0) * 240 / max(1, Y1 - Y0))),
                  (int((t2 - X0) * 240 / max(1, X1 - X0)), int((t3 - Y0) * 240 / max(1, Y1 - Y0))), (0, 0, 255), 1)
    cell = np.hstack([a, z])
    cv2.rectangle(cell, (0, 0), (560, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def sheets(a):
    run, scenes, stops, args = load_run(a.run)
    ds = Path(args["dataset"]) if Path(args["dataset"]).is_absolute() else Path.home() / "sci_ws" / args["dataset"]
    name = ds.name
    L = stop_list(scenes, stops)
    uniq = {}
    for s in L:
        uniq.setdefault(sid(name, s), dict(s, stages=[]))["stages"].append(s["stage"])
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    items = sorted(uniq.items(), key=lambda kv: kv[1]["t"])
    index = []
    cells = []
    for n, (k, s) in enumerate(items):
        img = frame_at(ds, s["stamp"])
        cells.append(panel(img, s["box"], s["tile"], f"{n:03d} t={s['t']:.0f} z={s['z']:.1f} {','.join(sorted(set(s['stages'])))}"))
        index.append(dict(n=n, id=k, t=s["t"], stages=sorted(set(s["stages"])), spot=s["spot"], box=s["box"], tile=s["tile"]))
    for p in range(0, len(cells), 8):
        grid = cells[p:p + 8]
        while len(grid) % 2:
            grid.append(np.zeros_like(cells[0]))
        rows = [np.hstack(grid[i:i + 2]) for i in range(0, len(grid), 2)]
        cv2.imwrite(str(out / f"stops_{p // 8:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "stops_index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(items)} unique stops -> {out}")


def metrics(a):
    labels = json.loads(Path(a.labels).read_text())
    res = {}
    for runp in a.run:
        run, scenes, stops, args = load_run(runp)
        name = Path(args["dataset"]).name
        L = stop_list(scenes, stops)
        pos_spots = {k for k, v in labels.items() if k.startswith(name + "|") and v.startswith("POS")}
        per = {}
        for st in STAGES:
            S = [s for s in L if s["stage"] == st]
            lab = [labels.get(sid(name, s), "UNLABELLED") for s in S]
            c = Counter(lab)
            npos = sum(v for k, v in c.items() if k.startswith("POS") and "UNREADABLE" not in k)
            npos_any = sum(v for k, v in c.items() if k.startswith("POS"))
            nlab = sum(v for k, v in c.items() if k != "UNLABELLED")
            # distinct physical positives reached: by label id key of the stop's spot group (spot ids are run-local)
            spots_pos = {s["spot"] for s, l in zip(S, lab) if l.startswith("POS") and s["spot"] is not None}
            per[st] = dict(stops=len(S), labelled=nlab, pos=npos_any, pos_readable=npos,
                           precision=round(npos_any / nlab, 3) if nlab else None,
                           readable_precision=round(npos / nlab, 3) if nlab else None,
                           false_stops=nlab - npos_any, fp_breakdown={k: v for k, v in c.items() if not k.startswith("POS")},
                           distinct_pos_spots=len(spots_pos))
        a0 = per["A0"]["false_stops"]
        for st in STAGES[1:]:
            per[st]["false_stop_reduction_vs_A0"] = round(1 - per[st]["false_stops"] / a0, 3) if a0 else None
        res[name] = per
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sheets"); s.add_argument("run"); s.add_argument("--out", default=str(Path.home() / "sci_ws" / "datasets" / "labels"))
    m = sp.add_parser("metrics"); m.add_argument("run", nargs="+"); m.add_argument("--labels", required=True); m.add_argument("--out", default=None)
    a = ap.parse_args()
    {"sheets": sheets, "metrics": metrics}[a.cmd](a)
