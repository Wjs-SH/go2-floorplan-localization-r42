#!/usr/bin/env python3
"""Hard-negative feature study (development split only). For every labelled spot, image features are computed on all
observations with SigLIP >= sim_floor (frames re-read from the extracted bag, read-only) and summarised per spot.
Reports per-feature spot-level AUC (POS vs NEG) and the fraction of POS spots a threshold would remove.

Features (tile = best SigLIP tile, ctx = 2.5x context crop):
  ctx_green      fraction of saturated green/cork-like hue in context (felt/cork bulletin boards)
  ctx_white_big  fraction of bright low-saturation pixels in context (paper sheets, glossy panels)
  tile_edges     Canny edge density in tile (text and plate borders)
  tile_strokes   number of small dark/bright connected components (character-like blobs) per 1000 px
  tile_contrast  p95 - p5 grey level
  sim_std        std of SigLIP score over the spot's observations (flicker)
  plane_rms      depth-plane residual
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

WS = Path.home() / "sci_ws"
sys.path.insert(0, str(WS / "tools"))
import rescore as RS  # noqa: E402
from probe_study import auc  # noqa: E402


def crop(img, b, k=1.0, smin=0):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(max(b[2] - b[0], b[3] - b[1]) * k, smin)
    x0, y0, x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
    return img[y0:y1, x0:x1] if k > 1 else img[int(max(0, b[1])):int(b[3]), int(max(0, b[0])):int(b[2])]


def feats(img, tile):
    t = crop(img, tile)
    c = crop(img, tile, 2.5, 120)
    if t.size == 0 or c.size == 0:
        return None
    hc = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    green = ((hc[..., 0] >= 35) & (hc[..., 0] <= 90) & (hc[..., 1] >= 60) & (hc[..., 2] >= 40)).mean()
    cork = ((hc[..., 0] >= 8) & (hc[..., 0] <= 25) & (hc[..., 1] >= 80) & (hc[..., 2] >= 60)).mean()
    white = ((hc[..., 1] <= 30) & (hc[..., 2] >= 200)).mean()
    g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    edges = (cv2.Canny(g, 60, 150) > 0).mean()
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    small = int(((st[1:, cv2.CC_STAT_AREA] >= 4) & (st[1:, cv2.CC_STAT_AREA] <= 0.05 * g.size)).sum()) if n > 1 else 0
    return dict(ctx_green=float(green + cork), ctx_white_big=float(white), tile_edges=float(edges),
                tile_strokes=1000.0 * small / g.size, tile_contrast=float(np.percentile(g, 95) - np.percentile(g, 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(WS / "experiments" / "dev_S1F_v3"))
    ap.add_argument("--labels", default=str(WS / "datasets" / "labels" / "v3" / "labels_dev_flat.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    sc, st, ds = RS.load(a.run)
    name = ds.name
    lab = {int(k.split("spot")[1]): v for k, v in json.loads(Path(a.labels).read_text()).items() if k.startswith(name + "|")}
    rows = list(csv.DictReader(open(ds / "index.csv")))
    stamps = np.array([float(r["stamp"]) for r in rows])
    per = defaultdict(list)
    cache = {}
    for s in sc:
        for r in s["regions"]:
            k = r["spot"]
            if k not in lab or lab[k] == "AMBIGUOUS" or r["sim"] < 0.078:
                continue
            i = int(np.argmin(np.abs(stamps - s["stamp"])))
            if i not in cache:
                cache = {i: cv2.imread(str(ds / rows[i]["rgb"]))}
            f = feats(cache[i], r.get("tile") or r["box"])
            if f:
                f.update(sim=r["sim"], plane_rms=r["feat"]["plane"].get("rms_m"))
                per[k].append(f)
    keys = sorted(per)
    y = np.array([1 if lab[k].startswith("POS") else 0 for k in keys])
    out = dict(run=a.run, n_spots=len(keys), n_pos=int(y.sum()), features={})
    for fn in ("ctx_green", "ctx_white_big", "tile_edges", "tile_strokes", "tile_contrast", "plane_rms"):
        v = np.array([np.nanmedian([o[fn] for o in per[k] if o.get(fn) is not None] or [np.nan]) for k in keys])
        ok = ~np.isnan(v)
        out["features"][fn] = dict(auc_pos_high=auc(y[ok], v[ok]), pos_median=float(np.median(v[ok][y[ok] == 1])),
                                   neg_median=float(np.median(v[ok][y[ok] == 0])),
                                   pos_values=[round(float(x), 4) for x in v[ok][y[ok] == 1]],
                                   neg_values=[round(float(x), 4) for x in v[ok][y[ok] == 0]])
    v = np.array([np.std([o["sim"] for o in per[k]]) for k in keys])
    out["features"]["sim_std"] = dict(auc_pos_high=auc(y, v), pos_median=float(np.median(v[y == 1])), neg_median=float(np.median(v[y == 0])))
    out["per_spot"] = {int(k): dict(label=lab[k], n=len(per[k]), **{fn: round(float(np.median([o[fn] for o in per[k] if o.get(fn) is not None] or [np.nan])), 4)
                                                                     for fn in ("ctx_green", "ctx_white_big", "tile_edges", "tile_strokes", "tile_contrast")})
                       for k in keys}
    txt = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    for fn, d in out["features"].items():
        print(f"{fn:14s} AUC(pos high) {d['auc_pos_high']:.3f}  pos med {d['pos_median']:.4f}  neg med {d['neg_median']:.4f}")
    for k, d in out["per_spot"].items():
        print(k, d)


if __name__ == "__main__":
    main()
