#!/usr/bin/env python3
"""SEM1 ROI ablation on STORED frames with the INSTALLED SigLIP (STORED_MODEL) and the INSTALLED RapidOCR detector.
Same source frame / same model / same prompts (roomstd) / same preprocessing for every arm; only the ROI changes.

  C0  the region box the run actually scored (legacy tiles 100 px + max over tiles) -> reproduction check vs logged sim,
      + EXACT encoder input dump (uint8 224x224 RGB PNG + fp16 tensor sha256) for a capped sample
  C1  human tight bbox (ORACLE_ROI)            -> only from human-reviewed CSV; none yet -> NOT_RUN (HUMAN_REVIEW_REQUIRED)
  C2  C1 + limited context / rectification     -> same dependency
  C3  RapidOCR PP-OCRv4 DB detector ONLY (no cls/rec) on the full frame, text lines grouped into sign ROIs (auto)
  C4  C0 region U C3 ROIs overlapping it (bounded union; candidate count logged)
Text present != valid sign: C3 ROIs are candidates; posters with text are reported as such (false-poster rate).
Labels are PROVISIONAL (Claude v3) spot labels of the v3 runs; results are diagnostics, not accuracy claims.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent
sys.path.insert(0, str(WS / "src" / "sci_visual_observation"))
sys.path.insert(0, str(WS / "src" / "sci_legacy_e5"))
LAB = Path("/home/unitree/sci_ws/datasets/labels/v3")
EXP = Path("/home/unitree/sci_ws/experiments")
DS = {"S-1F": dict(labels=LAB / "labels_S-1F_v3.json", frames=Path("/home/unitree/sci_ws/datasets/bags/S-1F")),
      "S-6F": dict(labels=LAB / "labels_S-6F_v3.json", frames=Path("/home/unitree/sci_ws/datasets/bags/S-6F"))}
POS = ("POS_ROOM", "POS_NAME")


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def group_lines(polys, W, H, gap=0.6, pad=0.15):
    """Merge detected text lines that are close (vertical gap < gap*line height, horizontal overlap) into sign ROIs."""
    boxes = [[float(min(p[:, 0])), float(min(p[:, 1])), float(max(p[:, 0])), float(max(p[:, 1]))] for p in polys]
    groups = []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        h = b[3] - b[1]
        for g in groups:
            gh = max(1.0, min(h, g["lh"]))
            if (b[0] <= g["box"][2] + gap * gh and b[2] >= g["box"][0] - gap * gh and b[1] <= g["box"][3] + gap * gh and b[3] >= g["box"][1] - gap * gh):
                g["box"] = [min(g["box"][0], b[0]), min(g["box"][1], b[1]), max(g["box"][2], b[2]), max(g["box"][3], b[3])]
                g["n"] += 1
                g["lh"] = min(g["lh"], h)
                break
        else:
            groups.append(dict(box=list(b), n=1, lh=h))
    out = []
    for g in groups:
        x0, y0, x1, y1 = g["box"]
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        out.append(dict(box=[max(0.0, x0 - px), max(0.0, y0 - py), min(W - 1.0, x1 + px), min(H - 1.0, y1 + py)], n_lines=g["n"]))
    return out


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-max", type=int, default=60)
    ap.add_argument("--human-bbox-csv", default=None, help="C1/C2: human tight bbox CSV (item,x0,y0,x1,y1,human_verified)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "encoder_input").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KCI_SIGN_PROMPTS", "roomstd")
    import torch
    import siglip_worker as SW
    t0 = time.time()
    m = SW.SciSiglip()
    load_s = time.time() - t0
    from rapidocr_onnxruntime import RapidOCR
    dets = {nt: RapidOCR(intra_op_num_threads=nt, inter_op_num_threads=1) for nt in (1, 2)}
    rows, det_ms = [], {1: [], 2: []}
    n_dump = 0
    human = {}
    if a.human_bbox_csv and Path(a.human_bbox_csv).exists():
        for r in csv.DictReader(open(a.human_bbox_csv)):
            if str(r.get("human_verified", "")).lower() == "true":
                human[r["item"]] = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]

    def score(bgr, boxes):
        s = m.score_sci(bgr, boxes)
        torch.cuda.synchronize()
        return [float(v) for v in s], m._last_tile_input

    for fl, cfg in DS.items():
        lab = json.loads(cfg["labels"].read_text())
        run = EXP / Path(lab["run"]).name
        prompts = json.loads((run / "siglip_prompts.json").read_text())["env"]
        assert prompts == os.environ["KCI_SIGN_PROMPTS"], (run, prompts)
        idx = list(csv.DictReader(open(cfg["frames"] / "index.csv")))
        st = np.array([float(r["stamp"]) for r in idx])
        best = {}
        for line in open(run / "scenes.jsonl"):
            r = json.loads(line)
            for g in r.get("regions") or []:
                k = f"{fl}|spot{g.get('spot')}"
                if g.get("spot") is None or k not in lab["labels"]:
                    continue
                if k not in best or g["sim"] > best[k]["sim"]:
                    best[k] = dict(sim=float(g["sim"]), box=g["box"], tile=g.get("tile") or g["box"], stamp=r.get("stamp"), t=r["t"])
        for k, o in sorted(best.items()):
            v3 = lab["labels"][k]
            fi = int(np.argmin(np.abs(st - float(o["stamp"]))))
            bgr = cv2.imread(str(cfg["frames"] / idx[fi]["rgb"]))
            H, W = bgr.shape[:2]
            c0, (crops, subs, owner, x) = score(bgr, [o["box"]])
            rec = dict(item=k, floor=fl, v3=v3, kind="POS" if v3 in POS else ("NEG" if v3.startswith("NEG") else "EXCLUDED"),
                       frame=idx[fi]["rgb"], t=o["t"], region_box=o["box"], logged_sim=round(o["sim"], 5), c0_sim=round(c0[0], 5),
                       c0_repro_abs_diff=round(abs(c0[0] - o["sim"]), 6), c0_n_tiles=len(crops), region_px=[round(o["box"][2] - o["box"][0]), round(o["box"][3] - o["box"][1])])
            if n_dump < a.dump_max:
                d = out / "encoder_input" / k.replace("|", "_")
                d.mkdir(parents=True, exist_ok=True)
                tiles = []
                for i, (c, b) in enumerate(zip(crops, subs)):
                    cv2.imwrite(str(d / f"tile{i:02d}_224_rgb_as_png.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
                    tiles.append(dict(i=i, box_src=[float(v) for v in b], shape=list(c.shape), rgb_uint8_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
                                      tensor_fp16_sha256=hashlib.sha256(x[i].detach().cpu().numpy().tobytes()).hexdigest()))
                (d / "meta.json").write_text(json.dumps(dict(item=k, source_frame=str(cfg["frames"] / idx[fi]["rgb"]),
                                                             source_sha256=hashlib.sha256((cfg["frames"] / idx[fi]["rgb"]).read_bytes()).hexdigest(),
                                                             region_box=o["box"], c0_sim=c0[0], prompts=prompts, model=os.environ.get("SCI_SIGLIP_MODEL"),
                                                             preprocess="BGR->RGB; square s=max(w,h,32) around tile centre clipped to frame; cv2.INTER_LINEAR 224; (x/255-0.5)/0.5 fp16 GPU",
                                                             kind="EXACT_ENCODER_INPUT (not a context preview)", tiles=tiles), indent=1))
                n_dump += 1
                rec["encoder_dump"] = str(d.relative_to(out))
            # C3: detector only, timed for 1 and 2 threads (same frame)
            polys = None
            for nt, eng in dets.items():
                t1 = time.perf_counter()
                res, _ = eng(bgr, use_det=True, use_cls=False, use_rec=False)
                det_ms[nt].append((time.perf_counter() - t1) * 1e3)
                if polys is None:
                    polys = [np.array(p, np.float32) for p in (res or [])]
            rois = group_lines(polys, W, H)
            hit = [r_ for r_ in rois if overlaps(r_["box"], o["box"])]
            rec.update(c3_n_text_lines=len(polys), c3_n_rois_frame=len(rois), c3_hit=bool(hit))
            if hit:
                sims, _ = score(bgr, [h["box"] for h in hit])
                j = int(np.argmax(sims))
                rec.update(c3_sim=round(sims[j], 5), c3_roi=[round(v, 1) for v in hit[j]["box"]], c3_roi_px=[round(hit[j]["box"][2] - hit[j]["box"][0]), round(hit[j]["box"][3] - hit[j]["box"][1])],
                           c3_lines=hit[j]["n_lines"], c3_iou_region=round(iou(hit[j]["box"], o["box"]), 3))
                rec["c4_sim"] = round(max(c0[0], sims[j]), 5)
                rec["c4_n_candidates"] = 1 + len(hit)
            else:
                rec.update(c3_sim=None, c4_sim=round(c0[0], 5), c4_n_candidates=1)
            # C3r: detector on the candidate region only (native pixels, padded 50 %, upscaled x3) -> tight text ROI
            bx = o["box"]
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            X0, Y0 = int(max(0, bx[0] - 0.5 * bw)), int(max(0, bx[1] - 0.5 * bh))
            X1, Y1 = int(min(W, bx[2] + 0.5 * bw)), int(min(H, bx[3] + 0.5 * bh))
            sub = cv2.resize(bgr[Y0:Y1, X0:X1], None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            t1 = time.perf_counter()
            res, _ = dets[2](sub, use_det=True, use_cls=False, use_rec=False)
            rec["c3r_det_ms_2thr"] = round((time.perf_counter() - t1) * 1e3, 1)
            pl = [np.array(p_, np.float32) / 3.0 + np.array([X0, Y0], np.float32) for p_ in (res or [])]
            rr = [r_ for r_ in group_lines(pl, W, H) if overlaps(r_["box"], bx)]
            rec["c3r_hit"] = bool(rr)
            if rr:
                sims, _ = score(bgr, [h["box"] for h in rr])
                j = int(np.argmax(sims))
                rec.update(c3r_sim=round(sims[j], 5), c3r_roi=[round(v, 1) for v in rr[j]["box"]], c3r_lines=rr[j]["n_lines"])
            else:
                rec["c3r_sim"] = None
            if k in human:
                c1, _ = score(bgr, [human[k]])
                rec["c1_sim"] = round(c1[0], 5)
            rows.append(rec)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "roi_ablation_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    summ = dict(model_load_s=round(load_s, 1), n_items=len(rows), n_encoder_dumps=n_dump,
                c0_repro_max_abs_diff=max(r["c0_repro_abs_diff"] for r in rows),
                det_ms={nt: dict(n=len(v), p50=round(float(np.median(v)), 1), p95=round(float(np.percentile(v, 95)), 1), max=round(max(v), 1)) for nt, v in det_ms.items()},
                human_bbox_items=len(human))
    for fl in DS:
        R = [r for r in rows if r["floor"] == fl]
        P = [r for r in R if r["kind"] == "POS"]
        N_ = [r for r in R if r["kind"] == "NEG"]
        summ[fl] = dict(n_pos=len(P), n_neg=len(N_),
                        auc_c0=auc([r["c0_sim"] for r in P], [r["c0_sim"] for r in N_]),
                        auc_c4=auc([r["c4_sim"] for r in P], [r["c4_sim"] for r in N_]),
                        c3_coverage_pos=f"{sum(r['c3_hit'] for r in P)}/{len(P)}", c3_text_on_neg=f"{sum(r['c3_hit'] for r in N_)}/{len(N_)}",
                        c3_text_on_poster=f"{sum(r['c3_hit'] for r in N_ if r['v3'] in ('NEG_POSTER', 'NEG_BOARD'))}/{sum(1 for r in N_ if r['v3'] in ('NEG_POSTER', 'NEG_BOARD'))}",
                        c3r_coverage_pos=f"{sum(r['c3r_hit'] for r in P)}/{len(P)}", c3r_text_on_neg=f"{sum(r['c3r_hit'] for r in N_)}/{len(N_)}",
                        c3r_text_on_poster=f"{sum(r['c3r_hit'] for r in N_ if r['v3'] in ('NEG_POSTER', 'NEG_BOARD'))}",
                        auc_c3r_or_c0=auc([r["c3r_sim"] if r["c3r_sim"] is not None else r["c0_sim"] for r in P], [r["c3r_sim"] if r["c3r_sim"] is not None else r["c0_sim"] for r in N_]),
                        c3r_det_ms_p50=float(np.median([r["c3r_det_ms_2thr"] for r in R])),
                        auc_c3_where_hit=auc([r["c3_sim"] for r in P if r["c3_sim"] is not None], [r["c3_sim"] for r in N_ if r["c3_sim"] is not None]),
                        c1="NOT_RUN: HUMAN_REVIEW_REQUIRED (no human-verified tight bbox)" if not human else "see rows")
    (out / "roi_ablation_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
