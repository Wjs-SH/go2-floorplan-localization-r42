#!/usr/bin/env python3
"""SEM1 follow-up: human box loader (UI export -> validated records) and selectable ROI arms on the SAME source frame.

Loader (CPU): HUMAN_BBOX.csv (RFC 4180, optional UTF-8 BOM) or the UI JSON state.
  key = (run_id, frame_sha256, physical_target_id or box_id, visit_id, box_revision); the newest revision per box wins.
  refused: HUMAN_BOX_FRAME_MISMATCH (file sha / width / height differ from the export), BOX_OUT_OF_RANGE, BOX_TOO_SMALL,
           DUPLICATE_REVISION_CONFLICT. human_verified=false -> kept but EXCLUDED_NOT_HUMAN_VERIFIED; test_only -> TEST_ONLY.
  The frame is the one the human drew on (frame_path + sha), never "the best-score frame of the item".
Arms (--arms, only the listed ones run; unlisted engines are never imported or constructed):
  C0       the region the pipeline scored on THIS frame (scene log), if any overlaps the human box; else C0_NOT_ON_FRAME
  C1_tile  human box, legacy 100 px tiling + max          (ORACLE_ROI_DIAGNOSTIC)
  C1_whole human box as ONE square crop                     (ORACLE_ROI_DIAGNOSTIC, sub-arm: input change is explicit)
  C2_padP  human box + P x box size context, one crop       (P from --c2-pad, DEV comparison setting)
  C2_rect  homography of the human 4-corner polygon to a rectangle (only if a polygon exists; else NOT_RUN)
  C3/C3r   RapidOCR detector arms (imported only if listed)
Scorer: --scorer fake (CPU, deterministic, for wiring tests) | siglip (installed model; GPU; run ONLY under the
sem1 supervisor with a separate approval - not in this follow-up).
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = Path("/home/unitree/sci_ws/experiments")
DATA = {"S-1F": Path("/home/unitree/sci_ws/datasets/bags/S-1F"), "S-6F": Path("/home/unitree/sci_ws/datasets/bags/S-6F")}
MIN_SIDE = 6.0


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ loader
def _rows_from_json(doc, items):
    st = doc["state"]
    by_item = {i["item"]: i for i in items}
    out = []
    for item, si in st.get("items", {}).items():
        it = by_item.get(item)
        if it is None:
            continue
        for f in it["frames"]:
            fs = si.get("frames", {}).get(f["frame_sha256"])
            if not fs:
                continue
            for b in fs.get("boxes", []):
                out.append(dict(dataset_id=it["dataset_id"], run_id=it["run_id"], item=item, physical_target_id=b.get("physical_target_id", ""),
                                visit_id=b.get("visit_id", ""), frame_path=f["frame_path"], frame_id=f["frame_id"], frame_sha256=f["frame_sha256"],
                                width=f["width"], height=f["height"], x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"], visual_type=b.get("visual_type"),
                                localization_role=b.get("localization_role"), text_readability=b.get("text_readability"), reviewer=st.get("reviewer", ""),
                                reviewed_at=b.get("reviewed_at"), human_verified="true" if b.get("human_verified") else "false",
                                label_revision=doc.get("label_revision"), box_id=b["box_id"], box_revision=b.get("revision", 1),
                                polygon=json.dumps(b["polygon"]) if b.get("polygon") else "", frame_status=fs.get("status") or "",
                                target_in_model_crop=b.get("target_in_model_crop", "UNSURE"), test_only="true" if b.get("test_only") else "false",
                                note=b.get("note", "")))
    return out


def load_human(path, review_root, items=None):
    """Return (accepted, rejected). review_root = folder the UI frame_path values are relative to."""
    review_root = Path(review_root)
    p = Path(path)
    if p.suffix == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("schema") != "sem1-human-bbox-1":
            raise ValueError("SCHEMA_MISMATCH")
        rows = _rows_from_json(doc, items or json.loads((review_root / "ui_items.json").read_text())["items"])
    else:
        rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig"), newline="")))
    acc, rej, seen = {}, [], {}
    for r in rows:
        why = []
        fp = review_root / r["frame_path"]
        if not fp.exists():
            why.append("FRAME_FILE_MISSING")
        else:
            if sha_file(fp) != r["frame_sha256"]:
                why.append("HUMAN_BOX_FRAME_MISMATCH:sha256")
            import cv2
            im = cv2.imread(str(fp))
            if im is None or int(r["width"]) != im.shape[1] or int(r["height"]) != im.shape[0]:
                why.append("HUMAN_BOX_FRAME_MISMATCH:size")
        try:
            x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
        except (TypeError, ValueError):
            why.append("BOX_NOT_NUMERIC")
            x0 = y0 = x1 = y1 = float("nan")
        W, H = float(r["width"]), float(r["height"])
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 <= x0 or y1 <= y0:
            why.append("BOX_OUT_OF_RANGE")
        elif min(x1 - x0, y1 - y0) < MIN_SIDE:
            why.append("BOX_TOO_SMALL")
        rec = dict(r, box=[x0, y0, x1, y1], box_revision=int(r.get("box_revision") or 1),
                   polygon=json.loads(r["polygon"]) if r.get("polygon") else None,
                   human_verified=str(r.get("human_verified", "")).lower() == "true", test_only=str(r.get("test_only", "")).lower() == "true")
        if why:
            rej.append(dict(rec, reject=why))
            continue
        bid = (r["run_id"], r["frame_sha256"], r.get("box_id"))
        prev = seen.get(bid)
        if prev is not None:
            if prev["box_revision"] == rec["box_revision"] and prev["box"] != rec["box"]:
                rej.append(dict(rec, reject=["DUPLICATE_REVISION_CONFLICT"]))
                continue
            if prev["box_revision"] > rec["box_revision"]:
                continue
        seen[bid] = rec
    for rec in seen.values():
        rec["key"] = "|".join(str(x) for x in (rec["run_id"], rec["frame_sha256"][:16], rec.get("physical_target_id") or rec.get("box_id"),
                                                 rec.get("visit_id") or "-", rec["box_revision"]))
        rec["status"] = "TEST_ONLY" if rec["test_only"] else ("ACCEPTED" if rec["human_verified"] else "EXCLUDED_NOT_HUMAN_VERIFIED")
        acc[rec["key"]] = rec
    return acc, rej


# ------------------------------------------------------------------------------------------------ crops
def square_crop(bgr, b, size=224):
    """Same geometry as SciSiglip._prep_u8 (square s=max(w,h,32) around the centre, clipped to the frame)."""
    import cv2
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    s = max(x1 - x0, y1 - y0, 32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = bgr.shape[:2]
    c = bgr[int(max(0, cy - s / 2)):int(min(H, cy + s / 2)), int(max(0, cx - s / 2)):int(min(W, cx + s / 2))]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR) if c.size else np.zeros((size, size, 3), np.uint8)


def pad_box(b, p, W, H):
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(0.0, b[0] - p * w), max(0.0, b[1] - p * h), min(W - 1.0, b[2] + p * w), min(H - 1.0, b[3] + p * h)]


def rectify(bgr, poly, out_w=224, out_h=224):
    import cv2
    src = np.array(poly, np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (out_w, out_h))


class FakeScorer:
    """Deterministic CPU stand-in: score = mean of the exact 224 crop / 255 (wiring/identity tests only)."""
    name = "FAKE_CPU_NOT_A_MODEL"

    def __init__(self):
        self.calls = []

    def tiled(self, bgr, b):
        nx, ny = max(1, math.ceil((b[2] - b[0]) / 100)), max(1, math.ceil((b[3] - b[1]) / 100))
        xs, ys = np.linspace(b[0], b[2], nx + 1), np.linspace(b[1], b[3], ny + 1)
        crops = [square_crop(bgr, [xs[i], ys[j], xs[i + 1], ys[j + 1]]) for i in range(nx) for j in range(ny)]
        self.calls.append(("tiled", len(crops)))
        return max(float(c.mean()) / 255 for c in crops), crops

    def whole(self, bgr, b):
        c = square_crop(bgr, b)
        self.calls.append(("whole", 1))
        return float(c.mean()) / 255, [c]

    def image(self, crop224):
        self.calls.append(("image", 1))
        return float(crop224.mean()) / 255, [crop224]


class SiglipScorer:                      # pragma: no cover - GPU path, not executed in this follow-up
    name = "SIGLIP_INSTALLED_FP16"

    def __init__(self):
        sys.path.insert(0, str(HERE.parent / "src" / "sci_visual_observation"))
        import siglip_worker as SW
        self.m = SW.SciSiglip()

    def _forward(self, crops_rgb):
        import torch
        x = self.m._to_gpu(crops_u8=crops_rgb)
        with torch.no_grad():
            e = self.m.model.get_image_features(pixel_values=x)
            en = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
            return (en @ self.m.txt.T).max(1).values.cpu().numpy()

    def tiled(self, bgr, b):
        s = self.m.score_sci(bgr, [b])
        return float(s[0]), self.m._last_tile_input[0]

    def whole(self, bgr, b):
        import cv2
        c = cv2.cvtColor(square_crop(bgr, b), cv2.COLOR_BGR2RGB)
        return float(self._forward([c])[0]), [c]

    def image(self, crop224):
        import cv2
        c = cv2.cvtColor(crop224, cv2.COLOR_BGR2RGB)
        return float(self._forward([c])[0]), [c]


def c0_region_on_frame(rec, review_root):
    """The region the pipeline scored on exactly this frame (scene log), overlapping the human box."""
    items = json.loads((Path(review_root) / "ui_items.json").read_text())["items"]
    it = next((i for i in items if i["item"] == rec["item"]), None)
    f = next((x for x in (it or {}).get("frames", []) if x["frame_sha256"] == rec["frame_sha256"]), None)
    if f is None or f.get("frame_index") is None:
        return None, "C0_FRAME_NOT_IN_ITEM"
    rows = list(csv.DictReader(open(DATA[rec["dataset_id"]] / "index.csv")))
    stamp = float(rows[int(f["frame_index"])]["stamp"])
    best, bi = None, 0.0
    for line in open(EXP / rec["run_id"] / "scenes.jsonl"):
        r = json.loads(line)
        if r.get("stamp") is None or abs(float(r["stamp"]) - stamp) > 0.06:
            continue
        for g in r.get("regions") or []:
            b = g["box"]
            x0, y0, x1, y1 = max(b[0], rec["box"][0]), max(b[1], rec["box"][1]), min(b[2], rec["box"][2]), min(b[3], rec["box"][3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            if inter > bi:
                best, bi = dict(box=b, logged_sim=g["sim"], scene_t=r["t"]), inter
    return (best, None) if best else (None, "C0_NOT_ON_FRAME")


def run_arms(acc, review_root, arms, scorer, c2_pads=(0.25, 0.5)):
    import cv2
    rows = []
    for key, rec in sorted(acc.items()):
        img = cv2.imread(str(Path(review_root) / rec["frame_path"]))
        H, W = img.shape[:2]
        base = dict(key=key, status=rec["status"], item=rec["item"], dataset_id=rec["dataset_id"], run_id=rec["run_id"],
                    frame_path=rec["frame_path"], frame_sha256=rec["frame_sha256"], physical_target_id=rec.get("physical_target_id"),
                    visit_id=rec.get("visit_id"), visual_type=rec.get("visual_type"), localization_role=rec.get("localization_role"),
                    human_box=rec["box"], scorer=scorer.name)
        for arm in arms:
            out = dict(base, arm=arm)
            if arm == "C0":
                reg, why = c0_region_on_frame(rec, review_root)
                if reg is None:
                    out.update(result="NOT_RUN", why=why)
                else:
                    s, crops = scorer.tiled(img, reg["box"])
                    out.update(result="SCORED", score=s, roi=reg["box"], logged_sim=reg["logged_sim"], n_crops=len(crops))
            elif arm == "C1_tile":
                s, crops = scorer.tiled(img, rec["box"])
                out.update(result="SCORED", score=s, roi=rec["box"], n_crops=len(crops), label="ORACLE_ROI_DIAGNOSTIC")
            elif arm == "C1_whole":
                s, crops = scorer.whole(img, rec["box"])
                out.update(result="SCORED", score=s, roi=rec["box"], n_crops=1, label="ORACLE_ROI_DIAGNOSTIC",
                           crop_sha256=hashlib.sha256(crops[0].tobytes()).hexdigest())
            elif arm.startswith("C2_pad"):
                p = float(arm[len("C2_pad"):])
                roi = pad_box(rec["box"], p, W, H)
                s, crops = scorer.whole(img, roi)
                out.update(result="SCORED", score=s, roi=roi, pad=p, n_crops=1, label="ORACLE_ROI_DIAGNOSTIC+CONTEXT")
            elif arm == "C2_rect":
                if not rec.get("polygon") or len(rec["polygon"]) != 4:
                    out.update(result="NOT_RUN", why="NO_HUMAN_POLYGON (rectangle box only: perspective not corrected)")
                else:
                    s, crops = scorer.image(rectify(img, rec["polygon"]))
                    out.update(result="SCORED", score=s, polygon=rec["polygon"], n_crops=1, label="ORACLE_ROI_DIAGNOSTIC+RECTIFIED")
            else:
                out.update(result="NOT_RUN", why=f"arm {arm} not handled here")
            rows.append(out)
    return rows


def expand_arms(spec, pads):
    out = []
    for a in spec.split(","):
        a = a.strip()
        if a == "C1":
            out += ["C1_tile", "C1_whole"]
        elif a == "C2":
            out += [f"C2_pad{p}" for p in pads] + ["C2_rect"]
        elif a:
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="HUMAN_BBOX.csv or UI JSON export")
    ap.add_argument("--review-root", required=True)
    ap.add_argument("--arms", default="C0,C1,C2")
    ap.add_argument("--c2-pad", default="0.25,0.5")
    ap.add_argument("--scorer", choices=["fake", "siglip"], default="fake")
    ap.add_argument("--include", choices=["accepted", "accepted+test"], default="accepted")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pads = [float(x) for x in a.c2_pad.split(",") if x]
    arms = expand_arms(a.arms, pads)
    if any(x in ("C3", "C3r") for x in arms):
        raise SystemExit("C3/C3r are run by sem1_crop_lab.py (RapidOCR); not imported here")
    acc, rej = load_human(a.human, a.review_root)
    want = ("ACCEPTED",) if a.include == "accepted" else ("ACCEPTED", "TEST_ONLY")
    use = {k: v for k, v in acc.items() if v["status"] in want}
    scorer = FakeScorer() if a.scorer == "fake" else SiglipScorer()
    rows = run_arms(use, a.review_root, arms, scorer)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(created=time.strftime("%F %T"), arms=arms, scorer=scorer.name, n_loaded=len(acc), n_rejected=len(rej),
                                   statuses={s: sum(1 for v in acc.values() if v["status"] == s) for s in ("ACCEPTED", "TEST_ONLY", "EXCLUDED_NOT_HUMAN_VERIFIED")},
                                   rejected=[dict(key=r.get("box_id"), item=r.get("item"), reject=r["reject"]) for r in rej], rows=rows), indent=1, default=str))
    print(json.dumps(dict(arms=arms, loaded=len(acc), used=len(use), rejected=len(rej), rows=len(rows), scorer=scorer.name)))


if __name__ == "__main__":
    main()
