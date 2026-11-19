#!/usr/bin/env python3
"""scale_survey.py (R35): px/m of one floor from measured real lengths. Changes nothing: it reports, the scale file is edited only after a separate decision.
Input CSV: name,x1,y1,x2,y2,measured_m[,method]   (x,y = structure-map px of the same corners that were measured with tape or laser).
usage: scale_survey.py --floor 4 --csv segments.csv [--out result.json]"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

SCALE_FILE = Path(__file__).resolve().with_name("semantic_scale_v1.json")
MIN_SEGMENTS, MAX_SPREAD_REL, MIN_LEN_M = 3, 0.02, 3.0


def analyse(rows, floor, scale_file=SCALE_FILE):
    segs, problems = [], []
    for r in rows:
        name = r.get("name") or f"seg{len(segs) + 1}"
        try:
            px = math.hypot(float(r["x2"]) - float(r["x1"]), float(r["y2"]) - float(r["y1"])); m = float(r["measured_m"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{name}: unreadable row"); continue
        if m <= 0 or px <= 0:
            problems.append(f"{name}: zero or negative length"); continue
        if m < MIN_LEN_M:
            problems.append(f"{name}: {m} m is short (< {MIN_LEN_M} m): a 1 px or 1 cm error matters more")
        segs.append(dict(name=name, px=round(px, 1), measured_m=m, px_per_m=round(px / m, 3), method=r.get("method")))
    file_val = None
    try:
        file_val = ((json.loads(Path(scale_file).read_text()).get("floors") or {}).get(str(floor)) or {}).get("px_per_m")
    except (OSError, ValueError):
        pass
    if not segs:
        return dict(floor=floor, status="NO_DATA", segments=[], problems=problems, scale_file_px_per_m=file_val)
    v = [s["px_per_m"] for s in segs]
    w = sum(s["px"] for s in segs) / sum(s["measured_m"] for s in segs)
    spread = (max(v) - min(v)) / w
    for s in segs:
        s["diff_from_combined_pct"] = round(100 * (s["px_per_m"] / w - 1), 2)
    status = "NEEDS_MORE_SEGMENTS" if len(segs) < MIN_SEGMENTS else "INCONSISTENT" if spread > MAX_SPREAD_REL else "CONSISTENT"
    return dict(floor=floor, status=status, n=len(segs), combined_px_per_m=round(w, 3), median_px_per_m=round(st.median(v), 3), spread_pct=round(100 * spread, 2), segments=segs, problems=problems, scale_file_px_per_m=file_val,
                diff_from_scale_file_pct=None if not file_val else round(100 * (w / file_val - 1), 2), rule=f">= {MIN_SEGMENTS} segments, spread <= {MAX_SPREAD_REL:.0%}",
                note="Report only. No map or scale file was changed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, required=True); ap.add_argument("--csv", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    res = analyse(list(csv.DictReader(open(a.csv))), a.floor)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
