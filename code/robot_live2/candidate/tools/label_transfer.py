#!/usr/bin/env python3
"""Transfer spot labels between runs of the same recording by world position (odom frame of the bag).

Spot ids are run-local (they depend on the candidate stream). A labelled spot is represented by the mean xyz of its
observations; a spot of the new run inherits the label of the nearest labelled spot within --radius metres whose
observation time windows overlap (|t| gap <= --max-dt s). Unmatched spots that matter (max sim >= floor, or any
legacy-pass / stop) are listed so they can be labelled manually (rescore.py spots ... renders them).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def spot_table(run):
    acc = defaultdict(lambda: dict(xyz=[], t=[], sim=[]))
    for l in open(Path(run) / "scenes.jsonl"):
        s = json.loads(l)
        for r in s["regions"]:
            if r["spot"] is None or r["xyz"] is None:
                continue
            a = acc[r["spot"]]
            a["xyz"].append(r["xyz"][:2]); a["t"].append(s["t"]); a["sim"].append(r["sim"])
    return {k: dict(xy=np.mean(v["xyz"], 0), t0=min(v["t"]), t1=max(v["t"]), max_sim=max(v["sim"])) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", required=True)
    ap.add_argument("--dst-run", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.8)
    ap.add_argument("--max-dt", type=float, default=20.0)
    ap.add_argument("--floor", type=float, default=0.078)
    a = ap.parse_args()
    lab = json.loads(Path(a.labels).read_text())
    src, dst = spot_table(a.src_run), spot_table(a.dst_run)
    labelled = {k: src[int(key.split("spot")[1])] for key, k in ((key, key) for key in lab if key.startswith(a.ds + "|"))
                if int(key.split("spot")[1]) in src}
    out, need = {}, []
    for k, v in dst.items():
        best, bd = None, a.radius
        for key, s in labelled.items():
            d = float(np.linalg.norm(v["xy"] - s["xy"]))
            gap = max(0.0, max(v["t0"], s["t0"]) - min(v["t1"], s["t1"]))
            if d < bd and gap <= a.max_dt:
                best, bd = key, d
        if best is not None:
            out[f"{a.ds}|spot{k}"] = lab[best]
        elif v["max_sim"] >= a.floor:
            need.append(k)
    keep = {k: v for k, v in lab.items() if not k.startswith(a.ds + "|")}
    keep.update(out)
    Path(a.out).write_text(json.dumps(keep, indent=1))
    print(json.dumps(dict(transferred=len(out), need_manual=sorted(need)), indent=1))


if __name__ == "__main__":
    main()
