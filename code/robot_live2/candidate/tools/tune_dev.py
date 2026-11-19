#!/usr/bin/env python3
"""Grid search of interpretable SCI policy parameters on the DEVELOPMENT run only (S-1F). The held-out test run is
never touched here. Objective: maximise stop precision of stage A3 subject to positive-spot recall >= A0 recall on dev;
ties -> more positives, then fewer stops. Prints the top configs and writes the best to config/e5_policy_tuned.json."""
import itertools
import json
import sys
from pathlib import Path

WS = Path.home() / "sci_ws"
sys.path.insert(0, str(WS / "tools"))
sys.path.insert(0, str(WS / "src" / "sci_visual_observation"))
import policy as P  # noqa: E402
import rescore as RS  # noqa: E402
from features import Geometry  # noqa: E402


def score(stops, labels, ds):
    lab = [labels.get(f"{ds}|spot{s['spot']}", "UNLABELLED") for s in stops]
    nl = sum(1 for l in lab if l not in ("UNLABELLED", "AMBIGUOUS"))
    pos = sum(1 for l in lab if l.startswith("POS"))
    reached = {s["spot"] for s, l in zip(stops, lab) if l.startswith("POS")}
    return dict(stops=len(stops), labelled=nl, pos=pos, prec=pos / nl if nl else 0.0, reached=len(reached))


def main():
    run, ds = WS / "experiments" / "dev_S1F_v3", "S-1F"
    labels = json.loads((WS / "datasets" / "labels" / "v3" / "labels_dev_flat.json").read_text())
    sc, st, _ = RS.load(run)
    geom = Geometry()
    a0 = [s for s in st if s.get("stage", "A0") == "A0"]
    by_t = {s["t"]: s for s in sc}
    a0s = []
    for e in a0:
        sp = next((r["spot"] for r in by_t[e["t"]]["regions"] if abs(r["box"][0] - e["picks"][0][0]) < 0.6 and abs(r["box"][1] - e["picks"][0][1]) < 0.6), None)
        a0s.append(dict(spot=sp))
    base = score(a0s, labels, ds)
    print("A0", base)
    grid = dict(min_pass=[1, 2], min_scenes=[2, 3], neg_set=["none", "poster_board_dir_light", "all"], neg_margin=[0.0, 0.01],
                ctx=["off", "board_poster_dir"], sat=[0.15, 0.3], obl=[65, 70, 75, 80], dmax=[5.0, 6.0], react=[1.0, 1.5])
    NEG = dict(none=[], poster_board_dir_light=["poster", "board", "directory", "light"], all=["poster", "board", "directory", "glare", "light", "elevator"])
    CTX = dict(off=[], board_poster_dir=["poster", "board", "directory"])
    res = []
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        g = dict(zip(keys, vals))
        cfg = P.load_cfg(None)
        cfg["A1"].update(min_pass=g["min_pass"], min_scenes=g["min_scenes"])
        cfg["A2"].update(neg_classes=NEG[g["neg_set"]], neg_margin=g["neg_margin"], ctx_classes=CTX[g["ctx"]],
                         glare_sat_max=g["sat"], glare_spec_max=max(0.3, g["sat"]))
        cfg["A3"].update(obliquity_max_deg=g["obl"], d_max_m=g["dmax"], stop_reaction_s=g["react"])
        stops = RS.replay_policy(sc, cfg, geom)
        r = score(stops.get("A3", []), labels, ds)
        r1 = score(stops.get("A1", []), labels, ds)
        r2 = score(stops.get("A2", []), labels, ds)
        res.append(dict(g=g, A3=r, A2=r2, A1=r1, cfg=cfg))
    ok = [r for r in res if r["A3"]["reached"] >= base["reached"]]
    ok.sort(key=lambda r: (-r["A3"]["prec"], -r["A3"]["reached"], r["A3"]["stops"]))
    print(f"{len(res)} configs, {len(ok)} keep A3 recall >= A0 ({base['reached']} positive spots)")
    for r in ok[:12]:
        print(json.dumps(r["g"]), "A3", r["A3"], "| A2", r["A2"])
    if not ok:
        ok = sorted(res, key=lambda r: (-r["A3"]["reached"], -r["A3"]["prec"]))
        print("no config meets recall constraint; best by recall:", json.dumps(ok[0]["g"]), ok[0]["A3"])
    best = ok[0]
    (WS / "config" / "e5_policy_tuned.json").write_text(json.dumps(dict(best["cfg"], _tuning=dict(
        split="dev S-1F only (experiments/dev_S1F_v3)", grid=grid, chosen=best["g"], dev_A3=best["A3"], dev_A0=base,
        n_configs=len(res), n_meeting_recall=len([r for r in res if r["A3"]["reached"] >= base["reached"]]))), indent=1))
    print("wrote config/e5_policy_tuned.json")


if __name__ == "__main__":
    main()
