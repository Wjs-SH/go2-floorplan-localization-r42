#!/usr/bin/env python3
"""SEM1 crop-budget comparison on STORED per-region scores (STORED_CPU simulation; no model run).

Why a simulation is valid here: a budget only decides WHICH whole regions are scored in a scene; the score of a region
that IS scored does not depend on the others (same crop, same model). Deferred regions simply give no fresh evidence
in that scene (typed DEFERRED, no p90 sample, no persistence vote). Spot association is taken from the log.
Policies per scene (regions = the run's logged regions, cost = tiles + 1 context upper bound):
  fixed_<N>  : e5rt.budget.plan_budget (waited first, cheap first, left/right alternation)
  value_<N>  : runtime-observable value order: spots near the threshold that still need confirmation, never-scored
               (new) spots, waiting time, then cost; keeps >= 1 region of exploration reserve for never-scored spots
Temporal confirmation: legacy mean3 (msim>=thr, sim>=floor, >=2 scenes) with thr = running p90 of FRESH sims only
(legacy_p90) or the DEV-fixed 0.092 (fixed_thr; PROVISIONAL, not activatable at runtime).
Metrics per provisional labelled spot: verified?, first-confirmation delay from the spot's first observation, FP spots,
crops/scene and GPU-ms estimate (13.5 ms/crop measured E5-RT1). Labels: Claude v3 (provisional) of the v3 runs.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "sci_visual_observation"))
from e5rt.budget import plan_budget, n_tiles, select_value, semantic_confirmed_rule  # noqa: E402

LAB = Path("/home/unitree/sci_ws/datasets/labels/v3")
EXP = Path("/home/unitree/sci_ws/experiments")
DS = {"dev_S1F_v3": ("S-1F", LAB / "labels_S-1F_v3.json"), "test_S6F_v3": ("S-6F", LAB / "labels_S-6F_v3.json")}
MS_PER_CROP = 13.5


def load(run):
    out = []
    for l in open(EXP / run / "scenes.jsonl"):
        r = json.loads(l)
        if r.get("regions"):
            out.append(dict(t=float(r["t"]), regs=[dict(box=g["box"], sim=float(g["sim"]), spot=g.get("spot")) for g in r["regions"]]))
    return out


def candidates(regs, state, now, cost):
    """Causal candidate list for e5rt.budget.select_value (identical construction to the runtime wrapper):
    only already-scored evidence (state = FRESH sims up to the previous scene), never future scores or labels."""
    out = []
    for i, g in enumerate(regs):
        s = state.get(g["spot"]) if g["spot"] is not None else None
        out.append(dict(idx=i, cost=cost[i], new=s is None, sims=list(s["sims"]) if s else [], waited_s=(now - s["last_fresh"]) if s else 0.0,
                        semantic_confirmed=bool(s and s.get("semantic_confirmed")), observed=False))
    return out


def simulate(scenes, policy, budget, fixed_thr=None, floor=0.078, warm=50):
    seen, state, prev_def = [], {}, []
    conf_t, first_t, crops, obs_t, thr_trace = {}, {}, [], {}, []
    for sc in scenes:
        regs, now = sc["regs"], sc["t"]
        boxes = [g["box"] for g in regs]
        cost = [n_tiles(b) + 1 for b in boxes]
        thr0 = fixed_thr if fixed_thr is not None else (float(np.quantile(seen, 0.9)) if len(seen) >= warm else math.inf)
        if budget is None or sum(cost) <= budget:
            send = list(range(len(regs)))
        elif policy == "fixed":
            send, _, _ = plan_budget(boxes, budget, prev_def, width=640)
        else:
            send, _, _ = select_value(candidates(regs, state, now, cost), budget, thr0)
        prev_def = [boxes[i] for i in range(len(regs)) if i not in send]
        crops.append(sum(cost[i] for i in send))
        for i in send:
            seen.append(regs[i]["sim"])
        thr = fixed_thr if fixed_thr is not None else (float(np.quantile(seen, 0.9)) if len(seen) >= warm else math.inf)
        thr_trace.append(thr)
        for i, g in enumerate(regs):
            sp = g["spot"]
            if sp is None:
                continue
            first_t.setdefault(sp, now)
            obs_t.setdefault(sp, []).append(now)
            if i not in send:
                continue
            s = state.setdefault(sp, dict(sims=[], scenes=set(), last_fresh=now))
            s["sims"].append(g["sim"]); s["scenes"].add(now); s["last_fresh"] = now
            if not s.get("semantic_confirmed") and semantic_confirmed_rule(s["sims"], len(s["scenes"]), thr, floor):
                s["semantic_confirmed"] = True
                conf_t.setdefault(sp, now)
    return conf_t, first_t, crops, obs_t, thr_trace


def visits(times, gap=10.0):
    v, cur = [], [times[0]]
    for t in times[1:]:
        if t - cur[-1] > gap:
            v.append(cur)
            cur = [t]
        else:
            cur.append(t)
    v.append(cur)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for run, (fl, lp) in DS.items():
        lab = json.loads(lp.read_text())
        assert Path(lab["run"]).name == run, "LABEL_RUN_MISMATCH"
        L = {int(k.split("spot")[1]): v for k, v in lab["labels"].items()}
        scenes = load(run)
        for thr_mode, fthr in (("legacy_p90", None), ("fixed_0.092_DIAGNOSTIC", 0.092)):
            for policy, budget in (("full", 40), ("fixed", 16), ("value", 16)):
                conf, first, crops, obs, thr_tr = simulate(scenes, policy, budget, fthr)
                pos = [s for s, v in L.items() if v in ("POS_ROOM", "POS_NAME")]
                neg = [s for s, v in L.items() if v.startswith("NEG")]
                tp = [s for s in pos if s in conf]
                delays = sorted(round(conf[s] - first[s], 1) for s in tp if s in first)
                vis_tot = vis_hit = 0
                for s in pos:
                    for vv in visits(sorted(obs.get(s, []))) if obs.get(s) else []:
                        vis_tot += 1
                        vis_hit += int(s in conf and vv[0] <= conf[s] <= vv[-1])
                rows.append(dict(run=run, floor=fl, thr_mode=thr_mode, policy=policy, budget=budget, n_pos=len(pos), n_neg=len(neg),
                                 tp=len(tp), fn=len(pos) - len(tp), fp=sum(1 for s in neg if s in conf),
                                 confirmed_unlabelled_or_excluded=sum(1 for s in conf if s not in pos and s not in neg),
                                 n_spots_seen=len(first), missed_positive_spots=sorted(s for s in pos if s not in conf),
                                 first_confirm_delay_s=dict(n=len(delays), p50=float(np.median(delays)) if delays else None,
                                                            p95=float(np.percentile(delays, 95)) if delays else None, max=max(delays) if delays else None),
                                 positive_visits=vis_tot, positive_visits_confirmed_within=vis_hit,
                                 crops_per_scene=dict(p50=float(np.median(crops)), p95=float(np.percentile(crops, 95))),
                                 thr_final=None if not np.isfinite(thr_tr[-1]) else round(thr_tr[-1], 4),
                                 evidence="STORED_CPU simulation; shared e5rt.budget.select_value; logged spot association; semantic layer only (L4/L5 NOT applied); labels PROVISIONAL"))
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r["run"], r["thr_mode"][:10], r["policy"], r["budget"], "tp", r["tp"], "fn", r["fn"], "fp", r["fp"], "unlab", r["confirmed_unlabelled_or_excluded"],
              "delay", r["first_confirm_delay_s"], "visits", f"{r['positive_visits_confirmed_within']}/{r['positive_visits']}", "crops", r["crops_per_scene"]["p50"], "thr", r["thr_final"])


if __name__ == "__main__":
    main()
