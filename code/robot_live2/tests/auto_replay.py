#!/usr/bin/env python3
"""R39 OFFLINE: run auto_fsm over a real run's recorded E3 history + E2 passes + odometry. e3_history keeps the top 6 hypotheses only, so rival floors below rank 6 are missing: a replay can only under-report comparison, which leans to UNDETERMINED/AMBIGUOUS.
usage: auto_replay.py RUN [RUN..]"""
import json, math, sys
from pathlib import Path
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate/live2"))
import auto_fsm as A
def jl(p):
    try:
        return [json.loads(l) for l in open(p)]
    except OSError:
        return []
def replay(run):
    X = OPR / "experiments" / run; S = X / "state"; t0 = json.loads((X / "experiment.json").read_text())["created_wall"]
    H = jl(S / "e3_history.jsonl"); P = jl(S / "e2/e2_passes.jsonl"); LG = jl(S / "e7_long.jsonl")
    E4H = jl(S / "e4_history.jsonl")
    tr = [json.loads(l) for l in open(X / "archive/field_eval/odom_trace.jsonl")] if (X / "archive/field_eval/odom_trace.jsonl").exists() else []
    F = A.AutoFSM(); out = []; orig = F._log
    F._log = lambda now, kind, a, b, why, ass, pm: out.append(dict(t=round(now - t0, 1), kind=kind, frm=a, to=b, why=why, gap=ass.get("gap"), lead=ass.get("leader_floor"), riv=ass.get("rival_floor")))
    F.state = None
    for r in H:
        now = r["wall"]
        coarse = sorted({p["floor"] for p in P if p.get("kind") == "coarse_floor_done" and p["wall"] <= now and p.get("pass_id") == 1})
        exact = sorted({p["floor"] for p in P if p.get("kind") == "floor_done" and p["wall"] <= now and p.get("pass_id") == 1})
        pts = [q for q in tr if q.get("wall", 0) <= now] if tr else []
        path = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(pts, pts[1:])) if pts and "x" in pts[0] else (now - t0) * 0.3
        L = next((l for l in reversed(LG) if l["wall"] <= now), None)
        e3 = dict(hypotheses=[dict(h, id=h["id"]) for h in r.get("top") or []])
        m4 = next((x["metrics"] for x in reversed(E4H) if x["wall"] <= now), {})      # recent merit is recorded for E4's best only; ppm is not in the history
        for h in e3["hypotheses"]:
            if h["id"] == m4.get("best_id"):
                h["recent"] = m4.get("best_recent")
        F.tick(e2=dict(floors_total=9, floors_done=exact, coarse=dict(floors_done=coarse)), e3=e3, long_d=L, path_m=path, now=now)
    return dict(run=run, n=len(H), t_end=round(H[-1]["wall"] - t0, 1) if H else None, transitions=[o for o in out if o["kind"] == "CLASS"], final=F.cls)
if __name__ == "__main__":
    for run in sys.argv[1:]:
        r = replay(run); print(run, "end", r["t_end"], "final", r["final"])
        for o in r["transitions"]:
            print("  ", o["t"], o["frm"], "->", o["to"], "lead", o["lead"], "riv", o["riv"], "gap", o["gap"], "|", (o["why"] or "")[:70])
