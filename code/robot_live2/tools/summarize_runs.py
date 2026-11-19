#!/usr/bin/env python3
"""Per-run summary of the REPLAY / LIVE histories (evaluation only; the truth floor never feeds E2/E3/E4).

  summarize_runs.py NAME:STATE_DIR:TRUTH_FLOOR ... > summary.json
"""
import json
import sys
from collections import Counter
from pathlib import Path


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return []


def main():
    out = {}
    for spec in sys.argv[1:]:
        name, sd, truth = spec.split(":")
        sd, truth = Path(sd), int(truth)
        e4 = jl(sd / "e4_history.jsonl")
        e3 = jl(sd / "e3_history.jsonl")
        e2ev = jl(sd / "e2" / "e2_passes.jsonl")
        st = json.loads((sd / "e2" / "e2_state.json").read_text()) if (sd / "e2" / "e2_state.json").exists() else {}
        # time share of each E4 state (history rows are written on change or on every E3 step)
        dur = Counter()
        for a, b in zip(e4, e4[1:]):
            dur[a["state"]] += b["wall"] - a["wall"]
        tot = sum(dur.values()) or 1.0
        complete = [r for r in e4 if (r.get("metrics") or {}).get("search_complete") and r["metrics"].get("best_floor") is not None]
        lead_ok = sum(1 for r in complete if r["metrics"]["best_floor"] == truth)
        wrong_conf = [r for r in e4 if r["state"] == "CONFIDENT_GEOMETRY" and (r.get("metrics") or {}).get("best_floor") != truth]
        right_conf = [r for r in e4 if r["state"] == "CONFIDENT_GEOMETRY" and (r.get("metrics") or {}).get("best_floor") == truth]
        # rank of the best truth-floor hypothesis among ACTIVE+SUSPENDED in the E3 history top lists (top 6 only)
        ranks = []
        for h in e3:
            fl = [t["floor"] for t in h.get("top") or []]
            ranks.append(fl.index(truth) + 1 if truth in fl else None)
        in_top6 = sum(1 for r in ranks if r is not None)
        # student floor verdict on the first pass window (per-floor best ranked by the student's twoway + wall_pass rule)
        best = {}
        for ev in e2ev:
            if ev.get("kind") == "floor_done":
                for c in ev.get("candidates") or []:
                    if c.get("is_student_best") and ev.get("pass_id") == 1:
                        best[c["floor"]] = c
        if not best:
            for c in st.get("candidates") or []:
                if c.get("is_student_best"):
                    best[c["floor"]] = c
        order = sorted(best, key=lambda f: (((best[f].get("wall_pass") or 0) > 0), -best[f]["twoway"])) if best else []
        passes = [dict(pass_id=e["pass_id"], why=e.get("why"), sec=None) for e in e2ev if e.get("kind") == "pass_start"]
        for p in passes:
            done = [e for e in e2ev if e.get("kind") == "pass_done" and e["pass_id"] == p["pass_id"]]
            p["sec"] = done[0]["sec"] if done else None
        fl = [e for e in e2ev if e.get("kind") == "floor_done"]
        out[name] = dict(truth_floor=truth, e4_time_share={k: round(v / tot, 3) for k, v in dur.items()}, e4_rows=len(e4),
                         after_full_search_rows=len(complete), leader_is_truth_share=round(lead_ok / len(complete), 3) if complete else None,
                         wrong_confident_rows=len(wrong_conf), right_confident_rows=len(right_conf),
                         first_wrong_confident=None if not wrong_conf else dict(wall=wrong_conf[0]["wall"], floor=wrong_conf[0]["metrics"]["best_floor"],
                                                                                 gap=wrong_conf[0]["metrics"].get("gap")),
                         truth_in_e3_top6_share=round(in_top6 / len(ranks), 3) if ranks else None,
                         student_pass1_floor_order=order, student_pass1_verdict=order[0] if order else None,
                         student_pass1_truth_rank=(order.index(truth) + 1) if truth in order else None,
                         e2_passes=passes, e2_floor_sec=[round(e["sec"], 1) for e in fl if e.get("sec") is not None][:24],
                         e3_kf_ms_p95_last=(e3[-1].get("timing") or {}).get("kf_ms_p95") if e3 else None,
                         final=None if not e4 else dict(state=e4[-1]["state"], best_floor=(e4[-1].get("metrics") or {}).get("best_floor")))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
