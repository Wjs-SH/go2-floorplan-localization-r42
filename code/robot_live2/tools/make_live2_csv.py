#!/usr/bin/env python3
"""Build the LIVE2 result CSVs from what the processes actually wrote (no value is invented).

  make_live2_csv.py --run NAME:MODE:STATE_DIR[:LOG_GLOB] ... --out REPORT_DIR [--truth-floor NAME=F]

MODE is REPLAY or LIVE (copied into every row). --truth-floor is EVALUATION ONLY (recorded dataset label), never an input
of E2/E3/E4; it fills the 'truth_floor' / 'top1_floor_correct' columns so wrong candidates are not hidden.
"""
import argparse
import csv
import glob
import json
from pathlib import Path


def jl(p):
    out = []
    try:
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def write(path, rows, cols=None):
    cols = cols or sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    return len(rows)


E2C = ["run", "mode", "pass_id", "t_now", "floor", "rank_in_floor", "rank_building", "px", "py", "x_m", "y_m", "phi_deg", "ppm",
       "beta_deg", "mirror", "merit", "twoway", "wall_pass", "wall_cost_m", "T2", "T3", "T4", "X_cross_m", "inlier_10cm",
       "is_student_best", "map_version", "odom_session", "fit_t", "window_s", "created_wall", "status", "truth_floor", "floor_correct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth-floor", action="append", default=[])
    a = ap.parse_args()
    truth = dict(t.split("=") for t in a.truth_floor)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    e2r, e3r, e4r, resr = [], [], [], []
    for spec in a.run:
        name, mode, sd = spec.split(":")[:3]
        logs = spec.split(":")[3] if spec.count(":") >= 3 else None
        sd = Path(sd)
        tf = truth.get(name)
        # ---- E2: per-floor candidates from the pass log, else the final state (older runs)
        seen = set()
        cands = []
        for ev in jl(sd / "e2" / "e2_passes.jsonl"):
            if ev.get("kind") == "floor_done" and ev.get("candidates"):
                for c in ev["candidates"]:
                    cands.append(dict(c, pass_id=ev.get("pass_id"), t_now=ev.get("t_now")))
        st = rj(sd / "e2" / "e2_state.json") or {}
        if not cands:
            for c in st.get("candidates") or []:
                cands.append(dict(c, t_now=st.get("t_now")))
        by_pass = {}
        for c in cands:
            k = (c.get("pass_id"), c["floor"], c["px"], c["py"], c["phi_deg"])
            if k in seen:
                continue
            seen.add(k)
            by_pass.setdefault(c.get("pass_id"), []).append(c)
        for pid, cs in by_pass.items():
            order = sorted(cs, key=lambda c: ((c.get("wall_pass") or 0) > 0, -(c["twoway"] if c["twoway"] is not None else -1e9), c["merit"]))
            fl_rank = {}
            for i, c in enumerate(order):
                fl_rank[c["floor"]] = fl_rank.get(c["floor"], 0) + 1
                e2r.append(dict(c, run=name, mode=mode, rank_building=i + 1, rank_in_floor=fl_rank[c["floor"]], truth_floor=tf,
                                floor_correct=None if tf is None else int(str(c["floor"]) == str(tf))))
        # ---- E3
        for h in jl(sd / "e3_history.jsonl"):
            top = h.get("top") or []
            row = dict(run=name, mode=mode, wall=h["wall"], kf_sid=(h.get("last_kf") or {}).get("sid"), kf_t=(h.get("last_kf") or {}).get("t"),
                       kf_eval_n=(h.get("last_kf") or {}).get("n_eval"), kf_ms=(h.get("last_kf") or {}).get("ms"),
                       alive=h.get("alive"), active=h.get("active"), suspended=h.get("suspended"), dropped=h.get("dropped"),
                       kf_pending=h.get("kf_pending"), kf_ms_p95=(h.get("timing") or {}).get("kf_ms_p95"), truth_floor=tf)
            for i in range(3):
                t = top[i] if i < len(top) else {}
                row.update({f"top{i+1}_id": t.get("id"), f"top{i+1}_floor": t.get("floor"), f"top{i+1}_score": t.get("score"),
                            f"top{i+1}_status": t.get("status"), f"top{i+1}_n_kf": t.get("n_kf")})
            row["top1_floor_correct"] = None if tf is None or not top else int(str(top[0]["floor"]) == str(tf))
            e3r.append(row)
        for ev in jl(sd / "e3_events.jsonl"):
            e3r.append(dict(run=name, mode=mode, wall=ev["wall"], event=ev["kind"], event_detail={k: v for k, v in ev.items() if k not in ("wall", "kind")}))
        # ---- E4
        for h in jl(sd / "e4_history.jsonl"):
            m = h.get("metrics") or {}
            tt = h.get("top_target") or {}
            e4r.append(dict(run=name, mode=mode, wall=h["wall"], state=h["state"], reasons=" | ".join(h.get("reasons") or []),
                            best_id=m.get("best_id"), best_floor=m.get("best_floor"), best_score=m.get("best_score"), gap=m.get("gap"),
                            competitor_floor=m.get("competitor_floor"), separation=m.get("separation"),
                            floors_within_gap=m.get("floors_within_gap_amb"), floors_done=m.get("floors_done"), floors_total=m.get("floors_total"),
                            n_active=m.get("n_active"), path_since_fit_m=m.get("path_since_best_fit_m"), best_n_kf=m.get("best_n_kf"),
                            odom_age_s=m.get("odom_last_rx_age_s"), feed_age_s=m.get("feed_last_sample_age_s"), priority=h.get("priority"),
                            n_targets=h.get("n_targets"), top_target=None if not tt else f"{tt.get('floor')}F {tt.get('label')} {tt.get('observability')}",
                            e4_ms=(h.get("timing_ms") or {}).get("e4"), e3_tick_ms=(h.get("timing_ms") or {}).get("e3_tick"), truth_floor=tf,
                            best_floor_correct=None if tf is None or m.get("best_floor") is None else int(str(m["best_floor"]) == str(tf))))
        # ---- resources + latency
        for r in jl(sd / "res_history.jsonl"):
            resr.append(dict(run=name, mode=mode, source="e34_sampler", **r))
        for ev in jl(sd / "e2" / "e2_passes.jsonl"):
            if ev.get("kind") in ("floor_done", "pass_done", "held", "child_stopped_memory", "floor_failed"):
                resr.append(dict(run=name, mode=mode, source="e2_" + ev["kind"], wall=ev["wall"], floor=ev.get("floor"), sec=ev.get("sec"),
                                 fit_s=ev.get("fit_s"), open_s=ev.get("open_s"), score_s=ev.get("score_s"), rss_peak_mb=ev.get("rss_peak_mb"),
                                 why=ev.get("why") or ev.get("error")))
        for lp in sorted(glob.glob(logs)) if logs else []:
            mod = Path(lp).name.split("_")[0]
            for k, s in enumerate(jl(lp)):
                if s.get("kind") == "SAMPLE" and k % 4 == 0:
                    resr.append(dict(run=name, mode=mode, source=f"supervisor_{mod}", wall=s.get("t"), temp=s.get("temp"),
                                     mem_avail_mb=s.get("avail"), module_rss_mb=s.get("own_rss_mb")))
    n = dict(e2=write(out / "E2_TOPK_RESULTS.csv", e2r, E2C), e3=write(out / "E3_INCREMENTAL_RESULTS.csv", e3r),
             e4=write(out / "E4_AMBIGUITY_RESULTS.csv", e4r), res=write(out / "RESOURCE_AND_LATENCY.csv", resr))
    print(json.dumps(n))


if __name__ == "__main__":
    main()
