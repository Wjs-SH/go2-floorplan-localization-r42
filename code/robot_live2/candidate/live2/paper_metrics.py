#!/usr/bin/env python3
"""paper_metrics.py (R35): one row per experiment for the paper, from the archived run. Read only on the run; writes evaluation/<exp>/PAPER_METRICS.json. Nothing here feeds E2..E7.
Rules: no fix -> no time-to-fix. A position / heading error at the fix exists only with an independent GT mark near the fix time. The start-pose error is a different quantity and is named so."""
import json
import statistics as st
import time
from pathlib import Path

GT_FIX_MAX_DT_S = 30.0


def _j(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _jl(p):
    out = []
    try:
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _pct(v, q):
    v = sorted(x for x in v if x is not None)
    return None if not v else v[min(len(v) - 1, int(q * len(v)))]


def build(rd, ed, e8=None, tm=None):
    rd, ed = Path(rd), Path(ed)
    x = _j(rd / "experiment.json") or {}
    st_ = rd / "state"
    t0 = x.get("created_wall")
    rel = lambda w: None if w is None or t0 is None else round(float(w) - float(t0), 1)
    e8 = e8 or _j(ed / "E8_SUMMARY.json") or {}
    tm = tm or _j(ed / "time_metrics.json") or {}
    e2 = _j(st_ / "e2/e2_state.json") or {}
    passes = _jl(st_ / "e2/e2_passes.jsonl")
    e3h = _jl(st_ / "e3_history.jsonl")
    e4h = _jl(st_ / "e4_history.jsonl")
    ms = (_j(st_ / "stage_milestones.json") or {}).get("first") or {}
    obs8 = _jl(st_ / "e8_observations.jsonl")
    prec = _jl(st_ / "e6_precision_results.jsonl")
    e7 = _j(st_ / "e7_state.json") or {}
    shadow = _jl(st_ / "e7_semantic_shadow.jsonl")
    early = _jl(st_ / "e7_shadow_early.jsonl")
    gts = _jl(ed / "gt.jsonl")
    fix = e7.get("global_decision") == "GLOBAL_CONFIRMED" and t0 is not None and float(e7.get("written_wall") or 0) >= float(t0) and not x.get("replay")
    t_fix = e7.get("written_wall") if fix else None
    lag = [float(h["wall"]) - float(h["last_kf"]["t"]) for h in e3h if h.get("last_kf") and h["last_kf"].get("t")]
    pend = [h.get("kf_pending") for h in e3h]
    ambig = [e for e in e4h if e.get("state") == "AMBIGUOUS"]
    gaps = [(e.get("metrics") or {}).get("partial_gap") for e in e4h]
    read = [o for o in obs8 if o.get("event") == "READ"]
    n_read_ok = sum(1 for o in read if o.get("ocr_lines"))
    aim = [r for r in prec if r.get("status")]
    aim_ok = [r for r in aim if r.get("status") in ("CAPTURED", "CAPTURE_STORED")]
    start = next((g for g in gts if g.get("kind") == "start"), None) or (gts[0] if gts else None)
    gt_fix = None
    if t_fix is not None:
        near = [g for g in gts if g.get("kind") != "start" and g.get("wall") is not None and abs(float(g["wall"]) - float(t_fix)) <= GT_FIX_MAX_DT_S]
        gt_fix = min(near, key=lambda g: abs(float(g["wall"]) - float(t_fix))) if near else None
    fo = _j(st_ / "fix_overlay.json") or {}
    err = dict(position_error_m=None, heading_error_deg=None, basis="no independent GT mark near the fix" if fix else "no fix")
    if gt_fix and fo.get("pose_px") and fo.get("ppm") and gt_fix.get("px") is not None:
        d = ((fo["pose_px"][0] - gt_fix["px"]) ** 2 + (fo["pose_px"][1] - gt_fix["py"]) ** 2) ** 0.5 / float(fo["ppm"])
        err = dict(position_error_m=round(d, 2), basis=f"GT mark {gt_fix.get('gt_id')} at {rel(gt_fix.get('wall'))} s", heading_error_deg=None if gt_fix.get("yaw_deg") is None or fo.get("heading_deg") is None else round((float(fo["heading_deg"]) - float(gt_fix["yaw_deg"]) + 180) % 360 - 180, 1))
    last_ev = next((p for p in reversed(_jl(st_ / "e8_pairs.jsonl")) if p.get("verdict")), {})
    out = dict(kind="LIVE2_PAPER_METRICS", written=time.strftime("%Y-%m-%d %H:%M:%S"), used_by_inference=False, run_id=x.get("exp_id"), class_="FIELD_FIX" if fix else "NO_FIX", start_time=x.get("created"), end_time=x.get("ended"), duration_s=rel(x.get("ended_wall")), end_reason=x.get("end_reason"),
               zone_label=x.get("zone_label"), building=x.get("building"), start_gt=None if not start else {k: start.get(k) for k in ("floor", "px", "py", "yaw_deg", "gt_id")},
               e2_first_candidate_s=(e8.get("e2") or {}).get("first_candidate_s"), provisional_s=rel((ms.get("PROVISIONAL") or {}).get("wall")), e2_floors_done=len(e2.get("floors_done") or []), e2_floors_total=e2.get("floors_total"), e2_all_floors_s=(e8.get("e2") or {}).get("all_floors_exact_s"), e2_passes_regional=len(e2.get("regional") or []),
               e2_restarts=e2.get("restarts") or 0, e3_kf_pending=dict(p50=_pct(pend, .5), p95=_pct(pend, .95), max=_pct(pend, 1.0)), e3_kf_lag_s=dict(p50=None if not lag else round(st.median(lag), 1), p95=None if not lag else round(_pct(lag, .95), 1), max=None if not lag else round(max(lag), 1)),
               e3_leader_lag_s=dict(p50=_pct([h.get("leader_kf_lag_s") for h in e3h], .5), max=_pct([h.get("leader_kf_lag_s") for h in e3h], 1.0)), e4_final_state=(e4h[-1].get("state") if e4h else None), e4_ambiguous_samples=len(ambig), e4_min_partial_gap=_pct(gaps, 0.0),
               e5_stop_requests=sum(1 for _ in _jl(rd / "archive/observe_run/live2_stop_requests.jsonl")), e5_inference_scenes=sum(1 for _ in open(rd / "archive/observe_run/scenes.jsonl")) if (rd / "archive/observe_run/scenes.jsonl").exists() else None,
               pantilt_attempts=len(aim), pantilt_captured=len(aim_ok), pantilt_success_rate=None if not aim else round(len(aim_ok) / len(aim), 3), ocr_reads=len(read), ocr_with_text=n_read_ok, ocr_success_rate=None if not read else round(n_read_ok / len(read), 3),
               e8_valid_independent=last_ev.get("n_valid_independent"), e8_room_readings=last_ev.get("n_room_readings"), e8_last_verdict=last_ev.get("verdict"), e8_candidates=[c.get("id") for c in last_ev.get("candidates") or []], e8_residuals_m=[c.get("residuals_m") for c in last_ev.get("candidates") or []],
               e7_decision=e7.get("global_decision"), e7_structural_class=e7.get("structural_class"), e7_why=e7.get("why"), e7_best=(e7.get("e7") or {}).get("best"), e7_gap_m=(e7.get("e7") or {}).get("gap_m"), e7_rivals=[dict(id=r.get("candidate_id"), floor=r.get("floor_id"), score=r.get("alignment_score"), convergence=r.get("convergence")) for r in (e7.get("e7") or {}).get("results") or []],
               shadow_verdicts=[s.get("verdict") for s in shadow], shadow_last_blocked_by=(shadow[-1].get("blocked_by") if shadow else None), early_e7_windows=len(early), early_e7_leader_best=sum(1 for w in early if w.get("leader_best")),
               global_fix=bool(fix), time_to_fix_raw_s=tm.get("TTF_raw_s") if fix else None, time_to_fix_normalized_s=tm.get("TTF_normalized_s") if fix else None, fix_error=err, start_backprojected_note="The E8_SUMMARY start error is a back-projection to the start mark: not the error at the fix.",
               failure_reason=None if fix else (e7.get("why") or last_ev.get("why") or x.get("end_reason")))
    (ed / "PAPER_METRICS.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out
