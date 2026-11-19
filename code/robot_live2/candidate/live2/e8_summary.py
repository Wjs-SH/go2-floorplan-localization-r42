"""e8_summary.py — per-run E2..E8 result summary from the ARCHIVED experiment (2026-10-30). Read-only on the run; writes
evaluation/<exp>/E8_SUMMARY.json + E8_SUMMARY.md and returns the dict (console.evaluate_experiment puts it under "e8").
Nothing here is fed back to E2..E7; GT (start mark) is used only for the error lines.
result_class:
  FIELD_FIX           a GLOBAL_CONFIRMED E7 record of a live (non-replay) session
  OFFLINE_FIX         the same from a replay session (never reported as a field E8 success)
  NO_FIX              no GLOBAL_CONFIRMED record (partial stages listed under 'stages')
"""
import json
import math
import time
from pathlib import Path


def _j(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _jl(p, n=None):
    try:
        L = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []
    return L[-n:] if n else L


def _rel(w, t0):
    return None if w is None or t0 is None else round(float(w) - float(t0), 1)


def summarize(rd, ed, ev=None):
    rd, ed = Path(rd), Path(ed)
    st = rd / "state"
    X = _j(rd / "experiment.json") or {}
    t0 = X.get("created_wall")
    e2 = _j(st / "e2" / "e2_state.json") or {}
    passes = _jl(st / "e2" / "e2_passes.jsonl")
    e3 = _j(st / "e3_state.json") or {}
    e4h = _jl(st / "e4_history.jsonl")
    e4 = _j(st / "e4_state.json") or {}
    e7 = _j(st / "e7_state.json") or {}
    sp = _j(st / "structure_path.json") or {}
    shadow = _jl(st / "e7_shadow.jsonl")
    rec = _j(st / "semantic_recovery.json")
    pol = _j(st / "e7_policy.json")
    e8p = _jl(st / "e8_pairs.jsonl")
    e8o = _jl(st / "e8_observations.jsonl")
    lio = _jl(st / "lio_series.jsonl")
    sess = str(e2.get("odom_session") or "")
    out = dict(kind="LIVE2_E8_SUMMARY", exp_id=X.get("exp_id"), written=time.strftime("%Y-%m-%d %H:%M:%S"), used_by_inference=False,
               input=X.get("input"), building=X.get("building"), siglip=X.get("siglip"), created=X.get("created"), ended=X.get("ended"),
               end_reason=X.get("end_reason"), replay=sess.startswith("REPLAY"), lidar="Hesai XT16 (FAST-LIO)",
               imu="Go2 body IMU (bridge + L1-frame adapter)" if X.get("input") == "body" else "L1 /utlidar/imu")
    fd = [p for p in passes if p.get("kind") == "floor_done"]
    out["e2"] = dict(first_candidate_s=_rel(min((p["wall"] for p in passes if p.get("kind") in ("coarse_floor_done", "floor_done")),
                                                default=None), t0),
                     all_floors_exact_s=_rel(next((p["wall"] for p in passes if p.get("kind") == "pass_done"), None), t0)
                     if len(e2.get("floors_done") or []) >= (e2.get("floors_total") or 9) else None,
                     floors_done=e2.get("floors_done"), floors_total=e2.get("floors_total"), device=e2.get("device"),
                     n_candidates=len(e2.get("candidates") or []), n_floor_fits=len(fd),
                     regional=[{k: r.get(k) for k in ("request_id", "floor", "status", "n", "sec", "why_failed")} for r in e2.get("regional") or []],
                     map_versions={k: v.get("map_version") for k, v in (e2.get("per_floor") or {}).items()})
    H = sorted([h for h in e3.get("hypotheses") or [] if h.get("status") in ("ACTIVE", "SUSPENDED")], key=lambda h: h.get("score", 9e9))
    out["e3"] = dict(top=[{k: h.get(k) for k in ("id", "floor", "score", "pose_m", "pose_px", "heading_deg", "n_kf", "status")} for h in H[:3]],
                     n_alive=e3.get("n_alive"))
    conf = next((h for h in e4h if h.get("state") == "CONFIDENT_GEOMETRY"), None)
    out["e4"] = dict(first_confident_s=_rel((conf or {}).get("wall"), t0), first_confident_leader=((conf or {}).get("metrics") or {}).get("best_id"),
                     final_state=e4.get("state"), states_seen=sorted({h.get("state") for h in e4h if h.get("state")}))
    out["e5_e6"] = dict(observations=len([o for o in e8o if o.get("event") in ("STORED", "READ")]),
                        rejected=len([o for o in e8o if o.get("event") == "REJECTED"]),
                        readings=[dict(observation_id=o.get("observation_id"), ocr=[t for t, _ in (o.get("ocr_lines") or [])][:6],
                                       vlm=(o.get("vlm") or {}).get("status") if isinstance(o.get("vlm"), dict) else o.get("vlm"))
                                  for o in e8o if o.get("event") == "READ"][:6],
                        pair_verdicts=[p.get("verdict") for p in e8p if p.get("verdict")][-6:])
    out["semantic_recovery"] = rec
    out["e7"] = dict(policy=(pol or {}).get("structure"), global_decision=e7.get("global_decision"), path=e7.get("path") or e7.get("source"),
                     why=e7.get("why"),
                     structure_path={k: sp.get(k) for k in ("leader", "floor", "status", "mode", "why", "evidence", "retro", "t_confident")} if sp else None,
                     shadow_would_confirm=[{k: s.get(k) for k in ("leader", "floor", "why")} for s in shadow][-3:])
    fix = e7.get("global_decision") == "GLOBAL_CONFIRMED"
    out["result_class"] = ("OFFLINE_FIX" if out["replay"] else "FIELD_FIX") if fix else "NO_FIX"
    if fix:
        hid = e7.get("semantic_hypothesis")
        h = next((x for x in e3.get("hypotheses") or [] if x.get("id") == hid), None) or {}
        out["fix"] = dict(hypothesis=hid, floor=h.get("floor"), pose_px=h.get("pose_px"), pose_m=h.get("pose_m"), heading_deg=h.get("heading_deg"),
                          ppm=h.get("ppm"), first_fix_s=_rel(e7.get("written_wall"), t0), path=e7.get("path") or e7.get("source"), why=e7.get("why"))
        out["time_to_fix_s"] = out["fix"]["first_fix_s"]
    else:
        out["fix"], out["time_to_fix_s"] = None, None
    tm = (ev or {}).get("time_metrics") or {}          # e6_settle.time_metrics (console evaluation): operator waits
    out["manual_wait_s"] = tm.get("sum_manual_wait_s")
    if out["time_to_fix_s"] is not None and out["manual_wait_s"]:
        out["time_to_fix_without_manual_wait_s"] = round(out["time_to_fix_s"] - out["manual_wait_s"], 1)
    temps, mems = [], []
    for f in rd.glob("*.sup.jsonl"):
        for r in _jl(f):
            if isinstance(r.get("temp"), (int, float)):
                temps.append(r["temp"])
            if isinstance(r.get("avail"), (int, float)):
                mems.append(r["avail"])
    hz = sorted(s.get("hz") for s in lio if s.get("hz") is not None)
    out["resources"] = dict(temp_max_c=round(max(temps), 1) if temps else None, mem_available_min_mb=round(min(mems)) if mems else None,
                            fastlio_hz_p50=hz[len(hz) // 2] if hz else None, fastlio_hz_p5=hz[int(len(hz) * 0.05)] if hz else None,
                            fastlio_hz_min=hz[0] if hz else None, gpu="not used (E2 CPU)" if e2.get("device") != "gpu" else "E2 GPU")
    start = (ev or {}).get("start_gt")
    if not start:
        out["ground_truth"] = "GROUND_TRUTH_UNAVAILABLE"
    else:
        out["ground_truth"] = dict(start=start)
        if fix and out["fix"].get("floor") is not None:
            out["ground_truth"]["floor_correct"] = out["fix"]["floor"] == start.get("floor")
            hh = next((x for x in e3.get("hypotheses") or [] if x.get("id") == out["fix"]["hypothesis"]), None) or {}
            r_ = hh.get("rec") or {}
            try:
                rows = (rd / "archive" / "live2_store" / "odom_link.csv").read_text().splitlines()[1:]
                x0, y0, yaw0 = (float(v) for v in rows[0].split(",")[2:5])
                xN, yN, yawN = (float(v) for v in rows[-1].split(",")[2:5])
                phi = math.radians(float(r_["phi"]))
                ax, ay = float(r_["anchor"][0]), float(r_["anchor"][1])
                dx, dy = x0 - ax, y0 - ay
                sx = float(r_["pos"][0]) + float(r_["ppm"]) * (math.cos(phi) * dx - math.sin(phi) * dy)
                sy = float(r_["pos"][1]) + float(r_["ppm"]) * (math.sin(phi) * dx + math.cos(phi) * dy)
                gpx = start.get("px") or (start.get("map_px") or [None, None])
                gx, gy = (gpx if isinstance(gpx, (list, tuple)) else (start.get("px"), start.get("py")))
                e_px = math.hypot(sx - float(gx), sy - float(gy))
                out["ground_truth"].update(start_implied_px=[round(sx, 1), round(sy, 1)], start_error_px=round(e_px, 1),
                                           start_error_m_at_hyp_scale=round(e_px / float(r_["ppm"]), 2),
                                           note="start pose implied by the FIX transform vs the start mark; heading error not computed "
                                                "(odom->body heading contract unverified)")
            except (OSError, KeyError, ValueError, TypeError, IndexError) as ex:
                out["ground_truth"]["error_not_computed"] = f"{type(ex).__name__}: {ex}"
    # 2026-10-31 timeline (s after the experiment start; raw walls are in the source files) — paper metrics
    msF = (_j(st / "stage_milestones.json") or {}).get("first") or {}
    amb = next((h for h in e4h if h.get("state") == "AMBIGUOUS"), None)
    tl = dict(experiment_start=0.0, first_e2_candidate=out["e2"]["first_candidate_s"],
              first_provisional=_rel((msF.get("PROVISIONAL") or {}).get("wall"), t0),
              first_e4_confident=out["e4"]["first_confident_s"], first_e4_ambiguous=_rel((amb or {}).get("wall"), t0),
              e7_structure_start=_rel(sp.get("started_wall"), t0) if sp else None, e7_structure_end=_rel(sp.get("ended_wall"), t0) if sp else None,
              fix=out["time_to_fix_s"])
    man = _jl(st / "e6_manual_events.jsonl")
    obs_t = []
    for o in _jl(st / "e6_observations.jsonl"):
        if o.get("event") not in ("created", "STOPPED_TARGET_READY"):
            continue
        oid = o.get("observation_id")
        row = next((r for r in obs_t if r["observation_id"] == oid), None)
        if row is None:
            row = dict(observation_id=oid); obs_t.append(row)
        if o.get("event") == "created":
            row["e5_event"] = _rel(o.get("detection_timestamp") or o.get("wall"), t0); row["stop_request"] = _rel(o.get("created_wall") or o.get("wall"), t0)
        else:
            row["stop_confirmed"] = _rel(o.get("stopped_wall") or o.get("wall"), t0)
    for row in obs_t:
        ph = [m for m in man if m.get("observation_id") == row["observation_id"] and m.get("kind") == "PHASE"]
        first = lambda name: next((_rel(m.get("wall"), t0) for m in ph if m.get("phase") == name), None)   # noqa: E731
        row.update(aim_start=first("AIMING"), aim_ready=first("AIM_READY") or first("WAIT_MANUAL_FOCUS"), focus_wait_start=first("WAIT_MANUAL_FOCUS"),
                   capture_requested=first("CAPTURE_REQUESTED"), capture_saved=first("CAPTURE_SAVED"), resume_home=first("HOME_CONFIRMED"))
        rd_ = next((x for x in e8o if x.get("observation_id") == row["observation_id"] and x.get("event") == "READ"), None)
        row["ocr_read_done"] = _rel((rd_ or {}).get("wall"), t0)
        row["ocr_s"] = ((rd_ or {}).get("reading") or {}).get("ocr_s") if isinstance((rd_ or {}).get("reading"), dict) else None
        if row.get("focus_wait_start") is not None and row.get("capture_requested") is not None:
            row["manual_focus_wait_s"] = round(row["capture_requested"] - row["focus_wait_start"], 1)
    tl["observations"] = obs_t
    for r in (_j(ed / "time_metrics.json") or {}).get("observations") or (_j(ed / "time_metrics.json") or {}).get("rows") or []:
        row = next((x for x in obs_t if x["observation_id"] == r.get("observation_id")), None)
        if row is not None:
            row["operator_settle_approval"] = _rel(r.get("t_manual_approval"), t0)
            if r.get("t_manual_approval") and r.get("t_stop_confirmed"):
                row["manual_settle_wait_s"] = round(float(r["t_manual_approval"]) - float(r["t_stop_confirmed"]), 1)
    out["timeline"] = tl
    out["metrics"] = dict(first_provisional_s=tl["first_provisional"], first_semantic_observation_s=next((r.get("ocr_read_done") for r in obs_t
                                                                                                       if r.get("ocr_read_done") is not None), None),
                          time_to_fix_s=out["time_to_fix_s"], manual_settle_wait_s=sum(r.get("manual_settle_wait_s") or 0 for r in obs_t) or None,
                          manual_focus_wait_s=sum(r.get("manual_focus_wait_s") or 0 for r in obs_t) or None,
                          ttf_minus_manual_waits_s=None if out["time_to_fix_s"] is None else round(out["time_to_fix_s"] - sum(
                              (r.get("manual_settle_wait_s") or 0) + (r.get("manual_focus_wait_s") or 0) for r in obs_t), 1),
                          note="ttf_minus_manual_waits_s is a procedure-corrected index, NOT a fully automatic system's time")
    stages = dict(fastlio=bool(lio), e2_all_floors=out["e2"]["all_floors_exact_s"] is not None, e3_hypotheses=bool(H),
                  e4_confident=conf is not None, e6_reading=bool(out["e5_e6"]["readings"]),
                  e7_structure=(sp or {}).get("status"), e7_semantic=e7.get("global_decision"), fix=fix)
    out["stages"] = stages
    ed.mkdir(parents=True, exist_ok=True)
    (ed / "E8_SUMMARY.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    md = [f"# E8 요약 — {out['exp_id']} ({out['result_class']})", "",
          f"- 입력 {out['input']} · {'재생(오프라인)' if out['replay'] else '현장'} · 종료 {out['end_reason']}",
          f"- E2: 최초 후보 {out['e2']['first_candidate_s']} s · 9/9 {out['e2']['all_floors_exact_s']} s · 지역 재탐색 {len(out['e2']['regional'])}회",
          f"- E3 1위: {(out['e3']['top'] or [{}])[0].get('id')} {(out['e3']['top'] or [{}])[0].get('floor')}F (미확정 후보)",
          f"- E4: 최초 확신 {out['e4']['first_confident_s']} s · 마지막 {out['e4']['final_state']}",
          f"- E6/E8: 관측 {out['e5_e6']['observations']} · 판독 {len(out['e5_e6']['readings'])} · 쌍 판정 {out['e5_e6']['pair_verdicts']}",
          f"- 의미정보 후보 복구: {(rec or {}).get('status')} {(rec or {}).get('why') or ''}",
          f"- E7: {out['e7']['global_decision']} · 구조 경로 {((out['e7']['structure_path'] or {}).get('status'))} "
          f"({((out['e7']['policy'] or {}).get('mode'))})",
          f"- **FIX: {'있음 — ' + str(out['fix']) if fix else 'NO_FIX'}** · time-to-fix {out['time_to_fix_s']} s",
          f"- 정답: {out['ground_truth'] if isinstance(out['ground_truth'], str) else out['ground_truth'].get('start_error_m_at_hyp_scale', 'FIX 없음 → 오차 없음')}",
          f"- 시간: 최초 잠정 위치 {out['metrics']['first_provisional_s']} s · 최초 의미 관측 {out['metrics']['first_semantic_observation_s']} s · "
          f"수동 진동 대기 {out['metrics']['manual_settle_wait_s']} s · 수동 초점 대기 {out['metrics']['manual_focus_wait_s']} s",
          f"- 자원: 최고 {out['resources']['temp_max_c']} °C · 최소 여유 {out['resources']['mem_available_min_mb']} MB · "
          f"FAST-LIO p50 {out['resources']['fastlio_hz_p50']} Hz / 최저 {out['resources']['fastlio_hz_min']} Hz"]
    (ed / "E8_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out
