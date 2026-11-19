"""op_clock.py — operator-normalized experiment clock (2026-11-01, r33). An ANALYSIS index, never a measurement.

Two clocks over the same host wall clock (nothing is modified: ROS / FAST-LIO / sensor stamps, E2..E7 timings, safety timers,
OCR / VLM times, E7 time and the real time-to-fix stay exactly as recorded):
  T_raw         = t - t_run_start
  T_normalized  = T_raw - (completed SETTLE + completed FOCUS intervals, actual)  + N_settle x 1 s + N_focus x 3 s
                  - (the part of a still-open manual interval)                    <- the clock PAUSES while the operator works
Intervals, from the records the console and the E6 bridge already write (one per observation / attempt):
  SETTLE  start = the settle gate's STATE WAITING_FOR_SETTLE_APPROVAL (the moment the gate confirmed the stop and asked the operator;
                  the robot's own deceleration before it is NOT removed)
          end   = COMMAND approve accepted (= [① 진동 안정 확인] passed the safety checks)          -> completed, counts +1 s
          cancelled / moved / expired / experiment ended                                          -> NOT completed: real time kept
  FOCUS   start = PHASE WAIT_MANUAL_FOCUS (aim done, manual focus begins; again after every [다시 촬영])
          end   = COMMAND capture / capture_anyway accepted (= [② 초점 확인 완료 · 사진 촬영] accepted)  -> completed, counts +3 s
          the observation ends (cancel / moved / abort / aim failed / timeout) before ②          -> NOT completed: real time kept
Robustness: events out of order or for another experiment are ignored; a command without an open interval is ignored; the removed
time is the UNION of the intervals (no double subtraction); everything is clipped to [t_run, t]. Pure functions, no device."""
import json
from pathlib import Path

SETTLE_NORM_S, FOCUS_NORM_S = 1.0, 3.0
SETTLE_END_NOT_DONE = ("USER_CANCELLED", "ROBOT_MOVED", "SETTLE_APPROVAL_EXPIRED", "EXPERIMENT_ENDED")
FOCUS_END_NOT_DONE = ("USER_CANCELLED", "USER_ABORTED", "ROBOT_MOVED", "MANUAL_TIMEOUT", "AIM_FAILED", "DEVICE_ERROR",
                      "HOME_CONFIRMED", "HOME_NOT_CONFIRMED")
NOTE = "수동조작 기준 보정시간: 진동 안정화 1초·수동 초점 3초를 가정한 분석값 (실제 측정 시간이 아님)"


def _jl(p):
    out = []
    try:
        with open(p) as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def intervals(state_dir, experiment_id=None):
    """-> list of dict(kind, observation_id, start, end, completed, why) in time order (open ones have end None)."""
    st = Path(state_dir)
    out = []
    ok_exp = lambda e: experiment_id is None or e.get("experiment_id") in (None, experiment_id)   # noqa: E731
    # settle: one interval per observation id
    open_s = {}
    for e in sorted((e for e in _jl(st / "e6_settle_events.jsonl") if ok_exp(e) and e.get("wall")), key=lambda e: e["wall"]):
        oid = e.get("observation_id")
        if e.get("kind") == "STATE" and e.get("state") == "WAITING_FOR_SETTLE_APPROVAL":
            if oid not in open_s and not any(i["observation_id"] == oid and i["kind"] == "SETTLE" for i in out):
                open_s[oid] = dict(kind="SETTLE", observation_id=oid, start=float(e["wall"]), end=None, completed=False, why="open")
        elif e.get("kind") == "COMMAND" and e.get("action") == "approve" and e.get("accepted") is True and oid in open_s:
            i = open_s.pop(oid); i.update(end=float(e["wall"]), completed=True, why="approved"); out.append(i)
        elif e.get("kind") == "STATE" and e.get("state") in SETTLE_END_NOT_DONE and oid in open_s:
            i = open_s.pop(oid); i.update(end=float(e["wall"]), completed=False, why=e.get("state")); out.append(i)
    out += list(open_s.values())
    # focus: one interval per WAIT_MANUAL_FOCUS entry (retake = a new interval of the same observation)
    open_f = {}
    for e in sorted((e for e in _jl(st / "e6_manual_events.jsonl") if ok_exp(e) and e.get("wall")), key=lambda e: e["wall"]):
        oid = e.get("observation_id")
        if e.get("kind") == "PHASE" and e.get("phase") == "WAIT_MANUAL_FOCUS":
            if oid not in open_f:
                open_f[oid] = dict(kind="FOCUS", observation_id=oid, start=float(e["wall"]), end=None, completed=False, why="open")
        elif e.get("kind") == "COMMAND" and e.get("accepted") is True and e.get("action") in ("capture", "capture_anyway") and oid in open_f:
            i = open_f.pop(oid); i.update(end=float(e["wall"]), completed=True, why=e.get("action")); out.append(i)
        elif e.get("kind") == "PHASE" and e.get("phase") in FOCUS_END_NOT_DONE and oid in open_f:
            i = open_f.pop(oid); i.update(end=float(e["wall"]), completed=False, why=e.get("phase")); out.append(i)
    out += list(open_f.values())
    return sorted(out, key=lambda i: i["start"])


def _union_len(spans):
    tot, cur = 0.0, None
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + ((cur[1] - cur[0]) if cur else 0.0)


def clock(iv, t_run, t):
    """Clocks at time t (t_run = experiment start). Intervals completed AFTER t are open at t (paused, not yet normalized)."""
    if t_run is None or t is None:
        return None
    t = float(t); raw = max(0.0, t - float(t_run))
    done, paused, n = [], [], dict(SETTLE=0, FOCUS=0)
    act = dict(SETTLE=0.0, FOCUS=0.0)
    for i in iv:
        a = max(float(t_run), i["start"])
        if a >= t:
            continue
        if i["completed"] and i["end"] <= t:
            done.append((a, min(i["end"], t))); n[i["kind"]] += 1; act[i["kind"]] += min(i["end"], t) - a
        elif i["end"] is None or i["end"] > t:                     # still open at t: the normalized clock is paused
            paused.append((a, t, i))
    removed = _union_len(done + [(a, b) for a, b, _ in paused])
    norm = raw - removed + n["SETTLE"] * SETTLE_NORM_S + n["FOCUS"] * FOCUS_NORM_S
    p = paused[-1][2] if paused else None
    return dict(raw_s=round(raw, 1), normalized_s=round(max(0.0, norm), 1), N_settle=n["SETTLE"], N_focus=n["FOCUS"],
                T_settle_s=round(act["SETTLE"], 1), T_focus_s=round(act["FOCUS"], 1), removed_s=round(removed, 1),
                paused=None if not p else dict(kind=p["kind"], observation_id=p["observation_id"], since=p["start"]),
                rule=f"T_raw - 완료된 수동 구간 + N_settle×{SETTLE_NORM_S:.0f}s + N_focus×{FOCUS_NORM_S:.0f}s", note=NOTE)


def summary(state_dir, t_run, t_end, t_fix=None, experiment_id=None):
    """End-of-run record: clocks at the end and, only with a FIX, the raw and normalized time-to-fix."""
    iv = intervals(state_dir, experiment_id)
    end = clock(iv, t_run, t_end)
    out = dict(kind="LIVE2_OPERATOR_CLOCK", t_run_start=t_run, t_end=t_end, at_end=end, intervals=iv, note=NOTE,
               settle_norm_s=SETTLE_NORM_S, focus_norm_s=FOCUS_NORM_S)
    if t_fix:
        f = clock(iv, t_run, t_fix)
        out.update(t_fix=t_fix, TTF_raw_s=f["raw_s"], TTF_normalized_s=f["normalized_s"], at_fix=f)
    else:
        out.update(t_fix=None, TTF_raw_s=None, TTF_normalized_s=None, why="NO_FIX — no time-to-fix (raw or normalized)")
    return out


BUSY_PHASES = ("REPOSITION_NEEDED", "WAIT_BODY_SETTLE", "WAIT_AIM_SETTLE", "AIMING", "AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_REQUESTED", "CAPTURING", "CAPTURE_SAVED", "CAPTURE_REVIEW", "CAPTURE_FAILED", "RETURNING_HOME")


def ops_timer(state_dir, experiment_id, t0, now):
    """R39: the third clock, only for the page's 'still looking' line: real time minus the E6 hardware work (settle wait, aim, focus, review, reposition, HOME), which the operator should not see a time notice in.
    It is not a measurement, not a time-to-fix and it stops nothing: no safety timer, stamp or FIX time reads it. -> dict(ops_s, ops_paused)."""
    st = Path(state_dir)
    spans, cur = [], None
    for e in sorted((e for e in _jl(st / "e6_manual_events.jsonl") if e.get("kind") == "PHASE" and e.get("wall") and e["wall"] <= now and e.get("experiment_id") in (None, experiment_id)), key=lambda e: e["wall"]):
        if e.get("phase") in BUSY_PHASES:
            cur = float(e["wall"]) if cur is None else cur
        elif cur is not None:
            spans.append([cur, float(e["wall"])]); cur = None
    settle = {}
    for e in sorted((e for e in _jl(st / "e6_settle_events.jsonl") if e.get("wall") and e["wall"] <= now and e.get("experiment_id") in (None, experiment_id)), key=lambda e: e["wall"]):
        o = e.get("observation_id")
        if e.get("kind") == "STATE" and e.get("state") == "WAITING_FOR_SETTLE_APPROVAL":
            settle.setdefault(o, float(e["wall"]))
        elif o in settle and (e.get("kind") == "COMMAND" or e.get("state") in SETTLE_END_NOT_DONE):
            spans.append([settle.pop(o), float(e["wall"])])
    open_now = cur is not None or bool(settle)
    if cur is not None:
        spans.append([cur, now])
    spans += [[v, now] for v in settle.values()]
    removed, last = 0.0, t0
    for a, b in sorted(spans):
        a, b = max(a, last), min(b, now)
        if b > a:
            removed += b - a; last = b
    return dict(ops_s=round(max(0.0, now - t0 - removed), 1), ops_paused=open_now)
