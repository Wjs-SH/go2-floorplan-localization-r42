"""obs_flow.py — LIVE2 per-observation operator state after a STOP request (display only; nothing is sent to the robot).

2026-10-22 field: the STOP sound was heard, but the operator had no signal for WHEN to drive on. STOP_CONFIRMED alone is
not "go": in precision mode E6 (aim -> capture -> HOME -> OCR) and E7 may still run on that stop.

States (newest observation wins):
  STOP_REQUESTED        E6 tracker record WAIT_STOP (operator should stop)
  STOP_CONFIRMED        stopped (fresh odometry) — observation about to start
  E6_OBSERVING          precision mode: no precision result for this observation_id yet
  E7_VERIFYING          precision result usable + decisive -> E7 job for this observation_id not finished
  OBSERVATION_COMPLETE  this observation's required processing is done (any outcome incl. NO_FIX / blurry / no label)
  RESUME_READY          OBSERVATION_COMPLETE (or observe-only hold elapsed) AND sensors / safety OK -> operator may drive
  HOLD_ERROR            device / focus fault, E6 timeout, odometry stale, safety pending -> do not drive on, check status
  (manual focus, 2026-10-26: AIMING -> WAIT_MANUAL_FOCUS -> CAPTURING -> RETURNING_HOME from e6_manual.json; HOME not
   confirmed / operator abort / device error -> HOLD_ERROR; RESUME_READY only after HOME_CONFIRMED and the stored record)
  STOP_CLEARED          request expired without a stop (EXPIRED_NO_STOP) or older than the display window
Observe-only mode (no precision bridge): nothing E6 will run, so RESUME_READY follows STOP_CONFIRMED after OBSERVE_HOLD_S
(the E5 frames of the stop are taken) — the operator is never left waiting for an E6 that does not exist.
"""

OBSERVE_HOLD_S, E6_TIMEOUT_S, E7_TIMEOUT_S, SHOW_S = 3.0, 150.0, 90.0, 60.0
HOLD_STATUSES = ("FOCUS_FAULT_BLOCKED", "DEVICE_ERROR_BLOCKED", "BLOCKED", "REFUSED_NO_VALID_APPROVAL", "USER_ABORTED")
# 2026-10-26 manual focus (e6_manual.json phases) -> operator states. RESUME_READY needs HOME_CONFIRMED + the stored record.
MANUAL_FLOW = dict(REPOSITION_NEEDED="REPOSITION_NEEDED", WAIT_BODY_SETTLE="AIMING", WAIT_AIM_SETTLE="AIMING", AIMING="AIMING", AIM_READY="WAIT_MANUAL_FOCUS", WAIT_MANUAL_FOCUS="WAIT_MANUAL_FOCUS",
                   CAPTURE_FAILED="WAIT_MANUAL_FOCUS", CAPTURE_REQUESTED="CAPTURING", CAPTURING="CAPTURING",
                   CAPTURE_SAVED="RETURNING_HOME", RETURNING_HOME="RETURNING_HOME")
MANUAL_HOLD = ("HOME_NOT_CONFIRMED", "DEVICE_ERROR", "USER_ABORTED", "MANUAL_TIMEOUT")


PAIR_READ_TIMEOUT_S = 150.0
PAIR_MSG = dict(WAITING_RESUME="observation stored (OCR in the background, PROVISIONAL only) — drive on to the next plate",
                SEARCHING="drive on — E5 is looking for the next plate",
                MORE_OBSERVATION_REQUIRED="another independent plate is needed — drive on",
                GLOBAL_FIX="position fixed (pair + E7) — you may drive on",
                CANDIDATE_MISSING="readings fit the map but no live candidate — re-search, you may drive on",
                NO_FIX="no unique position from these observations — you may drive on",
                PAIR_IDLE="observation not counted — you may drive on")


def pair_flow(out, oid, rec, pair, now):
    """E8 pair policy (2026-10-22): guidance comes from e8_pair, never before it has handled THIS observation."""
    if oid not in (pair.get("processed") or []):
        return out("E6_OBSERVING", "E6 capture being stored — stay stopped", (rec or {}).get("wall"))
    ps = pair.get("state") or ""
    if ps == "HOLD_ERROR":
        return out("HOLD_ERROR", f"E8: {pair.get('why')} — do not drive on, check the status", now)
    if ps in ("PAIR_READING", "PAIR_CONSISTENCY_CHECK", "REGION_CANDIDATE_READY", "E7_VERIFYING"):
        # R36 (11-02 field: photo stored, HOME done, still "stay stopped" for OCR): reading and E7 run in the background on the stored image and the LiDAR store. Nothing there needs the robot still.
        return out("RESUME_READY", f"{ps}: photo stored, HOME confirmed — reading continues in the background, you may drive on", pair.get("since"))
    if pair.get("resume_ok"):
        key = next((k for k in PAIR_MSG if ps.endswith(k)), None)
        return out("RESUME_READY", f"{ps}: {PAIR_MSG.get(key, 'you may drive on')}", pair.get("since"))
    return out("E6_OBSERVING", f"E8 {ps}", pair.get("since"))


def flow(e6_recent, precision, updates, e7, precision_mode, lio_ok, safety_pending, now, e6_blocked=None, pair=None, manual=None):
    """-> dict(state, observation_id, why, since) for the newest observation, or None.
    e6_recent: e6_state.json 'recent'; precision: {observation_id: precision record}; updates: {observation_id: [rows of
    e6_hypothesis_updates.jsonl]} (the e34 link writes rows for EVERY processed observation, also NO_EVIDENCE); e7: e7_state.json.
    e6_blocked: why the automatic E6 is blocked (device / focus fault, bridge gone) -> every stop after it is HOLD_ERROR:
    no automatic retry, no "drive on" guidance (2026-10-22)."""
    obs = [o for o in (e6_recent or []) if o.get("observation_id")]
    if not obs:
        return None
    o = max(obs, key=lambda x: float(x.get("created_wall") or 0))
    oid, st = o["observation_id"], o.get("state")
    out = lambda s, why, since=None: dict(state=s, observation_id=oid, why=why, since=since, precision_mode=precision_mode)  # noqa: E731
    if st == "WAIT_STOP":
        return out("STOP_REQUESTED", "STOP requested — stop with the joystick", o.get("created_wall"))
    if st != "STOPPED_TARGET_READY":
        return out("STOP_CLEARED", f"{st}: no stop for this request", o.get("wall"))
    t_stop = float(o.get("stopped_wall") or 0)
    man = manual if (manual or {}).get("observation_id") == oid else None
    if man:                                   # the operator's focus time is not E6 run time: age from the last manual phase
        t_stop = max(t_stop, float(man.get("phase_since") or 0))
    if now - t_stop > SHOW_S + (E6_TIMEOUT_S + E7_TIMEOUT_S if precision_mode else 0):
        return out("STOP_CLEARED", "old observation", t_stop)
    if not lio_ok or safety_pending:
        return out("HOLD_ERROR", "odometry not OK or safety check pending — do not drive on", now)
    if man and man.get("phase") in MANUAL_HOLD:
        return out("HOLD_ERROR", f"E6 {man['phase']}: {man.get('why') or ''} — do not drive on, check the head", man.get("phase_since"))
    if e6_blocked:
        return out("HOLD_ERROR", f"E6 blocked: {e6_blocked} — no retry; end the experiment and check the head", now)
    if man and man.get("phase") in MANUAL_FLOW:
        return out(MANUAL_FLOW[man["phase"]], f"{man['phase']}: {man.get('why') or ''}", man.get("phase_since"))
    if not precision_mode:
        if now - t_stop < OBSERVE_HOLD_S:
            return out("STOP_CONFIRMED", "observe-only: E5 frames of this stop", t_stop)
        return out("RESUME_READY", "observe-only mode (E6 off): you may drive on", t_stop + OBSERVE_HOLD_S)
    rec = (precision or {}).get(oid)
    if rec is None and man and man.get("phase") in ("HOME_CONFIRMED", "USER_CANCELLED", "ROBOT_MOVED", "MANUAL_TIMEOUT"):
        return out("E6_OBSERVING", f"{man['phase']}: record being stored — stay stopped", man.get("phase_since"))
    if rec is not None and rec.get("home_confirmed") is False:
        return out("HOLD_ERROR", "E6 HOME not confirmed — do not drive on, check the head", rec.get("wall"))
    if rec is None:
        if now - t_stop > E6_TIMEOUT_S:
            return out("HOLD_ERROR", f"no E6 result {E6_TIMEOUT_S:.0f} s after the stop — check the precision bridge", now)
        return out("E6_OBSERVING", "E6 aim / capture / OCR running — stay stopped", t_stop)
    status = rec.get("status")
    if status in HOLD_STATUSES:
        return out("HOLD_ERROR", f"E6 {status}: {rec.get('why') or ''} — operator checks the head", rec.get("wall"))
    if pair is not None:
        return pair_flow(out, oid, rec, pair, now)
    usable = status == "CAPTURED" and bool((rec.get("image_quality") or {}).get("ok"))
    rows = (updates or {}).get(oid)
    if rows is None:                                    # e34 link has not processed this record yet (~1 s tick)
        return out("E6_OBSERVING", "semantic match running", rec.get("wall"))
    if any(r.get("decisive") and r.get("verdict") == "SEMANTIC_CONSISTENT" for r in rows):   # the link requests E7
        e7_done = bool(e7) and e7.get("observation_id") == oid
        if not e7_done:
            if now - float(rec.get("wall") or t_stop) > E7_TIMEOUT_S:
                return out("RESUME_READY", f"E7 not finished in {E7_TIMEOUT_S:.0f} s — it continues in the background", now)
            return out("E7_VERIFYING", "E7 LiDAR verification running — stay stopped", rec.get("wall"))
        return out("RESUME_READY", f"observation complete: {e7.get('global_decision')}", e7.get("written_wall"))
    return out("RESUME_READY", f"observation complete ({status}{'' if usable else ', not usable'}) — move to observe again",
               rec.get("wall"))
