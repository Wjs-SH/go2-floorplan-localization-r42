"""Orthogonal operator/trial/resource gates for E5-RT (candidate).

Student deploy (baseline) gate: observe = heartbeat_ok AND active, where the web writes active = (trial state ==
'walking'). Pressing [멈춤] moves the trial to 'waiting'/'computing' (floor computation, 221 s on 10-17) and therefore
switched the whole visual search off. Here the states are separated:

  observe_enabled      visual search / tracking / SigLIP may run           (NOT tied to global matching)
  request_enabled      new STOP_REQUESTs may be issued to the operator       (only while the trial is walking)
  actuator_armed       new physical actions (aim/focus/HOME) may start      (heartbeat + armed + walking)
  operator_link_alive  the web heartbeat is fresh
  trial_logging_active a trial is open (walking/waiting/computing)
  global_match_state   idle | computing | done | unknown (informational; resource manager may defer, with reason)

A gate file without 'trial_state' (old viewer) falls back to the legacy semantics with reason LEGACY_GATE_SCHEMA, so
an old web UI never widens what E5 is allowed to do.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

TRIAL_OPEN = ("walking", "waiting", "computing")


def read_gate_file(path):
    p = Path(path)
    try:
        st = p.stat()
        d = json.loads(p.read_text())
        return d, st.st_mtime
    except (OSError, ValueError):
        return None, None


def gate_state(d, mtime, now=None, heartbeat_s=5.0, mode="walking", gate_on=True, arm_without_ui=False,
               resource_defer=None):
    """Pure. d = parsed gate json (or None), mtime = its file mtime. Returns a dict of orthogonal states + reasons.
    mode: 'walking' (default field mode), 'static_bench' (observe always, arm only if armed), 'shadow' (observe always,
    never request/arm: SHADOW_ONLY), 'always' (gate_on False)."""
    now = time.time() if now is None else now
    age = None if mtime is None else now - mtime
    hb = d is not None and age is not None and 0 <= age <= heartbeat_s or (d is not None and age is not None and -1.0 <= age < 0)
    trial = (d or {}).get("trial_state")
    legacy_schema = d is not None and trial is None
    if legacy_schema:
        trial = "walking" if d.get("active") else "unknown"
    walking = hb and trial == "walking"
    reasons = []
    if d is None:
        reasons.append("NO_GATE_FILE")
    elif not hb:
        reasons.append(f"HEARTBEAT_STALE:{age:.1f}s" if age is not None else "HEARTBEAT_UNKNOWN")
    if legacy_schema:
        reasons.append("LEGACY_GATE_SCHEMA")
    gm = (d or {}).get("global_match_state") or ("computing" if trial == "computing" else ("done" if trial == "done" else "idle" if hb else "unknown"))
    if not gate_on or mode == "always":
        observe = True
    elif mode in ("static_bench", "shadow"):
        observe = True
    elif legacy_schema:
        observe = walking                                   # old viewer: keep the old (conservative) meaning
    else:
        observe = hb and trial in TRIAL_OPEN                # search keeps running while the floor is computed
    request = walking and mode != "shadow"
    # E5RT2: a trial state is NOT an actuator grant. Missing/false always fails closed.
    armed_flag = (d or {}).get("armed") is True
    emergency = (d or {}).get("estop") is True or (d or {}).get("operator_abort") is True
    # E5RT2 (integration): a grant written by the candidate viewer carries its own expiry; the writer rewrites the file
    # every 0.3 s, so the reader must check the grant itself (an expired / malformed grant is not re-validated by a fresh
    # heartbeat).  A file without expiry keeps the reviewed-patch semantics (explicit armed:true + fresh heartbeat).
    grant_ok, exp = True, (d or {}).get("arm_expires_t")
    if armed_flag and (exp is not None or (d or {}).get("schema") == "e5rt-gate-2"):
        if not isinstance(exp, (int, float)) or exp != exp or now > exp or not (d or {}).get("arm_grant_id"):
            grant_ok = False
            reasons.append("ARM_GRANT_EXPIRED_OR_INVALID")
    armed = hb and armed_flag and grant_ok and (mode == "static_bench" or walking) and mode != "shadow" and not emergency
    if arm_without_ui:
        reasons.append("ARM_WITHOUT_UI_IGNORED:explicit_fresh_grant_required")
    if not armed_flag:
        reasons.append("NO_EXPLICIT_ACTUATOR_GRANT")
    if emergency:
        armed = request = False
        reasons.append("OPERATOR_ABORT_OR_ESTOP")
    defer = None
    if resource_defer:
        defer = str(resource_defer)
        reasons.append("RESOURCE_DEFERRED:" + defer)
    if observe and not request:
        reasons.append("TRIAL_GATE_BLOCKED:requests_off_trial_" + str(trial))
    return dict(observe_enabled=bool(observe), request_enabled=bool(request), actuator_armed=bool(armed),
                operator_link_alive=bool(hb), trial_logging_active=bool(hb and trial in TRIAL_OPEN),
                global_match_state=gm, trial_state=trial, heartbeat_age_s=age, mode=mode if gate_on else "always",
                resource_defer=defer, reasons=reasons, why=(d or {}).get("why", ""), legacy_schema=legacy_schema,
                # legacy keys kept for the existing UI/status consumers
                observe=bool(observe), arm=bool(armed), heartbeat_ok=bool(hb),
                # E5RT2 tri-state safety inputs: True/False only if the writer states them explicitly, else None (UNKNOWN)
                estop_state=_tri((d or {}).get("estop")) if hb else None,
                operator_abort_state=_tri((d or {}).get("operator_abort")) if hb else None,
                arm_grant_id=(d or {}).get("arm_grant_id") if armed else None)


def _tri(v):
    return v if isinstance(v, bool) else None


def gate_payload(state, now=None, grant=None, operator_abort=None, estop=None):
    """What the (candidate) web viewer writes: legacy 'active' + explicit trial/global-match state.
    E5RT2: armed only with a live, explicit, expiring operator grant (dict(id, expires_t)) while walking. operator_abort is
    written as a bool when the viewer knows it; estop is NOT known to the viewer and is therefore omitted (UNKNOWN)."""
    now = time.time() if now is None else now
    d = dict(active=state == "walking", trial_state=state or "idle",
             global_match_state="computing" if state == "computing" else ("done" if state == "done" else "idle"),
             t=now, schema="e5rt-gate-1" if grant is None and operator_abort is None else "e5rt-gate-2")
    if grant is not None or operator_abort is not None:
        live = bool(grant and state == "walking" and grant.get("expires_t", 0) >= now and not operator_abort)
        d.update(armed=live, arm_grant_id=(grant or {}).get("id") if live else None,
                 arm_expires_t=(grant or {}).get("expires_t") if live else None, operator_abort=bool(operator_abort))
    if isinstance(estop, bool):
        d["estop"] = estop
    return d
