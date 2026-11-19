"""siglip_gate.py — LIVE2 E4 -> SigLIP inference decision (pure python; used by live1/observe_node.py).

The model is loaded once and stays resident (reload costs 12-18 s); this only decides whether a scene runs proposals +
SigLIP inference. Modes (env LIVE2_SIGLIP_MODE):
  e4      (default) infer only while a FRESH E4 request asks for it (mode ACTIVE, stamp <= STALE_S old).
          Missing / stale / IDLE_OK request -> no inference (the scene is skipped and counted with its reason).
  eval    explicit SigLIP stand-alone evaluation: infer every scene regardless of E4. Recorded as EVAL_MODE and never as
          E4 integration (status kind, scene counters and the web label all say so).
  shadow  infer every scene and count what the e4 mode would have skipped (diagnosis only).
"""
import json
import time

STALE_S = 5.0
DUTY_T, DUTY_EVERY_S = 76.0, 3      # >= 76 C: infer in 1 of every 3 s (FAST-LIO / lio2 first); the keeper still pauses at 80 C
MODES = ("e4", "eval", "shadow")


def read_request(path):
    try:
        return json.loads(open(path).read())
    except (OSError, ValueError):
        return None


def decide(req, mode="e4", now=None, temp_c=None):
    """-> (infer: bool, why: str). `req` = parsed siglip_request.json or None; temp_c = current max SoC temperature."""
    now = time.time() if now is None else now
    ok, why = _decide(req, mode, now)
    if ok and temp_c is not None and temp_c >= DUTY_T and int(now) % DUTY_EVERY_S != 0:
        return False, f"THERMAL_DUTY {temp_c:.1f} >= {DUTY_T} C (1 of {DUTY_EVERY_S} s)"
    return ok, why


def _decide(req, mode, now):
    if mode == "eval":
        return True, "EVAL_MODE (stand-alone SigLIP evaluation, not E4 integration)"
    if req is None:
        e4 = (False, "E4_REQUEST_MISSING")
    elif now - float(req.get("stamp", 0)) > STALE_S:
        e4 = (False, f"E4_REQUEST_STALE {now - float(req.get('stamp', 0)):.1f}s")
    elif req.get("mode") == "ACTIVE":
        e4 = (True, f"E4_REQUEST_ACTIVE {req.get('e4_state')} {req.get('priority')}")
    else:
        e4 = (False, f"E4_REQUEST_{req.get('mode')}")
    if mode == "shadow":
        return True, "SHADOW (would " + ("infer" if e4[0] else "skip") + ": " + e4[1] + ")"
    return e4
