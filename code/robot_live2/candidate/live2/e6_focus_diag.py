#!/usr/bin/env python3
"""e6_focus_diag.py — ONE supervised focus-drive diagnosis through the EXISTING r1 contract (no aim, no HOME, no new endpoint).

Why (2026-10-22 field 418 test): 21 pulses / 3000 ms commanded, image change 1.1-1.37 (static noise) in both directions ->
SUSPECTED_DRIVE_FAULT. Before blaming the autofocus algorithm the drive chain must be separated:
  Nano received the F command?   -> raw Nano reply lines per pulse (e6_af_student tap, firmware 'focus=' field)
  motor shaft / gear / focus ring -> the OPERATOR watches them during the pulses (answers are recorded with --observed)
  IMX477 focus change             -> per-pulse sharpness + motion-image change on fresh frames
The request is the student /focus (KCI stepwise behind e6_af_student) at the CURRENT head pose, so its first pulses are the
student schedule (+300 ms x3 at PWM 180, then reversal / lash probes). The cap comes from approval B drive_ms_max (e.g. 1800:
both directions, < the 3000 ms no-response latch). Every gate stays: approval B + focus_allowed in the bridge AND at the console,
the persistent fault lock (clear it first with e6_aim_console_ctl clear-fault), pulse / PWM / time limits.

  e6_focus_diag.py --state <state> [--console URL]                       one diagnosis, prints + appends e6_focus_diag.jsonl
  e6_focus_diag.py --state <state> --direct-pulses                       FOCUS_DIAGNOSTIC (2026-10-22): fixed small F pulses
                     (+,+,-,-; caps PWM 160 / 200 ms / 800 ms, 1 run per approval) through the console's /live2/focus_diag
                     route — needs approval B limits.focus_diagnostic, NOT focus_mechanical_check; never AF / aim / HOME
  e6_focus_diag.py --state <state> --observed <diag_id> --shaft Y|N|? --gear Y|N|? --ring Y|N|?   operator observations
Classification: MOTOR_NO_RESPONSE / MOTOR_ROTATES_LENS_STATIC / LENS_ROTATES_IMAGE_STATIC / MOTOR_AND_FOCUS_RESPOND /
HARDWARE_STATE_UNKNOWN (image evidence alone never decides shaft / gear / ring).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "live1", HERE.parent / "src" / "sci_visual_observation", HERE.parent / "src"):
    sys.path.insert(0, str(p))

RESP_MOVED = 2.0


def classify(image_response, shaft, gear, ring):
    """Operator answers Y / N / ? + image evidence -> hardware state."""
    if shaft == "N":
        return "MOTOR_NO_RESPONSE"
    if shaft == "Y" and ring == "N":
        return "MOTOR_ROTATES_LENS_STATIC"
    if ring == "Y" and image_response is False:
        return "LENS_ROTATES_IMAGE_STATIC"
    if ring == "Y" and image_response is True:
        return "MOTOR_AND_FOCUS_RESPOND"
    return "HARDWARE_STATE_UNKNOWN"


def pulses_of(focus):
    out = []
    seen = set()
    for m in focus.get("measurements") or []:
        p = m.get("pulse")
        if not p or p.get("n") in seen:
            continue
        seen.add(p.get("n"))
        out.append(dict(n=p.get("n"), dir=p.get("dir"), pwm=p.get("pwm"), ms=p.get("ms"), moved=p.get("moved"),
                        response=p.get("response"), nano_reply=p.get("nano_reply"), sharp_after=m.get("sharp"),
                        frame_seq=m.get("frame_seq"), t_mono=p.get("t_mono")))
    cum = 0
    for p in out:
        cum += p["ms"] or 0
        p["cum_ms"] = cum
    return out


TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "REJECTED_BUSY", "PROTOCOL_ERROR", "REFUSED")


def direct_pulses(a, log, http=None, sleep=time.sleep):
    """Submit the console-side FOCUS_DIAGNOSTIC once and wait for its record (the console enforces every limit)."""
    import urllib.error
    import urllib.request

    def get(path):
        if http:
            return http(path)
        try:
            with urllib.request.urlopen(a.console + path, timeout=5.0) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode() or "{}")
    rid = time.strftime("live2diag-%H%M%S")
    sub = get(f"/live2/focus_diag?req={rid}&owner=live2_focus_diag")
    rec = sub
    t0 = time.time()
    while rec.get("status") not in TERMINAL and time.time() - t0 < 60:
        sleep(0.5)
        rec = get(f"/request?id={rid}")
    det = (rec.get("result") or {}).get("detail") or {}
    out = dict(kind="LIVE2_E6_FOCUS_DIRECT_DIAG", diag_id=rid, t=time.strftime("%F %T"), request_status=rec.get("status"),
               reason=rec.get("reason_code"), device_state=rec.get("device_state"), error=rec.get("error"), diagnostic=det,
               classification="HARDWARE_STATE_UNKNOWN" if not det else
               ("MOTOR_AND_FOCUS_RESPOND" if det.get("evidence", "").startswith("E_") else
                "HARDWARE_STATE_UNKNOWN (A: Nano not replying)" if det.get("evidence", "").startswith("A_") else "HARDWARE_STATE_UNKNOWN"),
               note="image evidence only: shaft / gear / ring are not visible to the IMX477")
    with open(log, "a") as f:
        f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(out, ensure_ascii=False, default=str)[:6000])
    return 0 if rec.get("status") == "COMPLETED" else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--console", default="http://127.0.0.1:8088")
    ap.add_argument("--deadline-s", type=float, default=60.0)
    ap.add_argument("--observed", default=None)
    ap.add_argument("--direct-pulses", action="store_true")
    ap.add_argument("--shaft", choices=["Y", "N", "?"], default="?")
    ap.add_argument("--gear", choices=["Y", "N", "?"], default="?")
    ap.add_argument("--ring", choices=["Y", "N", "?"], default="?")
    a = ap.parse_args()
    state = Path(a.state)
    log = state / "e6_focus_diag.jsonl"
    if a.observed:
        rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        d = next((r for r in rows if r.get("diag_id") == a.observed and r.get("kind") == "LIVE2_E6_FOCUS_DIAG"), None)
        if d is None:
            sys.exit(f"no diagnosis {a.observed}")
        rec = dict(kind="LIVE2_E6_FOCUS_DIAG_OBSERVED", diag_id=a.observed, wall=time.time(), shaft=a.shaft, gear=a.gear, ring=a.ring,
                   image_response=d.get("image_response"),
                   classification=classify(d.get("image_response"), a.shaft, a.gear, a.ring), source="OPERATOR_OBSERVATION")
        with open(log, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False))
        return 0
    if a.direct_pulses:
        return direct_pulses(a, log)
    import e6_precision_bridge as B
    if os.environ.get("SCI_PANTILT_APPROVED") != "1":
        sys.exit("REFUSED: SCI_PANTILT_APPROVED=1 not set")
    ok, why = B.hw_gate("focus", mode="focus_diag")
    if not ok:
        sys.exit(f"REFUSED focus diagnosis: {why}")
    from aim_client import AimClient
    cl = AimClient(a.console)
    t0 = time.time()
    did = time.strftime("FD_%H%M%S")
    res = cl.run("focus", {}, "live2diag", int(t0) % 1000, 1, a.deadline_s)
    foc = res.get("focus") or {}
    ps = pulses_of(foc)
    img = any((p.get("moved") or 0) >= RESP_MOVED or p.get("response") for p in ps) if ps else None
    rec = dict(kind="LIVE2_E6_FOCUS_DIAG", diag_id=did, t=time.strftime("%F %T"), wall=t0, total_s=round(time.time() - t0, 1),
               request_status=res.get("status"), device_state=res.get("device_state"), reason=res.get("reason_code"),
               focus_status=foc.get("status"), focus_reason=foc.get("reason"), drive_ms=foc.get("drive_ms"), pulses=ps,
               effective_limits=foc.get("effective_limits"), ever_response=foc.get("ever_response"),
               sharp_start=next((m.get("sharp") for m in foc.get("measurements") or []), None),
               image_response=img, nano_replied=bool(ps) and all(p.get("nano_reply") for p in ps),
               classification="HARDWARE_STATE_UNKNOWN (needs the operator's shaft / gear / ring observation: --observed)",
               no_home="this diagnosis never sends HOME or aim")
    with open(log, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(rec, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
