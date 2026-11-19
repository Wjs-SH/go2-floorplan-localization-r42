#!/usr/bin/env python3
"""LIVE1 OPERATOR_FREE_DRIVE SigLIP lifetime keeper. Owns ONLY the observe (SigLIP SHADOW) child it starts.

  - SigLIP reads the sensor-hub ring (no ROS); RGB / odometry / accumulation never depend on it.
  - start / resume only if: approval A valid for 'observe' (same boot, unexpired, text hash), hub index fresh,
    max temp < 72 C, MemAvailable >= 4300 MB, no other siglip_worker / e5_onboard running
  - pause (orderly SIGINT to its own child group, SIGKILL after 15 s) if temp >= 80 C or MemAvailable < 2500 MB
    (limits never relaxed), then wait for recovery and resume while the approval is valid
  - never runs past the approval expiry (child run limit = remaining approval time); approval expired -> ENDED, no restart
  - 3 consecutive error exits -> ERROR (no crash loop)
State for the web: <owned>/observe.keeper.json  (siglip = STARTING | RUNNING | PAUSED_THERMAL | PAUSED_MEMORY |
WAITING_HUB | WAITING_RESOURCES | ENDED_APPROVAL | ERROR | STOPPED)"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live1_common as C  # noqa: E402

START_T, ABORT_T, START_MB, ABORT_MB = 72.0, 80.0, 4300.0, 2500.0
RESUME_AFTER_MEM_T = 76.0   # R41: a memory pause says nothing about heat. Under E2 the board sits at 73-76 C, so the 72 C cold-start guard kept SigLIP off for the rest of the run after one VLM load (load5: +857 s to the end). Only after PAUSE:MEM; a thermal pause still cools to 72; the 80 C / 2500 MB aborts are unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--child-cmd", required=True, help="bash command; LIVE1_RUN_S is appended by the keeper")
    ap.add_argument("--log-prefix", required=True)
    a = ap.parse_args()
    term = [False]
    signal.signal(signal.SIGTERM, lambda *x: term.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *x: term.__setitem__(0, True))
    st = dict(siglip="STARTING", why=None, runs=[], consecutive_errors=0)

    def w(**k):
        st.update(k, wall=time.time(), keeper_pid=os.getpid())
        C.atomic_json(a.state, st)

    t_keeper = time.time()

    def authorized(ap_):
        """Legacy A: valid. STANDING: valid, or SUSPENDED (suspend = no NEW experiment; the running one keeps SigLIP)."""
        return ap_["valid"] or (ap_.get("kind") == "STANDING" and ap_.get("state") == "SUSPENDED")

    def remaining(ap_):
        """Legacy A: until the approval expiry. STANDING (no expiry): the per-experiment observe limit from keeper start."""
        if ap_.get("expires_epoch"):
            return float(ap_["expires_epoch"]) - time.time() - 30.0
        lim = float(((ap_.get("limits") or {}).get("observe") or {}).get("max_run_s") or ap_.get("experiment_max_s") or 0)
        return t_keeper + lim - time.time() - 30.0

    def blockers():
        ap_ = C.approval_status("A", "observe")
        if not authorized(ap_):
            return "ENDED_APPROVAL", ap_["why"], None
        rem = remaining(ap_)
        if rem < 60:
            return "ENDED_APPROVAL", "LESS_THAN_60S_LEFT_IN_APPROVAL", None
        idx = Path(a.hub) / "index.json"
        if not idx.exists() or time.time() - idx.stat().st_mtime > 2.0:
            return "WAITING_HUB", "SENSOR_HUB_INDEX_NOT_FRESH", rem
        t, m = C.max_temp(), C.mem_available_mb()
        limit = RESUME_AFTER_MEM_T if st["runs"] and str(st["runs"][-1].get("end_reason") or "").startswith("PAUSE:MEM") else START_T
        if t is None or t >= limit:
            return "WAITING_RESOURCES", f"TEMP {t} >= {limit} (start guard)", rem
        if m < START_MB:
            return "WAITING_RESOURCES", f"MEM {m:.0f} < {START_MB} (start guard)", rem
        dup = [p for p in C.find_pids("siglip_worker.py") + C.find_pids("e5_onboard.py") + C.find_pids("live1/observe_node.py")
               if not any(k in C.cmdline(p) for k in ("observe_keeper.py", "live1_supervisor.py", "bash -c"))]
        if dup:
            return "WAITING_RESOURCES", f"OTHER_SIGLIP_RUNNING {dup}", rem
        return None, None, rem

    child = None
    while not term[0]:
        if child is None:
            state, why, rem = blockers()
            if state == "ENDED_APPROVAL":
                w(siglip="ENDED_APPROVAL", why=why)
                break
            if state:
                w(siglip=state, why=why)
                time.sleep(2.0)
                continue
            if st["consecutive_errors"] >= 3:
                w(siglip="ERROR", why="3 consecutive error exits; not restarting")
                break
            run_s = int(rem)
            log = open(f"{a.log_prefix}_run{len(st['runs']) + 1}.out", "w")
            child = subprocess.Popen(["bash", "-c", f"export LIVE1_RUN_S={run_s} LIVE1_RUN_S_APPROVED_MAX={run_s}; " + a.child_cmd],
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            st["runs"].append(dict(pid=child.pid, start_wall=time.time(), run_s=run_s, log=log.name))
            w(siglip="RUNNING", why=None, child_pid=child.pid)
            continue
        rc = child.poll()
        if rc is not None:
            r = st["runs"][-1]
            r.update(end_wall=time.time(), rc=rc, end_reason=r.get("end_reason") or ("RUN_LIMIT_OR_EXIT" if rc == 0 else f"EXIT_{rc}"))
            st["consecutive_errors"] = 0 if rc == 0 or r["end_reason"].startswith("PAUSE") else st["consecutive_errors"] + 1
            child = None
            w(siglip="WAITING_RESOURCES", why=f"last run ended: {r['end_reason']}")
            time.sleep(2.0)
            continue
        t, m = C.max_temp(), C.mem_available_mb()
        reason = None
        if t is not None and t >= ABORT_T:
            reason = ("PAUSED_THERMAL", f"TEMP {t} >= {ABORT_T}")
        elif m is not None and m < ABORT_MB:
            reason = ("PAUSED_MEMORY", f"MEM {m:.0f} < {ABORT_MB}")
        elif not authorized(C.approval_status("A", "observe")):
            reason = ("ENDED_APPROVAL", "approval no longer valid")
        if reason:
            st["runs"][-1]["end_reason"] = "PAUSE:" + reason[1] if reason[0] != "ENDED_APPROVAL" else "APPROVAL"
            _stop(child)
            child = None
            st["runs"][-1].update(end_wall=time.time())
            w(siglip=reason[0], why=reason[1])
            if reason[0] == "ENDED_APPROVAL":
                break
            time.sleep(5.0)
            continue
        w(siglip="RUNNING", why=None, temp=t, mem=m)
        time.sleep(0.5)
    if child is not None:
        _stop(child)
        st["runs"][-1].update(end_wall=time.time(), end_reason="STOP_REQUESTED")
    if st["siglip"] not in ("ENDED_APPROVAL", "ERROR"):
        w(siglip="STOPPED", why="stop requested")


def _stop(child):
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(5)


if __name__ == "__main__":
    main()
