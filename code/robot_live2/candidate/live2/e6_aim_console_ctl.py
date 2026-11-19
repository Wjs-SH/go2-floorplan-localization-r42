#!/usr/bin/env python3
"""e6_aim_console_ctl.py — LIVE2 wrapper around the UNCHANGED student aim_console (pan-tilt + DC focus + IMX477 owner).

Student facts (code): opening /dev/ttyUSB0 resets the Nano (DTR/hupcl; calib_console.Nano may open up to 4 times) and it
boots to its EEPROM pose (95.4/157.2), then aim_console immediately commands HOME (pan_zero/tilt_zero 95.35/157.19,
aim_console.py:1057-1060); its exit path only sends F0 (focus stop, :1135-1140) — servos hold the LAST pose, so the next
start jumps from there to the boot pose. aim_console has no single-instance guard (a 2nd process would reset the Nano).

This wrapper adds only guards around the student program:
  start   refused unless: hardware approval B valid ('pantilt'), nothing holds /dev/ttyUSB0 or the IMX477, :8088 free,
          no other instance (pid file), temp < 72 C and MemAvailable >= 4300 MB. Then ONE student console start
          (--no-eeprom-sync, calibration limits pan 50-150 / tilt 115-180, KCI AF profile, 45 s AF budget). Never restarted
          automatically (each start = Nano reset).
  status  /state of the console + pid + port holders
  home    /home through the student r1 contract (single worker, queued)
  stop    /home first (unless the device reports an error), wait, then SIGINT (student exit: F0), verify the port is free.
Nothing here writes EEPROM, flashes firmware, runs stty, or sends /goto /tilt_sweep /pin_test /focus_profile /trim.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
L = CAND.parent
KC = CAND / "external" / "sci_event" / "code"
PORT_DEV = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AC00BLLM-if00-port0"
CAM_DEV = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_IMX477_HQ_Camera_UC517-video-index0"
URL = "http://127.0.0.1:8088"
PIDF = L / "runtime" / "owned" / "e6_aim_console.pid"
# R34: no --use-depth. LIVE2 always sends the distance in /aim?z=, so the console's own depth subscription was never used, and a second
# process reading colour + aligned depth stops aligned depth for EVERY reader (read-only reproduction 3/3, tools/depth_probe2.py; field 11-01).
ARGS = ["--config", str(CAND / "config" / "pantilt_calib_multi_20260928_trim1.REF_COPY.json"), "--no-eeprom-sync",
        "--af-budget-s", "45", "--pan-lim", "50", "150", "--tilt-lim", "115", "180"]
AF_PROFILE = KC.parent / "configs" / "af_profile_kci_20261017.json"
LAUNCHER = HERE / "e6_aim_console_live2.py"     # same student console + LIVE2 focus guards (always; strategy via --af-strategy)
sys.path.insert(0, str(CAND / "live1"))


def holders(dev):
    try:
        r = subprocess.run(["fuser", os.path.realpath(dev)], capture_output=True, text=True, timeout=5)
        return r.stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return ["UNKNOWN"]


def get(path, timeout=2.0):
    try:
        with urllib.request.urlopen(URL + path, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:                       # noqa: BLE001
        return dict(_error=str(e))


def alive_pid():
    try:
        pid = int(PIDF.read_text().split()[0])
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def approval_now():
    """Hardware approval B at THIS moment (each command re-checks; an approval valid at start may expire mid-session)."""
    import live1_common as C
    return C.approval_status("B", "pantilt")


def audit(event, **kw):
    sys.path.insert(0, str(HERE))
    import e6_precision_bridge as B
    B.audit(event, tool="e6_aim_console_ctl", **kw)


def manual_console():
    """The running console was started with --focus-mode manual (pid file third field)."""
    try:
        return PIDF.read_text().split()[2] == "manual_no_motor"
    except (OSError, IndexError):
        return False


def preflight():
    import live1_common as C
    ap = C.approval_status("B", "pantilt")
    t, m = C.max_temp(), C.mem_available_mb()
    chk = dict(approval_B=dict(ok=ap["valid"], why=ap.get("why")), port_free=dict(ok=not holders(PORT_DEV), holders=holders(PORT_DEV)),
               camera_free=dict(ok=not holders(CAM_DEV), holders=holders(CAM_DEV)), console_port_free=dict(ok="_error" in get("/state", 1.0)),
               single_instance=dict(ok=alive_pid() is None), resources=dict(ok=t is not None and t < 72 and m >= 4300, temp=t, mem_mb=round(m)),
               af_profile=dict(ok=AF_PROFILE.exists(), path=str(AF_PROFILE)))
    return all(v["ok"] for v in chk.values()), chk


FAULT_FILE = L / "runtime" / "e6_focus_fault.json"
FAULT_HISTORY = L / "runtime" / "e6_focus_fault_history"


def clear_fault(note):
    """Lift the persistent focus lock WITHOUT deleting the evidence: the fault record is moved into e6_focus_fault_history/
    together with the clearance (approval B file + text hash, boot, operator note). Needs, at this moment: approval B valid
    ('pantilt', this boot) with focus_mechanical_check=PASSED and focus_fault_cleared=true, and no aim console running
    (its in-memory latch would still block). Nothing moves."""
    import hashlib
    import live1_common as C
    if not FAULT_FILE.exists():
        print("no focus fault latched")
        return 0
    ap = C.approval_status("B", "pantilt")
    lim = ap.get("limits") or {}
    why = None if ap.get("valid") else f"approval B {ap.get('why')}"
    why = why or (None if lim.get("focus_mechanical_check") == "PASSED" else "approval B focus_mechanical_check != PASSED")
    why = why or (None if lim.get("focus_fault_cleared") is True else "approval B focus_fault_cleared != true")
    why = why or (None if alive_pid() is None else "aim console running: stop it first (its latch is in memory)")
    if len(note.strip()) < 10:
        why = why or "--note: say what was checked / repaired (>= 10 characters)"
    if why:
        audit("REFUSED_CLEAR_FAULT", why=why)
        print(f"REFUSED clear-fault: {why}")
        return 3
    FAULT_HISTORY.mkdir(parents=True, exist_ok=True)
    rec = json.loads(FAULT_FILE.read_text())
    ar = C.read_json(Path(ap["file"])) or {}
    out = FAULT_HISTORY / f"{time.strftime('%Y%m%d_%H%M%S')}_CLEARED.json"
    out.write_text(json.dumps(dict(fault=rec, fault_file_sha256=hashlib.sha256(FAULT_FILE.read_bytes()).hexdigest(),
                                   cleared_wall=time.time(), cleared=time.strftime("%F %T"), boot_id=C.boot_id(),
                                   approval_file=ap["file"], approval_text_sha256=ar.get("user_text_sha256"), note=note),
                              ensure_ascii=False, indent=1))
    FAULT_FILE.rename(FAULT_HISTORY / f"{out.stem}_original_e6_focus_fault.json")
    audit("FOCUS_FAULT_CLEARED", record=str(out), approval_file=ap["file"])
    print(f"focus fault cleared (evidence kept): {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "start", "status", "home", "stop", "clear-fault"])
    ap.add_argument("--no-home", action="store_true", help="stop: skip the HOME before shutdown (no servo command at all)")
    ap.add_argument("--note", default="", help="clear-fault: what the operator checked / repaired")
    ap.add_argument("--af-strategy", choices=["student_stepwise", "focus_v2"], default="student_stepwise",
                    help="start: AF search inside the guarded launcher (default: the student KCI stepwise, unchanged)")
    ap.add_argument("--focus-mode", choices=["auto", "manual"], default="auto",
                    help="start: manual = MANUAL_FIXED_NO_MOTOR (2026-10-26): the console refuses every focus-motor job; the "
                         "lens is set by hand. A latched DC-focus fault then does not block the servo HOME (motor not used)")
    ap.add_argument("--focus-v2", action="store_true",
                    help="start: student console with the LIVE2 focus search (e6_aim_console_live2.py, student files unchanged)")
    a = ap.parse_args()
    if a.cmd == "preflight":
        ok, chk = preflight()
        print(json.dumps(dict(ok=ok, checks=chk), ensure_ascii=False, indent=1))
        return 0 if ok else 3
    if a.cmd == "start":
        ok, chk = preflight()
        if not ok:
            print(json.dumps(dict(refused=True, checks=chk), ensure_ascii=False, indent=1))
            return 3
        if a.focus_v2:
            a.af_strategy = "focus_v2"
        if a.focus_mode == "manual" and a.focus_v2:
            print(json.dumps(dict(refused=True, why="--focus-mode manual excludes --focus-v2 (no DC focus motor)")))
            return 3
        if a.focus_v2:                                  # Focus V2 drives the DC focus motor: only after a passed mechanical check
            sys.path.insert(0, str(HERE))
            import e6_precision_bridge as B
            f_ok, f_why = B.focus_allowed(approval_now())
            if not f_ok:
                print(json.dumps(dict(refused=True, why=f"--focus-v2 refused: {f_why}"), ensure_ascii=False))
                return 3
        out = L / "runtime" / "e6_pantilt_out" / time.strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "aim_console.log", "w")
        env = dict(os.environ, KCI_AF_SKIP_IF_SHARP="1", LIVE2_AF_STRATEGY=a.af_strategy)
        if a.focus_mode == "manual":
            env["LIVE2_FOCUS_MODE"] = "MANUAL_FIXED_NO_MOTOR"
        cmd = ["bash", "-c", "set +u; source /opt/ros/foxy/setup.bash >/dev/null 2>&1; source $HOME/unitree_ros2/setup.sh >/dev/null 2>&1; "
               f"cd {KC} && exec taskset -c 2-5 /usr/bin/python3 {LAUNCHER} " + " ".join(ARGS) +
               f" --af-profile {AF_PROFILE} --out-root {out}"]
        ap_ = approval_now()
        audit("SENT", kind="console_start", af_strategy=a.af_strategy, approval_file=ap_.get("file"), approval_expires=ap_.get("expires_epoch"),
              note="opens /dev/ttyUSB0 -> Nano DTR reset -> boot pose -> HOME")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        PIDF.parent.mkdir(parents=True, exist_ok=True)
        PIDF.write_text(f"{p.pid} {out} {'manual_no_motor' if a.focus_mode == 'manual' else a.af_strategy}\n")
        for _ in range(60):
            st = get("/state", 1.0)
            if "_error" not in st:
                print(json.dumps(dict(started=True, pid=p.pid, out=str(out), state=st), ensure_ascii=False)[:800])
                return 0
            if p.poll() is not None:
                print(f"aim_console exited rc={p.returncode}; log {out}/aim_console.log")
                return 4
            time.sleep(1.0)
        print("aim_console did not answer /state within 60 s (left running for inspection; use stop)")
        return 5
    if a.cmd == "clear-fault":
        return clear_fault(a.note)
    if a.cmd == "status":
        print(json.dumps(dict(pid=alive_pid(), port_holders=holders(PORT_DEV), camera_holders=holders(CAM_DEV), state=get("/state")),
                         ensure_ascii=False)[:1500])
        return 0
    if a.cmd == "home":
        ap_ = approval_now()
        if not ap_["valid"]:
            audit("REFUSED_NO_VALID_APPROVAL", kind="home", approval_file=ap_.get("file"), why=ap_.get("why"))
            print(f"REFUSED home: hardware approval B {ap_.get('why')} — register a valid approval first")
            return 3
        audit("SENT", kind="home", approval_file=ap_.get("file"), approval_expires=ap_.get("expires_epoch"))
        print(json.dumps(get(f"/home?req=live2-home-{int(time.time())}&owner=live2_e6", 5.0), ensure_ascii=False))
        return 0
    if a.cmd == "stop":
        pid = alive_pid()
        st = get("/state", 1.0)
        ap_ = approval_now()
        if a.no_home:
            print("home skipped (--no-home: no servo command; student exit sends F0 = focus stop only)")
        elif FAULT_FILE.exists() and not manual_console():   # latched focus fault: same rule as e6_pipeline.after_focus
            # (a console in MANUAL_FIXED_NO_MOTOR never drives the focus motor: the DC-focus lock stays, HOME is a servo move)
            audit("NOT_SENT_FOCUS_FAULT", kind="home(stop)", fault_file=str(FAULT_FILE))
            print(f"home skipped: focus fault latched ({FAULT_FILE.name}) -> operator checks the head")
        elif not ap_["valid"]:
            audit("REFUSED_NO_VALID_APPROVAL", kind="home(stop)", approval_file=ap_.get("file"), why=ap_.get("why"))
            print(f"home skipped: hardware approval B {ap_.get('why')} -> operator checks the head (shutdown itself continues)")
        elif "_error" not in st and str(st.get("device_state")) not in ("UNKNOWN", "ERROR"):
            audit("SENT", kind="home(stop)", approval_file=ap_.get("file"), approval_expires=ap_.get("expires_epoch"))
            print("home:", json.dumps(get(f"/home?req=live2-stop-{int(time.time())}&owner=live2_e6", 5.0))[:300])
            time.sleep(3.0)
        else:
            print("home skipped: device", st.get("device_state", st.get("_error")), "-> operator checks the head")
        audit("SHUTDOWN_SIGINT", kind="stop", pid=pid, note="student exit path sends F0 only (focus stop); always allowed")
        if pid:
            os.killpg(os.getpgid(pid), signal.SIGINT)
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
        print("port holders after stop:", holders(PORT_DEV) or "none")
        return 0


if __name__ == "__main__":
    sys.exit(main())
