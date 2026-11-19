#!/usr/bin/env python3
"""LIVE1 control (called by LROOT/scripts/live1_ctl.sh). Default / `check` / `status` never start or stop anything.

  check [--graph]           gate report (process / port / resource / approval / contract / code identity); --graph adds a
                            DISCOVERY-ONLY ROS graph probe (no subscription)
  status                    owned LIVE1 components only
  start MODULE [opts]       MODULE in fastlio | lio | observe | web. Needs a valid user approval record listing MODULE
                            (web --test-only excepted). Every module runs under live1_supervisor (own session, owner record).
  stop MODULE|all           only LIVE1-owned processes, identity = PID + /proc starttime; stop-file + SIGTERM to the supervisor,
                            which stops its child tree (SIGINT, then SIGKILL after 10 s for software only)
  measure STAGE SECONDS     resource/FAST-LIO aggregate row -> reports/LIVE_FIELD1_RESOURCE_METRICS.csv
  baseline-a1 SECONDS       FAST-LIO-only baseline (Hz p50, inter-arrival p95) -> runtime/state/baseline_A1.json
  record-approval KIND ...  write the record of a USER-TYPED approval (verbatim text file, boot-bound, expiry)
  confirm-contract ...      write the verified odom frame contract (draft + human confirmer + evidence)
Never: sensor drivers / ROPUS services, /lidar_points subscription, Go2 commands. Pan-tilt / focus / HOME / serial / IMX477
only through module e6 (2026-10-22, E8 automatic path): refused unless hardware approval B ('pantilt', this boot) is valid;
it starts the UNCHANGED student aim console through live2/e6_aim_console_ctl.py (its own preflight, single instance) and the
precision bridge (--execute --focus-mode from_approval), which re-checks approval B before every command.
"""
import argparse
import csv
import hashlib
import json
import shutil
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live1_common as C  # noqa: E402

L = C.LROOT
CAND = C.CAND
HOME = Path.home()
SCI = HOME / "sci_ws"
FASTLIO_WS = SCI / "external" / "fastlio_ws"
FASTLIO_SRC = FASTLIO_WS / "src" / "FAST_LIO_ROS2"
FASTLIO_BIN = FASTLIO_WS / "install" / "fast_lio" / "lib" / "fast_lio" / "fastlio_mapping"
FASTLIO_CFG = SCI / "config" / "fastlio" / "fastlio_go2_esy_extrinsic.yaml"      # verified SCI config, read-only
DDS_LIVE = SCI / "config" / "fastlio" / "cyclonedds_live.xml"                     # verified SCI DDS profile, read-only
# 'fastlio --input body' (2026-10-30): Go2 body IMU -> sci_alt_imu bridge -> same FAST-LIO binary (alt config, /sci_alt_imu/*)
# -> l1_adapter_node -> /Odometry + /cloud_registered in the L1-IMU frame contract. Everything downstream (lio2 with FASTLIO_CFG,
# hub, E2..E8) is unchanged. Default stays 'l1'. Files live in the separate alt workspace (read-only from here).
ALT_WS = HOME / "sci_ws_alt_imu_20261029_204740"
BODY_LAUNCHER = ALT_WS / "scripts" / "body_fastlio_live2.sh"
BODY_ADAPTER_BIN = ALT_WS / "install" / "sci_alt_imu_bridge" / "lib" / "sci_alt_imu_bridge" / "l1_adapter_node"
BODY_BRIDGE_BIN = ALT_WS / "install" / "sci_alt_imu_bridge" / "lib" / "sci_alt_imu_bridge" / "bridge_node"
BODY_CALIB = ALT_WS / "config" / "calib" / "alt_imu_extrinsic_DYNFIELD_20261030_CANDIDATE.yaml"
INPUT_MODES = ("l1", "body")
E6_MODES = ("AUTO", "MANUAL_SETTLE")                    # 2026-10-30: e6 run mode (e6_settle); AUTO = existing path
MODEL = SCI / "models" / "siglip-base-patch16-224"
SCI_PY = SCI / ".venv" / "bin" / "python"
WEB_PORT = 8770
VENV_PY = SCI / "external" / "sci_live" / "venv_gpu" / "bin" / "python"           # student venv (numpy/scipy/torch), read-only
STORE_LIVE, STORE_REPLAY = ((L / "runtime" / "fake_store", L / "runtime" / "fake_store_replay") if C.FAKE else
                             (Path("/dev/shm/live2_store"), Path("/dev/shm/live2_store_replay")))
LIVE2_MODULES = ("fastlio", "lio", "lio2", "hub", "observe", "rec", "e2", "e34", "e6", "replay", "web", "console")
E6_CTL = CAND / "live2" / "e6_aim_console_ctl.py"
CTL_DIR = L / "runtime" / "ctl"                                                     # operator token, console audit
KNOWN_PORTS = {8765: "student live_viewer", 8088: "student aim_console", 8095: "student E5 UI", 8790: "ROPUS control gateway"}
SUP = CAND / "live1" / "live1_supervisor.py"
PROTECT_ARGS = sum((["--protect", p] for p in ("realsense2_camera_node", "hesai_ros_driver_node", "go2_control_runner", "robot_state_publisher")), [])
ROS_ENV = ("set +u; source /opt/ros/foxy/setup.bash >/dev/null 2>&1; "
           # 2026-10-19: under systemd (no ~/.bashrc) plain Foxy CycloneDDS segfaulted the graph probe; load the Unitree
           # ROS 2 env explicitly, as every interactive shell on this robot does (.bashrc -> unitree_ros2/setup.sh)
           f"[ -f {HOME}/unitree_ros2/setup.sh ] && source {HOME}/unitree_ros2/setup.sh >/dev/null 2>&1; "
           "export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 "
           f"RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOG_DIR={L}/runtime/roslog PYTHONDONTWRITEBYTECODE=1; ")


def say(*a):
    print("[live1_ctl]", *a, flush=True)


def port_in_use(port):
    out = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True).stdout
    return any(line.split()[3].rsplit(":", 1)[-1] == str(port) for line in out.splitlines() if len(line.split()) > 3)


def graph():
    out = L / "evidence" / f"graph_probe_{time.strftime('%H%M%S')}.json"
    r = subprocess.run(["bash", "-c", ROS_ENV + f"unset CYCLONEDDS_URI; timeout 30 /usr/bin/python3 {CAND}/live1/graph_probe.py --out {out}"],
                       capture_output=True, text=True)
    d = C.read_json(out)
    if d is None:
        return dict(ok=False, why="GRAPH_PROBE_FAILED", stderr=r.stderr[-400:])
    k = d["key"]
    npub = lambda t: len(k.get(t, {}).get("publishers", []))  # noqa: E731
    nsub = lambda t: len(k.get(t, {}).get("subscribers", []))  # noqa: E731
    return dict(ok=True, file=str(out), lidar_points_pubs=npub("/lidar_points"), lidar_points_subs=nsub("/lidar_points"),
                imu_pubs=npub("/utlidar/imu"), odom_pubs=npub("/Odometry"), odom_subs=nsub("/Odometry"),
                color_pubs=npub("/camera/color/image_raw"), color_subs=nsub("/camera/color/image_raw"),
                depth_pubs=npub("/camera/aligned_depth_to_color/image_raw"), info_pubs=npub("/camera/color/camera_info"),
                cloud_registered_pubs=npub("/cloud_registered"), cloud_registered_subs=nsub("/cloud_registered"),
                nodes=d["nodes"])


def code_identity():
    """Static import resolution (importlib find_spec, nothing executed) with the exact sys.path observe_node will use."""
    py = f"""
import importlib.util as u, sys, json, os
w=os.environ['SCI_WS']
sys.path[:0]=[w+'/external/sci_event/e5_onboard', w+'/src/sci_visual_observation', w+'/src/sci_sensor_bridge', w+'/src/sci_pantilt', w+'/src/sci_common']
out={{}}
for m in ('e5_onboard','e5_node','sources','ros_readonly','planner','e5rt','odom_base','timing'):
    s=u.find_spec(m); out[m]=None if s is None else s.origin
print(json.dumps(out))"""
    env = dict(os.environ, SCI_WS=str(CAND))
    r = subprocess.run([str(SCI_PY), "-I", "-B", "-c", py], capture_output=True, text=True, env=env)
    try:
        mods = json.loads(r.stdout)
    except ValueError:
        return dict(ok=False, why="IMPORT_RESOLUTION_FAILED", stderr=r.stderr[-300:])
    outside = {k: v for k, v in mods.items() if not v or not str(Path(v).resolve()).startswith(str(CAND.resolve()) + "/")}
    sci_env = (CAND / "src" / "sci_bringup" / "sci_env.sh").read_text()
    files = [CAND / "live1" / n for n in ("ctl.py", "live1_common.py", "lio_monitor.py", "observe_node.py", "live1_web.py",
                                         "live1_supervisor.py", "frame_probe.py", "graph_probe.py", "web/index.html")]
    files += [CAND / "src/sci_sensor_bridge/ros_readonly.py", CAND / "external/sci_event/e5_onboard/e5_onboard.py",
              CAND / "src/sci_visual_observation/e5_node.py"]
    return dict(ok=not outside, modules=mods, outside_candidate=outside,
                sci_env_overwrites_SCI_WS="export SCI_WS=$HOME/sci_ws" in sci_env,
                sci_env_note="LIVE1 never sources sci_env.sh; SCI_WS is set explicitly to the candidate and checked in observe_node",
                file_sha256={str(p.relative_to(L)): C.sha256(p) for p in files if p.exists()})


def approval_gates(g, module, run_s, accum):
    """Approval A must list the module (the console uses the 'web' grant), the run time must fit its approved limit."""
    apmod = "web" if module == "console" else module
    ap = C.approval_status("A", apmod)
    g["approval_A"] = dict(ok=ap["valid"], **ap)
    lim = (ap.get("limits") or {}).get(apmod, {})
    if run_s is not None:
        g["run_limit"] = dict(ok=lim.get("max_run_s") is not None and run_s <= float(lim["max_run_s"]), requested=run_s,
                              approved=lim.get("max_run_s"))
    if accum:
        g["accum_approved"] = dict(ok=bool(lim.get("accum_cloud_registered")), approved=lim.get("accum_cloud_registered"))


def fake_gates(module, run_s, accum, test_only):
    """FAKE workspace only: robot-specific gates are fixtures (fake_inject.json can fail any of them); approval, run limit
    and ownership are evaluated exactly as in the real workspace."""
    inj = C.read_json(L / "fake_inject.json") or {}
    g = dict(workspace_marker=dict(ok=C.MARKER.exists(), fake=True))
    for k in ("protected_processes", "student_components_not_running", "resources_start", "ports"):
        g[k] = dict(ok=True, fake=True)
    if module:
        g["no_duplicate_" + module] = dict(ok=not C.owner_alive(C.owner_record(module)), note="FAKE: owner record only")
        if not ((module in ("web", "e2", "e34") and test_only) or module in ("rec", "replay", "console")):
            approval_gates(g, module, run_s, accum)
        for name in (inj.get("gate_fail") or {}).get(module, []):
            g[name] = dict(ok=False, fake_injected=True)
    return g


def python_runs(pid, script):
    """True when pid is a python interpreter whose script argument ends with `script` (2026-10-19: a shell command line
    that merely contained 'live2/console.py' blocked the web console start)."""
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except OSError:
        return False
    argv = [a.decode(errors="replace") for a in argv if a]
    return bool(argv) and Path(argv[0]).name.startswith("python") and any(a.endswith(script) for a in argv[1:])


def gates(module=None, need_graph=False, run_s=None, accum=False, test_only=False, keep=False, port=None, input_mode="l1"):
    if C.FAKE:
        return fake_gates(module, run_s, accum, test_only)
    port = port or WEB_PORT
    g = {}
    g["workspace_marker"] = dict(ok=C.MARKER.exists())
    prot = {n: C.comm_count(n) for n in C.PROTECTED}
    g["protected_processes"] = dict(ok=all(prot[n] == c for n, c in C.PROTECTED.items()), counts=prot,
                                    note="LIVE1 never signals these; expected exactly one each")
    student = {p: C.find_pids(p) for p in C.CONFLICTS["student"]}
    g["student_components_not_running"] = dict(ok=not any(student.values()), found={k: v for k, v in student.items() if v})
    res = C.resources()
    g["resources_start"] = dict(ok=res["max_temp_c"] is not None and res["max_temp_c"] < 72.0 and res["mem_available_mb"] >= 4300,
                                max_temp_c=res["max_temp_c"], mem_available_mb=round(res["mem_available_mb"]),
                                rule="start: temp < 72 C and MemAvailable >= 4300 MB")
    free_gb = shutil.disk_usage(str(L)).free / 2 ** 30
    g["storage"] = dict(ok=free_gb >= 5.0, free_gb=round(free_gb, 1), rule="experiment start needs >= 5 GB free (records, logs)")
    g["ports"] = dict(ok=not port_in_use(port) or C.owner_alive(C.owner_record("web")) or C.owner_alive(C.owner_record("console")),
                      live1_web=port,
                      in_use={p: n for p, n in KNOWN_PORTS.items() if port_in_use(p)})
    g["frame_contract"] = dict(ok=C.read_json(C.CONTRACT) is not None, path=str(C.CONTRACT),
                               note="missing -> geometry/STOP evaluation stays off (semantic search on a flagged identity pose)")
    if module:
        conf = {p: C.find_pids(p) for p in C.CONFLICTS.get(module, [])}
        if module == "console":                 # only a python process RUNNING console.py/live1_web.py counts (an editor,
            conf = {p: [x for x in v if python_runs(x, p)] for p, v in conf.items()}  # tail, grep or shell line does not)
        g["no_duplicate_" + module] = dict(ok=not any(conf.values()) and not C.owner_alive(C.owner_record(module)),
                                            found={k: v for k, v in conf.items() if v})
        if not ((module in ("web", "e2", "e34") and test_only) or module in ("rec", "replay", "console")):
            # console (UI V2) = file/HTTP only: it shows "approval needed" and every module it starts passes ctl's own gate
            # rec / replay = file-only (no ROS/GPU/device); e2 / e34 / web with --test-only run on the REPLAY tree only
            approval_gates(g, module, run_s, accum)
    if module == "fastlio":
        g["fastlio_build"] = dict(ok=FASTLIO_BIN.exists() and FASTLIO_CFG.exists(), bin=str(FASTLIO_BIN),
                                  bin_sha256=C.sha256(FASTLIO_BIN) if FASTLIO_BIN.exists() else None,
                                  cfg_sha256=C.sha256(FASTLIO_CFG) if FASTLIO_CFG.exists() else None,
                                  log_redirect="Log/ bind-mounted to LROOT/runtime/fastlio/Log inside a private mount namespace",
                                  input=input_mode)
        if input_mode == "body":
            fs = [BODY_LAUNCHER, BODY_ADAPTER_BIN, BODY_BRIDGE_BIN, BODY_CALIB]
            g["fastlio_body_input"] = dict(ok=all(f.exists() for f in fs), missing=[str(f) for f in fs if not f.exists()],
                                           sha256={f.name: C.sha256(f) for f in fs if f.exists()},
                                           note="body IMU (/lowstate) -> bridge -> FAST-LIO (alt cfg) -> l1_adapter_node -> /Odometry,/cloud_registered; "
                                                "launcher refuses without a /lowstate publisher or with any existing /sci_alt_imu publisher")
    if module == "observe":
        g["model_present"] = dict(ok=(MODEL / "model.safetensors").exists(), model=str(MODEL))
    if module == "observe" or need_graph:
        g["code_identity"] = code_identity()
    if need_graph or module in ("fastlio", "lio", "lio2", "observe", "hub"):
        gr = graph()
        g["ros_graph"] = gr
        if gr.get("ok"):
            if module == "fastlio" and input_mode == "body":
                g["fastlio_topology"] = dict(ok=gr["lidar_points_pubs"] == 1 and gr["odom_pubs"] == 0 and gr["cloud_registered_pubs"] == 0,
                                             lidar_points_pubs=gr["lidar_points_pubs"], odom_pubs=gr["odom_pubs"], input="body",
                                             rule="one Hesai publisher, no other /Odometry or /cloud_registered producer "
                                                  "(the /lowstate body IMU is checked by the body launcher)")
            elif module == "fastlio":
                g["fastlio_topology"] = dict(ok=gr["lidar_points_pubs"] == 1 and gr["imu_pubs"] == 1 and gr["odom_pubs"] == 0,
                                             lidar_points_pubs=gr["lidar_points_pubs"], lidar_points_subs_before=gr["lidar_points_subs"],
                                             imu_pubs=gr["imu_pubs"], odom_pubs=gr["odom_pubs"],
                                             rule="one Hesai publisher, one IMU publisher, no other /Odometry producer")
            if module == "hub" or (module == "observe" and not keep):
                g["camera_topology"] = dict(ok=gr["color_pubs"] == 1 and gr["depth_pubs"] == 1 and gr["info_pubs"] == 1 and gr["color_subs"] == 0,
                                            color_pubs=gr["color_pubs"], color_subs_before=gr["color_subs"], depth_pubs=gr["depth_pubs"])
            if module in ("lio", "lio2"):
                g["odometry_present"] = dict(ok=gr["odom_pubs"] == 1, odom_pubs=gr["odom_pubs"],
                                             note="lio monitor reads FAST-LIO output only; 0 publishers -> NO_ODOMETRY")
    return g


def print_gates(g):
    for k, v in g.items():
        if isinstance(v, dict) and "ok" in v:
            extra = {kk: vv for kk, vv in v.items() if kk not in ("ok", "file_sha256", "modules", "nodes")}
            tag = "PASS" if v["ok"] else ("WARN" if k == "frame_contract" else "FAIL")
            say(f"{tag}  {k}  {json.dumps(extra, ensure_ascii=False, default=str)[:300]}")


def write_owner(module, proc, extra):
    s = C.proc_stat(proc.pid)
    rec = dict(module=module, pid=proc.pid, start=s["start"], pgid=s["pgid"], started_wall=C.now_wall(), boot_id=C.boot_id(), **extra)
    C.atomic_json(C.OWNED / f"{module}.json", rec)
    return rec


def launch(module, child_cmd, sup_args, extra):
    C.OWNED.mkdir(parents=True, exist_ok=True)
    stopf = C.OWNED / f"{module}.stop"
    if stopf.exists():
        stopf.unlink()
    log = L / "logs" / f"{module}_{time.strftime('%Y%m%d_%H%M%S')}"
    (L / "logs").mkdir(parents=True, exist_ok=True)
    if C.FAKE:                                   # test workspace: same supervisor + ownership, fake child, no robot dependence
        mx = sup_args[sup_args.index("--max-s") + 1] if "--max-s" in sup_args else "600"
        sup_args = ["--max-s", mx, "--start-temp", "150", "--abort-temp", "150", "--start-mb", "0", "--abort-mb", "0", "--wait-start-s", "1"]
        child_cmd = f"exec /usr/bin/python3 -I {CAND}/live2/fake_module.py --module {module} --root {L}"
    cmd = ["/usr/bin/python3", "-I", str(SUP), "--log", f"{log}.sup.jsonl", "--module", module, "--stop-file", str(stopf),
           "--state", str(C.OWNED / f"{module}.sup_state.json"), *sup_args, "--", "bash", "-c", child_cmd]
    out = open(f"{log}.out", "w")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(L))
    rec = write_owner(module, p, dict(log=f"{log}.out", sup_log=f"{log}.sup.jsonl", cmd=child_cmd, **extra))
    say(f"started {module}: supervisor pid {p.pid} (owner record {C.OWNED / (module + '.json')}) log {log}.out")
    return rec


HUB = (L / "runtime" / "fake_hub") if C.FAKE else Path("/dev/shm/live1_hub")


def start_e6(a):
    """E8 automatic E6: approval B -> student aim console (e6_aim_console_ctl: port/camera free, single instance, temp < 72 C,
    MemAvailable >= 4300 MB) -> precision bridge under live1_supervisor. Run time never past the approval B expiry."""
    ap = C.approval_status("B", "pantilt")
    if not ap["valid"]:
        say(f"REFUSED start e6: hardware approval B {ap['why']} (pan-tilt / focus / IMX477 need a B record for THIS boot)")
        return 3
    sys.path.insert(0, str(CAND / "live2"))
    import e6_precision_bridge as B
    fm, why = B.focus_mode_from_approval(ap)
    if fm is None:
        say(f"REFUSED start e6: {why}")
        return 3
    if C.owner_alive(C.owner_record("e6")):
        say("REFUSED start e6: already running")
        return 3
    run_s = min(float(a.run_s), float(ap.get("expires_epoch") or 0) - time.time() - 30.0)
    if run_s < 60:
        say(f"REFUSED start e6: approval B expires in {run_s + 30:.0f} s")
        return 3
    if not C.FAKE:
        r = subprocess.run(["/usr/bin/python3", "-I", "-B", str(E6_CTL), "start", "--af-strategy", "student_stepwise"]
                           + (["--focus-mode", "manual"] if fm == "MANUAL_FIXED_NO_MOTOR" else []),
                           capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            say(f"REFUSED start e6: student aim console not started (rc {r.returncode}): {(r.stdout + r.stderr)[-600:]}")
            return 3
        say("e6: student aim console started (Nano reset -> boot pose -> HOME)"
            + (" — MANUAL_FIXED_NO_MOTOR: focus motor refused at the console" if fm == "MANUAL_FIXED_NO_MOTOR" else ""))
    vlm = "1" if ((ap.get("limits") or {}).get("e6_vlm") is True) else "0"
    child = (f"export SCI_PANTILT_APPROVED=1 SCI_E6_VLM={vlm} PYTHONDONTWRITEBYTECODE=1; exec taskset -c 4 /usr/bin/python3 -s "
             f"{CAND}/live2/e6_precision_bridge.py --state {C.STATE} --hub {HUB} --execute --focus-mode from_approval --max-s {run_s:.0f}"
             f" --settle-mode {getattr(a, 'e6_mode', 'AUTO') or 'AUTO'}"
             + (" --reading deferred" if os.environ.get("LIVE2_E8_POLICY", "E8_PAIR") == "E8_PAIR" else ""))
    rec = launch("e6", child, ["--max-s", str(run_s + 30), "--no-resource-abort", "--start-temp", "72", "--start-mb", "4300",
                               "--wait-start-s", "5"], dict(run_s=run_s, focus_mode=fm, approval_B=ap.get("file"), vlm=vlm == "1",
                                                            e6_mode=getattr(a, "e6_mode", "AUTO") or "AUTO"))
    return 0 if rec else 1


def stop_e6_console():
    """After the bridge: student aim console HOME (if the device is clean and approval B valid) then SIGINT (student exit: F0)."""
    pidf = L / "runtime" / "owned" / "e6_aim_console.pid"
    try:
        os.kill(int(pidf.read_text().split()[0]), 0)
    except (OSError, ValueError, IndexError):
        return 0                                 # no student aim console running (FAKE workspaces never start one)
    br = C.read_json(C.STATE / "e6_bridge.json") or {}
    blocked = br.get("state") in ("BLOCKED", "BLOCKED_ENDED", "LOST")
    if blocked:                                  # device / focus fault: no automatic HOME (pipeline rule) — operator checks
        say(f"e6: HOME NOT sent at shutdown: E6 blocked ({br.get('why')}) — operator checks the head before any motion")
    r = subprocess.run(["/usr/bin/python3", "-I", "-B", str(E6_CTL), "stop"] + (["--no-home"] if blocked else []),
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    say(f"e6: aim console stop rc {r.returncode}: {(r.stdout + r.stderr)[-300:]}")
    return r.returncode


def start(a):
    m = a.module
    if m == "e6":
        return start_e6(a)
    run_s = a.run_s
    if m == "web" and getattr(a, "replay", False):
        a.test_only = True                     # REPLAY web = stored-input view, labelled TEST_ONLY, no approval needed
    if m == "console":
        run_s = min(float(run_s), 43200.0)                       # console: no approval needed, own 12 h cap
        a.run_s = run_s
    elif not ((m in ("web", "e2", "e34") and a.test_only) or m == "replay"):
        ap0 = C.approval_status("A", None if m == "rec" else m)
        if ap0.get("expires_epoch"):
            rem = float(ap0["expires_epoch"]) - time.time() - 30.0      # never run past the approval expiry
            if rem < run_s:
                say(f"run_s {run_s:.0f} capped to approval remaining {rem:.0f}s")
                run_s = max(0.0, rem)
                a.run_s = run_s
    inp = getattr(a, "input", "l1") or "l1"
    if inp not in INPUT_MODES or (inp != "l1" and m != "fastlio"):
        say(f"REFUSED start {m}: --input {inp} (only 'fastlio' takes --input l1|body)")
        return 3
    if (getattr(a, "e6_mode", "AUTO") or "AUTO") != "AUTO" and m != "e6":
        say(f"REFUSED start {m}: --e6-mode {a.e6_mode} (only 'e6' takes --e6-mode)")
        return 3
    g = gates(m, run_s=run_s if m in ("fastlio", "observe", "lio", "lio2", "hub", "e2", "e34") else None, accum=a.accum,
              test_only=a.test_only, keep=getattr(a, "keep", False), port=getattr(a, "port", None), input_mode=inp)
    print_gates(g)
    hard = [k for k, v in g.items() if isinstance(v, dict) and "ok" in v and not v["ok"] and k not in ("frame_contract",)
            and not (k == "ports" and m not in ("web", "console")) and not (k == "resources_start" and m in ("web", "console", "e34", "replay"))
            and not (k == "resources_start" and m == "e2")]                     # e2 has its own per-floor memory/thermal hold
    C.atomic_json(L / "reports" / f"gates_start_{m}_{time.strftime('%H%M%S')}.json", g)
    if hard:
        say(f"REFUSED start {m}: {hard}")
        return 3
    tag = time.strftime("%Y%m%d_%H%M%S")
    if m == "fastlio":
        rt = L / "runtime" / "fastlio"
        (rt / "Log").mkdir(parents=True, exist_ok=True)
        (rt / "PCD").mkdir(parents=True, exist_ok=True)
        binds = f"mount --bind {rt}/Log {FASTLIO_SRC}/Log && " + (f"mount --bind {rt}/PCD {FASTLIO_SRC}/PCD && " if (FASTLIO_SRC / "PCD").is_dir() else "")
        if inp == "body":                    # same namespace + Log/PCD bind mount as L1; the launcher runs bridge, FAST-LIO, adapter
            brun = rt / f"body_{tag}"
            inner = f"cd {rt} && " + binds + f"exec bash {BODY_LAUNCHER} {brun} {BODY_CALIB}"
            child = (ROS_ENV + f"source {HOME}/unitree_ros2/setup.sh >/dev/null 2>&1; source {HOME}/go2_ws/install/setup.bash >/dev/null 2>&1; "
                     f"source {FASTLIO_WS}/install/local_setup.bash; export CYCLONEDDS_URI=file://{DDS_LIVE}; "
                     f"P=$(ros2 pkg prefix fast_lio); case \"$P\" in {FASTLIO_WS}/*) ;; *) echo BLOCK_PREFIX $P; exit 6;; esac; "
                     f"exec unshare -Urm bash -c '{inner}'")
            return 0 if launch(m, child, ["--max-s", str(run_s), "--no-resource-abort", "--start-temp", "72", "--start-mb", "4300",
                                          "--wait-start-s", "5", *PROTECT_ARGS],
                               dict(run_s=run_s, input="body", body_run_dir=str(brun), body_calib=str(BODY_CALIB),
                                    body_calib_sha256=C.sha256(BODY_CALIB))) else 1
        inner = (f"cd {rt} && " + binds +
                 f"( /usr/bin/python3 {FASTLIO_SRC}/scripts/imu_correct.py & "
                 "ros2 run tf2_ros static_transform_publisher 0 0 0 1 0 0 0 camera_init map_viz --ros-args -r __node:=camera_init_to_map_viz & "
                 f"exec ros2 run fast_lio fastlio_mapping --ros-args -r __node:=laser_mapping --params-file {FASTLIO_CFG} )")
        child = (ROS_ENV + f"source {HOME}/unitree_ros2/setup.sh >/dev/null 2>&1; source {HOME}/go2_ws/install/setup.bash >/dev/null 2>&1; "
                 f"source {FASTLIO_WS}/install/local_setup.bash; export CYCLONEDDS_URI=file://{DDS_LIVE}; "
                 f"P=$(ros2 pkg prefix fast_lio); case \"$P\" in {FASTLIO_WS}/*) ;; *) echo BLOCK_PREFIX $P; exit 6;; esac; "
                 f"exec unshare -Urm bash -c '{inner}'")
        return 0 if launch(m, child, ["--max-s", str(run_s), "--no-resource-abort", "--start-temp", "72", "--start-mb", "4300",
                                      "--wait-start-s", "5", *PROTECT_ARGS], dict(run_s=run_s, input="l1")) else 1
    if m == "lio":
        acc = " --accum" if a.accum else ""
        child = ROS_ENV + f"export CYCLONEDDS_URI=file://{DDS_LIVE}; exec /usr/bin/python3 -s {CAND}/live1/lio_monitor.py --state {C.STATE} --max-s {run_s}{acc}"
        return 0 if launch(m, child, ["--max-s", str(run_s + 5), "--abort-temp", "80", "--abort-mb", "2500", "--wait-start-s", "5",
                                      *PROTECT_ARGS], dict(run_s=run_s, accum=a.accum)) else 1
    if m == "observe":
        shm = L / "runtime" / "e5_shm"
        shm.mkdir(parents=True, exist_ok=True)
        lim = (C.approval_status("A", m).get("limits") or {}).get(m, {})
        cprof = CAND / "config" / "e5rt_profiles"
        contract_env = f"export SCI_SEM1_ODOM_FRAME_CONTRACT={C.CONTRACT}; " if C.CONTRACT.exists() else "unset SCI_SEM1_ODOM_FRAME_CONTRACT; "
        child = (ROS_ENV + "unset CYCLONEDDS_URI; set -a; " + f". {cprof}/kci_20261017_field.env; . {cprof}/e5rt_candidate.env; set +a; "
                 f"export SCI_WS={CAND} SCI_SHM={shm} SCI_SIGLIP_MODEL={MODEL} SCI_E5_OBSERVE_MODE=shadow SCI_KCI_GATE=1 SCI_SEM1_WOULD_STOP=1 "
                 f"SCI_AIM_CONSOLE=http://127.0.0.1:1 SCI_PANTILT_URL=http://127.0.0.1:1 SCI_TRIAL_POINTER={shm}/trial_current.json "
                 f"SCI_PANTILT_CALIB={CAND}/config/pantilt_calib_multi_20260928.REF_COPY.json SCI_PY={SCI_PY} "
                 f"HF_HOME={L}/runtime/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME={L}/runtime/cache "
                 f"CUDA_CACHE_PATH={L}/runtime/cuda_cache SCI_SEM1_VISUAL_ONLY_IDENTITY=1 LIVE1_HUB_DIR={HUB} " +
                 ("" if a.keep else f"LIVE1_RUN_S={run_s} LIVE1_RUN_S_APPROVED_MAX={lim.get('max_run_s')}") + "; " + contract_env +
                 f"export LIVE2_E4_FEED={C.STATE}/e4_feed.json LIVE2_SIGLIP_MODE={siglip_mode()}; "   # E4 request drives SigLIP inference
                 "unset SCI_PANTILT_APPROVED SCI_ARM_WITHOUT_UI PYTHONNOUSERSITE; "
                 f"cd {CAND}; exec taskset -c 0,1 {SCI_PY} {CAND}/live1/observe_node.py --source {'shared_raw' if a.keep else 'ros_readonly'} "
                 f"--stage P2 --workers 2 --run-name live1_observe_{tag} --save-frames --e4-file {C.STATE}/e4_feed.json")
        if a.keep:
            if not (HUB / "index.json").exists():
                say("REFUSED observe --keep: sensor hub not running (start hub first)")
                return 3
            kcmd = (f"exec /usr/bin/python3 -I {CAND}/live1/observe_keeper.py --hub {HUB} --state {C.OWNED}/observe.keeper.json "
                    f"--log-prefix {L}/logs/observe_keep_{tag} --child-cmd {shlex.quote(child.replace(' exec taskset', ' exec taskset'))}")
            return 0 if launch(m, kcmd, ["--max-s", str(run_s + 60), "--no-resource-abort", "--start-temp", "100", "--start-mb", "0",
                                         "--wait-start-s", "5", *PROTECT_ARGS], dict(run_s=run_s, shm=str(shm), keep=True)) else 1
        return 0 if launch(m, child, ["--max-s", str(run_s + 120), "--start-temp", "72", "--abort-temp", "80", "--start-mb", "4300",
                                      "--abort-mb", "2500", "--wait-start-s", "5", "--no-dup", "siglip_worker.py", "--no-dup", "e5_onboard.py",
                                      *PROTECT_ARGS], dict(run_s=run_s, shm=str(shm))) else 1
    if m == "hub":
        HUB.mkdir(parents=True, exist_ok=True)
        contract_env = f"export SCI_SEM1_ODOM_FRAME_CONTRACT={C.CONTRACT}; " if C.CONTRACT.exists() else "unset SCI_SEM1_ODOM_FRAME_CONTRACT; "
        child = (ROS_ENV + "unset CYCLONEDDS_URI; " + contract_env +
                 f"exec taskset -c 2,3 /usr/bin/python3 -s {CAND}/live1/sensor_hub.py --hub {HUB} --live-jpeg {HUB}/live.jpg --max-s {run_s}")
        return 0 if launch(m, child, ["--max-s", str(run_s + 10), "--no-resource-abort", "--start-temp", "100", "--start-mb", "1000",
                                      "--wait-start-s", "5", *PROTECT_ARGS], dict(run_s=run_s, hub=str(HUB))) else 1
    if m == "rec":
        out = L / "runtime" / "field_eval" / tag
        child = (f"exec /usr/bin/python3 -I {CAND}/live1/field_recorder.py --state {C.STATE} --e5-shm {L}/runtime/e5_shm --out {out} "
                 f"--port 8771 --max-s {a.run_s}")
        return 0 if launch(m, child, ["--max-s", str(a.run_s + 10), "--start-temp", "100", "--start-mb", "500", "--abort-temp", "95",
                                      "--abort-mb", "300", "--wait-start-s", "5"], dict(out=str(out), port=8771)) else 1
    if m == "lio2":
        child = (ROS_ENV + f"export CYCLONEDDS_URI=file://{DDS_LIVE}; exec taskset -c 4 /usr/bin/python3 -s {CAND}/live2/lio_monitor2.py "
                 f"--state {C.STATE} --store {STORE_LIVE} --fastlio-cfg {FASTLIO_CFG} --max-s {run_s}")
        # CORE tier (like fastlio / hub): lio2 is the ONLY consumer that turns FAST-LIO /Odometry + /cloud_registered into the
        # odometry state, the map and the E2/E3 store. 2026-10-20 run X20261020_010413: its 80 C supervisor abort at 176 s cut
        # odometry / map / E2-E3 input while FAST-LIO itself kept running. Heat is now shed on the NON-core tier first (E2
        # in-flight fits stopped >= 78 C, SigLIP duty-cycled >= 76 C and paused at 80 C); lio2 logs alerts and keeps running.
        return 0 if launch(m, child, ["--max-s", str(run_s + 5), "--no-resource-abort", "--start-temp", "72", "--start-mb", "2500",
                                      "--wait-start-s", "5", *PROTECT_ARGS], dict(run_s=run_s, store=str(STORE_LIVE), tier="CORE")) else 1
    if m in ("e2", "e34"):
        rep = bool(a.test_only)
        state = L / "runtime" / ("state_REPLAY" if rep else "state")
        store = STORE_REPLAY if rep else STORE_LIVE
        state.mkdir(parents=True, exist_ok=True)
        if m == "e2":
            gpu = a.device == "gpu"
            if gpu and not ("e2_gpu" in (C.approval_status("A").get("modules") or [])):
                say("REFUSED e2 --device gpu: approval A does not list e2_gpu")
                return 3
            child = (("export LIVE2_E2_GPU_APPROVED=1; " if gpu else "export CUDA_VISIBLE_DEVICES=; ") +
                     "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; "
                     f"exec timeout {int(run_s) + 30} nice -n 10 taskset -c {a.cores} {VENV_PY} -I -B {CAND}/live2/e2_worker.py "
                     f"--building {a.building} --store {store} --out {state}/e2 --cache-dir {L}/runtime/e2_cache "
                     f"--procs {a.procs} --device {a.device}" + (f" --floors {a.floors}" if a.floors else "") + (f" --refresh-s {a.refresh_s}" if getattr(a, "refresh_s", 0) else ""))
            return 0 if launch(m, child, ["--max-s", str(run_s + 40), "--no-resource-abort", "--start-temp", "100", "--start-mb", "0",
                                          "--wait-start-s", "5"], dict(run_s=run_s, replay=rep, store=str(store), building=a.building)) else 1
        apb = C.approval_status("B", "pantilt")             # E8 pair reading: VLM only when approval B says e6_vlm (else OCR only)
        vlm_env = "SCI_E6_VLM=1 " if apb.get("valid") and (apb.get("limits") or {}).get("e6_vlm") is True else ""
        child = (f"export {vlm_env}OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; exec timeout {int(run_s) + 30} nice -n 5 taskset -c 5 "
                 f"/usr/bin/python3 -s {CAND}/live2/e34_node.py --store {store} --state {state} --building {a.building} --max-s {run_s} --hub {HUB}"
                 + (" --clock replay" if rep else ""))
        return 0 if launch(m, child, ["--max-s", str(run_s + 40), "--start-temp", "100", "--start-mb", "300", "--abort-temp", "95",
                                      "--abort-mb", "300", "--wait-start-s", "5"], dict(run_s=run_s, replay=rep, store=str(store))) else 1
    if m == "console":
        return start_console(a, run_s)
    if m == "replay":
        state = L / "runtime" / "state_REPLAY"
        state.mkdir(parents=True, exist_ok=True)
        child = (f"exec taskset -c 4 /usr/bin/python3 -s {CAND}/live2/replay_feed.py --data {a.data} --store {STORE_REPLAY} "
                 f"--state {state} --rate {a.rate} --hold-end-s {a.hold_end_s}")
        return 0 if launch(m, child, ["--max-s", str(a.run_s), "--start-temp", "100", "--start-mb", "300", "--abort-temp", "95",
                                      "--abort-mb", "300", "--wait-start-s", "5"], dict(replay=True, data=a.data)) else 1
    if m == "web":
        state = L / "runtime" / ("state_REPLAY" if a.replay else ("state_TEST_ONLY" if a.test_only else "state"))
        shm = L / "runtime" / ("e5_shm_TEST_ONLY" if a.test_only else "e5_shm")
        if a.replay:
            a.test_only = True
        child = (f"exec /usr/bin/python3 -I {CAND}/live1/live1_web.py --host {a.host} --port {WEB_PORT} --state {state} --e5-shm {shm} "
                 f"--trial-id {a.trial_id or tag}" + (" --test-only" if a.test_only else ""))
        env_state = dict(os.environ, LIVE1_STATE=str(state))
        os.environ.update(env_state)
        return 0 if launch(m, child, ["--max-s", str(a.run_s), "--start-temp", "100", "--start-mb", "500", "--abort-temp", "95",
                                      "--abort-mb", "300", "--wait-start-s", "5"], dict(test_only=a.test_only, port=WEB_PORT)) else 1
    say("unknown module")
    return 2


def start_console(a, run_s):
    """Operator console (live2/console.py: read-only panel + allow-listed control API). Started detached and NOT under
    live1_supervisor: the supervisor terminates every descendant it ever saw, so modules started from the console would
    die with it. The console itself is file/HTTP only; its run time is capped like the web (approval 'web' grant)."""
    port = a.port or WEB_PORT
    log = L / "logs" / f"console_{time.strftime('%Y%m%d_%H%M%S')}.out"
    (L / "logs").mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/python3", "-I", "-B", str(CAND / "live2" / "console.py"), "--host", a.host, "--port", str(port), "--max-s", str(run_s),
           "--root", str(L)]
    out = open(log, "w")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(L),
                         env=dict(os.environ, LIVE1_ROOT=str(L)))
    rec = write_owner("console", p, dict(log=str(log), cmd=" ".join(cmd), port=port, run_s=run_s, supervised=False))
    say(f"started console: pid {p.pid} port {port} (owner record {C.OWNED / 'console.json'}) log {log}")
    return 0 if rec else 1


def console_token():
    """New operator token for the web console (old sessions stay valid until they expire). Printed once, file mode 0600."""
    import secrets
    CTL_DIR.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(18)
    p = CTL_DIR / "operator_token"
    fd = os.open(str(p) + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    os.replace(str(p) + ".tmp", p)
    say(f"operator token / setup code written to {p} (0600; not printed — read it on the robot)")
    return 0


def siglip_mode():
    """observe inference policy: e4 (default; infer only on a fresh ACTIVE E4 request) | eval (stand-alone evaluation,
    labelled, not E4 integration) | shadow. Set explicitly with env LIVE2_SIGLIP_MODE when starting ctl / the console."""
    m = os.environ.get("LIVE2_SIGLIP_MODE", "e4")
    if m not in ("e4", "eval", "shadow"):
        raise SystemExit(f"LIVE2_SIGLIP_MODE {m!r} not in e4|eval|shadow")
    return m


def standing_grant(a):
    """STANDING_RESEARCH_AUTHORIZATION from the user's own text (file). Scope and limits are fixed in standing_auth.py;
    the device is the first trusted Tailscale device (the registered tablet)."""
    text = Path(a.text_file).read_text()
    dev = ((C.read_json(CTL_DIR / "auth" / "trusted_devices.json") or {}).get("devices") or [{}])[0]
    g = C.standing().grant(text, dev.get("name"), dev.get("stable_id") or dev.get("StableID"), a.recorded_by)
    say(f"standing grant {g['grant_id']} modules={g['modules']} excluded={g['excluded']} device={g['device']} (not boot-bound, no expiry)")
    return 0


def standing_status():
    st = C.standing().status()
    say(json.dumps({k: st.get(k) for k in ("valid", "why", "state", "grant_id", "modules", "device", "experiment_max_s", "created")},
                   ensure_ascii=False))
    return 0 if st["valid"] else 3


def standing_set(state, why):
    try:
        r = C.standing().set_state(state, "ctl", why)
    except (PermissionError, ValueError) as e:
        say(f"REFUSED standing {state}: {e}")
        return 3
    say(f"standing {r['grant_id']} -> {state}")
    return 0


def auth_retire_token():
    """After the account login is verified on the real tablet: the operator token no longer logs in (it remains only the
    robot-side setup code for a future re-enrollment). Existing token sessions are revoked."""
    sys.path.insert(0, str(CAND / "live2"))
    import auth as AUTH
    st = AUTH.AuthStore(CTL_DIR / "auth", CTL_DIR / "operator_token")
    if st.needs_enrollment():
        say("REFUSED: account 'recs' not enrolled yet — the token login is still the only way in")
        return 3
    d = st._sessions()
    d["sessions"] = {k: v for k, v in d["sessions"].items() if v.get("kind") != "legacy_token"}
    st._save_sessions(d)
    C.atomic_json(CTL_DIR / "auth" / "legacy_retired.json", dict(retired=C.now_wall(), by="ctl auth-retire-token"))
    say("legacy operator-token login retired; account login only")
    return 0


def auth_trust_device(ip):
    """Allow-list a Tailscale node for no-login CONTROL (device mode). The node is resolved by `tailscale whois` (read-only)
    and stored by its StableID; nothing in the Tailscale configuration is changed."""
    r = subprocess.run(["tailscale", "whois", "--json", ip], capture_output=True, text=True, timeout=5)
    try:
        d = json.loads(r.stdout)
        n, u = d["Node"], d.get("UserProfile") or {}
    except (ValueError, KeyError):
        say(f"REFUSED: tailscale whois {ip} failed: {r.stderr.strip()[:200]}")
        return 3
    tf = CTL_DIR / "auth" / "trusted_devices.json"
    (CTL_DIR / "auth").mkdir(parents=True, exist_ok=True)
    cur = C.read_json(tf) or dict(devices=[])
    cur["devices"] = [x for x in cur["devices"] if x.get("stable_id") != n["StableID"]] + [
        dict(stable_id=n["StableID"], name=n["Name"].split(".")[0], fqdn=n["Name"].rstrip("."), user=u.get("LoginName"),
             addresses=n.get("Addresses"), added=C.now_wall(), by="ctl auth-trust-device")]
    C.atomic_json(tf, cur)
    os.chmod(tf, 0o600)
    say(f"trusted for control: {n['Name'].split('.')[0]} ({n['StableID']}, {u.get('LoginName')})")
    return 0


def auth_untrust_device(name):
    tf = CTL_DIR / "auth" / "trusted_devices.json"
    cur = C.read_json(tf) or dict(devices=[])
    keep = [x for x in cur["devices"] if name not in (x.get("name"), x.get("stable_id"))]
    say(f"removed {len(cur['devices']) - len(keep)} device(s)")
    cur["devices"] = keep
    C.atomic_json(tf, cur)
    return 0


def auth_devices():
    for x in (C.read_json(CTL_DIR / "auth" / "trusted_devices.json") or {}).get("devices", []):
        say(f"{x.get('name')}  {x.get('stable_id')}  {x.get('user')}  added {x.get('added')}")
    say(f"mode: {(C.read_json(CTL_DIR / 'auth' / 'mode.json') or {}).get('mode', 'device')}")
    return 0


def auth_set_mode(mode):
    (CTL_DIR / "auth").mkdir(parents=True, exist_ok=True)
    C.atomic_json(CTL_DIR / "auth" / "mode.json", dict(mode=mode, set=C.now_wall()))
    say(f"auth mode = {mode} ({'no login; trusted Tailscale devices control' if mode == 'device' else 'account id/password login'})")
    return 0


def auth_dev_account():
    """Optional development account recs / recs for account mode only (weak by design, flagged dev)."""
    sys.path.insert(0, str(CAND / "live2"))
    import auth as AUTH
    st = AUTH.AuthStore(CTL_DIR / "auth", CTL_DIR / "operator_token")
    d = st.accounts
    rec = st._hash_record("recs", pw_version=int((d["users"].get("recs") or {}).get("pw_version", 0)) + 1)
    rec["dev_weak_password"] = True
    d["users"]["recs"] = rec
    st._save_accounts(d)
    st.logout_all("recs")
    say("development account recs/recs set (account mode only; device mode does not use it)")
    return 0


def auth_reset():
    """Robot-side password reset: removes the 'recs' account and all its sessions; re-enrollment needs a fresh setup code."""
    sys.path.insert(0, str(CAND / "live2"))
    import auth as AUTH
    AUTH.AuthStore(CTL_DIR / "auth", CTL_DIR / "operator_token").reset()
    say("account 'recs' removed; run console-token for a new setup code, then set the password on the tablet")
    return 0


def stop(module, wait_s=25.0):
    mods = ["e6", "observe", "e34", "e2", "replay", "rec", "hub", "lio2", "lio", "fastlio", "web", "console"] if module == "all" else [module]
    rc = 0
    for m in mods:
        rec = C.owner_record(m)
        if not rec:
            say(f"{m}: no owner record (nothing LIVE1-owned to stop)")
            continue
        if not C.owner_alive(rec):
            say(f"{m}: recorded pid {rec['pid']} not alive or starttime differs (PID reuse) -> NOT signalled")
            continue
        (C.OWNED / f"{m}.stop").write_text(C.now_wall())
        os.kill(rec["pid"], signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < wait_s and C.owner_alive(rec):
            time.sleep(0.3)
        st = C.read_json(C.OWNED / f"{m}.sup_state.json") or {}
        if C.owner_alive(rec):
            say(f"{m}: supervisor {rec['pid']} still alive after {wait_s}s -> left running, report to user (no blind kill)")
            rc = 1
        else:
            say(f"{m}: stopped ({st.get('end_reason')}, leftovers={st.get('leftovers')}) in {time.time() - t0:.1f}s")
    if "e6" in mods:                             # bridge stopped (or never ran): the aim console goes HOME and exits
        stop_e6_console()
    return rc


def status():
    for m in LIVE2_MODULES:
        rec = C.owner_record(m)
        st = C.read_json(C.OWNED / f"{m}.sup_state.json")
        say(f"{m:8s} owned={bool(rec)} alive={C.owner_alive(rec)} pid={(rec or {}).get('pid')} sup={json.dumps(st)[:200] if st else None}")
    return 0


def read_series(since):
    rows = []
    for p in (C.STATE / "lio_series.jsonl.1", C.STATE / "lio_series.jsonl"):
        if p.exists():
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["wall"] >= since:
                    rows.append(r)
    return rows


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return None if not v else v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def measure(stage, seconds, note=""):
    t0 = time.time()
    samples = []
    while time.time() - t0 < seconds:
        samples.append(C.resources())
        time.sleep(1.0)
    rows = read_series(t0)
    e5 = C.read_json(L / "runtime" / "e5_shm" / "status.json") or {}
    sc = (e5.get("scene") or {}).get("timing") or {}
    sh = e5.get("source_health") or {}
    row = dict(stage=stage, wall_start=time.strftime("%F %T", time.localtime(t0)), seconds=seconds, boot_id=C.boot_id(),
               odom_hz_p50=pct([r["hz"] for r in rows], 50), odom_hz_min=min([r["hz"] for r in rows if r["hz"]] or [None], key=lambda x: x or 0),
               odom_ia_p95_ms=pct([r["ia95"] for r in rows], 50), odom_age_p95_ms=pct([r["age95"] for r in rows], 50),
               odom_last_rx_age_max_s=max([r["last_rx_age"] for r in rows if r["last_rx_age"] is not None] or [None], key=lambda x: x or 0),
               lio_rows=len(rows), temp_max=max(s["max_temp_c"] for s in samples), mem_avail_min_mb=round(min(s["mem_available_mb"] for s in samples)),
               cpu_p50=pct([s["cpu_pct"] for s in samples], 50), gpu_p50=pct([s["gpu_load_pct"] for s in samples], 50),
               gpu_max=max([s["gpu_load_pct"] or 0 for s in samples]),
               color_hz=(sh.get("color") or {}).get("hz"), depth_hz=(sh.get("depth") or {}).get("hz"),
               e5_scene_ms_last=sc.get("scene_ms"), e5_siglip_ms_last=sc.get("siglip_infer_ms"),
               components=",".join(m for m in LIVE2_MODULES if C.owner_alive(C.owner_record(m))), note=note)
    out = L / "reports" / "LIVE_FIELD1_RESOURCE_METRICS.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    say(json.dumps(row, default=str))
    return row


def baseline_a1(seconds):
    row = measure("A1_baseline", seconds, note="FAST-LIO + lio monitor only")
    C.atomic_json(C.STATE / "baseline_A1.json", dict(hz_p50=row["odom_hz_p50"], ia_p95_ms=row["odom_ia_p95_ms"],
                                                    age_p95_ms=row["odom_age_p95_ms"], measured=row["wall_start"], seconds=seconds))
    return 0


def freedrive(a):
    """OPERATOR_FREE_DRIVE: start (or reuse) every module in order; each run time is capped to the approval expiry.
    No distance / angle script, frame probe optional. The operator drives with the joystick; nothing here moves the robot."""
    ap = C.approval_status("A")
    if not ap["valid"]:
        say(f"REFUSED freedrive: {ap['why']}")
        return 3
    rem = (float(ap["expires_epoch"]) - time.time() - 30.0) if ap.get("expires_epoch") else float(ap.get("experiment_max_s") or 1800)
    C.atomic_json(C.STATE / "session.json", dict(mode="OPERATOR_FREE_DRIVE", started_wall=C.now_wall(), started_epoch=time.time(),
                                                 approval=ap["file"], approval_expires_epoch=ap["expires_epoch"], boot_id=C.boot_id(),
                                                 note="no fixed distance or angle; operator joystick; Claude sends no motion command"))
    plan = [("fastlio", dict()), ("lio2", dict(accum=True)), ("hub", dict()), ("web", dict()), ("rec", dict()), ("observe", dict(keep=True)),
            ("e34", dict()), ("e2", dict())]
    for m, kw in plan:
        if C.owner_alive(C.owner_record(m)):
            say(f"{m}: already running (reused)")
            continue
        lim = float(((ap.get("limits") or {}).get(m) or {}).get("max_run_s", rem))      # per-module approved limit first
        ns = argparse.Namespace(module=m, run_s=min(rem, lim), accum=kw.get("accum", False), test_only=False, host="0.0.0.0", trial_id=None,
                                keep=kw.get("keep", False), building=a.building, procs=a.procs, device="cpu", cores="6,7",
                                floors=None, replay=False, data=None, rate=1.0, hold_end_s=0.0)
        rc = start(ns)
        if rc != 0:
            say(f"freedrive: {m} not started (rc {rc}); continuing with the modules that do not depend on it")
            if m in ("fastlio",):
                pass
        if m == "fastlio" and rc == 0:
            fl = Path(C.owner_record("fastlio")["log"])
            for _ in range(60):
                if fl.exists() and "Initialize the map kdtree" in fl.read_text(errors="replace"):
                    break
                time.sleep(1.0)
        if m == "hub" and rc == 0:
            for _ in range(30):
                if (HUB / "index.json").exists():
                    break
                time.sleep(0.5)
    status()
    return 0


def freedrive_end():
    """Stop every LIVE1 module (owned only) and write the session summary; the evaluation data stays on disk."""
    sess = C.read_json(C.STATE / "session.json") or {}
    rc = stop("all")
    sess.update(ended_wall=C.now_wall(), ended_epoch=time.time())
    C.atomic_json(C.STATE / "session.json", sess)
    out = L / "reports" / f"freedrive_session_{time.strftime('%Y%m%d_%H%M%S')}.json"
    C.atomic_json(out, dict(session=sess, components={m: C.read_json(C.OWNED / f"{m}.sup_state.json") for m in LIVE2_MODULES},
                            keeper=C.read_json(C.OWNED / "observe.keeper.json")))
    say(f"session saved {out}")
    return rc


def probe_frame(seconds):
    """G1-1 evidence: /Odometry + /tf_static samples while the OPERATOR walks straight and turns (joystick). Read-only."""
    ap = C.approval_status("A", "frame_probe")
    lim = (ap.get("limits") or {}).get("frame_probe", {})
    if not ap["valid"] or seconds > float(lim.get("max_run_s", 0)):
        say(f"REFUSED probe-frame: {ap['why'] or 'run limit'} (approved max {lim.get('max_run_s')})")
        return 3
    out = L / "evidence" / f"odom_frame_samples_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    say(f"collecting {seconds}s -> {out}. Operator: stand still 3 s, walk STRAIGHT forward >= 2 m (twice), then turn LEFT ~90 deg in place")
    r = subprocess.run(["bash", "-c", ROS_ENV + f"unset CYCLONEDDS_URI; exec timeout {seconds + 15} /usr/bin/python3 -s {CAND}/live1/frame_probe.py "
                        f"collect --out {out} --seconds {seconds}"])
    say(f"done rc={r.returncode}; next: frame_probe.py analyze --samples {out} --turn left --out <draft.json>")
    return r.returncode


def record_approval(a):
    text = Path(a.text_file).read_text()
    if len(text.strip()) < 20:
        say("REFUSED: approval text too short / empty")
        return 3
    limits = json.loads(a.limits) if a.limits else {}
    rec = dict(kind=a.kind, user_text=text, user_text_sha256=hashlib.sha256(text.encode()).hexdigest(), boot_id=C.boot_id(),
               granted_wall=C.now_wall(), granted_epoch=time.time(), expires_epoch=time.time() + 60 * a.minutes,
               modules=[m for m in a.modules.split(",") if m], limits=limits, recorded_by=a.recorded_by,
               note="record of an approval TYPED BY THE USER in the Claude Code chat; files/examples are never approvals")
    C.APPROVALS.mkdir(parents=True, exist_ok=True)
    p = C.APPROVALS / f"{a.kind}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    C.atomic_json(p, rec)
    say(f"approval record {p} expires {time.strftime('%F %T', time.localtime(rec['expires_epoch']))} modules={rec['modules']}")
    return 0


def confirm_contract(a):
    d = C.read_json(a.draft)
    if not d or not d.get("usable"):
        say(f"REFUSED: draft not usable: {None if not d else d.get('problems')}")
        return 3
    if not a.by.strip():
        say("REFUSED: confirmer required")
        return 3
    c = dict(frame_id=d["frame_id"], child_frame_id=d["child_frame_id"], body_transform=d["body_transform"], verified_by=a.by,
             evidence=dict(draft=str(Path(a.draft).resolve()), draft_sha256=C.sha256(a.draft), samples=d.get("evidence_samples"),
                           measurements=d.get("measurements")),
             confirmed_wall=C.now_wall(), boot_id=C.boot_id())
    C.atomic_json(C.CONTRACT, c)
    say(f"contract written {C.CONTRACT}: {c['body_transform']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    c = sp.add_parser("check")
    c.add_argument("--graph", action="store_true")
    sp.add_parser("status")
    s = sp.add_parser("start")
    s.add_argument("module", choices=["fastlio", "lio", "lio2", "observe", "web", "rec", "hub", "e2", "e34", "e6", "replay", "console"])
    s.add_argument("--port", type=int, default=None, help="web/console port (default 8770)")
    s.add_argument("--building", default="kci")
    s.add_argument("--procs", type=int, default=2)
    s.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    s.add_argument("--cores", default="6,7")
    s.add_argument("--refresh-s", type=float, default=0.0, help="e2 (R37 loc1): repeat coarse search every N s")
    s.add_argument("--floors", default=None, help="e2: space separated floor list (default: every floor of the building)")
    s.add_argument("--replay", action="store_true", help="web: show the REPLAY (stored input) tree")
    s.add_argument("--data", default=None, help="replay: recorded student dataset dir")
    s.add_argument("--rate", type=float, default=1.0)
    s.add_argument("--hold-end-s", type=float, default=0.0)
    s.add_argument("--keep", action="store_true", help="observe: OPERATOR_FREE_DRIVE keeper (SigLIP reads the sensor hub)")
    s.add_argument("--run-s", type=float, default=600.0)
    s.add_argument("--accum", action="store_true")
    s.add_argument("--test-only", action="store_true")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--trial-id", default=None)
    s.add_argument("--input", choices=INPUT_MODES, default="l1", help="fastlio only: l1 (default, L1 IMU) | body (Go2 body IMU via adapter)")
    s.add_argument("--e6-mode", choices=E6_MODES, default="AUTO",
                   help="e6 only: AUTO (default, existing path) | MANUAL_SETTLE (operator approves the pan-tilt after the body settled)")
    t = sp.add_parser("stop")
    t.add_argument("module", choices=["fastlio", "lio", "lio2", "observe", "web", "rec", "hub", "e2", "e34", "e6", "replay", "console", "all"])
    sp.add_parser("console-token")
    sp.add_parser("auth-retire-token")
    sp.add_parser("auth-reset")
    td = sp.add_parser("auth-trust-device")
    td.add_argument("tailscale_ip")
    ud = sp.add_parser("auth-untrust-device")
    ud.add_argument("name")
    sp.add_parser("auth-devices")
    am = sp.add_parser("auth-mode")
    am.add_argument("mode", choices=["device", "account"])
    sp.add_parser("auth-dev-account")
    m = sp.add_parser("measure")
    m.add_argument("stage")
    m.add_argument("seconds", type=float)
    m.add_argument("--note", default="")
    b = sp.add_parser("baseline-a1")
    b.add_argument("seconds", type=float)
    r = sp.add_parser("record-approval")
    r.add_argument("kind", choices=["A", "B"])
    r.add_argument("--text-file", required=True)
    r.add_argument("--minutes", type=float, required=True)
    r.add_argument("--modules", required=True)
    r.add_argument("--limits", default=None)
    r.add_argument("--recorded-by", default="claude-code (from the user's chat message)")
    fd = sp.add_parser("freedrive")
    fd.add_argument("--building", default="kci")
    fd.add_argument("--procs", type=int, default=2)
    sp.add_parser("freedrive-end")
    pf = sp.add_parser("probe-frame")
    pf.add_argument("seconds", type=float)
    sg = sp.add_parser("standing-grant", help="CLI only: record the user's STANDING_RESEARCH_AUTHORIZATION text")
    sg.add_argument("--text-file", required=True)
    sg.add_argument("--recorded-by", required=True)
    sp.add_parser("standing-status")
    for n in ("standing-suspend", "standing-resume", "standing-revoke"):
        sx = sp.add_parser(n)
        sx.add_argument("--why", default="")
    k = sp.add_parser("confirm-contract")
    k.add_argument("--draft", required=True)
    k.add_argument("--by", required=True)
    a = ap.parse_args()
    if a.cmd in (None, "check"):
        g = gates(need_graph=bool(getattr(a, "graph", False)))
        print_gates(g)
        out = L / "reports" / f"check_{time.strftime('%Y%m%d_%H%M%S')}.json"
        C.atomic_json(out, g)
        say(f"CHECK_ONLY: nothing was started or stopped. report {out}")
        return 0
    return dict(status=lambda: status(), start=lambda: start(a), stop=lambda: stop(a.module),
                measure=lambda: (measure(a.stage, a.seconds, a.note), 0)[1], **{"baseline-a1": lambda: baseline_a1(a.seconds)},
                **{"freedrive": lambda: freedrive(a), "freedrive-end": lambda: freedrive_end(), "probe-frame": lambda: probe_frame(a.seconds), "record-approval": lambda: record_approval(a), "confirm-contract": lambda: confirm_contract(a),
                   "console-token": lambda: console_token(), "auth-retire-token": lambda: auth_retire_token(),
                   "auth-reset": lambda: auth_reset(), "auth-trust-device": lambda: auth_trust_device(a.tailscale_ip),
                   "auth-untrust-device": lambda: auth_untrust_device(a.name), "auth-devices": lambda: auth_devices(),
                   "auth-mode": lambda: auth_set_mode(a.mode), "auth-dev-account": lambda: auth_dev_account(),
                   "standing-grant": lambda: standing_grant(a), "standing-status": lambda: standing_status(),
                   "standing-suspend": lambda: standing_set("SUSPENDED", a.why), "standing-resume": lambda: standing_set("ACTIVE", a.why),
                   "standing-revoke": lambda: standing_set("REVOKED", a.why)})[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
