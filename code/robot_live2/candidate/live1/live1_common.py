"""LIVE-FIELD1 shared helpers (stdlib only): paths, atomic JSON, resources, process identity, approval records.

Nothing here starts, stops or subscribes anything. Process signalling lives in live1_ctl (owner-verified only)."""
import glob
import hashlib
import json
import os
import socket
import time
from pathlib import Path

LROOT = Path(os.environ.get("LIVE1_ROOT", str(Path(__file__).resolve().parents[2])))
CAND = LROOT / "candidate"
RUNTIME = LROOT / "runtime"
STATE = Path(os.environ.get("LIVE1_STATE", str(RUNTIME / "state")))
OWNED = RUNTIME / "owned"
APPROVALS = RUNTIME / "approvals"
CONTRACT = RUNTIME / "contract" / "ODOM_FRAME_CONTRACT.json"
MARKER = next((LROOT / n for n in ("LIVE_FIELD2_WORKSPACE.json", "LIVE_FIELD1_WORKSPACE.json", "LIVE2_FAKE_WORKSPACE.json")
               if (LROOT / n).exists()), LROOT / "LIVE_FIELD1_WORKSPACE.json")   # LIVE2 workspace marker first
# FAKE = test workspace (tmp dir with only the fake marker): ctl replaces device children by live2/fake_module.py and the
# robot-specific gates by fixtures; approval / run-limit / ownership logic stays real. Never true in a real workspace.
FAKE = MARKER.name == "LIVE2_FAKE_WORKSPACE.json"
HZ = os.sysconf("SC_CLK_TCK")

# processes that belong to the robot / other projects: LIVE1 never signals them, only checks they stay alive
PROTECTED = {"hesai_ros_driver_node": 1, "realsense2_camera_node": 1, "robot_state_publisher": 1, "go2_control_runner": 1}
# components that must NOT already run when LIVE1 starts the same role (student / SEM1 / other copies)
CONFLICTS = {"fastlio": ["fastlio_mapping", "imu_correct.py", "laserMapping"],
             "observe": ["siglip_worker.py", "e5_onboard.py", "e5_node.py", "sem1_shadow_node.py", "live_siglip_worker"],
             "lio": ["live1/lio_monitor.py"],
             "web": ["live1/live1_web.py", "live2/console.py"],
             "console": ["live1/live1_web.py", "live2/console.py"],
             "rec": ["live1/field_recorder.py"],
             "hub": ["live1/sensor_hub.py"],
             "lio2": ["live1/lio_monitor.py", "live2/lio_monitor2.py"],
             "e2": ["live2/e2_worker.py"],
             "e34": ["live2/e34_node.py"],
             "replay": ["live2/replay_feed.py"],
             "student": ["rt_live_node.py", "live_viewer.py", "aim_console.py", "reader_onboard.py", "loc_worker.py",
                         "calib_console.py", "kci_field_up.sh", "sensors_go2.sh", "persistent_global_map_node"]}


def boot_id():
    return open("/proc/sys/kernel/random/boot_id").read().strip()


def now_wall():
    return time.strftime("%F %T %Z")


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ resources (read-only /proc and /sys)
def temps():
    out = {}
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            out[open(z + "/type").read().strip()] = int(open(z + "/temp").read()) / 1000.0
        except (OSError, ValueError):
            pass
    return out


def max_temp():
    t = [v for k, v in temps().items() if v > -40]
    return max(t) if t else None


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0
    return None


def gpu_load_pct():
    for p in ("/sys/devices/gpu.0/load", "/sys/devices/platform/gpu.0/load"):
        try:
            return int(open(p).read()) / 10.0
        except (OSError, ValueError):
            continue
    return None


_cpu_prev = [None]


def cpu_pct():
    """Whole-system CPU utilisation since the previous call (None on the first call)."""
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle, tot = v[3] + v[4], sum(v)
    prev, _cpu_prev[0] = _cpu_prev[0], (idle, tot)
    if prev is None or tot == prev[1]:
        return None
    return round(100.0 * (1 - (idle - prev[0]) / (tot - prev[1])), 1)


def resources():
    return dict(wall=time.time(), max_temp_c=max_temp(), temps=temps(), mem_available_mb=mem_available_mb(),
                gpu_load_pct=gpu_load_pct(), cpu_pct=cpu_pct(), loadavg=os.getloadavg())


# cmdline keyword -> role. Field run 2026-10-21 (EXP1): FAST-LIO fell to 5.8 Hz at 92-97 % system CPU and the records could
# not say which process took the CPU; this attributes it (read-only /proc, no signal, no priority change).
CPU_ROLES = (("fastlio_mapping", "fastlio"), ("lio_monitor2", "lio2"), ("sensor_hub", "hub"), ("siglip_worker", "siglip"),
             ("observe_node", "observe"), ("e2_worker", "e2"), ("loc_worker", "e2"), ("e34_node", "e34"),
             ("field_recorder", "rec"), ("realsense2_camera_node", "realsense"), ("hesai_ros_driver_node", "hesai"),
             ("console.py", "web"), ("llama-server", "vlm"), ("spawn_main", "mp_child"))   # mp_child: E2 fit children (spawn)
_proc_prev = [None]


def cpu_by_role():
    """CPU % (100 = one core) per role since the previous call, from /proc/<pid>/stat utime+stime; None on the first call."""
    tick = os.sysconf("SC_CLK_TCK")
    now, cur = time.monotonic(), {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            role = next((r for k, r in CPU_ROLES if k in cmd), None)
            if role is None:
                continue
            s = open(f"/proc/{d}/stat").read()
            f = s[s.rindex(")") + 2:].split()
            cur[(d, f[19])] = (role, int(f[11]) + int(f[12]))           # key: pid + starttime (pid reuse safe)
        except (OSError, ValueError, IndexError):
            continue
    prev, _proc_prev[0] = _proc_prev[0], (now, cur)
    if prev is None or now <= prev[0]:
        return None
    out = {}
    for k, (role, ticks) in cur.items():
        p = prev[1].get(k)
        if p is not None:
            out[role] = out.get(role, 0.0) + (ticks - p[1]) / tick / (now - prev[0]) * 100.0
    return {r: round(v, 1) for r, v in sorted(out.items(), key=lambda kv: -kv[1])}


# ------------------------------------------------------------------ process identity
def proc_stat(pid):
    try:
        s = open(f"/proc/{pid}/stat").read()
        rest = s[s.rindex(")") + 2:].split()
        return dict(state=rest[0], ppid=int(rest[1]), pgid=int(rest[2]), start=int(rest[19]), rss_mb=int(rest[21]) * 4096 / 2 ** 20)
    except (OSError, ValueError, IndexError):
        return None


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def argv(pid):
    try:
        return [a.decode(errors="replace") for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
    except OSError:
        return []


def find_pids(substr, exclude_self=True):
    """PIDs whose PROGRAM or a path-like ARGUMENT contains `substr`. A shell `-c` string, an editor or a grep that merely
    mentions the name is not a running instance (2026-10-19: an operator shell line containing 'live2/e2_worker.py'
    blocked the E2 start during a field run)."""
    me = {os.getpid(), os.getppid()}
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or (exclude_self and int(d) in me):
            continue
        av = argv(int(d))
        if not av or "live1_ctl" in " ".join(av) or os.path.basename(av[0]) in ("grep", "pgrep", "tail", "less", "vi", "vim", "nano", "code"):
            continue
        if any(substr in a and not any(ch.isspace() for ch in a) for a in av):
            out.append(int(d))
    return out


def comm_count(name):
    n = 0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if open(f"/proc/{d}/comm").read().strip() == name[:15]:
                    n += 1
            except OSError:
                pass
    return n


def owner_record(module):
    return read_json(OWNED / f"{module}.json")


def owner_alive(rec):
    """True only if the recorded PID still exists with the SAME /proc starttime (PID reuse safe)."""
    if not rec:
        return False
    s = proc_stat(rec["pid"])
    return bool(s) and s["start"] == rec["start"] and s["state"] != "Z"


# ------------------------------------------------------------------ approvals (records of USER-TYPED text only)
STANDING_DIR = RUNTIME / "ctl" / "auth"


def standing():
    import standing_auth as SA
    return SA.Standing(STANDING_DIR)


def approval_status(kind, module=None, now=None):
    """Run authorization for `kind` 'A' = research modules. Precedence:
      1. STANDING_RESEARCH_AUTHORIZATION (standing_auth.py): valid while ACTIVE and the module is in its fixed scope;
         not boot-bound, no expiry. Per-boot runtime state is NOT part of it (ctl gates re-check everything per start).
      2. legacy approval A record (boot-bound, expiring) — kept unchanged for modules outside the standing scope
         (e.g. frame_probe) and as the rollback path.
    A suspended / revoked standing grant does not fall back to an older A record for its own modules."""
    if kind == "A":
        st = standing().status(module)
        if st["valid"] or (st["state"] in ("SUSPENDED", "REVOKED") and (module is None or module in st.get("modules", []))):
            return st
    return legacy_approval_status(kind, module, now)


def legacy_approval_status(kind, module=None, now=None):
    """Newest approval record of `kind` ('A' / 'A_FASTLIO' / 'B'...). Valid = same boot, not expired, module listed."""
    now = time.time() if now is None else now
    recs = sorted(APPROVALS.glob(f"{kind}_*.json"))
    if not recs:
        return dict(valid=False, why="NO_APPROVAL_RECORD", kind=kind)
    r = read_json(recs[-1]) or {}
    why = None
    if r.get("boot_id") != boot_id():
        why = "APPROVAL_FOR_OTHER_BOOT"
    elif not r.get("user_text_sha256") or not r.get("user_text"):
        why = "APPROVAL_WITHOUT_USER_TEXT"
    elif hashlib.sha256(r["user_text"].encode()).hexdigest() != r["user_text_sha256"]:
        why = "APPROVAL_TEXT_HASH_MISMATCH"
    elif now > float(r.get("expires_epoch", 0)):
        why = "APPROVAL_EXPIRED"
    elif module is not None and module not in r.get("modules", []):
        why = f"MODULE_NOT_APPROVED:{module}"
    elif (r.get("limits") or {}).get("source") == "TABLET_EXPERIMENT_CONSENT":
        x = read_json(RUNTIME / "ctl" / "experiment.json") or {}      # 2026-10-31: valid only for the running experiment it names
        if x.get("exp_id") != r["limits"].get("exp_id") or x.get("state") not in ("STARTING", "RUNNING", "PARTIAL", "PAUSED"):
            why = f"CONSENT_FOR_OTHER_OR_ENDED_EXPERIMENT ({r['limits'].get('exp_id')})"
    return dict(valid=why is None, why=why, kind=kind, file=str(recs[-1]), expires_epoch=r.get("expires_epoch"),
                modules=r.get("modules", []), limits=r.get("limits", {}))


def host_addresses():
    out = {"hostname": socket.gethostname()}
    try:
        import subprocess
        txt = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=3).stdout
        for line in txt.splitlines():
            p = line.split()
            out[p[1]] = p[3].split("/")[0]
    except Exception:
        pass
    return out
