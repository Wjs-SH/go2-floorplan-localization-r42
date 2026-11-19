#!/usr/bin/env python3
"""WALK-diagnostic watchdog for live_walk_diag.sh (read-only: files only, no ROS, never commands the robot).
Unlike the static watchdog, robot motion is EXPECTED; it aborts only on sensor loss, FAST-LIO stall/divergence or resources:
  IMU bridge not READY / <400 Hz for 3 s, FAST-LIO not initialised in 25 s or no scan output for 3 s, speed state > 1.5 m/s
  (slow walk expected <= 0.5), single scan step > 0.35 m, |height along gravity| > 0.5 m (flat indoor floor), temp >= 80 C,
  MemAvailable < 2500 MB, duration cap. RUN/STOP_REQUEST (written by the 8776 tablet UI): "finish" -> DONE (operator finished
  all steps), anything else -> ABORT OPERATOR_STOP (software stop, NOT an emergency stop). Also logs FAST-LIO / bridge CPU (from owned pid files) and load average.
PROVISIONAL development gates, not validated performance thresholds. The operator stops the robot with the joystick.
  walk_diag_watchdog.py RUN --duration 180"""
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_diag_watchdog import mat_rows  # noqa: E402

# speed: real slow-walk replays (2026-10-30, 3 configs) show single-scan speed-state spikes up to 1.53 m/s (p99 0.72) -> abort on
# > 3.0 m/s instantly (lio_monitor-style) or > 1.5 m/s sustained for 10 scans (1 s)
G = dict(init_grace_s=25.0, health_bad_s=3.0, min_hz=400, lio_stall_s=3.0, max_speed_mps=1.5, spike_speed_mps=3.0, sustain_scans=10, max_step_m=0.35, max_height_m=0.5,
         temp_c=80.0, mem_mb=2500, max_duration_s=300.0)


def mat_full(path):
    out = []
    try:
        with open(path) as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 25:
                    out.append([float(x) for x in p[:25]])
    except OSError:
        pass
    return out


def height(rows):
    """displacement along the current gravity estimate (cols 22..24), FAST-LIO world is the start IMU attitude."""
    g = rows[-1][22:25]; n = math.sqrt(sum(x * x for x in g)) or 1.0
    d = [a - b for a, b in zip(rows[-1][4:7], rows[0][4:7])]
    return -sum(a * b / n for a, b in zip(d, g))


def decide(t, duration, health, health_bad_since, rows, last_row_wall, temp_c, mem_mb, n_checked=None):
    """n_checked = number of rows already judged at the previous 1 Hz tick: every NEW consecutive pair is checked for jumps
    (checking only the last pair missed a jump earlier in the same second — found by the 8776 walk e2e test 2026-10-30)."""
    if temp_c is not None and temp_c >= G["temp_c"]:
        return "ABORT", "TEMPERATURE %.0f C" % temp_c
    if mem_mb is not None and mem_mb < G["mem_mb"]:
        return "ABORT", "MEMORY %d MB" % mem_mb
    if t > G["init_grace_s"]:
        bad = health is None or health.get("health") != "READY" or health.get("publish_hz", 0) < G["min_hz"]
        if bad and health_bad_since is not None and t - health_bad_since >= G["health_bad_s"]:
            return "ABORT", "IMU_BRIDGE: %s %s Hz" % (None if health is None else health.get("health"), None if health is None else health.get("publish_hz"))
        if not rows:
            return "ABORT", "FASTLIO_NOT_INITIALISED within %.0f s" % G["init_grace_s"]
        if last_row_wall is not None and t - last_row_wall >= G["lio_stall_s"]:
            imu = "IMU bridge %s %s Hz" % (None if health is None else health.get("health"), None if health is None else health.get("publish_hz"))
            return "ABORT", "FASTLIO_STALLED %.1f s (%s)" % (t - last_row_wall, imu if bad else "IMU ok -> LiDAR input or FAST-LIO")
    if rows:
        v = math.hypot(*rows[-1][13:16])
        if v > G["spike_speed_mps"]:
            return "ABORT", "DIVERGENCE: speed state %.2f m/s" % v
        tail = rows[-G["sustain_scans"]:]
        if len(tail) == G["sustain_scans"] and all(math.hypot(*r[13:16]) > G["max_speed_mps"] for r in tail):
            return "ABORT", "DIVERGENCE: speed state > %.1f m/s for %d scans" % (G["max_speed_mps"], G["sustain_scans"])
        k0 = max(1, (n_checked or len(rows) - 1))
        steps = [math.dist(rows[i][4:7], rows[i - 1][4:7]) for i in range(k0, len(rows))]
        if steps and max(steps) > G["max_step_m"]:
            return "ABORT", "JUMP %.2f m in one scan" % max(steps)
        if len(rows[-1]) >= 25 and abs(height(rows)) > G["max_height_m"]:
            return "ABORT", "HEIGHT %.2f m along gravity (flat floor)" % height(rows)
    if t >= min(duration, G["max_duration_s"]):
        return "DONE", "duration %.0f s reached" % duration
    return "RUN", ""


def cpu_of(run, names, prev):
    out = {}
    for n in names:
        try:
            pid, st = (run / "owned" / (n + ".pid")).read_text().split()[:2]
            pg = int(pid); tot = 0
            for d in Path("/proc").iterdir():          # whole owned process group (unshare/bash wrappers + the real binary)
                if d.name.isdigit():
                    try:
                        s = (d / "stat").read_text().rsplit(")", 1)[1].split()
                        if int(s[2]) == pg:
                            tot += int(s[11]) + int(s[12])
                    except (OSError, ValueError, IndexError):
                        pass
            now = time.time()
            if n in prev:
                out[n] = round(100.0 * (tot - prev[n][0]) / os.sysconf("SC_CLK_TCK") / max(1e-3, now - prev[n][1]), 1)
            prev[n] = (tot, now)
        except (OSError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--duration", type=float, default=180.0)
    a = ap.parse_args(); run = Path(a.run)
    t0 = time.time(); bad_since = None; n_prev = 0; last_row_t = None; prev = {}; n_checked = None
    log = open(run / "watchdog.jsonl", "a")
    while True:
        t = time.time() - t0
        try:
            health = json.loads((run / "bridge_health.jsonl").read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            health = None
        ok_h = health is not None and health.get("health") == "READY" and health.get("publish_hz", 0) >= G["min_hz"]
        bad_since = None if ok_h else (bad_since if bad_since is not None else t)
        rows = mat_full(run / "fastlio_Log" / "mat_pre.txt")
        if len(rows) > n_prev:
            n_prev, last_row_t = len(rows), t
        try:
            temp = max(int(open(p).read()) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")) / 1000
        except (OSError, ValueError):
            temp = None
        mem = next((int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")), None)
        state, why = decide(t, a.duration, health, bad_since, rows, last_row_t, temp, mem, n_checked)
        n_checked = len(rows)
        req = run / "STOP_REQUEST"
        if state == "RUN" and req.exists():
            kind = req.read_text().strip()
            state, why = ("DONE", "operator finished all steps (tablet)") if kind == "finish" else ("ABORT", "OPERATOR_STOP (tablet, software stop)")
        r = rows[-1] if rows else None
        rec = dict(t=round(t, 1), state=state, why=why, imu=None if health is None else dict(h=health.get("health"), hz=health.get("publish_hz"), gaps=health.get("gaps")),
                   scans=len(rows), pos=None if r is None else [round(x, 3) for x in r[4:7]], speed=None if r is None else round(math.hypot(*r[13:16]), 3),
                   height=None if not rows else round(height(rows), 3), cpu=cpu_of(run, ("fastlio_alt", "bridge", "recorder", "walk_tap"), prev),
                   load1=float(open("/proc/loadavg").read().split()[0]), temp=temp, mem_mb=mem)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"{time.strftime('%T')} {state:5s} t={t:5.1f}s imu={rec['imu']} scans={len(rows)} pos={rec['pos']} v={rec['speed']} h={rec['height']} cpu={rec['cpu']} {why}", flush=True)
        if state != "RUN":
            (run / "verdict.json").write_text(json.dumps(dict(state=state, reason=why, t=round(t, 1), gates=G, last=rec), indent=1))
            sys.exit(0 if state == "DONE" else 3)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
