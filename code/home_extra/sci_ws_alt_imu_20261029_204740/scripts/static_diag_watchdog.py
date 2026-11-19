#!/usr/bin/env python3
"""Static-diagnostic watchdog for live_static_diag.sh (read-only: files only, no ROS, never commands the robot).
Inputs (in RUN): bridge_health.jsonl (bridge, 1 Hz), fastlio_Log/mat_pre.txt (FAST-LIO per scan), calib_monitor.json
(/lowstate summary: sticks, gyro), /sys thermal, /proc/meminfo. Writes RUN/watchdog.jsonl (1 Hz) and RUN/verdict.json, exits 0 on
DONE, 3 on ABORT (the shell script then stops every owned process). Thresholds are PROVISIONAL static-diagnostic gates.
  static_diag_watchdog.py RUN --duration 60"""
import argparse, json, math, os, sys, time
from pathlib import Path

G = dict(init_grace_s=25.0, health_bad_s=3.0, min_hz=400, lio_stall_s=3.0, max_pos_m=0.30, max_vel_mps=0.30, max_step_m=0.20,
         stick=0.20, gyro_p95=0.25, temp_c=80.0, mem_mb=2500, max_duration_s=300.0)


def mat_rows(path):
    out = []
    try:
        with open(path) as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 19:
                    out.append([float(x) for x in p[:19]])
    except OSError:
        pass
    return out


def decide(t, duration, health, health_bad_since, rows, last_row_wall, mon, temp_c, mem_mb, n_checked=None):
    """Pure decision. t = seconds since start. Returns (state, reason). state: RUN | DONE | ABORT"""
    if mon is not None and t > 2:
        if mon.get("stick_max_abs_1s", 0) >= G["stick"]:
            return "ABORT", "ROBOT_INPUT: remote stick %.2f (static diagnostic only)" % mon["stick_max_abs_1s"]
        if mon.get("gyro_norm_p95_1s", 0) >= G["gyro_p95"]:
            return "ABORT", "ROBOT_MOVED: gyro p95 %.2f rad/s" % mon["gyro_norm_p95_1s"]
    if temp_c is not None and temp_c >= G["temp_c"]:
        return "ABORT", "TEMPERATURE %.0f C" % temp_c
    if mem_mb is not None and mem_mb < G["mem_mb"]:
        return "ABORT", "MEMORY %d MB" % mem_mb
    if t > G["init_grace_s"]:
        if health is None or health.get("health") != "READY" or health.get("publish_hz", 0) < G["min_hz"]:
            if health_bad_since is not None and t - health_bad_since >= G["health_bad_s"]:
                return "ABORT", "IMU_BRIDGE: %s %s Hz" % (None if health is None else health.get("health"), None if health is None else health.get("publish_hz"))
        if not rows:
            return "ABORT", "FASTLIO_NOT_INITIALISED within %.0f s" % G["init_grace_s"]
        if last_row_wall is not None and t - last_row_wall >= G["lio_stall_s"]:
            return "ABORT", "FASTLIO_STALLED: no scan output for %.1f s (LiDAR or IMU input missing)" % (t - last_row_wall)
    if rows:
        r = rows[-1]
        pos, vel = r[4:7], r[13:16]
        if math.dist(pos, rows[0][4:7]) > G["max_pos_m"]:
            return "ABORT", "DRIFT %.2f m" % math.dist(pos, rows[0][4:7])
        if math.hypot(*vel) > G["max_vel_mps"]:
            return "ABORT", "VELOCITY %.2f m/s" % math.hypot(*vel)
        k0 = max(1, (n_checked or len(rows) - 1))          # every new consecutive pair since the previous 1 Hz tick
        steps = [math.dist(rows[i][4:7], rows[i - 1][4:7]) for i in range(k0, len(rows))]
        if steps and max(steps) > G["max_step_m"]:
            return "ABORT", "JUMP %.2f m" % max(steps)
    if t >= min(duration, G["max_duration_s"]):
        return "DONE", "duration %.0f s reached" % duration
    return "RUN", ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--duration", type=float, default=60.0)
    a = ap.parse_args()
    run = Path(a.run)
    t0 = time.time(); bad_since = None; n_prev = 0; last_row_t = None; n_checked = None
    log = open(run / "watchdog.jsonl", "a")
    while True:
        t = time.time() - t0
        try:
            health = json.loads((run / "bridge_health.jsonl").read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            health = None
        ok_h = health is not None and health.get("health") == "READY" and health.get("publish_hz", 0) >= G["min_hz"]
        bad_since = None if ok_h else (bad_since if bad_since is not None else t)
        rows = mat_rows(run / "fastlio_Log" / "mat_pre.txt")
        if len(rows) > n_prev:
            n_prev, last_row_t = len(rows), t
        try:
            mon = json.loads((run / "calib_monitor.json").read_text())
            if time.time() - mon.get("wall", 0) > 1.5:
                mon = dict(mon, stale=True)
        except (OSError, ValueError):
            mon = None
        try:
            temp = max(int(open(p).read()) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")) / 1000
        except (OSError, ValueError):
            temp = None
        mem = next((int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")), None)
        state, why = decide(t, a.duration, health, bad_since, rows, last_row_t, mon, temp, mem, n_checked)
        n_checked = len(rows)
        r = rows[-1] if rows else None
        rec = dict(t=round(t, 1), state=state, why=why, imu=None if health is None else dict(h=health.get("health"), hz=health.get("publish_hz"),
                   gaps=health.get("gaps"), resets=health.get("model_resets")), scans=len(rows),
                   pos=None if r is None else [round(x, 3) for x in r[4:7]], vel=None if r is None else [round(x, 3) for x in r[13:16]],
                   stick=None if mon is None else mon.get("stick_max_abs_1s"), gyro_p95=None if mon is None else mon.get("gyro_norm_p95_1s"),
                   temp=temp, mem_mb=mem)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"{time.strftime('%T')} {state:5s} t={t:5.1f}s imu={rec['imu']} scans={len(rows)} pos={rec['pos']} vel={rec['vel']} {why}", flush=True)
        if state != "RUN":
            (run / "verdict.json").write_text(json.dumps(dict(state=state, reason=why, t=round(t, 1), gates=G, last=rec), indent=1))
            sys.exit(0 if state == "DONE" else 3)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
