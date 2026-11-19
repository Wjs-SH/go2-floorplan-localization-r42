"""safety_guard.py — LIVE2 experiment-level safety end (pure python; called by the console watchdog every ~2 s).

Approved 2026-10-20 after the field runs: the core tier (FAST-LIO, lio2, sensor hub) no longer aborts itself at 80 C, so the
EXPERIMENT ends safely when a condition PERSISTS (existing limits, only a duration added; no limit is raised):
  THERMAL_SUSTAINED     max SoC temperature >= 80 C continuously for 60 s (after E2 / SigLIP have already shed load)
  DEVICE_THROTTLING     a real throttle state (cpu/gpu/cv/soc *-throttle-alert or thermal-cpufreq/devfreq cur_state > 0)
                        continuously for 10 s (fan and hot-surface-alert are NOT throttling)
  ODOMETRY_STALE        FAST-LIO odometry not received for 10 s by BOTH receivers (lio2 state and the hub ring); armed only
                        when the experiment is RUNNING, 15 s after that and after odometry was seen once (initialisation)
  CORE_MODULE_FAILED    a core research module (fastlio, lio2, hub, e34, rec) ended unexpectedly (E2 / SigLIP failures and
                        SigLIP thermal pauses are degradations, not safety ends)
  SENSOR_FAULT          a ROPUS sensor process (hesai, realsense, robot_state_publisher, go2_control_runner) missing or
                        restarted for 10 s continuously (transient driver restarts are tolerated)
  MEMORY_LOW            MemAvailable < 2500 MB continuously for 10 s
The end itself is the console's normal a_end: only LIVE2-owned modules are stopped, records are archived, the reason and a
snapshot are written. No robot command, no ROPUS service is touched.
"""
import glob
import os
import time

TEMP_LIMIT_C, TEMP_HOLD_S = 80.0, 60.0
THROTTLE_HOLD_S = 10.0
ODOM_STALE_S, ODOM_ARM_AFTER_RUNNING_S = 10.0, 15.0
SENSOR_HOLD_S = 10.0
MEM_LIMIT_MB, MEM_HOLD_S = 2500.0, 10.0
CORE_MODULES = ("fastlio", "lio2", "hub", "e34", "rec")
THROTTLE_TYPES = ("throttle-alert", "thermal-cpufreq", "thermal-devfreq")


def throttle_states(root="/sys/class/thermal"):
    """{cooling device type: cur_state} for the real throttling devices only (fan / hot-surface excluded)."""
    out = {}
    for d in glob.glob(f"{root}/cooling_device*"):
        try:
            t = open(f"{d}/type").read().strip()
            if any(k in t for k in THROTTLE_TYPES) and "hot-surface" not in t:
                out[t] = int(open(f"{d}/cur_state").read().strip())
        except (OSError, ValueError):
            pass
    return out


class SafetyGuard:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.since = {}                 # condition -> wall time it became continuously true
        self.odom_seen = False
        self.last = None

    def _hold(self, key, active, now):
        if not active:
            self.since.pop(key, None)
            return 0.0
        self.since.setdefault(key, now)
        return now - self.since[key]

    def reset(self):
        self.since.clear()
        self.odom_seen = False

    def check(self, s):
        """s = dict(state, running_since (wall or None), temp_c, mem_mb, throttle {type: cur_state}, odom_age_lio,
        odom_age_hub (s since last odometry per receiver, None = unknown), odom_seen (bool), modules {m: exp state},
        sensors_ok (bool)). -> dict(trigger=None|condition, why, pending=[(condition, held_s, need_s)], snapshot)."""
        now = self.clock()
        pend, trig = [], None

        def cond(key, active, need, why):
            nonlocal trig
            held = self._hold(key, active, now)
            if active:
                pend.append(dict(condition=key, held_s=round(held, 1), need_s=need, why=why))
                if held >= need and trig is None:
                    trig = (key, why)

        t = s.get("temp_c")
        cond("THERMAL_SUSTAINED", t is not None and t >= TEMP_LIMIT_C, TEMP_HOLD_S, f"max temp {t} >= {TEMP_LIMIT_C} C")
        thr = {k: v for k, v in (s.get("throttle") or {}).items() if v > 0}
        cond("DEVICE_THROTTLING", bool(thr), THROTTLE_HOLD_S, f"throttling {thr}")
        m = s.get("mem_mb")
        cond("MEMORY_LOW", m is not None and m < MEM_LIMIT_MB, MEM_HOLD_S, f"MemAvailable {m} < {MEM_LIMIT_MB} MB")
        cond("SENSOR_FAULT", s.get("sensors_ok") is False, SENSOR_HOLD_S, "ROPUS sensor process missing / restarted")
        if s.get("odom_seen"):
            self.odom_seen = True
        rs = s.get("running_since")
        armed = self.odom_seen and rs is not None and now - rs >= ODOM_ARM_AFTER_RUNNING_S
        ages = [a for a in (s.get("odom_age_lio"), s.get("odom_age_hub")) if a is not None]
        age = min(ages) if ages else float("inf")
        stale_for = age if armed else 0.0          # the receivers report the age itself: no extra timer needed
        if armed and stale_for >= ODOM_STALE_S:
            pend.append(dict(condition="ODOMETRY_STALE", held_s=round(stale_for, 1), need_s=ODOM_STALE_S, why=f"no odometry for {stale_for:.1f} s"))
            if trig is None:
                trig = ("ODOMETRY_STALE", f"no FAST-LIO odometry for {stale_for:.1f} s (lio2 {s.get('odom_age_lio')}, hub {s.get('odom_age_hub')})")
        elif armed and stale_for >= 2.0:
            pend.append(dict(condition="ODOMETRY_STALE", held_s=round(stale_for, 1), need_s=ODOM_STALE_S, why="odometry gap"))
        dead = [mm for mm, st in (s.get("modules") or {}).items() if mm in CORE_MODULES and st == "ENDED_UNEXPECTEDLY"]
        if dead and trig is None:
            trig = ("CORE_MODULE_FAILED", f"core module(s) ended unexpectedly: {dead}")
        self.last = dict(trigger=None if trig is None else trig[0], why=None if trig is None else trig[1], pending=pend,
                         armed_odometry=armed, checked=now,
                         limits=dict(temp_c=TEMP_LIMIT_C, temp_hold_s=TEMP_HOLD_S, throttle_hold_s=THROTTLE_HOLD_S,
                                     odom_stale_s=ODOM_STALE_S, sensor_hold_s=SENSOR_HOLD_S, mem_mb=MEM_LIMIT_MB, mem_hold_s=MEM_HOLD_S),
                         snapshot={k: s.get(k) for k in ("temp_c", "mem_mb", "throttle", "odom_age_lio", "odom_age_hub", "modules", "sensors_ok")})
        return self.last


def file_age(p):
    try:
        return time.time() - os.path.getmtime(p)
    except OSError:
        return None
