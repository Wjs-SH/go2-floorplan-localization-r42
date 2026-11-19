#!/usr/bin/env python3
"""Read-only resource sampling (Jetson sysfs/procfs) + admission control for optional GPU jobs (VLM).

Nothing here changes clocks, power mode, priorities of other processes or GPU partitioning (Jetson has no supported
per-process GPU share; we only decide whether to START a job)."""
from __future__ import annotations

import glob
import os
import threading
import time

GPU_LOAD = "/sys/devices/gpu.0/load"          # 0..1000


def _read(p, default=None):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":", 1)
        d[k] = int(v.split()[0]) // 1024
    return d


def temps():
    out = {}
    for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
        t, v = _read(z + "/type"), _read(z + "/temp")
        if t and v and v.lstrip("-").isdigit():
            out[t] = int(v) / 1000.0
    return out


class CpuSampler:
    def __init__(self):
        self.prev = self._stat()

    @staticmethod
    def _stat():
        a = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(a), a[3] + a[4]

    def sample(self):
        tot, idle = self._stat()
        dt, di = tot - self.prev[0], idle - self.prev[1]
        self.prev = (tot, idle)
        return 100.0 * (1 - di / dt) if dt > 0 else 0.0


def proc_rss_mb(pid):
    s = _read(f"/proc/{pid}/statm")
    return int(s.split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20 if s else None


class ResourceMonitor(threading.Thread):
    def __init__(self, period=1.0, watch_pids=None):
        super().__init__(daemon=True)
        self.period, self.watch, self.cpu, self.last = period, dict(watch_pids or {}), CpuSampler(), {}
        self.hist = []

    def run(self):
        while True:
            m = meminfo()
            g = _read(GPU_LOAD)
            s = dict(t=time.time(), cpu_pct=round(self.cpu.sample(), 1), gpu_pct=round(int(g) / 10, 1) if g else None,
                     mem_used_mb=m["MemTotal"] - m["MemAvailable"], mem_avail_mb=m["MemAvailable"],
                     swap_used_mb=m["SwapTotal"] - m["SwapFree"], temps=temps(), load1=os.getloadavg()[0],
                     rss_mb={k: proc_rss_mb(p) for k, p in self.watch.items()})
            self.last = s
            self.hist = (self.hist + [s])[-600:]
            time.sleep(self.period)


def admit_gpu_job(sample, need_mb, reserve_mb=2500, max_gpu_pct=85.0, max_temp_c=85.0):
    """Start an optional GPU job only if, after loading it, MemAvailable stays above reserve_mb and the GPU/thermals
    have headroom. Returns (ok, reason)."""
    if not sample:
        return False, "no_resource_sample"
    if sample["mem_avail_mb"] - need_mb < reserve_mb:
        return False, f"memory: avail {sample['mem_avail_mb']} MB - need {need_mb} < reserve {reserve_mb}"
    if sample.get("gpu_pct") is not None and sample["gpu_pct"] > max_gpu_pct:
        return False, f"gpu busy {sample['gpu_pct']}%"
    hot = max(sample.get("temps", {}).values() or [0])
    if hot > max_temp_c:
        return False, f"thermal {hot:.1f}C"
    return True, "ok"
