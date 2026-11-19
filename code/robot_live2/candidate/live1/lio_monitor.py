#!/usr/bin/env python3
"""LIVE1 FAST-LIO health + local accumulation monitor (read-only consumer of FAST-LIO OUTPUTS).

Subscribes:
  /Odometry          nav_msgs/Odometry      BEST_EFFORT KEEP_LAST(10)   always
  /cloud_registered  sensor_msgs/PointCloud2 BEST_EFFORT KEEP_LAST(1)   only with --accum (FAST-LIO output, NOT /lidar_points)
Never subscribes /lidar_points or any sensor driver topic, never publishes, never calls services.

Writes (atomic) into --state:
  lio_state.json   every 0.5 s : odom Hz / inter-arrival p50-p95-max / header->receive age / gaps / resets / frame ids /
                                 raw pose / decimated raw-frame track / accumulation stats / own CPU
  lio_series.jsonl one line per second (bounded) : for the A0..A5 resource comparison
  lio_map.png      every 2 s (with --accum) : top view of accumulated FAST-LIO points + track, RAW odom frame
The RAW odom frame (FAST-LIO camera_init) is not the robot body frame and not the building map frame.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live1_common as C  # noqa: E402


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]


JUMP_ABS_M, JUMP_V_MPS = 1.0, 3.0      # odometry step larger than 1 m + 3 m/s * dt = discontinuity (Go2 walks <= ~1.5 m/s here)


class OdomStats:
    """Pure state (testable without rclpy)."""
    # 2026-10-21 field data (EXP2 FAST-LIO 10 Hz steps <= 0.42 m, 2 Hz traces <= 1.33 m) clear this line by >= 0.89 m.

    def __init__(self, wall=time.time, mono=time.monotonic, track_max=20000):
        self.wall, self.mono = wall, mono
        self.lock = threading.Lock()
        self.rx = deque(maxlen=3000)              # (rx_mono, rx_wall, stamp)
        self.n = 0
        self.rejected = {}
        self.session = 0
        self.frames = None
        self.last = None                          # (stamp, x, y, z, yaw)
        self.track = deque(maxlen=track_max)      # raw-frame (x, y, stamp)
        self.path_m = 0.0                         # travelled distance with a 0.10 m deadband (10 Hz pose jitter is not motion)
        self.path_anchor = None
        self.gaps_over = 0                        # header-stamp gaps > 0.25 s

    def on_odom(self, st, x, y, z, qx, qy, qz, qw, frame_id, child):
        rx_m, rx_w = self.mono(), self.wall()
        with self.lock:
            self.n += 1
            if not all(math.isfinite(v) for v in (st, x, y, z, qx, qy, qz, qw)):
                self.rejected["NON_FINITE"] = self.rejected.get("NON_FINITE", 0) + 1
                return
            fr = (frame_id, child)
            if self.frames is not None and fr != self.frames:
                self.session += 1
                self.track.clear()
                self.last = None
                self.path_anchor = None
                self.rejected["FRAME_CHANGED_NEW_SESSION"] = self.rejected.get("FRAME_CHANGED_NEW_SESSION", 0) + 1
            self.frames = fr
            if self.last is not None:
                if st < self.last[0] - 0.5:
                    self.session += 1
                    self.track.clear()
                    self.rejected["TIME_BACKWARDS_RESET_NEW_SESSION"] = self.rejected.get("TIME_BACKWARDS_RESET_NEW_SESSION", 0) + 1
                elif st <= self.last[0]:
                    self.rejected["DUPLICATE_OR_REORDERED"] = self.rejected.get("DUPLICATE_OR_REORDERED", 0) + 1
                    return
                else:
                    if st - self.last[0] > 0.25:
                        self.gaps_over += 1
                    jump = math.dist((x, y, z), self.last[1:4])
                    if jump > JUMP_ABS_M + JUMP_V_MPS * (st - self.last[0]):   # position reset / re-init without a time jump:
                        self.session += 1                                     # never joined to the old trajectory (new session
                        self.track.clear()                                    # -> store reset, E3 hypotheses dropped, E2 re-search)
                        self.path_anchor = None
                        self.rejected["POSITION_JUMP_NEW_SESSION"] = self.rejected.get("POSITION_JUMP_NEW_SESSION", 0) + 1
                    # 2026-10-19: summing every 10 Hz update read 17 m for a parked robot -> low-pass (~1 s) + 0.10 m deadband
                    self.path_f = (x, y) if self.path_anchor is None else \
                        (self.path_f[0] + 0.1 * (x - self.path_f[0]), self.path_f[1] + 0.1 * (y - self.path_f[1]))
                    if self.path_anchor is None:
                        self.path_anchor = self.path_f
                    step = math.hypot(self.path_f[0] - self.path_anchor[0], self.path_f[1] - self.path_anchor[1])
                    if step >= 0.10:
                        self.path_m += step
                        self.path_anchor = self.path_f
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            self.rx.append((rx_m, rx_w, st))
            self.last = (st, x, y, z, yaw)
            if not self.track or math.hypot(x - self.track[-1][0], y - self.track[-1][1]) >= 0.05 or st - self.track[-1][2] >= 1.0:
                self.track.append((round(x, 3), round(y, 3), st))

    def summary(self, win_s=10.0):
        with self.lock:
            now_m, now_w = self.mono(), self.wall()
            rx = [r for r in self.rx if now_m - r[0] <= win_s]
            ia = [b[0] - a[0] for a, b in zip(rx, rx[1:])]
            age = [r[1] - r[2] for r in rx]
            last_rx_age = (now_m - self.rx[-1][0]) if self.rx else None
            hz = (len(rx) - 1) / (rx[-1][0] - rx[0][0]) if len(rx) >= 2 and rx[-1][0] > rx[0][0] else None
            return dict(n=self.n, session=self.session, rejected=dict(self.rejected), gaps_over_0p25s=self.gaps_over,
                        frame_id=self.frames[0] if self.frames else None, child_frame_id=self.frames[1] if self.frames else None,
                        hz_10s=None if hz is None else round(hz, 2), n_10s=len(rx),
                        interarrival_ms=dict(p50=_ms(pct(ia, 50)), p95=_ms(pct(ia, 95)), max=_ms(max(ia) if ia else None)),
                        header_to_rx_ms=dict(p50=_ms(pct(age, 50)), p95=_ms(pct(age, 95)), max=_ms(max(age) if age else None),
                                             note="host wall clock minus header stamp (lidar/IMU time base); meaningful only if clocks agree"),
                        last_rx_age_s=None if last_rx_age is None else round(last_rx_age, 3),
                        last_raw=None if self.last is None else dict(stamp=self.last[0], x=round(self.last[1], 3), y=round(self.last[2], 3),
                                                                     z=round(self.last[3], 3), yaw_deg=round(math.degrees(self.last[4]), 2)),
                        path_m=round(self.path_m, 2), track_n=len(self.track), wall=now_w)


def _ms(v):
    return None if v is None else round(1000.0 * v, 1)


DISPLAY_A = np.deg2rad(-47.3)


def display_xy(p):
    """DISPLAY ONLY. camera_init = the upside-down IMU frame at start, so raw x right / y up is a mirrored view from below.
    Reflect into (u = robot forward at start, v) with A = -47.3 deg (extrinsic chain; measured -45.3 deg on 2026-10-20 runs)
    and draw forward up: screen x = -v, screen y = u. Never used for computation (E2/E3 use odom_link)."""
    p = np.asarray(p, float).reshape(-1, 2)
    u = p[:, 0] * np.cos(DISPLAY_A) + p[:, 1] * np.sin(DISPLAY_A)
    v = p[:, 0] * np.sin(DISPLAY_A) - p[:, 1] * np.cos(DISPLAY_A)
    return np.stack([-v, u], 1)


class Accumulator:
    """Bounded 2D hit grid of FAST-LIO registered points in the RAW odom frame."""

    def __init__(self, res=0.1, max_cells=4_000_000, stride=3, dz_max=1.5):
        self.res, self.max_cells, self.stride, self.dz_max = res, max_cells, stride, dz_max
        self.cells = {}
        self.lock = threading.Lock()
        self.stats = dict(scans=0, points_used=0, process_ms=deque(maxlen=200), dropped_full=0, last_rx_mono=None, decode_errors=0,
                          header_stamps=deque(maxlen=200))

    def add_xyz(self, xyz, z_ref=None):
        t0 = time.perf_counter()
        p = xyz[:: self.stride]
        p = p[np.isfinite(p).all(axis=1)]
        if z_ref is not None:
            p = p[np.abs(p[:, 2] - z_ref) <= self.dz_max]
        ij = np.floor(p[:, :2] / self.res).astype(np.int64)
        if len(ij):
            key = ij[:, 0] * 1_000_003 + ij[:, 1]
            u, c = np.unique(key, return_counts=True)
            with self.lock:
                for k, n in zip(u.tolist(), c.tolist()):
                    if k in self.cells:
                        self.cells[k] += n
                    elif len(self.cells) < self.max_cells:
                        self.cells[k] = n
                    else:
                        self.stats["dropped_full"] += 1
        with self.lock:
            self.stats["scans"] += 1
            self.stats["points_used"] += int(len(p))
            self.stats["process_ms"].append(1000 * (time.perf_counter() - t0))
            self.stats["last_rx_mono"] = time.monotonic()

    def summary(self):
        with self.lock:
            pm = list(self.stats["process_ms"])
            lr = self.stats["last_rx_mono"]
            return dict(enabled=True, scans=self.stats["scans"], points_used=self.stats["points_used"], cells=len(self.cells),
                        res_m=self.res, process_ms_p50=None if not pm else round(pct(pm, 50), 2),
                        process_ms_p95=None if not pm else round(pct(pm, 95), 2), dropped_full=self.stats["dropped_full"],
                        decode_errors=self.stats["decode_errors"],
                        last_scan_age_s=None if lr is None else round(time.monotonic() - lr, 3))

    def render(self, track, size=520):
        import cv2
        with self.lock:
            if not self.cells and not track:
                return None
            ks = np.fromiter(self.cells.keys(), np.int64, len(self.cells))
            vs = np.fromiter(self.cells.values(), np.int64, len(self.cells))
        if len(ks):
            i = np.round(ks / 1_000_003).astype(np.int64)
            j = ks - i * 1_000_003
            xy = np.stack([i, j], 1) * self.res
        else:
            xy, vs = np.zeros((0, 2)), np.zeros(0)
        tr = np.array([(t[0], t[1]) for t in track]) if track else np.zeros((0, 2))
        xy, tr = display_xy(xy), display_xy(tr)               # picture only: un-mirrored, start heading up (raw data unchanged)
        allp = np.concatenate([xy, tr]) if len(tr) else xy
        lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
        span = float(max(hi - lo)) or 1.0
        sc = (size - 20) / span
        img = np.full((size, size, 3), 255, np.uint8)

        def px(p):
            return np.stack([10 + (p[:, 0] - lo[0]) * sc, size - 10 - (p[:, 1] - lo[1]) * sc], 1).astype(np.int32)
        if len(xy):
            q = px(xy)
            ok = (q[:, 0] >= 0) & (q[:, 0] < size) & (q[:, 1] >= 0) & (q[:, 1] < size)
            g = np.clip(255 - 40 * np.log1p(vs[ok]), 0, 230).astype(np.uint8)
            img[q[ok, 1], q[ok, 0]] = np.stack([g, g, g], 1)
        if len(tr) >= 2:
            cv2.polylines(img, [px(tr).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            cv2.circle(img, tuple(int(v) for v in px(tr[-1:])[0]), 5, (0, 160, 0), -1)
        cv2.putText(img, f"FAST-LIO odom, display: un-mirrored, start heading up | {span:.1f} m span | frame contract UNVERIFIED",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        return cv2.imencode(".png", img)[1].tobytes()


def cloud_xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z")):
        raise ValueError("NO_XYZ")
    n = msg.width * msg.height
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, memoryview)) else msg.data, np.uint8)
    rows = buf[: n * msg.point_step].reshape(n, msg.point_step)
    dt = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
    return np.stack([rows[:, off[k]: off[k] + 4].copy().view(dt).reshape(-1) for k in ("x", "y", "z")], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(C.STATE))
    ap.add_argument("--accum", action="store_true", help="also accumulate FAST-LIO /cloud_registered (needs approval)")
    ap.add_argument("--max-s", type=float, default=600.0)
    a = ap.parse_args()
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    rclpy.init()
    node = rclpy.create_node("live1_lio_monitor")
    od = OdomStats()
    acc = Accumulator() if a.accum else None

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        od.on_odom(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w, m.header.frame_id, m.child_frame_id)

    be = lambda d: QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=d,  # noqa: E731
                              durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(Odometry, "/Odometry", on_odom, be(10))
    subs = ["/Odometry"]
    if acc is not None:
        from sensor_msgs.msg import PointCloud2

        def on_cloud(m):
            try:
                xyz = cloud_xyz(m)
            except Exception:
                acc.stats["decode_errors"] += 1
                return
            z = od.last[3] if od.last else None
            acc.add_xyz(xyz, z_ref=z)
        node.create_subscription(PointCloud2, "/cloud_registered", on_cloud, be(1))
        subs.append("/cloud_registered")
    stop = [False]
    t_start = time.monotonic()
    me = os.getpid()
    cpu0 = [None]

    def own_cpu():
        s = open(f"/proc/{me}/stat").read().split(") ")[1].split()
        tot = (int(s[11]) + int(s[12])) / C.HZ
        prev, cpu0[0] = cpu0[0], (tot, time.monotonic())
        return None if prev is None else round(100 * (tot - prev[0]) / max(1e-6, cpu0[0][1] - prev[1]), 1)

    def writer():
        k = 0
        series = state / "lio_series.jsonl"
        while not stop[0]:
            time.sleep(0.5)
            k += 1
            s = od.summary()
            with od.lock:
                tr = list(od.track)
            doc = dict(kind="LIVE1_LIO_STATE", source="FAST-LIO /Odometry (read-only)", subscribed=subs, boot_id=C.boot_id(),
                       pid=me, written_wall=time.time(), odom=s, track_raw=[(t[0], t[1]) for t in tr[-3000:]],
                       accumulation=acc.summary() if acc else dict(enabled=False, why="ACCUM_NOT_ENABLED (needs --accum + approval)"),
                       frame_note="RAW FAST-LIO frame; not robot body, not building map")
            if k % 2 == 0:
                doc["own_cpu_pct"] = own_cpu()
                with open(series, "a") as f:
                    f.write(json.dumps(dict(wall=time.time(), hz=s["hz_10s"], ia95=s["interarrival_ms"]["p95"], age95=s["header_to_rx_ms"]["p95"],
                                            last_rx_age=s["last_rx_age_s"], n=s["n"], acc=(acc.summary() if acc else None),
                                            res=C.resources())) + "\n")
                if series.stat().st_size > 20 * 2 ** 20:
                    os.replace(series, series.with_suffix(".jsonl.1"))
            C.atomic_json(state / "lio_state.json", doc)
            if acc is not None and k % 4 == 0:
                png = acc.render(tr)
                if png:
                    C.atomic_bytes(state / "lio_map.png", png)
            if time.monotonic() - t_start > a.max_s:
                stop[0] = True
    th = threading.Thread(target=writer, daemon=True)
    th.start()
    try:
        while rclpy.ok() and not stop[0]:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    stop[0] = True
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
