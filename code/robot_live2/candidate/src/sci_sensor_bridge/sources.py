#!/usr/bin/env python3
"""Frame sources for the SCI E5 node.

ReplaySource : extracted bag folder (tools/extract_bag.py): rgb/*.jpg, depth/*.png, index.csv, tf.csv (odom->base_link)
RosSource    : live ROS 2 subscriber to the ALREADY RUNNING drivers (never starts/duplicates a driver):
               /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw, odometry topic (default
               /utlidar/robot_odom; /Odometry when FAST-LIO runs). Read-only subscriptions, sensor-data QoS,
               latest-wins buffers (bounded).
Both expose: poll() -> list of (stamp, seq, bgr); depth(stamp) -> (dimg, stamp); pose(stamp) -> ((x,y,yaw), stamp);
odom_hist (list of (t, x, y, yaw)); health() dict.
"""
from __future__ import annotations

import bisect
import csv
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def yaw_from_q(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class ReplaySource:
    def __init__(self, root, realtime=True, speed=1.0, t_start=0.0, t_end=None):
        self.root = Path(root)
        rows = list(csv.DictReader(open(self.root / "index.csv")))
        self.rows = [r for r in rows if float(r["stamp"]) - float(rows[0]["stamp"]) >= t_start
                     and (t_end is None or float(r["stamp"]) - float(rows[0]["stamp"]) <= t_end)]
        self.t0 = float(rows[0]["stamp"])
        tf = [r for r in csv.DictReader(open(self.root / "tf.csv")) if r["parent"] == "odom" and r["child"] == "base_link"]
        tf.sort(key=lambda r: float(r["stamp"]))
        self.pt = np.array([float(r["stamp"]) for r in tf])
        self.px = np.array([float(r["x"]) for r in tf])
        self.py = np.array([float(r["y"]) for r in tf])
        self.pyaw = np.unwrap(np.array([yaw_from_q(*(float(r[k]) for k in ("qx", "qy", "qz", "qw"))) for r in tf]))
        self.realtime, self.speed = realtime, speed
        self.i, self.wall0, self.alive = 0, None, True
        self.stamps = [float(r["stamp"]) for r in self.rows]
        self.odom_hist = []
        self.n_frames = len(self.rows)

    def poll(self):
        if self.i >= len(self.rows):
            self.alive = False
            return []
        if self.wall0 is None:
            self.wall0, self.s0 = time.time(), self.stamps[0]
        out = []
        while self.i < len(self.rows):
            st = self.stamps[self.i]
            if self.realtime and (st - self.s0) / self.speed > time.time() - self.wall0:
                break
            r = self.rows[self.i]
            out.append((st, int(r["idx"]), cv2.imread(str(self.root / r["rgb"]))))
            self.i += 1
            if not self.realtime and len(out) >= 1:
                break
        if out:
            t = out[-1][0]
            p = self.pose(t)[0]
            if p is not None:
                self.odom_hist = (self.odom_hist + [(t, *p)])[-200:]
        return out

    def depth(self, stamp):
        k = bisect.bisect_left(self.stamps, stamp)
        k = min(max(k, 0), len(self.stamps) - 1)
        if k > 0 and abs(self.stamps[k - 1] - stamp) < abs(self.stamps[k] - stamp):
            k -= 1
        return cv2.imread(str(self.root / self.rows[k]["depth"]), cv2.IMREAD_UNCHANGED), self.stamps[k]

    def pose(self, stamp):
        if not len(self.pt) or stamp < self.pt[0] - 0.5 or stamp > self.pt[-1] + 0.5:
            return None, None
        return (float(np.interp(stamp, self.pt, self.px)), float(np.interp(stamp, self.pt, self.py)),
                float(np.interp(stamp, self.pt, self.pyaw))), stamp

    def health(self):
        return dict(source="replay", root=str(self.root), progress=f"{self.i}/{len(self.rows)}", odom="REPLAY", depth="REPLAY")

    def close(self):
        pass


class RosSource:
    def __init__(self, odom_topic="/utlidar/robot_odom", color="/camera/color/image_raw",
                 depth="/camera/aligned_depth_to_color/image_raw", node_name="sci_e5_source"):
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        self.lock = threading.Lock()
        self.frames = deque(maxlen=30)                       # bounded: newest wins
        self.depths = deque(maxlen=20)
        self.odom = deque(maxlen=600)
        self.odom_hist = []
        self.rx_mono_ns = {}                                 # R1: frame seq -> receive monotonic ns (bounded)
        self.n_color = self.n_depth = self.n_odom = 0
        self.last_color_wall = self.last_depth_wall = self.last_odom_wall = 0.0
        self.odom_topic = odom_topic
        self.node.create_subscription(Image, color, self._on_color, qos_profile_sensor_data)
        self.node.create_subscription(Image, depth, self._on_depth, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, odom_topic, self._on_odom, qos_profile_sensor_data)
        self.alive = True
        self.th = threading.Thread(target=self._spin, daemon=True)
        self.th.start()

    @staticmethod
    def _img(m):
        a = np.frombuffer(m.data, np.uint8)
        if m.encoding in ("16UC1", "mono16"):
            return a.view(np.uint16).reshape(m.height, m.width).copy()
        a = a.reshape(m.height, m.width, -1)
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else a.copy()

    @staticmethod
    def _st(h):
        return h.stamp.sec + h.stamp.nanosec * 1e-9

    def _on_color(self, m):
        with self.lock:
            self.n_color += 1
            self.frames.append((self._st(m.header), self.n_color, self._img(m)))
            self.rx_mono_ns[self.n_color] = time.monotonic_ns()     # receive time, NOT exposure time
            if len(self.rx_mono_ns) > 300:
                self.rx_mono_ns.pop(next(iter(self.rx_mono_ns)))
            self.last_color_wall = time.time()

    def _on_depth(self, m):
        with self.lock:
            self.n_depth += 1
            self.depths.append((self._st(m.header), self._img(m)))
            self.last_depth_wall = time.time()

    def _on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self.lock:
            self.n_odom += 1
            self.odom.append((self._st(m.header), p.x, p.y, yaw_from_q(q.x, q.y, q.z, q.w)))
            self.last_odom_wall = time.time()

    def _spin(self):
        while self.alive and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def poll(self):
        with self.lock:
            out = list(self.frames)
            self.frames.clear()
            self.odom_hist = list(self.odom)[-200:]
        return out

    def depth(self, stamp):
        with self.lock:
            if not self.depths:
                return None, None
            st, d = min(self.depths, key=lambda x: abs(x[0] - stamp))
        return d, st

    def pose(self, stamp, max_gap=0.3):
        with self.lock:
            if not self.odom:
                return None, None
            ts = [o[0] for o in self.odom]
            k = bisect.bisect_left(ts, stamp)
            cand = [i for i in (k - 1, k) if 0 <= i < len(ts)]
            i = min(cand, key=lambda j: abs(ts[j] - stamp))
            if abs(ts[i] - stamp) > max_gap:
                return None, None
            o = self.odom[i]
        return (o[1], o[2], o[3]), o[0]

    def health(self):
        now = time.time()
        f = lambda t: "OK" if now - t < 1.0 else ("STALE" if t > 0 else "NO_DATA")
        with self.lock:
            ts = [o[0] for o in self.odom if o[0] >= (self.odom[-1][0] - 3.0)] if self.odom else []
        hz = round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 2 and ts[-1] > ts[0] else 0.0
        return dict(source="ros", odom_topic=self.odom_topic, color=f(self.last_color_wall), depth=f(self.last_depth_wall),
                    odom=f(self.last_odom_wall), odom_hz=hz, n_color=self.n_color, n_depth=self.n_depth, n_odom=self.n_odom)

    def close(self):
        self.alive = False
        try:
            self.node.destroy_node()
        except Exception:
            pass
