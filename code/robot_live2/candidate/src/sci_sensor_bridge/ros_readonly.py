#!/usr/bin/env python3
"""SEM1: explicit read-only ROS input (--source ros_readonly) for an approved, time-limited SHADOW.

Scope enforced in code (not only by convention):
  - ALLOWLIST: /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info, /Odometry.
    Any other topic raises (no /lidar_points, no IMU, no /utlidar/robot_odom substitution).
  - A topic is subscribed only if it is ALREADY published with the expected type (graph check at start). /Odometry
    absent -> VISUAL_ONLY: pose() returns (None, None) with reason unless visual_identity_pose=True, in which case the
    identity pose is returned and flagged (geometry/P2 STOP evaluation stays UNAVAILABLE upstream).
  - No publisher, no service client, no parameter writes. Never opens a device.
  - QoS: BEST_EFFORT + KEEP_LAST(1) (compatible with the RELIABLE/VOLATILE publishers). Note: a slower consumer does
    NOT reduce the bytes DDS delivers to this process; only the queue depth is bounded.
  - Callbacks only stamp + append to a bounded ring (count AND bytes). Decoding happens in poll()/depth() outside the
    executor. Gaps, out-of-order header stamps, resets and drops are counted.
Approval: SEM1 APPROVAL_RECORD.json must exist and not be expired (checked by the caller / supervisor).

FOLLOW-UP geometry contract (before any geometry shadow):
  - Odometry samples are validated IN THE CALLBACK (finite position, unit quaternion, future stamp, backwards = reset ->
    new session, duplicate, large jump, producer frame change -> new session); rejected samples never reach pose().
  - The odom frame contract (frame_id, child_frame_id and the verified transform to the robot BODY) must be supplied
    from a verified source (SCI_SEM1_ODOM_FRAME_CONTRACT json). Without it geometry is UNVERIFIED and pose() returns
    (None, "UNVERIFIED_FRAME_CONTRACT"). No rotation constant (e.g. 45.3 or 228 deg) is assumed; an input already in the
    body frame is never transformed twice.
  - odom_hist is a property: the latest validated samples under the lock (not refreshed only by poll()).
  - Images: rgb8 / bgr8 / mono8 / 16UC1 / mono16 honouring step (row padding) and is_bigendian; others are rejected.
"""
from __future__ import annotations

import json
import math
import os

import threading
import time
from collections import deque

import cv2
import numpy as np

import depth_freshness as DF

ALLOW = {"/camera/color/image_raw": "sensor_msgs/msg/Image",
         "/camera/aligned_depth_to_color/image_raw": "sensor_msgs/msg/Image",
         "/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
         "/Odometry": "nav_msgs/msg/Odometry"}


def check_allowed(topic):
    if topic not in ALLOW:
        raise PermissionError(f"ros_readonly: topic {topic} is outside the approved allowlist")
    return ALLOW[topic]


class _Ring:
    def __init__(self, max_n, max_bytes):
        self.q, self.max_n, self.max_bytes, self.bytes = deque(), max_n, max_bytes, 0
        self.dropped = 0

    def push(self, item, nbytes):
        self.q.append((item, nbytes))
        self.bytes += nbytes
        while self.q and (len(self.q) > self.max_n or self.bytes > self.max_bytes):
            _, b = self.q.popleft()
            self.bytes -= b
            self.dropped += 1


class RosReadonlySource:
    def __init__(self, odom_topic="/Odometry", visual_identity_pose=False, node_name="sci_sem1_readonly",
                 ring_frames=8, ring_bytes=40 * 1024 * 1024, graph_wait_s=3.0):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, CameraInfo
        check_allowed(odom_topic)
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name, start_parameter_services=False)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1,
                         durability=DurabilityPolicy.VOLATILE)
        self._init_state(ring_frames, ring_bytes, visual_identity_pose)
        t_end = time.time() + graph_wait_s
        names = {}
        while time.time() < t_end:                              # graph metadata only (no data)
            names = dict(self.node.get_topic_names_and_types())
            if all(t in names for t in list(ALLOW)[:3]):
                break
            time.sleep(0.2)
        self.subscribed, self.refused = [], {}
        for t, typ in ALLOW.items():
            if t == "/Odometry" and odom_topic != "/Odometry":
                continue
            have = names.get(t)
            if not have:
                self.refused[t] = "NOT_PUBLISHED"
                continue
            if typ not in have:
                self.refused[t] = f"TYPE_MISMATCH:{have}"
                continue
            if t == "/camera/color/image_raw":
                self.node.create_subscription(Image, t, self._on_color, qos)
            elif t == "/camera/aligned_depth_to_color/image_raw":
                self.node.create_subscription(Image, t, self._on_depth, qos)
            elif t == "/camera/color/camera_info":
                self.node.create_subscription(CameraInfo, t, self._on_info, qos)
            elif t == "/Odometry":
                from nav_msgs.msg import Odometry
                self.node.create_subscription(Odometry, t, self._on_odom, qos)
            self.subscribed.append(t)
        self.visual_only = "/Odometry" not in self.subscribed
        self.alive = True
        self.th = threading.Thread(target=self._spin, daemon=True)
        self.th.start()

    def _init_state(self, ring_frames=8, ring_bytes=40 * 1024 * 1024, visual_identity_pose=False, contract=None, wall=time.time):
        """All state (also used by fake-input tests: no rclpy needed)."""
        self.lock = threading.Lock()
        self.color = _Ring(ring_frames, ring_bytes)
        self.depths = _Ring(ring_frames, ring_bytes)
        self.depth_gate = DF.Tracker()
        self.odom = deque(maxlen=600)
        self.odom_session, self._odom_last, self._odom_frames = 0, None, None
        self.caminfo = None
        self.rx_mono_ns = {}
        self.stats = dict(color=dict(n=0, ooo=0, reset=0, gaps=[], last_st=None, rx=[]), depth=dict(n=0, ooo=0, reset=0, gaps=[], last_st=None, rx=[]),
                          odom=dict(n=0, accepted=0, rejected={}, gaps=[], rx=[], last_rx_wall=None, frames=None), caminfo=dict(n=0),
                          decode_rejected={})
        self.n_color = 0
        self.visual_identity_pose = visual_identity_pose
        self.wall = wall
        if contract is None:
            cp = os.environ.get("SCI_SEM1_ODOM_FRAME_CONTRACT")
            contract = json.loads(open(cp).read()) if cp and os.path.exists(cp) else None
        self.contract = contract
        self.geometry_status = "CONTRACT_PRESENT" if self._contract_ok(contract) else "UNVERIFIED_FRAME_CONTRACT"
        if not hasattr(self, "subscribed"):
            self.subscribed, self.refused = [], {}
        if not hasattr(self, "visual_only"):
            self.visual_only = True

    @staticmethod
    def _contract_ok(c):
        if not c:
            return False
        need = ("frame_id", "child_frame_id", "body_transform", "verified_by", "evidence")
        if any(not c.get(k) for k in need):
            return False
        t = c["body_transform"]
        num = lambda v: isinstance(v, (int, float)) and math.isfinite(v)  # noqa: E731
        if t.get("kind") == "identity":
            return True
        if t.get("kind") == "planar_yaw":
            return num(t.get("yaw_deg"))
        if t.get("kind") == "fastlio_mirror_lever_yaw":        # odom_base.fastlio_to_base_2d form; values from the contract only
            lv = t.get("lever_m")
            return num(t.get("delta_deg")) and isinstance(lv, (list, tuple)) and len(lv) == 2 and all(num(v) for v in lv)
        return False

    # ------------------------------------------------------------------ callbacks: stamp + bounded insert only
    @staticmethod
    def _st(h):
        return h.stamp.sec + h.stamp.nanosec * 1e-9

    def _rec(self, key, st):
        s = self.stats[key]
        s["n"] += 1
        if s["last_st"] is not None:
            if st < s["last_st"] - 0.5:
                s["reset"] += 1
            elif st <= s["last_st"]:
                s["ooo"] += 1
            else:
                s["gaps"] = (s["gaps"] + [st - s["last_st"]])[-2000:]
        s["last_st"] = st
        s["rx"] = (s["rx"] + [time.monotonic()])[-2000:]

    def _on_color(self, m):
        rx = time.monotonic_ns()
        st = self._st(m.header)
        with self.lock:
            self._rec("color", st)
            self.n_color += 1
            self.color.push((st, self.n_color, rx, m), len(m.data))
            self.rx_mono_ns[self.n_color] = rx
            if len(self.rx_mono_ns) > 300:
                self.rx_mono_ns.pop(next(iter(self.rx_mono_ns)))

    def _on_depth(self, m):
        rx = time.monotonic_ns()
        st = self._st(m.header)
        with self.lock:
            self._rec("depth", st)
            self.depths.push((st, rx, m), len(m.data))

    def _on_info(self, m):
        with self.lock:
            self.stats["caminfo"]["n"] += 1
            self.caminfo = dict(w=m.width, h=m.height, k=list(m.k), frame_id=m.header.frame_id)

    def _reject_odom(self, why):
        r = self.stats["odom"]["rejected"]
        r[why] = r.get(why, 0) + 1

    def _on_odom(self, m, future_tol_s=2.0, jump_m=1.5, jump_dt_s=0.5):
        rx_wall = self.wall()
        st = self._st(m.header)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        frames = (getattr(m.header, "frame_id", None), getattr(m, "child_frame_id", None))
        with self.lock:
            o = self.stats["odom"]
            o["n"] += 1
            o["rx"] = (o["rx"] + [time.monotonic()])[-2000:]
            o["last_rx_wall"] = rx_wall
            vals = (st, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                return self._reject_odom("NON_FINITE")
            qn = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
            if not 0.95 <= qn <= 1.05:
                return self._reject_odom("QUATERNION_NOT_UNIT")
            if st > rx_wall + future_tol_s:
                return self._reject_odom("FUTURE_STAMP")
            if self._odom_frames is not None and frames != self._odom_frames:
                self.odom_session += 1                       # producer / frame change: old history is not continued
                self.odom.clear()
                self._odom_last = None
                self._reject_odom("FRAME_CHANGED_NEW_SESSION")
            self._odom_frames = frames
            o["frames"] = dict(frame_id=frames[0], child_frame_id=frames[1])
            if self.contract and self._contract_ok(self.contract) and frames != (self.contract["frame_id"], self.contract["child_frame_id"]):
                return self._reject_odom("FRAME_CONTRACT_MISMATCH")
            if self._odom_last is not None:
                if st < self._odom_last - 0.5:
                    self.odom_session += 1
                    self.odom.clear()
                    self._reject_odom("TIME_BACKWARDS_RESET_NEW_SESSION")
                elif st <= self._odom_last:
                    return self._reject_odom("DUPLICATE_OR_REORDERED")
                elif self.odom:
                    last = self.odom[-1]
                    if st - last[0] < jump_dt_s and math.hypot(p.x - last[1], p.y - last[2]) > jump_m:
                        return self._reject_odom("JUMP")
                    o["gaps"] = (o["gaps"] + [st - last[0]])[-2000:]
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self._odom_last = st
            o["accepted"] += 1
            self.odom.append((st, p.x, p.y, yaw))
            self.last_odom_wall = rx_wall

    @property
    def odom_hist(self):
        """Latest validated samples IN THE BODY FRAME (same frame as pose()), read under the lock at call time.
        LIVE1 fix: this used to return RAW odom-frame samples while pose() returned body-frame poses, so the P2 decision
        reference (built from odom_hist) and the scenes (built from pose()) used different frames. Without a verified
        contract there is no body frame -> [] (P2 / STOP evaluation defers; never raw samples)."""
        if self.geometry_status != "CONTRACT_PRESENT":
            return []
        with self.lock:
            raw = list(self.odom)[-200:]
        return [(o[0],) + tuple(self.body_pose((o[1], o[2], o[3]))) for o in raw]

    @odom_hist.setter
    def odom_hist(self, v):
        pass

    def odom_raw_hist(self):
        """Validated samples in the RAW producer frame (diagnostics / frame-contract evidence only; never for geometry)."""
        with self.lock:
            return list(self.odom)[-200:]

    def body_pose(self, raw):
        """Raw odom (x, y, yaw) -> robot body per the VERIFIED contract; None if the contract is missing."""
        if self.geometry_status != "CONTRACT_PRESENT":
            return None
        t = self.contract["body_transform"]
        x, y, yaw = raw
        if t["kind"] == "identity":
            return raw
        if t["kind"] == "planar_yaw":
            a = math.radians(t["yaw_deg"])
            return (math.cos(a) * x - math.sin(a) * y, math.sin(a) * x + math.cos(a) * y, yaw + a)
        lx, ly = t["lever_m"]                                  # identical arithmetic to odom_base.fastlio_to_base_2d
        dl = math.radians(t["delta_deg"])
        c, s_ = math.cos(yaw), math.sin(yaw)
        bx, by = x + c * lx - s_ * ly, -(y + s_ * lx + c * ly)
        ys = -yaw
        yaw2 = ys + dl
        bx += -(math.cos(ys) * lx - math.sin(ys) * ly) + (math.cos(yaw2) * lx - math.sin(yaw2) * ly)
        by += -(math.sin(ys) * lx + math.cos(ys) * ly) + (math.sin(yaw2) * lx + math.cos(yaw2) * ly)
        return bx, by, yaw2

    def _spin(self):
        while self.alive and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    # ------------------------------------------------------------------ consumer side (decode here, not in callbacks)
    ENC = {"rgb8": (3, 1), "bgr8": (3, 1), "mono8": (1, 1), "16UC1": (1, 2), "mono16": (1, 2)}

    @classmethod
    def _img(cls, m):
        """Decode honouring encoding / step (row padding) / is_bigendian. Unsupported encodings raise."""
        if m.encoding not in cls.ENC:
            raise ValueError(f"UNSUPPORTED_ENCODING:{m.encoding}")
        ch, bpc = cls.ENC[m.encoding]
        row = m.width * ch * bpc
        if m.step < row or len(m.data) < m.step * m.height:
            raise ValueError(f"BAD_STEP_OR_SIZE:step={m.step},row={row},len={len(m.data)}")
        a = np.frombuffer(m.data, np.uint8)[:m.step * m.height].reshape(m.height, m.step)[:, :row]
        if bpc == 2:
            dt = np.dtype(">u2") if getattr(m, "is_bigendian", 0) else np.dtype("<u2")
            return np.ascontiguousarray(a).view(dt).reshape(m.height, m.width).astype(np.uint16)
        a = np.ascontiguousarray(a).reshape(m.height, m.width, ch) if ch > 1 else np.ascontiguousarray(a).reshape(m.height, m.width)
        if m.encoding == "rgb8":
            return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        return a.copy()

    def poll(self):
        with self.lock:
            items = [x for x, _ in self.color.q]
            self.color.q.clear()
            self.color.bytes = 0
        out = []
        for st, seq, rx, m in items:
            try:
                out.append((st, seq, self._img(m)))
            except ValueError as e:
                k = str(e).split(":")[0]
                self.stats["decode_rejected"][k] = self.stats["decode_rejected"].get(k, 0) + 1
        return out

    def depth(self, stamp):
        """Nearest aligned depth, or (None, None) when it is missing or too far from `stamp` (R34: never an old image)."""
        with self.lock:
            q = [x for x, _ in self.depths.q]
            if not q:
                self.depth_gate.judge(stamp, None)
                return None, None
            st, rx, m = min(q, key=lambda x: abs(x[0] - stamp))
            if not self.depth_gate.judge(stamp, st):
                return None, None
        return self._img(m), st

    def pose(self, stamp, max_gap=0.3):
        """Body pose nearest to the image stamp (|dt| <= max_gap) from VALIDATED samples of the current session.
        Visual-only (no odom subscription): identity only if explicitly flagged; geometry stays unavailable upstream.
        Missing frame contract: (None, None) -- never a guessed transform."""
        if self.visual_only:
            return ((0.0, 0.0, 0.0), stamp) if self.visual_identity_pose else (None, None)
        if self.geometry_status != "CONTRACT_PRESENT":
            # LIVE1: odom is received but the frame contract is not verified. Semantic search may run on an explicitly
            # FLAGGED identity pose (pose_mode IDENTITY_FLAGGED_CONTRACT_MISSING); odom_hist stays [] so geometry/P2 STOP
            # evaluation is unavailable. Never a guessed transform.
            return ((0.0, 0.0, 0.0), stamp) if self.visual_identity_pose else (None, None)
        with self.lock:
            if not self.odom:
                return None, None
            o = min(self.odom, key=lambda x: abs(x[0] - stamp))
        if abs(o[0] - stamp) > max_gap:
            return None, None
        return self.body_pose((o[1], o[2], o[3])), o[0]

    def pose_mode(self):
        if self.visual_only:
            return "IDENTITY_FLAGGED_GEOMETRY_UNAVAILABLE" if self.visual_identity_pose else "NONE"
        if self.geometry_status != "CONTRACT_PRESENT":
            return "IDENTITY_FLAGGED_CONTRACT_MISSING" if self.visual_identity_pose else "NONE_CONTRACT_MISSING"
        return "ODOMETRY_BODY_CONTRACT"

    def health(self):
        with self.lock:
            out = dict(source="ros_readonly", subscribed=list(self.subscribed), refused=dict(self.refused), depth_fresh=self.depth_gate.snapshot(),
                       visual_only=self.visual_only, pose_mode=self.pose_mode(), geometry_status=self.geometry_status,
                       caminfo=self.caminfo, odom_session=self.odom_session,
                       ring_dropped=dict(color=self.color.dropped, depth=self.depths.dropped))
            for k in ("color", "depth"):
                s = self.stats[k]
                g = sorted(s["gaps"])
                rx = s["rx"]
                out[k] = dict(n=s["n"], ooo=s["ooo"], reset=s["reset"],
                              hz=round((len(rx) - 1) / (rx[-1] - rx[0]), 2) if len(rx) > 2 and rx[-1] > rx[0] else None,
                              gap_p50_ms=round(g[len(g) // 2] * 1e3, 1) if g else None,
                              gap_p95_ms=round(g[int(len(g) * 0.95)] * 1e3, 1) if g else None,
                              gap_max_ms=round(g[-1] * 1e3, 1) if g else None)
            o = self.stats["odom"]
            g = sorted(o["gaps"])
            rx = o["rx"]
            out["odom"] = dict(n=o["n"], accepted=o["accepted"], rejected=dict(o["rejected"]), frames=o["frames"],
                               hz=round((len(rx) - 1) / (rx[-1] - rx[0]), 2) if len(rx) > 2 and rx[-1] > rx[0] else None,
                               header_gap_p50_ms=round(g[len(g) // 2] * 1e3, 1) if g else None,
                               header_gap_p95_ms=round(g[int(len(g) * 0.95)] * 1e3, 1) if g else None,
                               header_gap_max_ms=round(g[-1] * 1e3, 1) if g else None,
                               age_s=None if not self.odom or o["last_rx_wall"] is None else round(self.wall() - self.odom[-1][0], 3),
                               age_clock="wall(now) - header stamp (assumes ROS time == wall time; not verified)",
                               session=self.odom_session)
            out["geometry_status"] = self.geometry_status
            out["decode_rejected"] = dict(self.stats["decode_rejected"])
            out["ring_semantics"] = "ring_dropped = oldest buffered message evicted by a newer one in THIS process (not a DDS loss)"
            out["caminfo_n"] = self.stats["caminfo"]["n"]
        return out

    def close(self):
        self.alive = False
        try:
            self.th.join(timeout=1.0)
            self.node.destroy_node()
        except Exception:
            pass
