"""LIVE1 shared sensor hub: ONE read-only ROS receiver (ros_readonly allowlist) -> shared-memory frame ring.

Writer (live1/sensor_hub.py) owns the only RealSense/Odometry subscription of LIVE1 and writes:
  <hub>/slot_XX.color  raw BGR uint8 (h*w*3)        <hub>/slot_XX.depth  raw uint16 (h*w) nearest depth (or absent)
  <hub>/index.json     atomic: frames [{seq, stamp, rx_mono_ns, slot, w, h, depth_stamp, color_sha8}], odom samples
                       (validated RAW samples + session), caminfo, receiver health, writer pid / wall
Reader (SharedHubSource) implements the small source interface e5_node uses (poll / depth / pose / odom_hist / health /
alive / lock / odom_session / rx_mono_ns / last_odom_wall), so SigLIP runs without its own ROS subscription and its
lifetime is independent of the RGB / odometry path. Geometry follows the SAME frame-contract code as ros_readonly
(no contract -> odom_hist == [] and only a FLAGGED identity pose). A slot can be overwritten while being read; the
reader re-checks the index sequence after reading and drops a torn frame (counted)."""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ros_readonly import RosReadonlySource
import depth_freshness as DF

N_SLOTS = 16


def read_index(hub):
    try:
        with open(Path(hub) / "index.json") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class SharedHubSource(RosReadonlySource):
    """Reader. No rclpy import, no subscription."""

    def __init__(self, hub_dir, visual_identity_pose=True, max_index_age_s=2.0):
        self.hub = Path(hub_dir)
        self._init_state(visual_identity_pose=visual_identity_pose)
        self.max_index_age_s = max_index_age_s
        self.subscribed, self.refused = ["shared_hub:" + str(self.hub)], {}
        self.alive = True
        self.last_seq = None
        self.torn = 0
        self.hub_stale = 0
        self.depth_gate = DF.Tracker()
        self._idx = None
        self.visual_only = True
        self._refresh()

    # -------------------------------------------------------------- index -> state (odometry through the SAME validation)
    def _refresh(self):
        d = read_index(self.hub)
        if d is None:
            return None
        self._idx = d
        with self.lock:
            self.caminfo = d.get("caminfo")
            self.visual_only = not d.get("odom_subscribed", False)
            sess = d.get("odom_session", 0)
            if sess != self.odom_session:
                self.odom.clear()
                self.odom_session = sess
            have = self.odom[-1][0] if self.odom else -1e18
            for o in d.get("odom", []):
                if o[0] > have:
                    self.odom.append(tuple(o[:4]))
            if self.odom:
                self.last_odom_wall = d.get("odom_last_rx_wall") or time.time()
            self.stats["odom"]["frames"] = d.get("odom_frames")
        return d

    def _load(self, fr, kind):
        p = self.hub / f"slot_{fr['slot']:02d}.{kind}"
        try:
            b = p.read_bytes()
        except OSError:
            return None
        if kind == "color":
            if len(b) != fr["h"] * fr["w"] * 3:
                return None
            return np.frombuffer(b, np.uint8).reshape(fr["h"], fr["w"], 3).copy()
        if len(b) != fr["h"] * fr["w"] * 2:
            return None
        return np.frombuffer(b, "<u2").reshape(fr["h"], fr["w"]).copy()

    def _still_valid(self, fr):
        d2 = read_index(self.hub)
        if not d2:
            return False
        return any(f["seq"] == fr["seq"] and f["slot"] == fr["slot"] for f in d2.get("frames", []))

    def poll(self):
        d = self._refresh()
        if d is None or time.time() - float(d.get("wall", 0)) > self.max_index_age_s:
            self.hub_stale += 1
            return []
        out = []
        frames = [f for f in d.get("frames", []) if self.last_seq is None or f["seq"] > self.last_seq]
        for fr in frames[-2:]:                               # latest-first semantics: at most the 2 newest new frames
            img = self._load(fr, "color")
            if img is None or not self._still_valid(fr):
                self.torn += 1
                continue
            with self.lock:
                self.rx_mono_ns[fr["seq"]] = fr["rx_mono_ns"]
                if len(self.rx_mono_ns) > 300:
                    self.rx_mono_ns.pop(next(iter(self.rx_mono_ns)))
                self.stats["color"]["n"] += 1
                self.stats["color"]["rx"] = (self.stats["color"]["rx"] + [time.monotonic()])[-2000:]
            out.append((fr["stamp"], fr["seq"], img))
        if frames:
            self.last_seq = frames[-1]["seq"]
        return out

    def depth(self, stamp):
        """Nearest aligned depth, or (None, None) when it is missing or too far from `stamp` (R34: never an old image)."""
        d = self._idx or read_index(self.hub)
        if not d:
            self.depth_gate.judge(stamp, None)
            return None, None
        cands = [f for f in d.get("frames", []) if f.get("depth_stamp") is not None]
        if not cands:
            self.depth_gate.judge(stamp, None)
            return None, None
        fr = min(cands, key=lambda f: abs(f["depth_stamp"] - stamp))
        if not self.depth_gate.judge(stamp, fr["depth_stamp"]):
            return None, None
        img = self._load(fr, "depth")
        if img is None or not self._still_valid(fr):
            self.torn += 1
            return None, None
        return img, fr["depth_stamp"]

    def health(self):
        d = self._idx or {}
        h = dict(d.get("receiver_health") or {})
        h.update(source="shared_hub", hub=str(self.hub), hub_index_age_s=None if not d else round(time.time() - float(d.get("wall", 0)), 3),
                 hub_writer_pid=d.get("pid"), torn_frames=self.torn, hub_stale_polls=self.hub_stale,
                 visual_only=self.visual_only, pose_mode=self.pose_mode(), geometry_status=self.geometry_status,
                 odom_session=self.odom_session, reader_depth_fresh=self.depth_gate.snapshot())
        return h

    def close(self):
        self.alive = False


class HubWriter:
    """Used by live1/sensor_hub.py around a live RosReadonlySource (the only ROS receiver)."""

    def __init__(self, hub_dir, src, jpeg_path=None, jpeg_period_s=0.125):
        self.hub = Path(hub_dir)
        self.hub.mkdir(parents=True, exist_ok=True)
        self.src = src
        self.frames = deque(maxlen=N_SLOTS - 2)              # 2 spare slots: a reader never sees a slot reused within 2 writes
        self.seq_slot = 0
        self.jpeg_path, self.jpeg_period_s, self._last_jpeg = jpeg_path, jpeg_period_s, 0.0
        self.counts = dict(written=0, depth_attached=0, jpeg=0, write_ms=deque(maxlen=300))
        self.last_color_wall = None

    def step(self):
        import cv2
        new = self.src.poll()
        for st, seq, img in new[-1:]:                        # newest only: the hub never queues old frames
            t0 = time.perf_counter()
            slot = self.seq_slot % N_SLOTS
            self.seq_slot += 1
            h, w = img.shape[:2]
            _atomic(self.hub / f"slot_{slot:02d}.color", np.ascontiguousarray(img).tobytes())
            dimg, dst = self.src.depth(st)
            if dimg is not None:
                _atomic(self.hub / f"slot_{slot:02d}.depth", np.ascontiguousarray(dimg.astype("<u2")).tobytes())
                self.counts["depth_attached"] += 1
            self.frames.append(dict(seq=int(seq), stamp=float(st), rx_mono_ns=int(self.src.rx_mono_ns.get(seq, 0)), slot=slot, w=w, h=h,
                                    depth_stamp=None if dimg is None else float(dst), wall=time.time()))
            self.counts["written"] += 1
            self.counts["write_ms"].append(1000 * (time.perf_counter() - t0))
            self.last_color_wall = time.time()
            if self.jpeg_path and time.time() - self._last_jpeg >= self.jpeg_period_s:
                self._last_jpeg = time.time()
                im = img.copy()
                cv2.putText(im, f"LIVE RGB seq {seq} | hub", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                _atomic(self.jpeg_path, cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
                self.counts["jpeg"] += 1
        return len(new)

    def write_index(self):
        s = self.src
        with s.lock:
            odom = [list(o) for o in list(s.odom)[-200:]]
            sess = s.odom_session
            frames_ids = s.stats["odom"].get("frames")
            last_rx = s.stats["odom"].get("last_rx_wall")
        wm = sorted(self.counts["write_ms"])
        doc = dict(kind="LIVE1_SENSOR_HUB", pid=os.getpid(), wall=time.time(), frames=list(self.frames), odom=odom, odom_session=sess,
                   odom_frames=frames_ids, odom_last_rx_wall=last_rx, odom_subscribed="/Odometry" in s.subscribed,
                   caminfo=s.caminfo, receiver_health=s.health(), last_color_wall=self.last_color_wall,
                   writer=dict(written=self.counts["written"], depth_attached=self.counts["depth_attached"], jpeg=self.counts["jpeg"],
                               write_ms_p50=wm[len(wm) // 2] if wm else None, write_ms_p95=wm[int(len(wm) * 0.95)] if wm else None))
        _atomic(self.hub / "index.json", json.dumps(doc, default=str).encode())
        return doc


def _atomic(path, data):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
