#!/usr/bin/env python3
"""walk_tap.py RUN [--cloud-max-mb MB] [--cloud-max-s S] [--cloud-every e2|all] — read-only tap on the body-IMU FAST-LIO OUTPUTS during a walk diagnostic.

1. /sci_alt_imu/Odometry -> RUN/odom.csv (stamp, recv wall, x y z, qx qy qz qw), every message. This is the single position source
   of the tablet, the run report and re-analyses (calib_tools/walk_metrics.py), so all three show the same distance.
2. Only with --cloud-max-mb > 0: /sci_alt_imu/cloud_registered is stored UNDECODED (the exact CDR bytes FAST-LIO published:
   same frame_id 'camera_init', same header stamps, same fields) — i.e. exactly what LIVE2 lio_monitor2/e2_feed receive as
   /cloud_registered — but ONLY while RUN/CLOUD_WINDOW says "on <label>" (written by the tablet at step changes), and never past
   the MB / seconds caps. Default --cloud-every e2 keeps only what LIVE2 e2_feed consumes: bursts of E2_MERGE=3 consecutive
   scans every E2_SAMPLE_DT=1.5 s (e2_feed defaults, judged on the header stamps) -> 20 % of the scans. One dense XT16 cloud is
   ~2.6 MB (measured offline 2026-10-30: 55k points x 48 B), i.e. ~26 MB/s at full rate vs ~5.3 MB/s for e2 bursts.
   The cloud subscription exists only while a window is open (no cloud copies outside the windows). Output RUN/cloud_window_bag/ is a rosbag2 (sqlite3) with the windowed clouds + every Odometry message
   (raw), so the offline chain bag -> e2_compat adapter -> e2_feed copy runs without the live system.
Never subscribes /lidar_points, never publishes. Subscriptions are BEST_EFFORT (like lio_monitor2): a slow tap drops its own
copies and can never back-pressure FAST-LIO. Windows / drops / sizes -> RUN/cloud_windows.jsonl + RUN/cloud_window_summary.json."""
import argparse, json, os, signal, sqlite3, struct, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calib_tools"))
import walk_metrics as WM  # noqa: E402

CLOUD_TOPIC, ODOM_TOPIC = "/sci_alt_imu/cloud_registered", "/sci_alt_imu/Odometry"
E2_MERGE, E2_SAMPLE_DT = 3, 1.5            # = e2_feed.E2Feed defaults (sample_dt=1.5, merge=3), LIVE2 lio_monitor2 passes sample_dt
QOS_YAML = ("- history: 1\n  depth: {d}\n  reliability: 2\n  durability: 2\n  deadline:\n    sec: 2147483647\n    nsec: 4294967295\n"
            "  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 2147483647\n"
            "    nsec: 4294967295\n  avoid_ros_namespace_conventions: false")


def cloud_info(data):
    """PointCloud2 CDR -> (stamp_ns, frame_id, height, width, point_step)."""
    b = memoryview(data)[4:].tobytes()
    sec, nsec = struct.unpack_from("<iI", b, 0)
    frame, o = WM._cdr_string(b, 8)
    o = (o + 3) & ~3
    h, w = struct.unpack_from("<II", b, o)
    o += 8; nf = struct.unpack_from("<I", b, o)[0]; o += 4
    for _ in range(nf):                                       # PointField: name, offset u32, datatype u8, count u32
        _, o = WM._cdr_string(b, o); o = (o + 3) & ~3; o += 4; o += 1; o = (o + 3) & ~3; o += 4
    o += 1; o = (o + 3) & ~3                                  # is_bigendian
    ps = struct.unpack_from("<I", b, o)[0]
    return sec * 1_000_000_000 + nsec, frame, h, w, ps


class WindowGate:
    """Decides whether a cloud is stored. Pure (no ROS): used by the tap and by the fake walk backend / tests."""

    def __init__(self, run, max_mb, max_s, clock=time.time, every="e2"):
        self.f = Path(run) / "CLOUD_WINDOW"; self.log = Path(run) / "cloud_windows.jsonl"
        self.max_b, self.max_s, self.clock, self.every = max_mb * 1e6, float(max_s), clock, every
        self.burst_t0 = None; self.burst_n = 0; self.last_stamp = None; self.skipped = 0
        self.cur = None; self.windows = []; self.bytes = 0; self.on_s = 0.0; self.cap_hit = None
        self._chk = 0.0; self._want = None

    def _event(self, **e):
        with open(self.log, "a") as f:
            f.write(json.dumps(dict(wall=self.clock(), **e)) + "\n")

    def poll(self):
        now = self.clock()
        if now - self._chk >= 0.1:
            self._chk = now
            try:
                t = self.f.read_text().split()
                self._want = t[1] if t and t[0] == "on" and len(t) > 1 else ("on" if t and t[0] == "on" else None)
            except OSError:
                self._want = None
        if self.cur and (self._want != self.cur["label"] or self.cap_hit):
            self.close(now, "file_off" if not self.cap_hit else "cap_" + self.cap_hit)
        if self._want and not self.cur and not self.cap_hit:
            self.burst_t0 = None; self.burst_n = 0; self.last_stamp = None
            self.cur = dict(label=self._want, start_wall=now, end_wall=None, clouds=0, bytes=0, stamps=[], frames=set(), points=[])
            self._event(ev="open", label=self._want)
        return self.cur is not None

    def wanted(self, stamp_ns):
        """e2 bursts: E2_MERGE consecutive scans (gap <= 0.35 s, e2_feed's rule) starting every E2_SAMPLE_DT on the header clock."""
        if self.every == "all":
            return True
        if self.last_stamp is not None and stamp_ns - self.last_stamp > 0.35e9:
            self.burst_n = E2_MERGE if self.burst_n else 0      # a gap breaks the current burst; next burst on schedule
        self.last_stamp = stamp_ns
        if 0 < self.burst_n < E2_MERGE:
            self.burst_n += 1; return True
        if self.burst_t0 is None or stamp_ns - self.burst_t0 >= E2_SAMPLE_DT * 1e9 - 0.05e9:
            self.burst_t0 = stamp_ns; self.burst_n = 1; return True
        self.skipped += 1
        return False

    def accept(self, nbytes, stamp_ns, frame, points):
        """call only after poll() returned True and wanted() said yes; False when this cloud would exceed a cap (window closed)."""
        now = self.clock()
        if self.bytes + nbytes > self.max_b:
            self.cap_hit = "mb"
        elif self.on_s + (now - self.cur["start_wall"]) > self.max_s:
            self.cap_hit = "s"
        if self.cap_hit:
            self.close(now, "cap_" + self.cap_hit); return False
        c = self.cur; c["clouds"] += 1; c["bytes"] += nbytes; c["stamps"].append(stamp_ns); c["frames"].add(frame); c["points"].append(points)
        self.bytes += nbytes
        return True

    def close(self, now, why):
        c = self.cur
        if not c:
            return
        self.cur = None; self.on_s += now - c["start_wall"]
        st = c.pop("stamps"); pts = c.pop("points")
        d = [(b - a) / 1e6 for a, b in zip(st, st[1:])]
        inb = [x for x in d if x < 350]                          # inside e2 bursts (between bursts the gap is ~1.2 s by design)
        bursts = 1 + sum(x >= 350 for x in d) if st else 0
        c.update(end_wall=now, why=why, frames=sorted(c["frames"]), first_stamp_ns=st[0] if st else None, last_stamp_ns=st[-1] if st else None,
                 every=self.every, bursts=bursts, max_gap_in_burst_ms=round(max(inb), 1) if inb else None,
                 gaps_over_150ms_in_burst=sum(x > 150 for x in inb), dup_or_back_stamps=sum(x <= 0 for x in d),
                 points_mean=round(sum(pts) / len(pts)) if pts else None, mb=round(c["bytes"] / 1e6, 2))
        self.windows.append(c); self._event(ev="close", **{k: v for k, v in c.items() if k != "start_wall"})

    def summary(self):
        return dict(kind="CLOUD_WINDOW_SUMMARY", topic=CLOUD_TOPIC, stored_format="sensor_msgs/msg/PointCloud2 CDR bytes as published (undecoded)",
                    every=self.every, e2_merge=E2_MERGE, e2_sample_dt=E2_SAMPLE_DT, skipped_by_e2_schedule=self.skipped,
                    windows=self.windows, clouds=sum(w["clouds"] for w in self.windows), mb=round(self.bytes / 1e6, 2),
                    window_on_s=round(self.on_s, 1), cap_mb=self.max_b / 1e6, cap_s=self.max_s, cap_hit=self.cap_hit)


class BagWriter:
    """Minimal rosbag2 sqlite3 writer (Foxy schema; this install has no rosbag2_py). `ros2 bag info/play` read it."""

    def __init__(self, d):
        self.d = Path(d); self.d.mkdir(parents=True, exist_ok=False)
        self.db = self.d / (self.d.name + "_0.db3")
        self.con = sqlite3.connect(str(self.db))
        self.con.executescript("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, serialization_format TEXT NOT NULL,"
                               " offered_qos_profiles TEXT NOT NULL);"
                               "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL);"
                               "CREATE INDEX timestamp_idx ON messages (timestamp ASC);")
        self.topics = {}; self.count = {}; self.t0 = None; self.t1 = None; self.last_commit = time.time()

    def topic(self, name, typ, depth):
        i = len(self.topics) + 1; self.topics[name] = (i, typ); self.count[name] = 0
        self.con.execute("INSERT INTO topics VALUES (?,?,?,?,?)", (i, name, typ, "cdr", QOS_YAML.format(d=depth)))

    def write(self, name, t_ns, data):
        self.con.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES (?,?,?)", (self.topics[name][0], t_ns, bytes(data)))
        self.count[name] += 1; self.t0 = t_ns if self.t0 is None else min(self.t0, t_ns); self.t1 = t_ns if self.t1 is None else max(self.t1, t_ns)
        if time.time() - self.last_commit > 1.0:
            self.con.commit(); self.last_commit = time.time()

    def close(self):
        self.con.commit(); self.con.close()
        tl = "".join(f"    - topic_metadata:\n        name: {n}\n        type: {t}\n        serialization_format: cdr\n"
                     f"        offered_qos_profiles: \"\"\n      message_count: {self.count[n]}\n" for n, (_, t) in self.topics.items())
        (self.d / "metadata.yaml").write_text(
            "rosbag2_bagfile_information:\n  version: 4\n  storage_identifier: sqlite3\n  relative_file_paths:\n"
            f"    - {self.db.name}\n  duration:\n    nanoseconds: {(self.t1 - self.t0) if self.t0 else 0}\n"
            f"  starting_time:\n    nanoseconds_since_epoch: {self.t0 or 0}\n  message_count: {sum(self.count.values())}\n"
            f"  topics_with_message_count:\n{tl}  compression_format: \"\"\n  compression_mode: \"\"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--cloud-max-mb", type=float, default=0.0); ap.add_argument("--cloud-max-s", type=float, default=60.0)
    ap.add_argument("--cloud-every", choices=("e2", "all"), default="e2")
    a = ap.parse_args()
    run = Path(a.run)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    rclpy.init()
    node = rclpy.create_node("sci_alt_walk_tap")
    be = lambda d: QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=d,  # noqa: E731
                              durability=DurabilityPolicy.VOLATILE)
    odo = open(run / "odom.csv", "a", buffering=1)
    if odo.tell() == 0:
        odo.write(WM.ODOM_CSV_HEADER + "\n")
    cloud = a.cloud_max_mb > 0
    bag = gate = None
    if cloud:
        bag = BagWriter(run / "cloud_window_bag")
        bag.topic(CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2", 5); bag.topic(ODOM_TOPIC, "nav_msgs/msg/Odometry", 50)
        gate = WindowGate(run, a.cloud_max_mb, a.cloud_max_s, every=a.cloud_every)
    stats = dict(odom=0, clouds_seen=0, clouds_stored=0, cloud_cb_ms_max=0.0, odom_frames=set(), cloud_subscribed_s=0.0, subscribe_events=0)

    def on_odom(data):
        now = time.time()
        s, fr, ch, *pq = WM.decode_odometry_cdr(data)
        odo.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f\n" % (s, now, *pq))
        stats["odom"] += 1; stats["odom_frames"].add(f"{fr}->{ch}")
        if bag:
            bag.write(ODOM_TOPIC, int(now * 1e9), data)

    def on_cloud(data):
        t0 = time.perf_counter(); stats["clouds_seen"] += 1
        if gate.poll():
            st, fr, h, w, ps = cloud_info(data)
            if gate.wanted(st) and gate.accept(len(data), st, fr, h * w):
                bag.write(CLOUD_TOPIC, time.time_ns(), data); stats["clouds_stored"] += 1
        stats["cloud_cb_ms_max"] = max(stats["cloud_cb_ms_max"], 1000 * (time.perf_counter() - t0))

    node.create_subscription(Odometry, ODOM_TOPIC, on_odom, be(50), raw=True)
    sub = [None, 0.0]

    def window_timer():                                    # cloud subscription only while a window is open
        on = gate.poll()
        if on and sub[0] is None:
            sub[0] = node.create_subscription(PointCloud2, CLOUD_TOPIC, on_cloud, be(5), raw=True); sub[1] = time.time()
            stats["subscribe_events"] += 1
        elif not on and sub[0] is not None:
            node.destroy_subscription(sub[0]); sub[0] = None; stats["cloud_subscribed_s"] += time.time() - sub[1]
    if cloud:
        node.create_timer(0.1, window_timer)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    print(f"[walk_tap] odom -> {run / 'odom.csv'}; cloud window {'ON cap %.0f MB / %.0f s' % (a.cloud_max_mb, a.cloud_max_s) if cloud else 'OFF'}", flush=True)
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        odo.close()
        summ = dict(stats, odom_frames=sorted(stats["odom_frames"]), cloud_cb_ms_max=round(stats["cloud_cb_ms_max"], 2), cloud_enabled=cloud)
        if cloud:
            if sub[0] is not None:
                stats["cloud_subscribed_s"] += time.time() - sub[1]
            summ["cloud_subscribed_s"] = round(stats["cloud_subscribed_s"], 1); summ["subscribe_events"] = stats["subscribe_events"]
            if gate.cur:
                gate.close(time.time(), "tap_stopped")
            bag.close()
            summ.update(gate.summary(), bag_dir=str(bag.d), bag_bytes=os.path.getsize(bag.db))
        (run / "cloud_window_summary.json").write_text(json.dumps(summ, indent=1))
        print("[walk_tap] stopped: " + json.dumps({k: v for k, v in summ.items() if k != "windows"}), flush=True)
        try:
            node.destroy_node(); rclpy.shutdown()
        except Exception:                                    # noqa: BLE001 — context may already be shut down by SIGINT
            pass


if __name__ == "__main__":
    main()
