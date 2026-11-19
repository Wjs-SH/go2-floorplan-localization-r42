#!/usr/bin/env python3
"""LIVE2 FAST-LIO health + accumulation + E2 input feed (read-only consumer of FAST-LIO OUTPUTS).

Same subscriptions as the LIVE1 lio monitor with --accum (approval item A-acc):
  /Odometry          nav_msgs/Odometry       BEST_EFFORT KEEP_LAST(10)
  /cloud_registered  sensor_msgs/PointCloud2 BEST_EFFORT KEEP_LAST(1)   FAST-LIO output, NOT /lidar_points
The one decoded cloud feeds both the 2D accumulation (LIVE1 Accumulator, unchanged) and the E2 feed (e2_feed.E2Feed),
so E2 adds no DDS reader. Never subscribes /lidar_points or a driver topic, never publishes, never calls a service.

Writes (atomic) into --state: lio_state.json / lio_series.jsonl / lio_map.png exactly like LIVE1, plus
  e2_feed_state.json  every 0.5 s : feed counters, sample latency, odom session, store span
and the student LiveStore (odom_link.csv, samples/, far_points.npz) under --store.
The E2 sample work runs in its own thread (queue depth 1): a slow sample never blocks the odometry callback.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
sys.path.insert(0, str(CAND / "live1"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAND / "external" / "sci_live" / "realtime"))
import live1_common as C  # noqa: E402
from lio_monitor import OdomStats, Accumulator, cloud_xyz  # noqa: E402
import e2_feed as EF  # noqa: E402
from live_store import LiveStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(C.STATE))
    ap.add_argument("--store", default="/dev/shm/live2_store")
    ap.add_argument("--fastlio-cfg", required=True, help="the yaml the running FAST-LIO uses (extrinsic_R/T)")
    ap.add_argument("--sample-dt", type=float, default=1.5)
    ap.add_argument("--no-accum", action="store_true")
    ap.add_argument("--max-s", type=float, default=600.0)
    a = ap.parse_args()
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    rclpy.init()
    node = rclpy.create_node("live2_lio_monitor")
    od = OdomStats()
    acc = None if a.no_accum else Accumulator()
    owner = C.read_json(C.OWNED / "fastlio.json") or {}
    _Feed, _fkw = EF.E2Feed, {}
    _clean = os.environ.get("LIVE2_E2_CLEAN")      # 2026-11-13 esy: structure-point cleaning (e2_feed_clean.py). Unset -> E2Feed exactly as before
    if not _clean:                                 # the console may be restarted without the variable -> a flag file switches it on too
        try:
            _clean = (Path.home() / "kci_loadtest_20261113" / "e2_clean.on").read_text().strip() or "1"
        except OSError:
            _clean = None
    if _clean:                                     # "1" = all, "nomove" = no moving-object rule, "noperson" = floor/ceiling + thin lines only
        try:
            if _clean == "pose":                   # 2026-11-13 esy: original structure points + real-time pose correction only (e2_feed_posefix.py)
                import e2_feed_posefix as _PF
                _Feed, _fkw = _PF.PoseFixE2Feed, {}
            else:
                import e2_feed_clean as _EC
                _Feed, _fkw = _EC.CleanE2Feed, dict(person=_clean != "noperson", move=_clean not in ("noperson", "nomove"))
        except Exception as _e:                    # module missing / broken -> original feed
            print(f"[lio2] e2_feed_clean unavailable, using E2Feed: {_e!r}", flush=True)
    feed = _Feed(a.store, EF.read_extrinsic(a.fastlio_cfg), sample_dt=a.sample_dt, live_store_cls=LiveStore,
                 session_info=dict(boot_id=C.boot_id(), owner=f"fastlio_pid{owner.get('pid')}_start{owner.get('start')}",
                                   fastlio_cfg=str(a.fastlio_cfg), fastlio_cfg_sha256=C.sha256(a.fastlio_cfg)), **_fkw)
    due = threading.Event()
    last_session = [None]

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        od.on_odom(st, p.x, p.y, p.z, q.x, q.y, q.z, q.w, m.header.frame_id, m.child_frame_id)
        changed = last_session[0] is not None and od.session != last_session[0]
        last_session[0] = od.session
        feed.on_odom(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                     session_changed=changed)

    def on_cloud(m):
        try:
            xyz = cloud_xyz(m)
        except Exception:
            if acc is not None:
                acc.stats["decode_errors"] += 1
            return
        if acc is not None:
            z = od.last[3] if od.last else None
            acc.add_xyz(xyz, z_ref=z)
        if feed.on_cloud(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec, xyz):
            due.set()

    be = lambda d: QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=d,  # noqa: E731
                              durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(Odometry, "/Odometry", on_odom, be(10))
    node.create_subscription(PointCloud2, "/cloud_registered", on_cloud, be(1))
    subs = ["/Odometry", "/cloud_registered"]
    stop = [False]
    t_start = time.monotonic()
    me = os.getpid()

    def feeder():
        while not stop[0]:
            if due.wait(0.2):
                due.clear()
                try:
                    feed.sample()
                except Exception as e:
                    feed.stats["last_error"] = f"{type(e).__name__}: {e}"

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
                       accumulation=acc.summary() if acc else dict(enabled=False, why="--no-accum"),
                       frame_note="RAW FAST-LIO frame; not robot body, not building map")
            C.atomic_json(state / "lio_state.json", doc)
            C.atomic_json(state / "e2_feed_state.json", dict(feed.summary(), written_wall=time.time(), pid=me))
            if k % 2 == 0:
                with open(series, "a") as f:
                    f.write(json.dumps(dict(wall=time.time(), hz=s["hz_10s"], ia95=s["interarrival_ms"]["p95"], age95=s["header_to_rx_ms"]["p95"],
                                            last_rx_age=s["last_rx_age_s"], n=s["n"], acc=(acc.summary() if acc else None),
                                            feed_ms_p95=feed.summary().get("sample_ms_p95"), res=C.resources())) + "\n")
                if series.stat().st_size > 20 * 2 ** 20:
                    os.replace(series, series.with_suffix(".jsonl.1"))
            if acc is not None and k % 4 == 0:
                png = acc.render(tr)
                if png:
                    C.atomic_bytes(state / "lio_map.png", png)
            if time.monotonic() - t_start > a.max_s:
                stop[0] = True
    threading.Thread(target=writer, daemon=True).start()
    threading.Thread(target=feeder, daemon=True).start()
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
