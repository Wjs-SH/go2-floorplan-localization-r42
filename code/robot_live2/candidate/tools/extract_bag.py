#!/usr/bin/env python3
"""Read-only extraction from a ROS 2 sqlite bag: RGB (jpg q95) + aligned depth (16-bit png) at a target rate,
camera_info, and all /tf transforms (csv). The bag is opened read-only (sqlite URI mode=ro)."""
import argparse, csv, json, sqlite3, os
from pathlib import Path
import numpy as np, cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage

ap = argparse.ArgumentParser(); ap.add_argument("db"); ap.add_argument("out"); ap.add_argument("--hz", type=float, default=2.0)
a = ap.parse_args(); out = Path(a.out); (out / "rgb").mkdir(parents=True, exist_ok=True); (out / "depth").mkdir(exist_ok=True)
con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
tid = {n: i for i, n in con.execute("select id,name from topics")}
def rows(topic):
    return con.execute("select timestamp,data from messages where topic_id=? order by timestamp", (tid[topic],))
def img(m):
    a_ = np.frombuffer(m.data, np.uint8)
    if m.encoding in ("16UC1", "mono16"): return a_.view(np.uint16).reshape(m.height, m.width)
    a_ = a_.reshape(m.height, m.width, -1)
    return cv2.cvtColor(a_, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else a_
for t in ("/camera/color/camera_info", "/camera/aligned_depth_to_color/camera_info"):
    for ts, d in rows(t):
        m = deserialize_message(d, CameraInfo); json.dump({"k": list(m.k), "d": list(m.d), "w": m.width, "h": m.height, "frame": m.header.frame_id}, open(out / (t.split("/")[2] + "_info.json"), "w")); break
# stream: keep only (rowid, stamp) for depth; fetch blobs on demand (bounded memory)
dids, dstamps = [], []
for rid, d in con.execute("select id,data from messages where topic_id=? order by timestamp", (tid["/camera/aligned_depth_to_color/image_raw"],)):
    h = deserialize_message(d, Image).header.stamp; dids.append(rid); dstamps.append(h.sec + h.nanosec * 1e-9)
dstamps = np.array(dstamps)
def depth_msg(k):
    return deserialize_message(con.execute("select data from messages where id=?", (dids[k],)).fetchone()[0], Image)
step, last, n = 1.0 / a.hz, -1e9, 0
w = csv.writer(open(out / "index.csv", "w")); w.writerow(["idx", "stamp", "rgb", "depth", "depth_dt"])
for ts, d in con.execute("select timestamp,data from messages where topic_id=? order by timestamp", (tid["/camera/color/image_raw"],)):
    m = deserialize_message(d, Image); st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if st - last < step: continue
    last = st; k = int(np.argmin(np.abs(dstamps - st))); dd = depth_msg(k)
    rn, dn = f"rgb/{n:05d}.jpg", f"depth/{n:05d}.png"
    cv2.imwrite(str(out / rn), img(m), [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(str(out / dn), img(dd))
    w.writerow([n, f"{st:.6f}", rn, dn, f"{dstamps[k]-st:.4f}"]); n += 1
tw = csv.writer(open(out / "tf.csv", "w")); tw.writerow(["stamp", "parent", "child", "x", "y", "z", "qx", "qy", "qz", "qw"])
# SCI recordings (scripts/record_e5_dataset.sh) carry odometry as nav_msgs/Odometry: write it as odom->base_link rows
from nav_msgs.msg import Odometry
for ot in ("/Odometry", "/utlidar/robot_odom"):
    if ot in tid:
        for ts, d in rows(ot):
            m = deserialize_message(d, Odometry); p, q = m.pose.pose.position, m.pose.pose.orientation
            tw.writerow([f"{m.header.stamp.sec + m.header.stamp.nanosec*1e-9:.6f}", "odom", "base_link", p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        break
for ts, d in rows("/tf"):
    for t in deserialize_message(d, TFMessage).transforms:
        p, q = t.transform.translation, t.transform.rotation
        tw.writerow([f"{t.header.stamp.sec + t.header.stamp.nanosec*1e-9:.6f}", t.header.frame_id, t.child_frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
print("frames", n)
