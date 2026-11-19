#!/usr/bin/env python3
"""LIVE1 G0 ROS graph probe — DISCOVERY ONLY.

Creates one rclpy participant, waits for discovery, then reads the graph (node names, topics/types, publisher and
subscriber endpoints with QoS). It creates NO subscription, NO publisher, NO service client and never touches data.
Writes JSON to --out. Exit 0 always (the content says what is missing)."""
import argparse
import json
import os
import socket
import time

KEY = ["/lidar_points", "/Odometry", "/path", "/cloud_registered", "/cloud_registered_body", "/utlidar/imu",
       "/utlidar/imu_corrected", "/camera/color/image_raw", "/camera/color/camera_info",
       "/camera/aligned_depth_to_color/image_raw", "/camera/aligned_depth_to_color/camera_info", "/tf", "/tf_static",
       "/floorplan_loc/pose", "/cmd_vel", "/api/sport/request"]


def qos_d(q):
    def name(v):
        return getattr(v, "name", str(v))
    return dict(reliability=name(q.reliability), durability=name(q.durability), history=name(q.history), depth=q.depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", type=float, default=4.0)
    a = ap.parse_args()
    import rclpy
    rclpy.init()
    node = rclpy.create_node("live1_graph_probe_discovery_only")
    t0 = time.time()
    while time.time() - t0 < a.wait:
        rclpy.spin_once(node, timeout_sec=0.2)
    topics = {n: t for n, t in node.get_topic_names_and_types()}
    nodes = sorted(f"{ns.rstrip('/')}/{n}" for n, ns in node.get_node_names_and_namespaces())
    key = {}
    for t in KEY:
        pubs = node.get_publishers_info_by_topic(t)
        subs = node.get_subscriptions_info_by_topic(t)
        key[t] = dict(present=t in topics, types=topics.get(t),
                      publishers=[dict(node=f"{p.node_namespace.rstrip('/')}/{p.node_name}", qos=qos_d(p.qos_profile)) for p in pubs],
                      subscribers=[dict(node=f"{s.node_namespace.rstrip('/')}/{s.node_name}", qos=qos_d(s.qos_profile)) for s in subs
                                   if s.node_name != "live1_graph_probe_discovery_only"])
    env = {k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI")}
    doc = dict(kind="LIVE1_GRAPH_PROBE", discovery_only=True, created_subscriptions=0, created_publishers=0,
               host=socket.gethostname(), boot_id=open("/proc/sys/kernel/random/boot_id").read().strip(),
               wall=time.strftime("%F %T %Z"), wait_s=a.wait, env=env, nodes=nodes, topics=topics, key=key)
    node.destroy_node()
    rclpy.shutdown()
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(dict(nodes=len(nodes), topics=len(topics),
                          key={t: (len(v["publishers"]), len(v["subscribers"])) for t, v in key.items() if v["present"]})))


if __name__ == "__main__":
    main()
