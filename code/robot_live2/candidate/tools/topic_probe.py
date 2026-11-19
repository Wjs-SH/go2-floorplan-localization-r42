#!/usr/bin/env python3
"""Measure receive rate of ROS 2 topics (read-only subscriber). usage: topic_probe.py SEC topic:type ..."""
import sys, time, importlib, rclpy
from rclpy.qos import qos_profile_sensor_data
def main():
    sec = float(sys.argv[1]); rclpy.init(); n = rclpy.create_node("sci_topic_probe")
    stamps = {}
    for spec in sys.argv[2:]:
        t, ty = spec.split(":"); mod, cls = ty.rsplit(".", 1)
        C = getattr(importlib.import_module(mod), cls); stamps[t] = []
        n.create_subscription(C, t, lambda m, t=t: stamps[t].append(time.time()), qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < sec: rclpy.spin_once(n, timeout_sec=0.05)
    for t, s in stamps.items():
        hz = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 else 0.0
        gap = max((b - a for a, b in zip(s, s[1:])), default=float("nan"))
        print(f"{t} n={len(s)} hz={hz:.2f} max_gap={gap:.3f}s")
    n.destroy_node(); rclpy.shutdown()
main()
