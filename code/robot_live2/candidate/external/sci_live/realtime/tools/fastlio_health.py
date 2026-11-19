# -*- coding: utf-8 -*-
"""fastlio_health.py — /Odometry 를 10초 받아 주기·헤더 빈틈·도착 지연·메시지 사이 점프·지금 위치를 본다(로봇에서, Unitree DDS 환경).
    source ~/unitree_ros2/setup.sh; CYCLONEDDS_URI=file://.../cyclonedds_live.xml python fastlio_health.py"""
import time, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
rclpy.init(); n = Node('odom_health'); g = []
n.create_subscription(Odometry, '/Odometry', lambda m: g.append((time.time(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)), QoSProfile(depth=100))
t0 = time.time()
while time.time() - t0 < 10: rclpy.spin_once(n, timeout_sec=0.05)
a = np.array(g); dh = np.diff(a[:, 1]); lag = a[:, 0] - a[:, 1]; step = np.linalg.norm(np.diff(a[:, 2:5], axis=0), axis=1)
print(f'FAST-LIO {len(a)/10:.1f} Hz · 헤더 간격 중앙 {np.median(dh):.3f}s 최대 {dh.max():.2f}s · 0.15초 넘는 빈틈 {int((dh>0.15).sum())}회 · 도착 지연 중앙 {np.median(lag):.2f}s 최대 {lag.max():.2f}s')
print(f'한 번에 움직인 거리 중앙 {np.median(step):.3f} m 최대 {step.max():.2f} m · 0.3 m 넘는 점프 {int((step>0.3).sum())}회 · 지금 xyz {np.round(a[-1,2:5],2).tolist()}')
