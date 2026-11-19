import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
dur = float(sys.argv[1])
rclpy.init(); n = Node('still_probe'); got = []
n.create_subscription(Odometry, '/Odometry', lambda m: got.append((time.time(), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)), QoSProfile(depth=50))
t0 = time.time()
while time.time() - t0 < dur: rclpy.spin_once(n, timeout_sec=0.1)
import numpy as np
g = np.array(got)
if len(g) < 2: print('FAST-LIO 자세 없음'); sys.exit()
d = np.linalg.norm(g[-1, 1:] - g[0, 1:])
print(f'{dur:.0f}초 정지 점검: 자세 {len(g)/dur:.1f} Hz · 위치 변화 {d:.3f} m · 지금 xyz ({g[-1,1]:.2f}, {g[-1,2]:.2f}, {g[-1,3]:.2f})')
