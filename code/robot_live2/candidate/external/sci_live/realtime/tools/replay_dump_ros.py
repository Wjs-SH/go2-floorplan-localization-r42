# -*- coding: utf-8 -*-
"""replay_dump_ros.py — a131_live 시험 데이터(스캔·자세 묶음)를 실제 토픽 형식으로 실제 시간 흐름대로 내보낸다.

    ROS_DOMAIN_ID=77 python replay_dump_ros.py ../../a131_live/testdata/S6_chunk0_55.npz [배속 1.0] [reliable|best_effort] [반복 1]

  /lidar_points : PointCloud2 (x, y, z, intensity float32 · ring uint16 · timestamp float64, point_step 26 — 녹화본과 같은 배치),
                  스캔 끝 시각에 발행, header.stamp = 녹화본 헤더 시각
  /Odometry     : nav_msgs/Odometry (camera_init→imu). 시험 데이터의 자세는 odom→base_link 라서
                  a131_live.fastlio_to_base 의 역변환으로 FAST-LIO 원 자세를 만든다(노드가 다시 바꾸면 원래 자세).
반복(부하 시험용): 13초 데이터를 시각을 이어 붙여 여러 번 낸다(자세는 매번 처음으로 돌아가 튄다 — 위치 결과는 의미 없고
구조점 추출 부하·메모리만 본다).
센서를 켜지 않고 노드의 ROS 연결·표본 만들기를 시험하는 용도다. 다른 노드와 섞이지 않게 도메인을 따로 쓴다.
ROS_LOCALHOST_ONLY=1 은 쓰지 않는다: FastDDS 가 공유메모리 전송을 끄고 UDP 로만 보내 1.6 MB 점군이 대부분 유실된다.
"""
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'a131_live'))
from a131_live import LEVER_IMU_BASE, fastlio_to_base  # noqa: E402


def base_to_fastlio(pt, qt, delta_deg=45.3):
    """fastlio_to_base(mirror=True) 의 역변환."""
    from scipy.spatial.transform import Rotation as Rot
    R_t = Rot.from_quat(qt)
    R_sy = R_t * Rot.from_euler('z', -delta_deg, degrees=True)
    pb_m = pt + R_sy.apply(LEVER_IMU_BASE) - R_t.apply(LEVER_IMU_BASE)
    pb = pb_m * np.array([1.0, -1.0, 1.0])
    q_imu = R_sy.as_quat() * np.array([-1.0, 1.0, -1.0, 1.0])
    p_imu = pb - Rot.from_quat(q_imu).apply(LEVER_IMU_BASE)
    return p_imu, q_imu


def stamp(ns):
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def main():
    d = np.load(sys.argv[1])
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    ps, px, pq = d['pose_stamp_ns'], d['pose_xyz'], d['pose_quat_xyzw']
    p_imu, q_imu = base_to_fastlio(px, pq)
    chk_p, chk_q = fastlio_to_base(p_imu, q_imu)
    err = float(np.abs(chk_p - px).max())
    print(f'역변환 확인: 위치 최대 차 {err:.2e} m', flush=True)

    import re
    ids = sorted({int(m.group(1)) for k in d.files for m in [re.match(r'scan(\d+)_x$', k)] if m})
    ev = [(int(ps[i]), 'odom', i) for i in range(len(ps))] + [(int(d[f'scan{i}_point_end_ns']), 'scan', i) for i in ids]
    ev.sort()
    rclpy.init()
    node = Node('a131_dump_replay')
    rel = sys.argv[3] if len(sys.argv) > 3 else 'best_effort'
    loops = int(sys.argv[4]) if len(sys.argv) > 4 else 1    # 실물 드라이버 설정에 맞춰 시험(reliable | best_effort)
    qs = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if rel == 'reliable' else ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, depth=10)
    pub_c = node.create_publisher(PointCloud2, '/lidar_points', qs)
    pub_o = node.create_publisher(Odometry, '/Odometry', QoSProfile(depth=50))
    fields = [PointField(name='x', offset=0, datatype=7, count=1), PointField(name='y', offset=4, datatype=7, count=1),
              PointField(name='z', offset=8, datatype=7, count=1), PointField(name='intensity', offset=12, datatype=7, count=1),
              PointField(name='ring', offset=16, datatype=4, count=1), PointField(name='timestamp', offset=18, datatype=8, count=1)]
    dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'],
                   'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f8'], 'offsets': [0, 4, 8, 12, 16, 18], 'itemsize': 26})
    time.sleep(2.0)                                 # 구독자가 붙을 시간
    span = ev[-1][0] - ev[0][0] + 100_000_000
    ev = [(t + k * span, kind, i) for k in range(loops) for t, kind, i in ev]
    t0_ev, wall0 = ev[0][0], time.time()
    for t, kind, i in ev:
        due = wall0 + (t - t0_ev) / 1e9 / speed
        w = due - time.time()
        if w > 0:
            time.sleep(w)
        if kind == 'odom':
            m = Odometry()
            m.header.stamp = stamp(t); m.header.frame_id = 'camera_init'; m.child_frame_id = 'body'
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, p_imu[i])
            (m.pose.pose.orientation.x, m.pose.pose.orientation.y,
             m.pose.pose.orientation.z, m.pose.pose.orientation.w) = map(float, q_imu[i])
            pub_o.publish(m)
        else:
            n = len(d[f'scan{i}_x'])
            shift = t - int(d[f'scan{i}_point_end_ns'])        # 반복할 때 더한 시각
            arr = np.zeros(n, dt)
            for k in ('x', 'y', 'z', 'intensity', 'ring', 'timestamp'):
                arr[k] = d[f'scan{i}_{k}']
            if shift:
                arr['timestamp'] = d[f'scan{i}_timestamp'] + shift / 1e9
            c = PointCloud2()
            c.header.stamp = stamp(int(d[f'scan{i}_header_stamp_ns']) + shift); c.header.frame_id = 'hesai_lidar'
            c.height, c.width = 1, n
            c.fields = fields; c.is_bigendian = False; c.point_step = 26; c.row_step = 26 * n; c.is_dense = False
            c.data = arr.tobytes()
            pub_c.publish(c)
    time.sleep(1.0)
    print(f'발행 끝: 스캔 {len(ids) * loops}개 · 자세 {len(ps) * loops}개 · {time.time() - wall0:.1f}s', flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
