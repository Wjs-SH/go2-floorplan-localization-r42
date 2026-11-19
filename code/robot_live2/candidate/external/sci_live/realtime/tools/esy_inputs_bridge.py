#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""esy_inputs_bridge.py — esy 수정 FAST-LIO 에 로스백 때와 같은 입력을 실시간으로 만들어 준다(2026-10-09).

esy 가 로스백 입력을 만든 방식(로봇 ~/fastlio_legodom_20261009: make_ref_bag.py · add_leg_odom.py)을 실시간으로 옮겼다.

  /utlidar/imu → /utlidar/imu_ref   (make_ref_bag.py)
      헤더 + shift 로 LiDAR 시계에 올리고, 자이로에서 바이어스를 뺀다. 축·부호·가속도는 그대로.
      shift   = median(LiDAR 헤더 − 그 LiDAR 수신 시각에서 보간한 IMU 헤더) — 처음 cal_s 초로 한 번 구해 고정
      바이어스 = 처음 bias_s 초 자이로 평균(그동안 로봇을 세워 둘 것)
      다른 점: 200 Hz 균일 재샘플은 안 한다. 실시간이라 뒤 샘플을 기다릴 수 없어 받은 샘플을 그대로 낸다.

  /utlidar/robot_pose → /leg_odom   (add_leg_odom.py)
      연속 두 위치의 3차원 거리 / 수신 시각 차(1 ms 하한) = 속력(항상 ≥ 0) → twist.twist.linear.x, pose 는 비움.
      로스백은 /tf odom→base_link 를 같은 식으로 차분했다. go2_ws 에서 그 변환을 내는 것은 go2_driver 하나이고
      /utlidar/robot_pose 로 만든다(z 만 +0.07, 차분엔 무관). 로스백의 /tf 가 정말 그것이었는지는 bag(USB)을 안 열어 미확인.
      다른 점: 로스백은 LiDAR 수신 시각에서 앞뒤 값을 보간했고, 여기선 위치가 올 때마다 최신 값을 낸다.
      FAST-LIO 는 /leg_odom 헤더를 안 보고 스캔 처리 때 가장 최근 값만 쓴다(laserMapping.cpp leg_odom_cbk).

LiDAR 는 shift 를 구하는 동안만 raw 로 구독(헤더 시각 8바이트만 읽음)하고, 끝나면 구독을 없앤다.
보정값은 --out 폴더 imu_ref_calib.json 에 남긴다. 비교용으로 imu_correct.py 식 오프셋(첫 메시지의 수신 − 헤더)도 같이 적는다.
--compare 면 /sportmodestate hypot(vx, vy) 를 같이 받아 leg_speed.csv 에 나란히 적는다(esy 가 "뜻이 같다"고 한 대안의 실측 비교).

    source ~/unitree_ros2/setup.sh; python3 tools/esy_inputs_bridge.py --out ../logs/live_<월일>
"""
import argparse
import json
import math
import os
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2

NS = 1_000_000_000


def stamp_ns(st):
    return st.sec * NS + st.nanosec


class Bridge(Node):
    def __init__(self, a):
        super().__init__('esy_inputs_bridge')
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        be = lambda d: QoSProfile(depth=d, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        rel = lambda d: QoSProfile(depth=d, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        # FAST-LIO 수정본의 IMU 구독은 KeepAll+reliable 이라 발행도 reliable 이어야 붙는다. /leg_odom 구독은 best_effort(어느 쪽이든 붙음).
        self.pub_imu = self.create_publisher(Imu, a.imu_out, rel(400))
        self.pub_leg = self.create_publisher(Odometry, a.leg_out, rel(20))
        # --- IMU 시각 보정 상태 ---
        self.cal = None                         # 끝나면 dict(shift_ns, bias, ...)
        self.lid = []                           # (수신 s, 헤더 s) — 보정 동안만
        self.imu_rh = []                        # (수신 s, 헤더 s) — 보정 동안만
        self.gyr0 = []                          # 처음 bias_s 초 자이로
        self.t_imu0 = None
        self.first_imu_offset = None            # imu_correct.py 식: 첫 메시지 (수신 − 헤더)
        self.last_out_ns = 0
        self.n_imu_in = self.n_imu_out = self.n_nonmono = 0
        self.last_imu_h = None; self.imu_dts = []
        self.sub_imu = self.create_subscription(Imu, a.imu_in, self.on_imu, be(400))
        self.sub_lid = self.create_subscription(PointCloud2, a.lidar, self.on_lidar_raw, be(5), raw=True)
        self.t_lid0 = None
        # --- 다리 속력 ---
        self.prev_pose = None
        self.n_leg = 0; self.spd_win = []
        self.sport = None
        self.sub_pose = self.create_subscription(PoseStamped, a.pose, self.on_pose, be(50))
        self.csv = None
        if a.compare:
            from unitree_go.msg import SportModeState
            self.create_subscription(SportModeState, '/sportmodestate', self.on_sport, be(50))
            self.csv = open(self.out / 'leg_speed.csv', 'w')
            self.csv.write('recv_s,dt_s,speed_pose_3d,sport_hypot_vxvy,sport_norm_vxyz\n')
        self.create_timer(0.5, self.check_cal)
        self.create_timer(10.0, self.report)
        self.t_start = time.time()
        self.get_logger().info(f'IMU {a.imu_in} → {a.imu_out} (보정 {a.cal_s}s · 바이어스 {a.bias_s}s, 로봇 세워 둘 것) · '
                               f'{a.pose} → {a.leg_out} · LiDAR {a.lidar}(보정 동안만) · 기록 {self.out}')

    # ---------- IMU ----------
    def on_imu(self, m):
        now = time.time()
        h_ns = stamp_ns(m.header.stamp)
        self.n_imu_in += 1
        if self.last_imu_h is not None and len(self.imu_dts) < 20000:
            self.imu_dts.append((h_ns - self.last_imu_h) / NS)
        self.last_imu_h = h_ns
        if self.t_imu0 is None:
            self.t_imu0 = now
            self.first_imu_offset = now - h_ns / NS
        if self.cal is None:
            self.imu_rh.append((now, h_ns / NS))
            if now - self.t_imu0 <= self.a.bias_s:
                g = m.angular_velocity
                self.gyr0.append((g.x, g.y, g.z))
            return
        o_ns = h_ns + self.cal['shift_ns']
        if o_ns <= self.last_out_ns:                       # 헤더가 거꾸로 오면 1 µs 뒤로 (imu_correct.py 와 같은 처리)
            o_ns = self.last_out_ns + 1000; self.n_nonmono += 1
        self.last_out_ns = o_ns
        m.header.stamp.sec = int(o_ns // NS); m.header.stamp.nanosec = int(o_ns % NS)
        b = self.cal['bias']
        m.angular_velocity.x -= b[0]; m.angular_velocity.y -= b[1]; m.angular_velocity.z -= b[2]
        self.pub_imu.publish(m)
        self.n_imu_out += 1

    def on_lidar_raw(self, blob):
        now = time.time()
        b = bytes(blob[:12])
        e = '<' if b[1] == 1 else '>'                      # CDR 캡슐 헤더 0x0001 = little endian
        sec = struct.unpack_from(e + 'i', b, 4)[0]; nsec = struct.unpack_from(e + 'I', b, 8)[0]
        if self.t_lid0 is None:
            self.t_lid0 = now
        self.lid.append((now, sec + nsec / NS))

    def check_cal(self):
        if self.cal is not None or self.t_lid0 is None or self.t_imu0 is None:
            return
        now = time.time()
        if now - self.t_lid0 < self.a.cal_s or now - self.t_imu0 < self.a.bias_s or len(self.gyr0) < 50:
            return
        L = np.array(self.lid); I = np.array(self.imu_rh)
        sel = (L[:, 0] >= I[0, 0]) & (L[:, 0] <= I[-1, 0])
        if sel.sum() < 20:
            return
        d = L[sel, 1] - np.interp(L[sel, 0], I[:, 0], I[:, 1])
        shift_esy = float(np.median(d))
        shift = shift_esy if self.a.shift is None else float(self.a.shift)   # --shift 를 주면 그 값(재서 얻은 값 시험용), 기록엔 둘 다
        bias = np.mean(np.array(self.gyr0), axis=0)
        lid_lat = float(np.median(L[:, 0] - L[:, 1])); imu_lat = float(np.median(I[:, 0] - I[:, 1]))
        self.cal = dict(shift_ns=int(round(shift * NS)), bias=bias.tolist())
        info = dict(
            made=time.strftime('%Y-%m-%d %H:%M:%S'), method='esy make_ref_bag.py: median(LiDAR 헤더 − LiDAR 수신시각에서 보간한 IMU 헤더)',
            shift_s=shift, shift_esy_method_s=shift_esy, shift_source='esy 방식' if self.a.shift is None else '--shift 고정값', skew_std_ms=float(np.std(d) * 1000), n_lidar=int(sel.sum()), n_imu=len(I), cal_s=self.a.cal_s,
            gyro_bias=bias.tolist(), bias_s=self.a.bias_s, n_bias=len(self.gyr0),
            gyro_bias_std=np.std(np.array(self.gyr0), axis=0).tolist(),
            imu_correct_style_offset_s=self.first_imu_offset,       # imu_correct.py: 첫 메시지 (ROS 현재 − IMU 헤더)
            lidar_recv_minus_header_median_s=lid_lat, imu_recv_minus_header_median_s=imu_lat,
            imu_in=self.a.imu_in, imu_out=self.a.imu_out, resample='없음(실시간, 받은 샘플 그대로)')
        (self.out / 'imu_ref_calib.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
        self.destroy_subscription(self.sub_lid); self.sub_lid = None
        self.lid = []; self.imu_rh = []
        self.get_logger().info(f'IMU 보정 끝: shift {shift:+.6f}s ({info["shift_source"]}, esy 방식 값 {shift_esy:+.6f}s) (skew std {info["skew_std_ms"]:.1f} ms, LiDAR {int(sel.sum())}개) · '
                               f'자이로 바이어스 {np.round(bias, 5).tolist()} (std {np.round(info["gyro_bias_std"], 5).tolist()}) · '
                               f'imu_correct 식 오프셋 {self.first_imu_offset:+.6f}s · 수신−헤더 LiDAR {lid_lat:.3f}s / IMU {imu_lat:.3f}s → '
                               f'이제 {self.a.imu_out} 발행, LiDAR 구독 해제')

    # ---------- 다리 속력 ----------
    def on_sport(self, m):
        v = m.velocity
        self.sport = (math.hypot(v[0], v[1]), math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))

    def on_pose(self, m):
        now = time.time()
        p = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
        if self.prev_pose is not None:
            t0, p0 = self.prev_pose
            dt = max(now - t0, 1e-3)
            spd = math.sqrt((p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2 + (p[2] - p0[2]) ** 2) / dt
            o = Odometry()
            o.header.stamp = self.get_clock().now().to_msg()
            o.header.frame_id = 'odom'; o.child_frame_id = 'base_link'
            o.twist.twist.linear.x = spd
            self.pub_leg.publish(o)
            self.n_leg += 1; self.spd_win.append(spd)
            if self.csv is not None:
                s2, s3 = self.sport if self.sport else (float('nan'), float('nan'))
                self.csv.write(f'{now:.4f},{dt:.4f},{spd:.4f},{s2:.4f},{s3:.4f}\n')
        self.prev_pose = (now, p)

    def report(self):
        el = time.time() - self.t_start
        dts = np.array(self.imu_dts) if self.imu_dts else np.array([0.0])
        med = float(np.median(dts)) if len(dts) else 0.0
        gaps = int(np.sum(dts > 3 * med)) if med > 0 else 0
        s = np.array(self.spd_win) if self.spd_win else np.array([0.0])
        if self.cal is None:
            self.get_logger().warn(f'{el:5.0f}s · IMU 보정 대기 중(아직 {self.a.imu_out} 안 냄): LiDAR {len(self.lid)}개 · IMU {len(self.imu_rh)}개 '
                                   f'· 바이어스 표본 {len(self.gyr0)}개 — LiDAR 드라이버가 켜져 있어야 끝난다')
        self.get_logger().info(
            f'{el:5.0f}s · IMU 받음 {self.n_imu_in} 냄 {self.n_imu_out} (헤더 간격 중앙 {med * 1000:.1f} ms, 3배 넘는 빈틈 {gaps}, 역순 {self.n_nonmono}) · '
            f'leg {self.n_leg} (10초 속력 중앙 {np.median(s):.2f} 최대 {s.max():.2f} m/s)'
            + (f' · sport hypot {self.sport[0]:.2f}' if self.sport else ''))
        self.spd_win = []; self.imu_dts = []
        if self.csv is not None:
            self.csv.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imu-in', default='/utlidar/imu')
    ap.add_argument('--imu-out', default='/utlidar/imu_ref')
    ap.add_argument('--lidar', default='/lidar_points')
    ap.add_argument('--pose', default='/utlidar/robot_pose')
    ap.add_argument('--leg-out', default='/leg_odom')
    ap.add_argument('--cal-s', type=float, default=10.0, help='shift 를 구할 LiDAR 수집 시간(초). esy 권고는 60초 창, 중앙값이라 10초(약 100스캔)로 둠')
    ap.add_argument('--bias-s', type=float, default=1.5, help='자이로 바이어스를 잴 처음 시간(초) — esy 와 같음')
    ap.add_argument('--shift', type=float, default=None, help='IMU 헤더에 더할 값(초)을 고정. 안 주면 esy 방식으로 구함(기본)')
    ap.add_argument('--out', default='.')
    ap.add_argument('--compare', action='store_true', help='/sportmodestate hypot(vx,vy) 를 leg_speed.csv 에 같이 적는다')
    a = ap.parse_args()
    rclpy.init()
    n = Bridge(a)
    stop = {'v': False}

    def _stop(*_):
        stop['v'] = True
    signal.signal(signal.SIGINT, _stop)                   # 비대화형 & 실행에서도 끝나게(Foxy 는 SIGINT 무시로 시작될 수 있음)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while rclpy.ok() and not stop['v']:
            rclpy.spin_once(n, timeout_sec=0.1)
    finally:
        if n.csv is not None:
            n.csv.close()
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
