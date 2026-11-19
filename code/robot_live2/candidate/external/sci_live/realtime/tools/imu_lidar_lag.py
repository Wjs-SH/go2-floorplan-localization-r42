# -*- coding: utf-8 -*-
"""imu_lidar_lag.py — IMU 헤더 시계와 LiDAR(FAST-LIO 자세) 시계 사이 실제 시각차를 걷는 동안 잰다(2026-10-09).

왜: esy 방식 shift = median(LiDAR 헤더 − 수신 시각에서 보간한 IMU 헤더) 는 두 토픽의 지연 차이까지 먹는다.
    로봇 실측(10-09 17:38): LiDAR 수신−헤더 0.105초(헤더 = 스캔 시작, 스캔 0.1초) · IMU 0.023초 → shift −0.082초.
    두 헤더가 같은 시계라면 IMU 를 실제보다 0.08초 앞당기는 셈이다. 진짜 차이는 움직임으로만 잴 수 있다.
어떻게: FAST-LIO 몸체 = IMU 좌표계라, /Odometry 연속 자세의 회전 증분(LiDAR 정합이 고정한 값)과
    /utlidar/imu 원본 자이로를 [t_k + lag, t_{k+1} + lag] 로 적분한 값을 축별로 맞대고, 오차가 가장 작은 lag 를 찾는다.
    결과 lag = "원본 IMU 헤더에 더해야 LiDAR 시계가 되는 값"(esy shift 와 같은 뜻). 회전이 있어야 잴 수 있다(모퉁이 몇 번이면 충분).
    한계: FAST-LIO 자세 자체가 지금 쓰는 shift 로 만든 IMU 를 섞어 나온 것이라, LiDAR 정합이 약한 구간에선 그쪽으로 끌릴 수 있다.

    python tools/imu_lidar_lag.py record <저장.npz> [초=150]     # 로봇: 걷는 동안 기록(끝나면 바로 분석도 찍음)
    python tools/imu_lidar_lag.py analyze <저장.npz>             # 어디서든 다시 분석
"""
import sys
import time

import numpy as np


def record(path, sec):
    import signal
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    rclpy.init()
    n = Node('jsh_imu_lidar_lag')
    q = QoSProfile(depth=500, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    imu, odo = [], []
    n.create_subscription(Imu, '/utlidar/imu', lambda m: imu.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time(),
                          m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)), q)
    n.create_subscription(Odometry, '/Odometry', lambda m: odo.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time(),
                          m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w,
                          m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)), q)
    stop = {'v': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True)); signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    t0 = time.time()
    print(f'기록 {sec:.0f}초: /utlidar/imu 원본 + /Odometry → {path}', flush=True)
    while time.time() - t0 < sec and not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.05)
    np.savez(path, imu=np.array(imu), odom=np.array(odo))
    print(f'저장: IMU {len(imu)} · 자세 {len(odo)} ({time.time() - t0:.0f}초)', flush=True)
    n.destroy_node(); rclpy.shutdown()


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def log_SO3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def analyze(path, used_shift=None):
    d = np.load(path)
    imu, odo = d['imu'], d['odom']
    if len(imu) < 500 or len(odo) < 50:
        print(f'자료 부족: IMU {len(imu)} · 자세 {len(odo)}'); return None
    ti = imu[:, 0]; g = imu[:, 2:5]
    o = np.argsort(ti); ti, g = ti[o], g[o]
    still = int(1.5 * 160)
    g = g - g[:still].mean(0)                                  # 처음 1.5초 바이어스(정지 가정 — 기록을 정지 상태에서 시작할 것)
    cum = np.vstack([np.zeros(3), np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(ti)[:, None], axis=0)])   # 자이로 적분(사다리꼴)
    to = odo[:, 0]
    R = [quat_to_R(q) for q in odo[:, 2:6]]
    dth = np.array([log_SO3(R[k].T @ R[k + 1]) for k in range(len(R) - 1)])   # 몸체 좌표 회전 증분(LiDAR 시계)
    ok = (np.diff(to) > 0.05) & (np.diff(to) < 0.25)
    rate = np.linalg.norm(dth, axis=1) / np.maximum(np.diff(to), 1e-3)
    turn = ok & (rate > 0.15)                                  # 회전하는 구간만(0.15 rad/s ≈ 8.6°/s 이상)
    print(f'자세 {len(odo)}개 · 회전 구간 {turn.sum()}개 · 최대 회전속도 {np.degrees(rate[ok].max()):.0f}°/s · 걸은 거리 '
          f'{np.linalg.norm(np.diff(odo[:, 6:9], axis=0), axis=1).sum():.1f} m')
    if turn.sum() < 20:
        print('회전이 너무 적어 잴 수 없음(모퉁이를 몇 번 돌아야 함)'); return None
    lags = np.arange(-0.30, 0.3001, 0.002)
    err = []
    for L in lags:                                             # IMU 헤더 + L = LiDAR 시계
        a = np.array([np.interp(to[:-1] - L, ti, cum[:, k]) for k in range(3)]).T
        b = np.array([np.interp(to[1:] - L, ti, cum[:, k]) for k in range(3)]).T
        err.append(np.mean(np.sum((b - a - dth)[turn] ** 2, axis=1)))
    err = np.array(err); i = int(np.argmin(err))
    j0, j1 = max(i - 3, 0), min(i + 4, len(lags))              # 바닥 근처 2차 맞춤으로 2 ms 칸 사이 보간
    c = np.polyfit(lags[j0:j1], err[j0:j1], 2); best = -c[1] / (2 * c[0]) if c[0] > 0 else lags[i]
    rms0 = np.sqrt(err[np.argmin(np.abs(lags))]); rmsb = np.sqrt(err[i])
    print(f'가장 맞는 lag = {best:+.4f} s (IMU 헤더 + lag = LiDAR 시계) · 회전 증분 오차 RMS {np.degrees(rmsb):.3f}° (lag 0 이면 {np.degrees(rms0):.3f}°)')
    for L in (-0.082, 0.0, 0.071):
        k = int(np.argmin(np.abs(lags - L)))
        print(f'   lag {L:+.3f} s 일 때 오차 RMS {np.degrees(np.sqrt(err[k])):.3f}°')
    if used_shift is not None:
        print(f'   지금 브리지 shift {used_shift:+.4f} s 와 차이 {best - used_shift:+.4f} s')
    return dict(best_lag_s=float(best), rms_deg=float(np.degrees(rmsb)), n_turn=int(turn.sum()))


if __name__ == '__main__':
    if sys.argv[1] == 'record':
        record(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 150.0)
        analyze(sys.argv[2])
    else:
        analyze(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
