# -*- coding: utf-8 -*-
"""live_preflight.py — 드라이버·FAST-LIO 를 띄운 직후, 실시간 노드를 켜기 전에 확인할 것.

    # 1) 로봇을 세워 둔 채로: 토픽·필드·주기·LiDAR 장착 TF
    python live_preflight.py static [--seconds 5]
    # 2) 직진 몇 m → 제자리 '좌회전' 한 바퀴쯤(반시계). 그동안 FAST-LIO 를 녹화하고 규약을 판정
    python live_preflight.py motion --seconds 40 [--save /dev/shm/flio_rec.npz]
    python live_preflight.py analyze /dev/shm/flio_rec.npz          # 녹화 파일로 다시 판정

규약(esy a131_live README): FAST-LIO /Odometry → odom→base_link 는 레버암 → y 거울 → 몸체 z 축 δ(기본 45.3°) 회전.
이 값은 노트북 FAST-LIO 설정에서 잰 것이라 로봇 FAST-LIO 에서 다시 확인해야 한다:
  - δ: fastlio_to_base(δ=0) 자세로 estimate_delta_deg 가 45° 근처인지(직진 구간 이동 방향의 몸체 기준 각)
  - 거울상: fastlio_to_base 자세의 yaw 가 좌회전(반시계)에서 커지는지
어긋나면 노드 파라미터 delta_deg / mirror 로 준다(코드에 부호 뒤집기를 넣지 않는다).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'a131_live'))

PF = {1: 'int8', 2: 'uint8', 3: 'int16', 4: 'uint16', 5: 'int32', 6: 'uint32', 7: 'float32', 8: 'float64'}
WANT = {'x': 7, 'y': 7, 'z': 7, 'intensity': 7, 'ring': 4, 'timestamp': 8}


def _ros(seconds, lidar_topic, odom_topic, want_tf):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    rclpy.init()
    node = Node('floorplan_preflight')
    got = dict(cloud=[], odom=[], tf=None)
    qs = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

    def on_cloud(m):
        if len(got['cloud']) < 3:                     # 처음 몇 개만 자세히 본다(나머지는 수만)
            from rt_live_node import cloud_arrays
            try:
                a = cloud_arrays(m)
                t = a['timestamp'][np.isfinite(a['timestamp'])]
                info = dict(n=m.width * m.height, step=m.point_step, frame=m.header.frame_id,
                            fields={f.name: (f.offset, f.datatype) for f in m.fields},
                            t_span=float(t.max() - t.min()) if len(t) else float('nan'),
                            t_end_minus_hdr=float(t.max() - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)) if len(t) else float('nan'),
                            ring=(int(a['ring'].min()), int(a['ring'].max())), finite=float(np.isfinite(a['x']).mean()))
            except Exception as e:
                info = dict(error=f'{type(e).__name__}: {e}', fields={f.name: (f.offset, f.datatype) for f in m.fields})
            got['cloud'].append(info)
        got.setdefault('cloud_t', []).append(time.time())

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        got['odom'].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                            m.header.frame_id, m.child_frame_id, time.time()))

    # 점군 구독은 발행자 QoS 에 맞춘다(노드와 같은 이유: 1.6 MB 를 best-effort 로 받으면 유실이 잦다)
    t_wait = time.time() + 3.0
    while not node.get_publishers_info_by_topic(lidar_topic) and time.time() < t_wait:
        rclpy.spin_once(node, timeout_sec=0.1)
    pubs0 = node.get_publishers_info_by_topic(lidar_topic)
    if pubs0 and all(p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in pubs0):
        qs = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    node.create_subscription(PointCloud2, lidar_topic, on_cloud, qs)
    node.create_subscription(Odometry, odom_topic, on_odom, QoSProfile(depth=50))
    buf = None
    if want_tf:
        try:
            import tf2_ros
            buf = tf2_ros.Buffer()
            tf2_ros.TransformListener(buf, node)
        except Exception as e:
            print(f'tf2_ros 없음: {e}')
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:                         # Ctrl-C / kill -INT: 녹화를 여기서 끝내고 판정
        print('녹화 중단 — 여기까지로 판정', flush=True)
    if buf is not None:
        try:
            tr = buf.lookup_transform('base_link', 'hesai_lidar', rclpy.time.Time())
            got['tf'] = tr.transform
        except Exception as e:
            got['tf'] = f'{type(e).__name__}: {e}'
    pubs = {t: [(p.node_name, str(p.qos_profile.reliability).split('.')[-1]) for p in node.get_publishers_info_by_topic(t)]
            for t in (lidar_topic, odom_topic)}
    node.destroy_node()
    rclpy.shutdown()
    return got, pubs


def static(a):
    got, pubs = _ros(a.seconds, a.lidar_topic, a.odom_topic, True)
    ok = True
    print(f'== 발행자: {pubs}')
    n_c = len(got.get('cloud_t', []))
    print(f'== 점군 {a.lidar_topic}: {a.seconds:.0f}초 동안 {n_c}개 ({n_c / a.seconds:.1f} Hz, 기대 10 Hz)')
    if not n_c:
        print('   !! 점군이 안 온다 — 드라이버·토픽 이름·DDS 환경(source ~/unitree_ros2/setup.sh) 확인'); ok = False
    for c in got['cloud'][:1]:
        print(f'   {c}')
        if 'error' in c:
            ok = False
        else:
            bad = [k for k, dt in WANT.items() if c['fields'].get(k, (None, None))[1] != dt]
            if bad:
                print(f'   !! 필드 형식이 녹화본과 다르다: {bad} (기대 {WANT})'); ok = False
            if not (0.05 < c['t_span'] < 0.15):
                print(f'   !! 점 시각 폭 {c["t_span"]:.3f}s — 한 바퀴 0.1초가 아니다(timestamp 가 절대초 float64 인지)'); ok = False
    od = got['odom']
    print(f'== FAST-LIO {a.odom_topic}: {len(od)}개 ({len(od) / a.seconds:.1f} Hz, 기대 10 Hz)')
    if not od:
        print('   !! FAST-LIO 가 안 온다 — IMU(/utlidar/imu_corrected)·점군 입력 확인'); ok = False
    else:
        print(f'   frame {od[-1][8]} → {od[-1][9]} · 헤더-수신 지연 {np.median([o[10] - o[0] for o in od]):+.3f}s '
              f'(점군과 같은 시계여야 함 — 노드는 스캔 끝 뒤 자세를 기다린다)')
        if got['cloud'] and 'error' not in got['cloud'][0]:
            print(f'   점군 마지막 점 - 점군 헤더 {got["cloud"][0]["t_end_minus_hdr"]:+.3f}s')
    tf = got['tf']
    if tf is None or isinstance(tf, str):
        print(f'== /tf_static base_link←hesai_lidar: 없음({tf}) — 노드 기본값(URDF 0.145,0,0.11 yaw 90°)을 쓴다')
    else:
        from scipy.spatial.transform import Rotation as Rot
        R = Rot.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
        t = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        from rt_live_node import STATIC_BASE_LIDAR
        M = np.asarray(STATIC_BASE_LIDAR).reshape(4, 4)
        dR, dt = float(np.abs(R - M[:3, :3]).max()), float(np.abs(t - M[:3, 3]).max())
        print(f'== /tf_static base_link←hesai_lidar: t={np.round(t, 3)} · 노드 기본값과 회전 차 {dR:.4f} · 위치 차 {dt:.4f} m')
        if dR > 1e-3 or dt > 5e-3:
            print('   !! 노드 기본값과 다르다 — static_base_lidar 파라미터로 실물 값을 줄 것'); ok = False
    print('\n정지 점검:', '통과' if ok else '문제 있음(위 !! 참고)')


def analyze(rec, delta_default=45.3):
    from a131_live import fastlio_to_base, estimate_delta_deg
    from scipy.spatial.transform import Rotation as Rot
    t, p, q = rec['t'], rec['p'], rec['q']
    print(f'== 녹화 {t[-1] - t[0]:.1f}초 · 자세 {len(t)}개 · 이동 거리 {np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)):.1f} m')
    raw_yaw = np.unwrap(Rot.from_quat(q).as_euler('zyx')[:, 0])
    # 회전 구간 목록(초당 20° 넘게 1초 이상) — 자리 잡느라 돈 회전과 판정용 좌회전을 구분하려고 전부 보여 준다
    pb1, qb1 = fastlio_to_base(p, q, delta_deg=0.0, mirror=True)
    yb = np.unwrap(Rot.from_quat(qb1).as_euler('zyx')[:, 0])
    rate = np.gradient(yb, t)
    on = np.abs(rate) > np.radians(20)
    segs, j = [], 0
    while j < len(t):
        if on[j]:
            k2 = j
            while k2 + 1 < len(t) and on[k2 + 1]:
                k2 += 1
            if t[k2] - t[j] >= 1.0:
                segs.append((t[j] - t[0], t[k2] - t[0], float(np.degrees(yb[k2] - yb[j]))))
            j = k2 + 1
        else:
            j += 1
    for a0, a1, dy in segs:
        print(f'   회전 구간 {a0:5.1f}~{a1:5.1f}초: 거울상 켬 기준 yaw {dy:+.0f}°')
    # 가장 많이 돈 3초 구간 = 제자리 회전 구간
    k = max(1, int(round(3.0 / max(np.median(np.diff(t)), 1e-3))))
    i = int(np.argmax(np.abs(raw_yaw[k:] - raw_yaw[:-k]))) if len(t) > k else 0
    res = {}
    for mirror in (True, False):
        pb, qb = fastlio_to_base(p, q, delta_deg=0.0, mirror=mirror)
        yaw = np.unwrap(Rot.from_quat(qb).as_euler('zyx')[:, 0])
        d, nwin, frac = estimate_delta_deg(t, pb[:, 0], pb[:, 1], yaw)
        turn = float(np.degrees(yaw[i + k] - yaw[i])) if len(t) > k else float('nan')
        res[mirror] = (d, nwin, frac, turn)
        print(f'   거울상 {"켬" if mirror else "끔"}: δ {d:6.1f}° (직진 창 {nwin}개, ±20° 안 {100 * frac:.0f}%) · '
              f'가장 많이 돈 3초 구간 yaw 변화 {turn:+.0f}°')
    d, nwin, frac, turn = res[True]
    print('\n판정(조작자가 그 구간에 좌회전=반시계로 돌렸다는 전제):')
    if nwin < 10:
        print(f'   δ: 직진 창 {nwin}개 — 부족. 0.25 m/s 넘게 몇 m 곧게 걸어 다시 녹화'); return
    if abs(turn) < 30:
        print(f'   거울상: 회전 {turn:+.0f}° — 부족. 제자리에서 크게(90° 넘게) 좌회전해 다시 녹화'); return
    mir_ok = turn > 0
    print(f'   거울상(mirror=true): {"맞음 — 좌회전에서 yaw 가 커진다" if mir_ok else "틀림 — 좌회전에서 yaw 가 작아진다 → mirror:=false 로 다시 판정"}')
    use = True if mir_ok else False
    d_use = res[use][0]
    print(f'   δ: {d_use:.1f}° (기본값 {delta_default}°, 녹화본 범위 44.7~45.7°) → '
          f'{"기본값 그대로" if abs(d_use - delta_default) < 3 else f"노드에 -p delta_deg:={d_use:.1f}"}'
          f'{"" if use else " -p mirror:=false"}')


def motion(a):
    print(f'{a.seconds:.0f}초 녹화 시작 — 직진 몇 m(0.25 m/s 넘게) → 제자리 좌회전(반시계) 90° 넘게', flush=True)
    got, _ = _ros(a.seconds, a.lidar_topic, a.odom_topic, False)
    od = got['odom']
    if len(od) < 20:
        print(f'FAST-LIO 자세가 {len(od)}개뿐 — 녹화 실패'); return
    rec = dict(t=np.array([o[0] for o in od]), p=np.array([o[1:4] for o in od]), q=np.array([o[4:8] for o in od]))
    if a.save:
        np.savez(a.save, **rec)
        print(f'저장 {a.save}')
    analyze(rec)


def main():
    import signal
    # 백그라운드(&)로 띄우면 SIGINT 가 '무시' 로 시작해 kill -INT 가 안 먹는다(2026-10-08 로봇에서 녹화가 저장 안 됨) → 직접 받는다
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('static', 'motion', 'analyze'))
    ap.add_argument('rec', nargs='?')
    ap.add_argument('--seconds', type=float, default=None)
    ap.add_argument('--save', default=None)
    ap.add_argument('--lidar-topic', default='/lidar_points')
    ap.add_argument('--odom-topic', default='/Odometry')
    a = ap.parse_args()
    if a.mode == 'static':
        a.seconds = a.seconds or 5.0
        static(a)
    elif a.mode == 'motion':
        a.seconds = a.seconds or 40.0
        motion(a)
    else:
        with np.load(a.rec) as f:
            analyze({k: f[k] for k in f.files})


if __name__ == '__main__':
    main()
