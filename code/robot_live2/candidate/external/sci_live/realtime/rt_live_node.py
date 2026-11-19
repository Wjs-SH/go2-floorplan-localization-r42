# -*- coding: utf-8 -*-
"""rt_live_node.py — Go2 실시간 층 판별·지도 위치추정 ROS2(Foxy) 노드.

    source /opt/ros/foxy/setup.bash
    ../venv_gpu/bin/python rt_live_node.py --ros-args -p building:=sw

받는 것 : LiDAR PointCloud2(기본 /lidar_points, 필드 x·y·z·intensity·ring·timestamp) · FAST-LIO Odometry(기본 /Odometry, camera_init→imu)
내는 것 : /floorplan_loc/pose  geometry_msgs/PoseStamped (frame floorplan_map_<층>F, 그 층 base_map.yaml 미터 좌표) — FAST-LIO 가 올 때마다
         /floorplan_loc/floor std_msgs/String JSON (다수결 층 vote · 바퀴 판정 verdict · 순위 · 점수 · 창 끝 시각과 지금까지의 지연)
         /floorplan_loc/status std_msgs/String JSON (표본 수·지연·작업자 상태) — 1초마다

흐름
  1. FAST-LIO 자세를 두 규약으로 바꿔 쌓는다(a131_live.fastlio_to_base: 구조점용 / fastlio_to_odom_link: 위치추정용).
  2. sample_dt(기본 1.5초)마다 최근 스캔 3개(직전 2개 + target)로 구조점 표본을 만든다(a131_live.A131Live, 별도 스레드, 프로세스 3개).
     target 스캔 끝 뒤의 자세가 하나 와야 만든다(esy: 끝부분 보간). 원거리 점도 target 스캔에서 만든다.
  3. 표본을 LiveStore(/dev/shm) 에 기존 주행 데이터 형식으로 쓴다 → loc_worker.py 두 개(층 판별 / 위치 보정)가 그것을 읽는다.
  4. 가장 최근 정합(pose_fit.json)으로 FAST-LIO 위치를 지도 위치로 옮겨 바로 발행한다(pose_tools.map_pose_px).
알고리즘(구조점 추출·위치추정·층 판별)은 손대지 않고 그대로 부른다.
"""
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

# 수치 연산 스레드는 1개로: 묶지 않으면 노드 본체 하나가 CPU 425% 를 써서 FAST-LIO·표본 처리·웹 화면까지 밀렸다
# (10-08 로봇 실측, 부하 12.4/8코어). 구조점·정제는 프로세스로 나눠 병렬이 된다.
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'a131_live'))
sys.path.insert(0, str(ROOT / 'a131_clean_live'))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from live_store import LiveStore  # noqa: E402
import r1_trial as T  # noqa: E402  R1: 발행 직전 실험 세대 확인(다른 실험·실패 결과를 /floorplan_loc 로 내지 않음)
from pose_tools import map_pose_px, load_map_meta, px_to_m, read_json  # noqa: E402

# 2026-10-13 esy: 정제 프로세스로 가는 파이프는 스캔 스레드와 초기화([시작]/[초기화]) 스레드가 같이 쓴다.
#   잠금 없이 동시에 보내면 바이트가 섞여 정제 쪽이 UnpicklingError 로 죽었다(10-13 415, "초기화 1" 직후). 보낼 때만 잠근다.
_CLEAN_TX = threading.Lock()


def _csend(conn, msg):
    with _CLEAN_TX:
        conn.send(msg)

# PointField datatype → numpy
_PF = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
# 녹화본(a1_3_1 전달본) 기준 base_link←hesai_lidar. 실물 /tf_static 과 다르면 파라미터로 준다.
STATIC_BASE_LIDAR = [0.0, -1.0, 0.0, 0.145, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.110, 0.0, 0.0, 0.0, 1.0]


def cloud_arrays(msg, need=('x', 'y', 'z', 'intensity', 'ring', 'timestamp')):
    """PointCloud2 → 필드별 배열(점 순서·개수 그대로, NaN 포함)."""
    fields = {f.name: f for f in msg.fields}
    miss = [n for n in need if n not in fields]
    if miss:
        raise KeyError(f'PointCloud2 필드가 없다: {miss} (있는 것: {sorted(fields)})')
    dt = np.dtype({'names': list(need), 'formats': [('>' if msg.is_bigendian else '<') + _PF[fields[n].datatype] for n in need],
                   'offsets': [fields[n].offset for n in need], 'itemsize': msg.point_step})
    n = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    return {k: np.array(arr[k]) for k in need}


def _quiet_stub_modules():
    """a131_live 가 matplotlib·rosbag2_py 자리에 넣는 가짜 모듈은 어떤 속성이든 가짜 객체를 돌려줘 __file__ 까지 가짜가 된다.
    rclpy 로거가 호출 위치를 찾으며 inspect.getmodule 로 sys.modules 를 훑다가 거기서 TypeError 로 죽으므로,
    그 모듈들의 __file__ 만 None 으로 둔다(inspect 가 건너뛴다). 가짜 모듈의 동작은 그대로다."""
    import types
    from a131_live import _D
    for m in list(sys.modules.values()):
        if isinstance(m, types.ModuleType) and isinstance(getattr(m, '__file__', None), _D):
            m.__file__ = None


def _cleaner_main(conn, static, stride):
    """정제 프로세스: esy LiveCleaner 를 받은 순서대로 돌린다(이력이 순서에 의존). observe 는 답 없음, clean 은 지운 점 마스크를 돌려준다.
    로봇에서 observe 가 스캔당 44 ms·clean 이 98 ms 라(10-08 실측) 표본 루프와 한 줄로 돌리면 3초 간격도 넘친다 → 따로 돈다."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)   # fork 로 물려받은 rclpy SIGTERM 핸들러는 죽지 않게 만든다(10-08 노트북에서 종료가 멈춤)
    from a131_clean_live import LiveCleaner
    C = LiveCleaner(static_matrix=static, support_stride=stride)
    obs_ms = deque(maxlen=50)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        kind, sid, x, y, z, inten, ring, t, pose = msg
        t0 = time.perf_counter()
        if kind == 'reset':                              # 초기화: 이력을 버리고 새로
            C = LiveCleaner(static_matrix=static, support_stride=stride)
            obs_ms.clear()
            continue
        if kind == 'obs':
            C.observe(sid, x, y, z, ring, t, pose=pose)
            obs_ms.append(1000 * (time.perf_counter() - t0))
        else:
            kill = C.clean(sid, x, y, z, inten, ring, t, pose=pose, return_mask=True)[-1]
            conn.send((sid, kill, 1000 * (time.perf_counter() - t0), float(np.median(obs_ms)) if obs_ms else None, dict(C.stats)))


class RtLiveNode(Node):
    def __init__(self):
        super().__init__('floorplan_rt_localizer')
        P = self.declare_parameter
        self.building = P('building', 'sw').value
        self.lidar_topic = P('lidar_topic', '/lidar_points').value
        self.odom_topic = P('odom_topic', '/Odometry').value
        self.cloud_rel = P('cloud_reliability', 'auto').value               # auto: 발행자 설정을 보고 맞춘다 | reliable | best_effort
        self.sample_dt = float(P('sample_dt', 3.0).value)   # 3초 = 위치추정 프레임 간격(dt 3초)과 같다. 정제까지 붙으면 1.5초는 밀린다(10-08 로봇 실측)
        self.scan_period_ns = int(float(P('scan_period', 0.1).value) * 1e9)   # LiDAR 회전 주기(XT16 10 Hz)
        self.workers = int(P('a131_workers', 3).value)
        self.exact_pca = bool(P('exact_pca', True).value)
        self.clean_on = bool(P('clean', True).value)                           # esy 정제(이중반사·사람·움직이는 물체)
        self.clean_stride = int(P('clean_stride', 2).value)                    # 면 위 리턴 살리기 기준점 간격
        self.delta_deg = float(P('delta_deg', 45.3).value)
        self.mirror = bool(P('mirror', True).value)
        # 2026-10-13 esy: 웹 왼쪽(FAST-LIO 좌표 누적) 그림 전용. 위치추정은 표본마다 점을 yaw+β(센서 방향, 이 로봇 데이터로 228°)로 놓는데
        #   그림은 FAST-LIO yaw 만 써서 표본마다 로봇 중심으로 β-180-δ(=2.7°) 덜 돌아 번져 보였다(415 실험 5 cm 칸 11281→9689). 위치추정 입력과 무관.
        self.viz_dyaw = float(np.deg2rad(float(P('viz_beta_deg', 228.0).value) - 180.0 - self.delta_deg))
        self.root = Path(P('store_root', '/dev/shm/rt_live').value)
        self.start_workers = bool(P('start_workers', True).value)
        self.floor_procs = int(P('floor_procs', 3).value)                     # 층 판별을 나눠 맡을 프로세스 수
        self.floor_trigger = P('floor_trigger', '').value                     # 실험 모드: 신호 파일이 바뀔 때마다 한 바퀴(웹 [시작] 버튼)
        if self.floor_trigger:
            try:
                os.remove(self.floor_trigger)                                 # 지난 실행의 신호가 남아 있으면 바로 돌아 버린다
            except OSError:
                pass
        self.viewer_port = int(P('viewer_port', 8765).value)                  # 브라우저 실시간 보기(0 = 끔)
        self.worker_python = P('worker_python', sys.executable).value
        static = [float(v) for v in P('static_base_lidar', STATIC_BASE_LIDAR).value]
        self.static = np.asarray(static, float).reshape(4, 4)

        self.lock = threading.Lock()
        self.odom = deque(maxlen=1200)            # (stamp_ns, p_imu(3), q_imu(4)) — 10 Hz 로 2분
        self.scans = deque(maxlen=48)             # dict(id, arrays, start_ns, end_ns, header_ns) — 정제 이력에 모든 스캔이 들어가도록 넉넉히
        self.scan_id = 0
        self.last_target = -1
        # R1: 실험 증거 영구 보관(기본 끔 — 켜려면 evidence_dir 파라미터). 실시간 저장소(keep_s 900 s)와 따로.
        self.evidence_dir = P('evidence_dir', '').value
        self.store = LiveStore(self.root / 'data', reset=True, archive=self._evidence_archive())
        self.out = self.root / 'out'
        if self.out.exists():
            import shutil
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        self.stats = dict(samples=0, sample_fail=0, last_sample_sec=None, last_error=None, scans_missed=0, scans_late=0)
        self.meta_cache = {}
        self.fit = None
        self.fit_mtime = 0.0

        from a131_live import A131Live
        import signal
        # 구조점 작업 프로세스는 Ctrl-C 를 받지 않게 한다(정리는 노드가 pool.terminate 로). 핸들러를 바꾸면 rclpy 의
        # 신호 처리를 덮으므로, 만들 때만 신호 마스크로 막는다 — fork 된 자식은 막힌 마스크를 물려받는다.
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            self.extractor = A131Live(static_matrix=self.static, workers=self.workers, exact_pca=self.exact_pca)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        _quiet_stub_modules()
        self.cleaner = None                       # 정제 프로세스와 잇는 파이프(없으면 정제 안 함)
        self.cleaner_proc = None
        if self.clean_on:
            import multiprocessing as mp
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                pc, cc = mp.get_context('fork').Pipe()
                self.cleaner_proc = mp.get_context('fork').Process(target=_cleaner_main, args=(cc, self.static, self.clean_stride),
                                                                   daemon=True)
                self.cleaner_proc.start()
                self.cleaner = pc
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        self.observed_upto = -1                   # 정제 이력에 넣은 마지막 스캔 번호
        self.cleaned = {}                         # 스캔 번호 → 정제된 xyz(표본 묶음이 겹칠 때 다시 쓰기)
        # 2026-10-15 esy: 정제 이력 넘기기 스레드. 표본 루프가 구조점(약 1.9 s)을 만드는 동안 새로 온 스캔을 바로 observe 로 넘긴다.
        #   전에는 그동안 쌓인 스캔(약 27개)을 다음 차례에 몰아 넘기며 스캔마다 정제 프로세스를 기다려(약 41 ms) 한 바퀴가 3 s 를 넘었고,
        #   한 번 밀리면 표본이 계속 약 2 s 늦은 채로 굳었다(10-15 로봇: 노드 켠 직후·E5 켠 뒤, 표본은 안 빠짐).
        #   정제 프로세스로 가는 메시지 순서·내용은 그대로다: 루프와 이 스레드가 clean_order 로 한 줄로 보내고, 이 스레드는
        #   다음 표본 묶음(대상 − 2 부터) 앞까지만 observe 한다(묶음 스캔은 루프가 clean 으로 보낸다).
        self.clean_order = threading.Lock()
        self._feed = None                         # (reset_seq, 다음 묶음 대상 스캔 번호)
        self.obs_feeder = bool(P('obs_feeder', True).value)
        self.live_scan_hz = float(P('live_scan_hz', 5.0).value)   # 웹 '지금 스캔'(화면 전용). 0 이면 끔

        self.cloud_sub = None
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, QoSProfile(depth=50))
        self._subscribe_cloud()
        if self.cloud_sub is None:
            self.create_timer(1.0, self._subscribe_cloud)
        self.pub_pose = self.create_publisher(PoseStamped, '/floorplan_loc/pose', 10)
        self.pub_floor = self.create_publisher(String, '/floorplan_loc/floor', 10)
        self.pub_status = self.create_publisher(String, '/floorplan_loc/status', 10)
        self.create_timer(1.0, self.on_status)
        self.create_timer(0.2, self.check_reset)
        self.reset_seq = 0

        self.procs = []
        if self.start_workers:
            env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
            for role in ('floor', 'pose'):
                lf = open(self.out / f'{role}.stdout', 'w')
                extra = (['--procs', str(self.floor_procs)] + (['--trigger', self.floor_trigger] if self.floor_trigger else [])
                         if role == 'floor' else [])
                self.procs.append(subprocess.Popen(
                    ['nice', '-n', '5', self.worker_python, str(HERE / 'loc_worker.py'), '--role', role,
                     '--building', self.building, '--store', str(self.root / 'data'), '--out', str(self.out)] + extra,
                    stdout=lf, stderr=subprocess.STDOUT, env=env))
        if self.viewer_port:
            lf = open(self.out / 'viewer.stdout', 'w')
            self.procs.append(subprocess.Popen(
                [self.worker_python, str(HERE / 'tools' / 'live_viewer.py'), '--root', str(self.root),
                 '--building', self.building, '--port', str(self.viewer_port)]
                + (['--trigger', self.floor_trigger] if self.floor_trigger else []), stdout=lf, stderr=subprocess.STDOUT))
        self._stop = False
        self.th = threading.Thread(target=self.sample_loop, daemon=True)
        self.th.start()
        if self.cleaner is not None and self.obs_feeder:
            self.th_feed = threading.Thread(target=self._feed_loop, daemon=True)
            self.th_feed.start()
        if self.live_scan_hz > 0:
            self.th_scan = threading.Thread(target=self._live_scan_loop, daemon=True)
            self.th_scan.start()
        self._write_run_config()
        self.get_logger().info(f'건물 {self.building} · LiDAR {self.lidar_topic} · FAST-LIO {self.odom_topic} · '
                               f'표본 {self.sample_dt}s · 구조점 프로세스 {self.workers} · 저장 {self.root}')

    # ---------- 받기 ----------
    def _subscribe_cloud(self):
        """점군 구독. 1.6 MB 메시지라 best-effort 는 전송 중 유실이 잦다 — 발행자가 reliable 이면 reliable 로 받는다
        (best-effort 구독은 둘 다와 연결되지만 연결 자체가 best-effort 가 된다). auto 는 발행자가 보일 때까지 기다린다."""
        if self.cloud_sub is not None:
            return
        rel = self.cloud_rel
        if rel == 'auto':
            pubs = self.get_publishers_info_by_topic(self.lidar_topic)
            if not pubs:
                return
            rel = 'best_effort' if any(p.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT for p in pubs) else 'reliable'
        pol = ReliabilityPolicy.RELIABLE if rel == 'reliable' else ReliabilityPolicy.BEST_EFFORT
        self.cloud_sub = self.create_subscription(PointCloud2, self.lidar_topic, self.on_cloud,
                                                  QoSProfile(reliability=pol, history=HistoryPolicy.KEEP_LAST, depth=5))
        self.stats['cloud_qos'] = rel
        self.get_logger().info(f'점군 구독 {self.lidar_topic} ({rel})')

    def on_odom(self, msg):
        t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with self.lock:
            if self.odom and t <= self.odom[-1][0]:
                return
            self.odom.append((t, np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])))
        self.publish_pose(t, np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]), msg.header.stamp)

    def on_cloud(self, msg):
        try:
            a = cloud_arrays(msg)
        except Exception as e:
            self.stats['last_error'] = str(e)
            return
        t = a['timestamp'].astype(np.float64)
        ok = np.isfinite(t)
        if not ok.any():
            return
        tn = np.rint(t[ok] * 1e9).astype(np.int64)
        hdr = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        end_ns = int(tn.max())
        with self.lock:
            # 번호는 받은 순서가 아니라 스캔 끝 시각으로 매긴다: 중간 스캔이 전송 중 빠지면 번호도 건너뛰어,
            # 표본 묶음(직전 2개 + target)이 실제로 연속인 스캔만 되게 한다(녹화본의 scan_index 와 같은 뜻).
            if self.scans:
                prev = self.scans[-1]
                if end_ns <= prev['end_ns']:
                    self.stats['scans_late'] += 1
                    return
                k = max(1, int(round((end_ns - prev['end_ns']) / self.scan_period_ns)))
                self.stats['scans_missed'] += k - 1
                sid = prev['id'] + k
            else:
                sid = 0
            self.scans.append(dict(id=sid, a=a, start_ns=int(tn.min()), end_ns=end_ns, header_ns=int(hdr)))
            self.scan_id += 1

    # ---------- 표본 만들기 ----------
    def _pose_arrays(self):
        with self.lock:
            od = list(self.odom)
        st = np.array([o[0] for o in od], np.int64)
        p = np.stack([o[1] for o in od]); q = np.stack([o[2] for o in od])
        return st, p, q

    def sample_loop(self):
        """표본 루프. 어떤 예외가 나도 스레드가 죽지 않게 감싼다(10-08: 초기화 직후 예외로 스레드가 죽어 표본이 끊겼다)."""
        while not self._stop:
            try:
                self._sample_loop()
            except Exception as e:
                self.stats['loop_restarts'] = self.stats.get('loop_restarts', 0) + 1
                self.stats['last_error'] = f'loop {type(e).__name__}: {e}'
                self.get_logger().error(f'표본 루프 예외 — 다시 시작: {self.stats["last_error"]}')
                time.sleep(0.5)

    def _sample_loop(self):
        from a131_live import make_interpolator, fastlio_to_base, fastlio_to_odom_link, FarPoses, far_points
        from scipy.spatial.transform import Rotation as Rot  # noqa: F401
        # 표본은 스캔 번호 격자(N 스캔마다, 1.5초 = 15 스캔)에 맞춰 뽑는다. 위치추정은 창 안에서 3초 간격 프레임을
        # 요청하므로, 격자에 맞으면 요청 시각마다 정확히 표본이 있다(직전+1.5초 식이면 1.5~1.6초로 밀려 어긋난다).
        N = max(1, int(round(self.sample_dt * 1e9 / self.scan_period_ns)))
        id0 = None
        seq = 0
        while not self._stop:
            time.sleep(0.05)
            if seq != self.reset_seq:
                seq, id0 = self.reset_seq, None
            with self.lock:
                if len(self.scans) < 3 or len(self.odom) < 2:
                    continue
                last_pose = self.odom[-1][0]
                ready = [s for s in self.scans if s['end_ns'] < last_pose]      # 끝 뒤의 자세가 온 스캔
            if len(ready) < 3:
                continue
            ids = {s['id'] for s in ready}
            full = [s for s in ready if s['id'] > self.last_target and s['id'] - 1 in ids and s['id'] - 2 in ids]
            if id0 is None:
                tgt = full[-1] if full else None
            else:
                on = [s for s in full if (s['id'] - id0) % N == 0]
                tgt = on[-1] if on else None
                nxt = self.last_target + N - (self.last_target - id0) % N        # 다음 격자 스캔 번호
                if tgt is None and full and (ready[-1]['id'] > nxt or nxt in ids):
                    # 격자 스캔이 빠졌거나(전송 유실) 묶음이 불완전하거나 처리 지연으로 지나갔다 → 가장 최근 것으로 다시 맞춘다
                    tgt = full[-1]
                    self.stats['grid_resets'] = self.stats.get('grid_resets', 0) + 1
            # 정제 이력: 다음 표본 묶음 앞까지의 스캔을 시간 순서대로 observe(건너뛰는 스캔도 움직이는 물체 판정에 필요)
            if self.cleaner is not None:
                limit = tgt['id'] - 2 if tgt is not None else (nxt - 2 if id0 is not None else None)
                with self.clean_order:
                    todo = [s for s in ready if limit is not None and self.observed_upto < s['id'] < limit]
                    if todo:
                        st, p_imu, q_imu = self._pose_arrays()
                        pb, qb = fastlio_to_base(p_imu, q_imu, delta_deg=self.delta_deg, mirror=self.mirror)
                        t1 = time.perf_counter()
                        for s in todo:
                            self._clean_send('obs', s, st, pb, qb)
                            self.observed_upto = s['id']
                        self.stats['observe_send_ms'] = round(1000 * (time.perf_counter() - t1) / len(todo), 1)
                        self.stats['observe_send_n'] = len(todo)
            if tgt is None:
                continue
            if id0 is None or (tgt['id'] - id0) % N:
                id0 = tgt['id']
            grp = [s for s in ready if s['id'] in (tgt['id'] - 2, tgt['id'] - 1, tgt['id'])]
            t0 = time.perf_counter()
            try:
                st, p_imu, q_imu = self._pose_arrays()
                pb, qb = fastlio_to_base(p_imu, q_imu, delta_deg=self.delta_deg, mirror=self.mirror)
                M = 2_000_000_000

                def interp(lo, hi):
                    m = (st >= lo - M) & (st <= hi + M)
                    return make_interpolator(st[m], pb[m], qb[m])
                items = []
                tc = time.perf_counter()
                if self.cleaner is not None:
                    # esy 정제(스캔 단계, 녹화본 정제와 같은 자리): 지운 점 = NaN, 점 개수·순서·ring·timestamp 그대로
                    asked = []
                    with self.clean_order:
                        for s in grp:
                            if s['id'] not in self.cleaned and s['id'] > self.observed_upto:
                                if self._clean_send('clean', s, st, pb, qb):
                                    asked.append(s)
                                self.observed_upto = s['id']
                            elif s['id'] not in self.cleaned:
                                self.stats['clean_skipped'] = self.stats.get('clean_skipped', 0) + 1   # 이미 observe 로 지나간 스캔
                        self._feed = (seq, tgt['id'] + N)       # 넘기기 스레드: 다음 묶음(대상 − 2) 앞까지 observe 해도 됨
                    for s in asked:
                        if self.cleaner is None:
                            break
                        try:
                            sid, kill, ms, obs_ms, cs = self.cleaner.recv()
                        except (EOFError, OSError) as e:
                            self._cleaner_dead(e)
                            break
                        a = s['a']
                        xyz = np.column_stack([a['x'], a['y'], a['z']]).astype(np.float64)
                        xyz[kill] = np.nan
                        self.cleaned[sid] = xyz
                        self.stats['clean_scan_ms'] = round(ms, 1)
                        self.stats['observe_ms'] = None if obs_ms is None else round(obs_ms, 1)
                        self.stats['clean_removed_pct'] = round(100 * (cs.get('ghost', 0) + cs.get('person', 0) + cs.get('moving_pts', 0))
                                                                / max(cs.get('points', 0), 1), 2)
                    for k in [k for k in self.cleaned if k < grp[-1]['id'] - 6]:
                        del self.cleaned[k]
                for s in grp:
                    a = s['a']
                    xyz = self.cleaned.get(s['id'])
                    if xyz is None:
                        xyz = np.column_stack([a['x'], a['y'], a['z']]).astype(np.float64)
                    meta = dict(scan_index=s['id'], point_start_ns=s['start_ns'], point_end_ns=s['end_ns'],
                                header_stamp_ns=s['header_ns'])
                    items.append((s['id'], xyz, a['intensity'], a['ring'], a['timestamp'].astype(np.float64),
                                  interp(s['start_ns'], s['end_ns']), meta))
                if self.clean_on:
                    self.stats['clean_wait_ms'] = round(1000 * (time.perf_counter() - tc), 1)
                self.extractor.add_scans(items)
                res = self.extractor.extract([s['id'] for s in grp], interp(grp[0]['start_ns'], grp[-1]['end_ns']))
                a = tgt['a']
                txyz = self.cleaned.get(tgt['id'])                                           # 원거리 점도 정제된 target 스캔에서
                fx, fy, fz = (a['x'], a['y'], a['z']) if txyz is None else (txyz[:, 0], txyz[:, 1], txyz[:, 2])
                far = far_points(fx, fy, fz, a['ring'], a['timestamp'], FarPoses(st / 1e9, pb, qb))
                # 위치추정 규약 자세를 표본 시각으로 보간(yaw 는 펼쳐서)
                ox, oy, oyaw = fastlio_to_odom_link(p_imu, q_imu)
                ts = res['stamp_ns']
                x = float(np.interp(ts, st, ox)); y = float(np.interp(ts, st, oy))
                yaw = float(np.interp(ts, st, np.unwrap(oyaw)))
                if seq != self.reset_seq:                      # 만드는 사이 초기화됐다 — 이전 데이터라 버린다
                    self.last_target = -1
                    continue
                sid = self.store.add(ts / 1e9, x, y, yaw, res['xy'], res['role'], far)
                if sid is not None:
                    # 시각화용: 구조점을 FAST-LIO 좌표(base 규약)로 옮겨 표본 옆에 둔다(위치추정 입력과는 무관)
                    R, pos = FarPoses(st / 1e9, pb, qb).pose_at(np.array([ts / 1e9]))
                    keep = res['role'] != 4
                    xyz = np.c_[res['xy'][keep].astype(np.float64), np.zeros(int(keep.sum()))]
                    # 2026-10-13 esy: 1.3.1 점은 이미 바닥에 눌린 2차원이라 방향(yaw)만 돌린다(위치추정 입력과 같은 방식).
                    #   roll·pitch 까지 쓰면 걷는 동안 표본마다 13~69 cm 밀려 화면이 번져 보였다. 옛 줄: (R[0].apply(xyz) + pos[0])[:, :2]
                    _yw = R[0].as_euler("zyx")[0] + self.viz_dyaw; _c, _s = np.cos(_yw), np.sin(_yw); _q = xyz[:, :2]
                    wxy = np.c_[pos[0][0] + _c * _q[:, 0] - _s * _q[:, 1], pos[0][1] + _s * _q[:, 0] + _c * _q[:, 1]].astype(np.float32)
                    d = self.store.samples / sid
                    _ex = {}   # 2026-10-13 esy: 가구/벽 구분용(웹 표시 전용) — 칸 최고 높이·거리. a131_live 가 내보낼 때만
                    if res.get('height_max_m') is not None and res.get('range_m') is not None:
                        _ex = dict(hmax=np.asarray(res['height_max_m'])[keep].astype(np.float32),
                                   rng=np.asarray(res['range_m'])[keep].astype(np.float32))
                    np.savez(d / '.local.tmp.npz', pts=wxy, pose=np.array([pos[0][0], pos[0][1], _yw]), **_ex)
                    os.replace(d / '.local.tmp.npz', d / 'local.npz')
                self.stats['samples'] = self.store.n
                self.stats['last_sample_sec'] = round(time.perf_counter() - t0, 2)
                self.stats['last_sample_lag_s'] = round((last_pose - ts) / 1e9, 2)
                if sid is None:
                    self.stats['sample_fail'] += 1
            except Exception as e:
                self.stats['sample_fail'] += 1
                self.stats['last_error'] = f'{type(e).__name__}: {e}'
                self.get_logger().warn(f'표본 실패: {self.stats["last_error"]}')
            self.last_target = tgt['id']

    def _live_scan_loop(self):
        """2026-10-15 esy: 웹 왼쪽 '지금 스캔'(화면 전용, 위치추정 입력과 무관). 최신 스캔을 초당 live_scan_hz 번 몸체 기준 평면 점으로
        줄여(벽 높이·15 m 안·최대 2500점) out/live_scan.npz 에 쓴다. 표본(3초마다, 만드는 데 약 2초)만 그리면 점이 화살표보다
        2~5초 늦게 툭툭 따라왔다. 좌표는 웹 그림(local.npz)과 같다: 최신 FAST-LIO 자세(base 규약) + viz_dyaw."""
        from a131_live import fastlio_to_base
        from scipy.spatial.transform import Rotation as Rot
        rng = np.random.default_rng(0)
        last = None
        while not self._stop:
            time.sleep(1.0 / self.live_scan_hz)
            try:
                with self.lock:
                    if not self.scans or not self.odom:
                        continue
                    s, o = self.scans[-1], self.odom[-1]
                if s['id'] == last:
                    continue
                last = s['id']
                a = s['a']
                xyz = np.column_stack([a['x'], a['y'], a['z']]).astype(np.float64)
                xyz = xyz[np.isfinite(xyz).all(1)]
                b = xyz @ self.static[:3, :3].T + self.static[:3, 3]           # 라이다 → 몸체
                r = np.hypot(b[:, 0], b[:, 1])
                b = b[(r > 0.5) & (r < 15.0) & (b[:, 2] > -0.15) & (b[:, 2] < 1.6)]
                if len(b) > 2500:
                    b = b[rng.choice(len(b), 2500, replace=False)]
                pb, qb = fastlio_to_base(o[1][None], o[2][None], delta_deg=self.delta_deg, mirror=self.mirror)
                yaw = Rot.from_quat(qb[0]).as_euler('zyx')[0] + self.viz_dyaw
                c, sn = np.cos(yaw), np.sin(yaw)
                w = np.c_[pb[0, 0] + c * b[:, 0] - sn * b[:, 1], pb[0, 1] + sn * b[:, 0] + c * b[:, 1]].astype(np.float32)
                tmp = self.out / '.live_scan.tmp.npz'
                np.savez(tmp, pts=w, sid=s['id'])
                os.replace(tmp, self.out / 'live_scan.npz')
            except Exception as e:
                self.stats['live_scan_error'] = f'{type(e).__name__}: {e}'
                time.sleep(1.0)

    def _feed_loop(self):
        """2026-10-15 esy: 표본 루프가 바쁜 동안 새 스캔을 정제 이력(observe)에 바로 넘긴다. 다음 묶음 앞까지만, 스캔 번호 순서대로."""
        from a131_live import fastlio_to_base
        while not self._stop:
            time.sleep(0.05)
            try:
                f = self._feed
                if f is None or self.cleaner is None or f[0] != self.reset_seq:
                    continue
                with self.lock:
                    if len(self.odom) < 2:
                        continue
                    last_pose = self.odom[-1][0]
                    ready = [s for s in self.scans if s['end_ns'] < last_pose and s['id'] < f[1] - 2]
                if not ready:
                    continue
                with self.clean_order:
                    if self._feed != f or self.cleaner is None:
                        continue                         # 그새 루프가 다음 묶음을 정했거나 초기화됨 → 다음 바퀴에
                    todo = [s for s in ready if s['id'] > self.observed_upto]
                    if not todo:
                        continue
                    st, p_imu, q_imu = self._pose_arrays()
                    pb, qb = fastlio_to_base(p_imu, q_imu, delta_deg=self.delta_deg, mirror=self.mirror)
                    t1 = time.perf_counter()
                    for s in todo:
                        if self.cleaner is None:
                            break
                        self._clean_send('obs', s, st, pb, qb)
                        self.observed_upto = s['id']
                    self.stats['feed_send_ms'] = round(1000 * (time.perf_counter() - t1) / len(todo), 1)
                    self.stats['feed_sent'] = self.stats.get('feed_sent', 0) + len(todo)
            except Exception as e:
                self.stats['feed_error'] = f'{type(e).__name__}: {e}'
                time.sleep(0.5)

    def _clean_send(self, kind, s, st, pb, qb):
        """스캔 하나를 정제 프로세스로. 자세는 a131 과 같은 보간기로 헤더 시각에서 구해 (x, y, yaw) 로 넘긴다(LiveCleaner._pose 와 같은 식)."""
        if self.cleaner is None:
            return
        from a131_live import make_interpolator
        a = s['a']
        m = (st >= s['header_ns'] - 2_000_000_000) & (st <= s['header_ns'] + 2_000_000_000)
        try:
            T = np.asarray(make_interpolator(st[m], pb[m], qb[m]).matrix_at(int(s['header_ns'])), dtype=np.float64)
        except Exception:
            # 그 스캔 근처 자세가 없다(초기화 직후 FAST-LIO 가 아직 자세를 안 낼 때 등) → 이력에 넣지 않는다
            self.stats['clean_no_pose'] = self.stats.get('clean_no_pose', 0) + 1
            return False
        pose = (float(T[0, 3]), float(T[1, 3]), float(np.arctan2(T[1, 0], T[0, 0])))
        try:
            _csend(self.cleaner, (kind, s['id'], a['x'], a['y'], a['z'], a['intensity'], a['ring'], s['header_ns'] / 1e9, pose))
        except (OSError, BrokenPipeError) as e:
            self._cleaner_dead(e)
            return False
        return True

    def _cleaner_dead(self, e):
        self.stats['clean_dead'] = f'{type(e).__name__}: {e}'
        self.get_logger().error(f'정제 프로세스가 죽었다 — 정제 없이 계속: {e}')
        self.cleaner = None

    # ---------- 내기 ----------
    def _last_odom_t(self):
        with self.lock:
            return self.odom[-1][0] / 1e9 if self.odom else float('nan')

    def _fit(self):
        p = self.out / 'pose_fit.json'
        try:
            mt = p.stat().st_mtime
        except OSError:
            return None
        if mt != self.fit_mtime:
            f = read_json(p)
            if f:
                self.fit, self.fit_mtime = f, mt
        return self.fit

    def publish_pose(self, t_ns, p_imu, q_imu, stamp):
        """발행이 실패해도 노드는 살아 있어야 한다(2026-10-08: 정보관 지도에 base_map.yaml 이 없어 첫 발행에서 노드가 죽었다)."""
        try:
            self._publish_pose(t_ns, p_imu, q_imu, stamp)
        except Exception as e:
            self.stats['pose_error'] = f'{type(e).__name__}: {e}'

    def _publish_pose(self, t_ns, p_imu, q_imu, stamp):
        from a131_live import fastlio_to_odom_link, fastlio_to_base
        from scipy.spatial.transform import Rotation as Rot
        ox, oy, oyaw = fastlio_to_odom_link(p_imu, q_imu)
        fit = self._fit()
        live = dict(t=t_ns / 1e9, odom_link=[float(ox[0]), float(oy[0]), float(oyaw[0])], map=None)
        if fit and not T.usable_now(fit, T.read_pointer(self.out)):        # R1: 다른 실험·실패·취소 세대의 정합으로 그리지 않음
            live['fit_rejected'] = T.classify(fit, T.read_pointer(self.out))[0]
            fit = None
        if fit:
            px, py, hd = map_pose_px(fit, float(ox[0]), float(oy[0]), float(oyaw[0]))
            m = int(fit['map'])
            if m not in self.meta_cache:
                try:
                    self.meta_cache[m] = load_map_meta(os.environ['SEMLOC_MAPS_ROOT'], m)
                except FileNotFoundError:
                    self.meta_cache[m] = None            # 미터 정보(base_map.yaml) 없는 지도 → 픽셀 좌표로 발행
            meta = self.meta_cache[m]
            msg = PoseStamped()
            msg.header.stamp = stamp
            if meta:
                msg.header.frame_id = f'floorplan_map_{m}F'
                msg.pose.position.x, msg.pose.position.y = px_to_m(meta, px, py)
            else:
                msg.header.frame_id = f'floorplan_map_{m}F_px'   # 이미지 픽셀(x 오른쪽, y 아래)
                msg.pose.position.x, msg.pose.position.y = float(px), float(py)
            msg.pose.orientation.z, msg.pose.orientation.w = float(np.sin(hd / 2)), float(np.cos(hd / 2))
            ok, why = T.guarded_publish(self.out, fit, lambda: self.pub_pose.publish(msg),      # R1: 발행 직전 마지막 확인
                                        lambda s, info: self.stats.__setitem__('pose_rejected', dict(info, status=s)))
            if ok:
                live.update(map=m, px=px, py=py, heading=hd, fit_t=fit['t'], trial_id=fit.get('trial_id'))
            else:
                live['fit_rejected'] = why
        # 시각화용 지금 자세(초당 5번까지): FAST-LIO 를 a131 과 같은 base 규약으로 바꾼 것도 함께
        now = time.time()
        if now - getattr(self, '_live_w', 0.0) >= 0.2:
            self._live_w = now
            pb, qb = fastlio_to_base(p_imu, q_imu, delta_deg=self.delta_deg, mirror=self.mirror)
            live['base'] = [float(pb[0, 0]), float(pb[0, 1]), float(Rot.from_quat(qb[0]).as_euler('zyx')[0])]
            tmp = self.out / '.live_odom.json.tmp'
            tmp.write_text(json.dumps(live))
            os.replace(tmp, self.out / 'live_odom.json')

    def _evidence_archive(self):
        if not self.evidence_dir:
            return None
        from live_store import RunEvidenceArchive
        return RunEvidenceArchive(Path(self.evidence_dir) / time.strftime('run_%Y%m%d_%H%M%S'))

    def _write_run_config(self):
        """실험 기록용: 이 실행의 조건(노드 파라미터 · FAST-LIO 설정 · 위치추정 설정 · 코드 판)을 out/run_config.json 에."""
        import hashlib
        cfg = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), building=self.building, delta_deg=self.delta_deg, mirror=self.mirror,
                   sample_dt=self.sample_dt, clean=self.clean_on, clean_stride=self.clean_stride, cloud_reliability=self.cloud_rel,
                   a131_workers=self.workers, exact_pca=self.exact_pca, floor_procs=self.floor_procs,
                   static_base_lidar=self.static.ravel().tolist())
        # FAST-LIO 설정·extrinsic·구현(go2ws/esy)·환경변수·IMU 보정(sensors_go2.sh 가 남긴 것). 화면 서버가 실험 보관 때 다시 읽어 덮는다.
        import fastlio_info
        cfg.update(fastlio_info.read())
        try:
            import yaml
            c = yaml.safe_load(open(ROOT / 'floor_id' / 'config' / 'buildings.yaml'))
            cfg['localizer'] = c.get('consts')
            cfg['scene_source'] = c.get('scene_source', 'lidar')
        except Exception:
            pass
        cfg['code_md5'] = {n: hashlib.md5((HERE / n).read_bytes()).hexdigest()[:10]
                           for n in ('rt_live_node.py', 'loc_worker.py', 'live_store.py')}
        for n, d in (('a131_live', ROOT / 'a131_live' / 'a131_live.py'), ('a131_clean_live', ROOT / 'a131_clean_live' / 'a131_clean_live.py')):
            try:
                cfg['code_md5'][n] = hashlib.md5(d.read_bytes()).hexdigest()[:10]
            except OSError:
                pass
        (self.out / 'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=1))

    def check_reset(self):
        """웹 [초기화]: FAST-LIO 를 끈 뒤 RESET 파일이 생긴다 → 쌓인 스캔·자세·표본·정제 이력·구조점 캐시를 비운다.
        스캔 번호가 0 부터 다시 시작하므로 구조점 캐시(scan_id 열쇠)를 꼭 비워야 한다."""
        req = self.out / 'RESET'
        if not req.exists():
            return
        with self.lock:
            self.odom.clear()
            self.scans.clear()
            self.scan_id = 0
            self.last_target = -1
            self.observed_upto = -1
            self.cleaned = {}
            self.store = LiveStore(self.root / 'data', reset=True, archive=self._evidence_archive())
            self.reset_seq += 1
            self.stats.update(samples=0, sample_fail=0, scans_missed=0, scans_late=0, grid_resets=0, clean_skipped=0)
        try:
            self.extractor.cache.clear()
        except AttributeError:
            pass
        if self.cleaner is not None:
            with self.clean_order:
                self._feed = None
                try:
                    _csend(self.cleaner, ('reset', None, None, None, None, None, None, None, None))
                except (OSError, BrokenPipeError) as e:
                    self._cleaner_dead(e)
        for n in ('pose_fit.json', 'floor.json', 'floor_progress.json', 'live_odom.json'):
            try:
                os.remove(self.out / n)
            except OSError:
                pass
        self.fit, self.fit_mtime = None, 0.0
        os.replace(req, self.out / 'RESET_DONE')
        self.get_logger().info(f'초기화 {self.reset_seq}: 데이터를 비웠다')

    def on_status(self):
        fl = read_json(self.out / 'floor.json')
        ptr = T.read_pointer(self.out)
        fl_cls = T.classify(fl, ptr)[0]
        if fl and fl_cls not in ('CURRENT', 'LEGACY_UNTRACKED'):              # R1: 실패·다른 실험 결과는 순위로 내지 않는다
            self.pub_floor.publish(String(data=json.dumps(dict(
                status=fl_cls, error=T.describe_failure(fl) if fl_cls == 'FAILED' else None, verdict=None, ranking=[],
                origin_trial=fl.get('trial_id'), trial_id=(ptr or {}).get('trial_id'), fix_status='NOT_CONFIRMED'), ensure_ascii=False)))
            fl = None
        if fl:
            self.pub_floor.publish(String(data=json.dumps(dict(
                status='OK', trial_id=fl.get('trial_id'), fix_status='NOT_CONFIRMED', result_kind='E2_FLOOR_RANKING_TOP1_UNVERIFIED',
                vote=fl.get('vote'), vote_counts=fl.get('vote_counts'), verdict=fl['verdict'], ranking=fl['ranking'],
                t_now=fl['t_now'], age_s=round(self._last_odom_t() - fl['t_now'], 1), pass_sec=fl['pass_sec'],
                twoway={str(c['map']): round(c['twoway'], 3) for c in fl['cands']},
                wall_cost_m={str(c['map']): round(c['wall_cost_m'], 3) for c in fl['cands']}), ensure_ascii=False)))
        fit = self._fit()
        if fit and not T.usable_now(fit, ptr):
            fit = None
        st = dict(self.stats, trial=(ptr or {}).get('trial_id'), span_s=round(self.store.span(), 1), scans=self.scan_id, odom=len(self.odom),
                  fit_map=fit['map'] if fit else None, fit_age_s=round(time.time() - fit['wall_time'], 1) if fit else None,
                  workers_alive=[p.poll() is None for p in self.procs])
        self.pub_status.publish(String(data=json.dumps(st, ensure_ascii=False)))
        self._n_status = getattr(self, '_n_status', 0) + 1
        if self._n_status % 5 == 0:                       # 5초마다 파일에도(현장 기록·ROS CLI 없이 확인)
            with open(self.out / 'status.jsonl', 'a') as f:
                f.write(json.dumps(dict(st, wall=round(time.time(), 1)), ensure_ascii=False) + '\n')

    def destroy_node(self):
        self._stop = True
        self.th.join(timeout=4.0)                        # 만들던 표본은 마저 끝낸다(표본 하나 ~1.6초)
        for p in self.procs:
            p.terminate()
        if getattr(self.extractor, 'pool', None) is not None:
            self.extractor.pool.terminate()
        if self.cleaner_proc is not None:
            try:
                if self.cleaner is not None:
                    _csend(self.cleaner, None)                  # 끝내라는 신호
            except (OSError, BrokenPipeError):
                pass
            self.cleaner_proc.join(timeout=2.0)
            if self.cleaner_proc.is_alive():
                self.cleaner_proc.terminate()
                self.cleaner_proc.join(timeout=1.0)
            if self.cleaner_proc.is_alive():
                self.cleaner_proc.kill()
        super().destroy_node()


def main():
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)    # 멈춤 진단: kill -USR1 <pid> → 모든 스레드 스택을 stderr 로
    # 지도 경로를 패키지 안으로(작업자와 같게). floor_id.setup_env 가 SEMLOC_MAPS_ROOT 등을 넣는다.
    sys.path.insert(0, str(ROOT / 'floor_id'))
    import floor_id as F
    rclpy.init()
    node = RtLiveNode()
    F.setup_env(F.load_cfg(), node.building)

    # 종료 신호를 직접 받는다. 비대화형 셸에서 & 로 띄우면 SIGINT 가 '무시' 로 시작하고, Foxy rclpy 는 그걸 되살리지 않아
    # Ctrl-C/kill -INT 가 버려졌다(2026-10-07 로봇에서 확인). launch 의 SIGTERM 도 같게 처리한다.
    # 1초 상태 타이머가 있어 대기 중에도 1초 안에 파이썬 핸들러가 돈다.
    def _quit(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)
    try:
        from rclpy.executors import ExternalShutdownException       # Humble 이상(Foxy 에는 없다)
    except ImportError:
        ExternalShutdownException = KeyboardInterrupt
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
