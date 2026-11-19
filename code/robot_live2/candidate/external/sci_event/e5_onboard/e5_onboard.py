#!/usr/bin/env python3
"""선배 SCI E5(e5_node, 로봇에서 SigLIP·판단·FSM)에 우리 것을 얹어 띄운다(2026-10-15). SCI 파일은 하나도 고치지 않는다.

  - 오돔: FAST-LIO /Odometry 를 로봇 몸체 기준으로(odom_base.py). 환경변수 SCI_ODOM_BASE=1, SCI_ODOM_DELTA_DEG(기본 45.3).
    SCI e5_node 는 오돔을 변환 없이 써서, /Odometry 를 그대로 주면 표지 월드 좌표가 거리의 약 0.77 배 틀어지고
    멈춘 뒤 다시 겨누는 방향이 45° 어긋난다(region_world·P2 predict_pose·view_from·plan_point 모두 같은 자세를 씀).
  - 조준(R1): r1-aim-1 요청 ID 로만 결과를 받는다(job 번호 증가 방식 제거). SCI_AIM_TIMEOUT_S 는 논리 deadline.
  - KCI 10-16: P2 반응 시간 2.5 s(SCI_P2_REACTION_S), FAST-LIO 속도 1 s 기울기(SCI_ODOM_SPEED_WIN_S) — 비우거나 0 이면 SCI 그대로.
  - 나머지(후보·P2 정지 정책·FSM·상태 파일·UI·판독 작업 큐)는 SCI 그대로.

  쓰임(로봇, SCI 파이썬): start_e5_onboard.sh 가 부른다. 인자는 e5_node.py 와 같다.
"""
import os
import sys
import time
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCI_SRC = Path(os.environ.get("SCI_WS", str(Path.home() / "sci_ws"))) / "src"
sys.path.insert(0, str(SCI_SRC / "sci_visual_observation"))
sys.path.insert(0, str(HERE))

import e5_node as N                      # noqa: E402  SCI 원본. import 때 sci_sensor_bridge·sci_pantilt 경로를 넣는다
import sources as SRC                    # noqa: E402
import planner as PL                     # noqa: E402
from odom_base import fastlio_to_base_2d  # noqa: E402
from e5rt import gates as RG              # noqa: E402  E5-RT (2026-10-17 candidate): e5_node 의 sys.path 에 sci_visual_observation 이 있음
from e5rt import home_policy as HP        # noqa: E402
from e5rt import live_render as LR        # noqa: E402
from e5rt import status_writer as SW      # noqa: E402
from e5rt.tracker import LKTracker        # noqa: E402
from e5rt import E5RT_VERSION             # noqa: E402
from e5rt import observation as OBS       # noqa: E402  E5RT2: one hand-off for planner + reader


# 2026-10-17 (KCI): R1 e5_node.atomic_write 는 임시 파일 이름이 고정(status.json.tmp)인데, R1 에서 이벤트 루프가 따로 된
#   스레드라 메인 루프(장면)와 이벤트 스레드가 동시에 status.json 을 쓰면 한쪽 os.replace 가 FileNotFoundError.
#   이벤트 스레드에서 나면 EVENT_LOOP_ERROR 로 넘어가지만, 메인 루프에서 나면 E5 가 통째로 죽는다(10-17 19:40:36 현장).
#   SCI 원본 파일은 그대로 두고 여기서 '쓰는 쪽마다 다른 임시 이름'으로 바꿔 끼운다(같은 동작, 경합만 없앰).
# E5-RT: 학생 방식(쓰는 쪽마다 고유 임시 이름)을 그대로 쓰되 실패 시 임시 파일을 지운다(SW.atomic_write_unique).
#   남은 문제(느린 스레드가 더 새 status.json 을 옛 내용으로 덮어씀)는 GatedNode.publish_status 의 잠금 + _rev 로 막는다.
def _atomic_write_unique(path, data):
    SW.atomic_write_unique(path, data)


N.atomic_write = _atomic_write_unique

# 2026-10-17 (KCI 현장): R1 촬영 품질 = 사진 '전체' 라플라시안 분산 >= 40. 문·벽·어두운 복도가 대부분인 사진은 표지판이 선명해도
#   26.8(가운데 43)로 '흐림' → 재촬영마다 초점을 처음부터 다시 돌려 렌즈가 틀어짐(60→12→6). 20 이면 26.8 통과, 진짜 흐린 6.3 은 걸림.
# E5-RT: the student value 20 is an EXPERIMENT OVERRIDE (full-image metric vs console centre metric not comparable). It is no
#   longer a hidden code default: set it explicitly (config/e5rt_profiles/kci_20261017_field.env). Unset/empty = R1 value 40.
QUALITY_SHARP_MIN = os.environ.get("SCI_QUALITY_SHARP_MIN", "")
if QUALITY_SHARP_MIN.strip() and hasattr(N, "QUALITY_SHARP_MIN"):
    N.QUALITY_SHARP_MIN = float(QUALITY_SHARP_MIN)


def _console_home(ev_id):   # E5-RT: kept for reference only, NOT called (HOME goes through the device owner)
    """KCI 10-17: 이벤트 끝에 R1 이 HOME 을 안 보냈을 때(웹 실험이 꺼져 arm 이 없으면 HOME 도 막힘) 콘솔에 정면 복귀를 직접 요청."""
    import urllib.request
    url = os.environ.get("SCI_PANTILT_URL", "http://127.0.0.1:8088").rstrip("/") + "/home"
    try:
        r = json.loads(urllib.request.urlopen(url, timeout=5).read())
        print(f"KCI: 이벤트 #{ev_id} 뒤 정면 복귀 요청 → {r.get('status')}", flush=True)
    except Exception as e:
        print(f"KCI: 이벤트 #{ev_id} 뒤 정면 복귀 요청 실패 {type(e).__name__}: {e}", flush=True)

ODOM_BASE = os.environ.get("SCI_ODOM_BASE", "0") == "1"
DELTA = float(os.environ.get("SCI_ODOM_DELTA_DEG", "45.3"))
AIM_TIMEOUT = float(os.environ.get("SCI_AIM_TIMEOUT_S", "60"))   # e5_node 가 논리 deadline 으로 읽는다
# 2026-10-16 (KCI, R1 위에 얹음): P2 반응 시간. SCI 1.5 초는 설정 파일에도 '측정 안 한 가정값'. 10-15 현장 실측 신호→실제 정지
# 중앙 약 2.3 초(9번 중 7번 0.6~2.9 초)라 1.5 초로 예측한 자리보다 더 가서 서고, 표지판이 옆·뒤로 빠졌다. 비우면 SCI 값 그대로.
P2_REACTION_S = os.environ.get("SCI_P2_REACTION_S", "2.5")
# 2026-10-17 (KCI 현장): P2 판독 조건(SCI 가정값, 로봇에서 안 잰 값). 비스듬함 55°(결정 땐 여유 5° 빼서 50°)면 벽 표지판(옆 1 m)은
#   앞 1.2 m 안에서만 통과 → RealSense 시야(좌우 27.8°) 밖이라 로봇이 그 전에 서면 신호가 영영 안 남(10-17 19:5x, 2~3 m 앞에서 61~69°로 대기만).
#   75°(결정 70°)면 옆 1 m 표지판은 앞 2.7 m(화면 안 20°)부터 통과. 거리 4 → 5 m(글자 크기 예상은 5 m 에서도 기준 15 px 의 9배).
#   70° 이상 비스듬한 글자를 OCR 이 읽는지는 아직 모름 — 이 시험으로 잰다. 비우면 SCI 값 그대로.
P2_OBLIQUITY_MAX = os.environ.get("SCI_P2_OBLIQUITY_MAX_DEG", "75")
P2_RANGE_MAX = os.environ.get("SCI_P2_RANGE_MAX_M", "5.0")
# 2026-10-16 (KCI): FAST-LIO 오돔 속도(R1 stop_policy 의 보조 근거·P2 예측·장면). SCI 는 0.5 초 앞뒤 두 자세 차이라, 서 있는 Go2 의
# 흔들림·FAST-LIO 잡음에도 0.05 m/s·3°/s 를 넘어 '멈춤 확인'이 1.8~14.5 초 늦었다(10-15). 문턱값은 그대로, 속도만 SPEED_WIN 초
# 동안의 기울기(최소제곱)로 잰다. R1 의 주 근거(몸체 속도 sportmodestate)가 연결되면 그쪽이 먼저 쓰인다. 0 이면 SCI 식 그대로.
SPEED_WIN = float(os.environ.get("SCI_ODOM_SPEED_WIN_S", "1.0"))
_sci_odom_speed = N.odom_speed


def robust_odom_speed(hist, t, win=0.5):
    """(선속도 m/s, yaw 속도 °/s) — SPEED_WIN 초 창의 x·y·yaw 를 시간에 대한 직선으로 맞춘 기울기. 점이 모자라면 SCI 식."""
    import math
    import numpy as np
    w = max(win, SPEED_WIN)
    h = [p for p in hist if t - w - 1e-6 <= p[0] <= t + 1e-6]
    if len(h) < 4 or h[-1][0] - h[0][0] < 0.3:
        return _sci_odom_speed(hist, t, win)
    a = np.asarray(h, float)
    tt = a[:, 0] - a[:, 0].mean()
    den = float(tt @ tt)
    vx, vy = float(tt @ (a[:, 1] - a[:, 1].mean())) / den, float(tt @ (a[:, 2] - a[:, 2].mean())) / den
    yw = np.unwrap(a[:, 3])
    vw = float(tt @ (yw - yw.mean())) / den
    return math.hypot(vx, vy), math.degrees(abs(vw))


if SPEED_WIN > 0:
    N.odom_speed = robust_odom_speed      # e5_node 가 import 한 이름(장면·P2·멈춤 확인 보조 근거)을 바꾼다


class BaseOdomRosSource(SRC.RosSource):
    """SCI RosSource 와 같고 오돔 자세만 몸체 기준으로 바꿔 쌓는다. 웹 실시간 사진용으로 가장 최근 컬러 한 장을 따로 잡아 둔다
    (poll() 이 frames 를 비우므로 거기서는 못 읽음)."""
    latest = None
    new_color = None                     # threading.Event — 실시간 사진 스레드를 새 컬러가 올 때 바로 깨움

    # 2026-10-17 (KCI 현장): SCI 는 odom_hist 를 장면 루프의 poll() 때만 채운다(1~2 초마다). R1 의 멈춤 확인은 이벤트 스레드가
    #   '지금 시각 기준 최근 1 초' 오돔으로 하므로, 묵은 목록이면 속도 = 없음 → 0.7 초 연속 정지가 계속 끊겨 20 초 뒤 실패
    #   (10-17 20:01·20:1x 두 번, 로봇은 속도 0.00 으로 서 있었음). 읽을 때마다 받은 오돔 그대로(최신)를 준다. poll() 의 대입은 무시.
    @property
    def odom_hist(self):
        lk = getattr(self, "lock", None)
        if lk is None:
            return []
        with lk:
            return list(self.odom)[-200:]

    @odom_hist.setter
    def odom_hist(self, value):
        pass

    def _on_color(self, m):
        super()._on_color(m)
        with self.lock:
            if self.frames:
                self.latest = self.frames[-1]
        if self.new_color is not None:
            self.new_color.set()

    def _on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        x, y, yaw = p.x, p.y, SRC.yaw_from_q(q.x, q.y, q.z, q.w)
        if ODOM_BASE:
            x, y, yaw = fastlio_to_base_2d(x, y, yaw, DELTA)
        st = self._st(m.header)
        ok, why = odom_sample_ok(st, x, y, yaw, self._odom_last_st, time.time())
        with self.lock:
            if not ok:                                   # E5-RT: NaN/Inf/future/backwards samples never reach the policy
                self.odom_rejected[why] = self.odom_rejected.get(why, 0) + 1
                if why == "time_backwards":              # odom reset / replay restart -> new session, old history dropped
                    self.odom_session += 1
                    self.odom.clear()
                    self._odom_last_st = None
                return
            self.n_odom += 1
            self.odom.append((st, x, y, yaw))
            self._odom_last_st = st
            self.last_odom_wall = time.time()

    _odom_last_st = None
    odom_session = 0

    @property
    def odom_rejected(self):
        d = self.__dict__.setdefault("_odom_rejected", {})
        return d

    def health(self):
        h = super().health()
        h["odom_frame"] = f"base(δ{DELTA:g}°)" if ODOM_BASE else "raw"
        h["odom_session"] = self.odom_session
        h["odom_rejected"] = dict(self.odom_rejected)
        return h


ODOM_FUTURE_TOL_S = float(os.environ.get("SCI_ODOM_FUTURE_TOL_S", "5.0"))   # lidar-stamped odom vs host clock skew margin


def odom_sample_ok(stamp, x, y, yaw, last_stamp, wall_now, future_tol_s=None, back_tol_s=0.5):
    """E5-RT V07/V10: reject non-finite pose/stamp and stamps in the future (ROS time on this host ~ wall clock);
    a stamp more than back_tol_s older than the last accepted one is an odometry reset/restart."""
    import math
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (stamp, x, y, yaw)):
        return False, "non_finite"
    future_tol_s = ODOM_FUTURE_TOL_S if future_tol_s is None else future_tol_s
    if stamp > wall_now + future_tol_s:
        return False, "future_stamp"
    if last_stamp is not None and stamp < last_stamp - back_tol_s:
        return False, "time_backwards"
    if last_stamp is not None and stamp <= last_stamp:
        return False, "duplicate_or_reordered"
    return True, None


class Planner(PL.PanTiltPlanner):
    """R1: SCI 계획기 그대로(r1-aim-1 요청 ID 로만 결과를 받음). 예전 '콘솔 job 번호 증가 = 성공' 방식은 지웠다
    (10-15 21:21 앞 이벤트 천장 사진을 다음 이벤트가 받은 race). 구형 콘솔은 PROTOCOL_MISMATCH 로 거부, 폴백 없음.
    조준 대기(SCI_AIM_TIMEOUT_S)는 e5_node 가 논리 deadline 으로 쓰고, 넘으면 /cancel 후 기기 idle 확인 전까지 RELEASE 안 함."""


# ---------- 찾기 스위치(R1): 웹(실시간 위치 보기)이 /dev/shm/sci_e5/kci_gate.json 에 실험 상태를 쓴다(heartbeat).
#   observe(SigLIP·후보·P2 계산 허용)와 arm(새 STOP·새 물리 동작 허용)을 나눈다.
#   - walking 모드(기본): heartbeat 신선 + active(걷는 중) 일 때만 observe·arm.
#   - static_bench 모드(SCI_E5_OBSERVE_MODE=static_bench): 정지 상태 관측. observe 는 항상, arm 은 heartbeat 신선 + armed 일 때만.
#   - always(SCI_KCI_GATE=0): 예전처럼 항상 observe; arm 은 여전히 heartbeat 필요(SCI_ARM_WITHOUT_UI=1 일 때만 예외, 시험용).
#   heartbeat(파일 mtime)가 5 초 넘게 끊기면 arm=False: 결과 첨부·deadline·오류 기록은 계속, 새 STOP·조준·HOME 은 안 함.
#   마지막 walking=true 를 붙잡아 두지 않는다(매번 다시 읽음, 캐시 0.5 초).
GATE = Path(os.environ.get("SCI_SHM", "/dev/shm/sci_e5")) / "kci_gate.json"
GATE_ON = os.environ.get("SCI_KCI_GATE", "1") == "1"
OBSERVE_MODE = os.environ.get("SCI_E5_OBSERVE_MODE", "walking")
HEARTBEAT_S = float(os.environ.get("SCI_GATE_HEARTBEAT_S", "5.0"))
_gate_cache = [0.0, None]


def gate_state(now=None, path=None, mode=None, gate_on=None, resource_defer=None):
    """E5-RT: orthogonal gates (e5rt.gates). Returns observe_enabled / request_enabled / actuator_armed /
    operator_link_alive / trial_logging_active / global_match_state (+ legacy keys observe, arm, heartbeat_ok).
    New web schema (trial_state): the search keeps running while the floor is computed ([멈춤] -> computing);
    only new STOP requests and physical actions need the trial to be walking. Old schema (active only): unchanged."""
    now = time.time() if now is None else now
    d, mtime = RG.read_gate_file(Path(path or GATE))
    return RG.gate_state(d, mtime, now=now, heartbeat_s=HEARTBEAT_S, mode=mode or OBSERVE_MODE,
                         gate_on=GATE_ON if gate_on is None else gate_on,
                         arm_without_ui=os.environ.get("SCI_ARM_WITHOUT_UI") == "1", resource_defer=resource_defer)


def gate():
    """(observe, why) for the search pipeline; cached 0.5 s, never latched."""
    now = time.time()
    if _gate_cache[1] is None or now - _gate_cache[0] >= 0.5:
        _gate_cache[:] = [now, gate_state(now)]
    g = _gate_cache[1]
    return g["observe"], g["why"]


def gate_full():
    gate()
    return _gate_cache[1]


class GatedFrontend:
    """SCI Frontend 를 감싼다. 쉬는 동안 prefetch(후보 만들기)·scene(SigLIP)을 부르지 않고 '후보 없음' 장면을 돌려준다."""

    def __init__(self, fe):
        self._fe, self._thr = fe, 0.0

    def __getattr__(self, k):
        return getattr(self._fe, k)

    def prefetch(self, t_next):
        if gate()[0]:
            return self._fe.prefetch(t_next)

    def scene(self, t, dimg, pose, allow_stop=True, buf=None):
        on, why = gate()
        if on:
            r = self._fe.scene(t, dimg, pose, allow_stop) if buf is None else self._fe.scene(t, dimg, pose, allow_stop, buf=buf)
            self._thr = r.get("thr", self._thr)
            return r
        import cv2
        import numpy as np
        fr = (buf if buf is not None else self._fe.buf).newest()
        bgr = fr["bgr"].copy() if fr is not None else np.zeros((480, 640, 3), np.uint8)
        cv2.putText(bgr, "SEARCH PAUSED", (8, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return dict(t=t, base=dict(bgr=bgr), regs=[], vp=None, thr=self._thr, legacy_ev=None,
                    timing=dict(paused=1.0, scene_ms=0.0), t_siglip_req=time.time(), regions=[])


SIGN_DISP = ("VERIFIED", "VERIFYING", "STOP_REQUEST", "ALREADY_OBSERVED")   # SigLIP 이 표지판으로 본 것(그 밖 후보는 안 그림)
LIVE_S = float(os.environ.get("SCI_LIVE_FRAME_S", "0.125"))                     # 웹 실시간 사진 주기(0 = 끄고 원래대로 장면마다)


def live_boxes_for(latest_frame_id, boxes_meta):
    """R1 §7.1: detection boxes are drawn on the live image ONLY if the live frame IS their source frame.
    Otherwise nothing is drawn on the live image (the inference panel shows the source frame + boxes + age)."""
    if not boxes_meta or boxes_meta.get("frame_id") is None or latest_frame_id is None:
        return []
    return list(boxes_meta["boxes"]) if int(latest_frame_id) == int(boxes_meta["frame_id"]) else []


E5RT_SIGLIP_CACHE = os.environ.get("SCI_E5RT_SIGLIP_CACHE", "off")      # off | cache  (B3 ablation; default off)
E5RT_TRACK_SCALE = float(os.environ.get("SCI_E5RT_TRACK_SCALE", "0.5"))    # LK tracking resolution factor (owner thread)
# E5RT2: total neural crops per SigLIP request (tiles + 1 context upper bound per region). Measured on the S-1F onboard
#   replay: max tiles+regions = 33 (E5-RT1 run: 29); 32 bound once and exposed a crash (fixed). Default 40 = never bound
#   on the measured data -> decision-preserving there; any value that binds is a policy change (evaluate separately). 0 = off.
E5RT2_MAX_CROPS = int(os.environ.get("SCI_E5RT2_MAX_CROPS", "40") or 0) or None
E5RT2_TRACK_SEM_MAX_AGE_S = float(os.environ.get("SCI_E5RT2_TRACK_SEM_MAX_AGE_S", "4.0"))   # geometry-only track lifetime
E5RT2_REQUIRE_CONFIRMED_GEOMETRY = os.environ.get("SCI_E5RT2_REQUIRE_CONFIRMED_GEOMETRY", "1") == "1"
SEED_P2_STATES = ("VERIFIED", "TRACK")          # verified + worth-verifying candidates get geometry tracks (no evidence)


class GatedNode(N.Node):
    """2026-10-15 밤 + R1: 웹 사진을 장면 처리와 떼어 냄. frame.jpg = 최신 RealSense 사진(LIVE_S 마다).
    R1: 장면 상자는 그 상자를 계산한 원본 프레임에만 그린다. 원본 프레임+상자는 frame_inference.jpg 로 따로 쓴다.
    E5-RT (candidate, 2026-10-17):
      - 실시간 사진의 상자 = 그 프레임에서 LK 로 '검증된' 추적 상자만(트랙 번호·의미 나이·추적 품질 표시). 학생 2.5 초 옛 상자는
        KCI_LIVE_BOXES=legacy 로만 재현(비교용).
      - HOME = 이 이벤트가 쓴 장치의 정리 동작. 같은 장치 소유자(device worker)로만 보내고, 장치 상태 UNKNOWN·비상정지·다른 작업
        소유 중에는 보내지 않는다(home_policy). 학생의 HTTP /home 직접 호출 스레드는 쓰지 않는다.
      - 찾기(observe)와 새 정지 요청(request)·물리 동작(arm)을 나눈다(e5rt.gates). 층 계산 중에도 찾기는 계속.
      - status.json 은 한 번에 한 스레드만 만들고 쓰며(_status_lock) _rev 가 단조 증가한다(옛 내용 역전 없음)."""
    _live_meta, _live_on = None, False

    _home_ctx, _seed_slot = None, None

    def _rt_init(self):
        """E5-RT runtime pieces (idempotent; also works for objects built with __new__ in tests)."""
        if "rt_tracker" in self.__dict__:
            return
        import threading
        self._status_lock = threading.Lock()
        self._rev = SW.RevisionedWriter()
        self.home_log = []
        self.rt_tracker = LKTracker(track_scale=E5RT_TRACK_SCALE)
        self._seed_lock = threading.Lock()             # E5RT2: guards rt_tracker / sem_by_track / _trk_latest
        self._trk_event = threading.Event()
        self._trk_latest = None                        # (frame meta, bgr) of the newest frame the owner tracked
        self.sem_by_track = {}
        self.track_stats = dict(frames=0, t0=None, last_fid=None, seeded=0, seed_rejected=0, caught_up_frames=0,
                                lost_in_catch_up=0, expired_no_semantic=0, update_mono=deque(maxlen=5000))

    def __init__(self, a, deps=None, **kk):
        self._rt_init()
        super().__init__(a, deps=deps, **kk)
        ov = {}
        if P2_REACTION_S.strip():
            old = self.p2cfg["timing"]["reaction_s"]
            self.p2cfg["timing"]["reaction_s"] = float(P2_REACTION_S)      # self.p2.c 와 같은 사전
            ov["p2.timing.reaction_s"] = dict(sci=old, kci=float(P2_REACTION_S), why="10-15 실측 신호→정지 중앙 약 2.3 s",
                                              status="EXPERIMENT_OVERRIDE_UNVALIDATED")
        for key, val, why in (("obliquity_max_deg", P2_OBLIQUITY_MAX, "55°(결정 50°)는 벽 표지판이 RealSense 화면 밖일 때만 통과(10-17 현장)"),
                              ("range_max_m", P2_RANGE_MAX, "벽 표지판 통과 구간을 복도 가운데(옆 1.5 m)에서도 화면 안에 두려고")):
            if val.strip():
                old = self.p2cfg["readability"][key]
                self.p2cfg["readability"][key] = float(val)                   # self.p2.c 와 같은 사전
                ov[f"p2.readability.{key}"] = dict(sci=old, kci=float(val), why=why, status="EXPERIMENT_OVERRIDE_UNVALIDATED",
                                                   note="OCR readability at this obliquity/range not measured")
        if SPEED_WIN > 0:
            ov["odom_speed"] = dict(sci="0.5 s 양 끝 차이", kci=f"{SPEED_WIN:g} s 기울기(최소제곱)", why="서 있을 때 흔들림으로 멈춤 확인 지연",
                                    status="KEPT_STUDENT_FIX")
        if QUALITY_SHARP_MIN.strip() and hasattr(N, "QUALITY_SHARP_MIN"):
            ov["capture_quality.sharp_min"] = dict(sci=40.0, kci=float(QUALITY_SHARP_MIN), metric="laplacian_var_cv32f_full_gray(full image)",
                                                   status="EXPERIMENT_OVERRIDE_UNVALIDATED",
                                                   note="console uses a CENTRE-ROI metric (30); full-image 20 vs centre 30 are not comparable")
        ov["live_boxes"] = dict(mode=os.environ.get("KCI_LIVE_BOXES", "tracked"), status="E5RT_TRACKED_OVERLAY",
                                note="legacy = student 2.5 s old boxes (comparison only)")
        ov["home_after_event"] = dict(policy="e5rt.home_policy", status="E5RT_REPLACES_STUDENT_ALWAYS_HOME")
        ov["siglip_cache"] = dict(mode=E5RT_SIGLIP_CACHE, status="E5RT_OPT_IN")
        ov["e5rt_version"] = E5RT_VERSION
        if os.environ.get("SCI_SEM1_WOULD_STOP", "0") == "1" or OBSERVE_MODE == "shadow":
            from e5rt.would_stop import WouldStopEvaluator          # SEM1 private counterfactual evaluator (shadow)
            self.would_stop = WouldStopEvaluator(path=str(self.run / "would_stop_private.jsonl"))
            ov["would_stop_evaluator"] = dict(status="PRIVATE_COUNTERFACTUAL", file="would_stop_private.jsonl")
        (self.run / "kci_overrides.json").write_text(json.dumps(ov, ensure_ascii=False, indent=1))
        print("e5_onboard 바꾼 값(KCI/E5-RT): " + json.dumps(ov, ensure_ascii=False), flush=True)

    # ------------------------------------------------------------------ HOME (cleanup of this event's own action)
    def _finish_observation(self, f, cap_status, reason, cap_rec=None):
        """R1 body unchanged; the R1 HOME condition calls can_start_physical() -> answered by home_policy here."""
        self._home_ctx = f
        try:
            super()._finish_observation(f, cap_status, reason, cap_rec=cap_rec)
        finally:
            self._home_ctx = None

    def home_allowed(self, f):
        self._rt_init()
        g = gate_full()
        other = bool(getattr(self.device, "busy", False) and getattr(self.device, "busy_key", None) not in (None, getattr(f, "key", None)))
        estop, abort = self.safety_inputs(g)
        ok, why = HP.home_decision(device_used=getattr(f, "device_used", False), device_idle_confirmed=getattr(f, "device_idle_confirmed", False),
                                   home_after=getattr(N, "HOME_AFTER", True), estop=estop, operator_abort=abort,
                                   owner_busy_other=other, armed=g["actuator_armed"],
                                   cleanup_without_arm=os.environ.get("SCI_E5RT_HOME_WITHOUT_ARM", "0") == "1")
        rec = dict(t=time.time(), event_key=getattr(f, "key", None), ok=ok, why=why, estop=estop, operator_abort=abort,
                   estop_signal="GATE_FILE" if g.get("estop_state") is not None else "NOT_AVAILABLE_IN_E5 (UNKNOWN -> no auto HOME)")
        self.home_log = (self.home_log + [rec])[-50:]
        try:
            self.store.set_fields(f.key, home_decision=rec)
        except Exception:
            pass
        return ok

    def safety_inputs(self, g=None):
        """E5RT2: tri-state E-stop / operator abort. A latched True from any source wins; otherwise the gate file value;
        absent everywhere = None (UNKNOWN). UNKNOWN is never filled with False."""
        g = g or gate_full()
        vals = []
        for latched, gv in ((getattr(self, "estop_latched", None), g.get("estop_state")),
                            (getattr(self, "operator_abort_latched", None), g.get("operator_abort_state"))):
            vals.append(True if (latched is True or gv is True) else (False if (gv is False and latched in (None, False)) else None))
        return vals[0], vals[1]

    def can_start_physical(self):
        if self._home_ctx is not None:
            return self.home_allowed(self._home_ctx)
        g = gate_full()
        estop, abort = self.safety_inputs(g)
        return bool(g["actuator_armed"]) and estop is not True and abort is not True

    def event_busy(self):
        """E5-RT: when new STOP requests are not allowed (trial not walking / shadow), the policies must not 'fire' a
        track (it would be marked observed without an observation) -> treat as busy."""
        return super().event_busy() or not gate_full()["request_enabled"]

    # ------------------------------------------------------------------ frontend (+ optional SigLIP cache)
    def _start_frontend(self):
        super()._start_frontend()
        if self.frontend is not None and not isinstance(self.frontend, GatedFrontend):
            fe = self.frontend
            if E5RT_SIGLIP_CACHE != "off" and getattr(fe, "vlm", None) is not None:
                from e5rt.rt_frontend import RtSiglipCache
                wrap = RtSiglipCache(fe.vlm, fe.buf, legacy_events=fe.L, mode=E5RT_SIGLIP_CACHE)
                fe.vlm = fe.L.vlm = wrap
            if getattr(fe, "vlm", None) is not None:          # E5RT2: real crop budget (outermost; typed DEFERRED_BUDGET)
                from e5rt.budget import BudgetedSiglip
                fe.vlm = fe.L.vlm = BudgetedSiglip(fe.vlm, legacy_events=fe.L, max_crops=E5RT2_MAX_CROPS,
                                                   policy=os.environ.get("SCI_SEM1_BUDGET_POLICY", "fair"))   # SEM1: value = opt-in
            self.frontend = GatedFrontend(fe)

    def p2_tick(self, src, stamp, rel):
        if gate()[0]:
            n0 = len(getattr(self, "tick_log", []))
            r = super().p2_tick(src, stamp, rel)
            ws = getattr(self, "would_stop", None)
            if ws is not None and len(getattr(self, "tick_log", [])) > n0:     # a real tick happened
                h = src.health() if hasattr(src, "health") else {}
                ws.on_tick(rel, self.p2_summary, context=dict(visual_only=h.get("visual_only"), pose_mode=h.get("pose_mode")))
            return r

    def loop(self, src):
        self._src_ref = src
        if LIVE_S > 0 and isinstance(src, BaseOdomRosSource):                 # 재생 소스는 원래대로(장면마다)
            import threading
            self._live_on = True
            src.new_color = threading.Event()
            threading.Thread(target=self._live_loop, args=(src,), daemon=True).start()
        return super().loop(src)

    def _sign_boxes(self, res, dec, act):
        out = []
        for r, d in zip(res["regions"], dec):
            if d["display"] in SIGN_DISP:
                out.append(([int(round(v)) for v in r["box"]], f"sign {r['sim']:.2f}" + (" seen" if d["display"] == "ALREADY_OBSERVED" else ""),
                            1 if d["display"] == "ALREADY_OBSERVED" else 2))
        if act and act.get("box"):
            out.append(([int(round(v)) for v in act["box"]], f"STOP {act.get('z', 0):.1f}m", 4))
        return out

    @staticmethod
    def _paint(im, boxes):
        import cv2
        for (x0, y0, x1, y1), lab, th in boxes:
            c = (0, 90, 255) if th >= 4 else (0, 150, 255)                       # 주황(BGR)
            cv2.rectangle(im, (x0, y0), (x1, y1), c, th)
            cv2.putText(im, lab, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        return im

    def draw(self, bgr, res, dec, t, act, frame=None):
        self._rt_init()
        import cv2
        boxes = self._sign_boxes(res, dec, act)
        fid = (frame or {}).get("frame_id")
        now = time.monotonic()
        self._live_meta = dict(frame_id=fid, boxes=boxes, t_drawn=time.time(), t_drawn_mono=now, header_stamp=(frame or {}).get("header_stamp"))
        # E5RT2: tracker seeding moved to after_scene() (owner thread, with bounded catch-up); draw() only renders.
        im = self._paint(bgr.copy(), boxes)                                     # source frame + its own boxes
        cv2.putText(im, f"INFERENCE frame {fid}", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        N.atomic_write(N.SHM / "frame_inference.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        if not self._live_on:                                                    # 실시간 스레드를 끈 경우만 장면마다
            if self.a.ui_scale != 1.0:
                im = cv2.resize(im, None, fx=self.a.ui_scale, fy=self.a.ui_scale, interpolation=cv2.INTER_AREA)
            N.atomic_write(N.SHM / "frame.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())

    # ------------------------------------------------------------------ E5RT2 live tracker (owner thread, geometry only)
    def track_frame(self, seq, st, img):
        """Every camera frame, on the owner thread (same thread as the P2 tick): LK update of the live tracks. Tracks
        without a fresh semantic refresh for E5RT2_TRACK_SEM_MAX_AGE_S are retired (geometry is not semantics)."""
        self._rt_init()
        import cv2
        fe = getattr(self, "frontend", None)
        it = (getattr(getattr(fe, "buf", None), "items", None) or [None])[-1]
        g = it["gray"] if (it is not None and "gray" in it and int(it["id"]) == int(seq)) else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        with self._seed_lock:
            try:
                self.rt_tracker.update(g, int(seq), float(st))
            except Exception as e:                                    # a tracker failure never stops the owner
                self.fault("TRACKER_ERROR", repr(e))
            for k, tr in self.rt_tracker.tracks.items():
                sm = self.sem_by_track.get(k)
                if tr["status"] == "LIVE" and sm is not None and now - sm["t_mono"] > E5RT2_TRACK_SEM_MAX_AGE_S:
                    tr["status"], tr["why"] = "TRACK_LOST", "expired_no_semantic_refresh"
                    self.track_stats["expired_no_semantic"] += 1
            self.rt_tracker.prune(keep_lost=16)
            ts = self.track_stats
            ts["frames"] += 1; ts["t0"] = ts["t0"] or now; ts["last_fid"] = int(seq)
            ts["update_mono"].append(now)
            self._trk_latest = (dict(frame_id=int(seq), header_stamp=float(st)), img)
        self._trk_event.set()

    def after_scene(self, res, decisions, p2recs, act, fmeta, dimg):
        """Seed geometry tracks from THIS scene's source frame (verified signs + candidates worth verifying), then catch
        up through the buffered frames source < f <= the newest frame the owner already tracked (never beyond it, never
        back). Existing tracks of other keys keep running; re-seeded keys are replaced."""
        self._rt_init()
        fid = fmeta.get("frame_id")
        dec = decisions.get(self.active_stage) or [{}] * len(res["regions"])
        now = time.monotonic()
        cands = []
        for i, (r, d, pr) in enumerate(zip(res["regions"], dec, p2recs or [None] * len(res["regions"]))):
            stop_here = bool(act and act.get("box") == r.get("box"))
            p2s = (pr or {}).get("state")
            if not (stop_here or d.get("display") in SIGN_DISP or p2s in SEED_P2_STATES):
                continue
            key = int(r["spot"]) if r.get("spot") is not None else -(int(fid or 0) * 100 + i + 1)
            pri = 0 if stop_here else (1 if (p2s == "VERIFIED" or d.get("display") in SIGN_DISP) else 2)
            cands.append((pri, -float(r["sim"]), key, r, d, p2s, stop_here))
        cands.sort(key=lambda c: c[:2])
        cands = cands[:self.rt_tracker.max_tracks]
        if fid is None or not cands:
            return
        items = list(getattr(getattr(self.frontend, "buf", None), "items", []) or [])
        src_i = next((j for j, it in enumerate(items) if int(it["id"]) == int(fid)), None)
        with self._seed_lock:
            if src_i is None or "gray" not in items[src_i]:
                self.track_stats["seed_rejected"] += len(cands)
                self.status["e5rt2_seed_last"] = dict(source_frame=fid, why="SOURCE_FRAME_EVICTED_NOT_SUBSTITUTED", n=len(cands))
                return
            latest = self.rt_tracker.prev[1] if self.rt_tracker.prev is not None else int(fid)
            tmp = LKTracker(track_scale=self.rt_tracker.s, max_tracks=self.rt_tracker.max_tracks)
            base = items[src_i]
            for _, _, key, r, d, p2s, stop_here in cands:
                tmp.seed(key, list(r.get("tile") or r["box"]), base["gray"], int(base["id"]), base["th"])
            n_cu = 0
            for it in items[src_i + 1:]:
                if int(it["id"]) > int(latest):                       # bounded: never past the newest tracked frame
                    break
                tmp.update(it["gray"], int(it["id"]), it["th"])
                n_cu += 1
            self.track_stats["caught_up_frames"] += n_cu
            for _, _, key, r, d, p2s, stop_here in cands:
                tr = tmp.tracks.get(key)
                if tr is None:
                    continue
                if tr["status"] == "LIVE" and tr["frame_id"] == int(latest):
                    self.rt_tracker.tracks[key] = tr
                else:
                    self.track_stats["lost_in_catch_up"] += 1
                    tr = dict(tr, status="TRACK_LOST", why=tr.get("why") or "lost_during_catch_up")
                    self.rt_tracker.tracks[key] = tr
                self.sem_by_track[key] = dict(display="STOP_REQUEST" if stop_here else (d.get("display") or p2s),
                                              sim=float(r["sim"]), t_mono=now, source_frame_id=int(fid),
                                              p2_state=p2s, score_status=r.get("score_status", "FRESH"),
                                              snapshot_id=fmeta.get("snapshot_id"))
            if self.rt_tracker.prev is None:
                self.rt_tracker.prev = tmp.prev
            self.track_stats["seeded"] += len(cands)
            self.status["e5rt2_seed_last"] = dict(source_frame=int(fid), latest_frame=int(latest), n=len(cands), caught_up=n_cu)

    def render_live(self, f):
        """f = (stamp, seq, bgr) of a frame the owner TRACKED. Boxes verified on this frame only (live_render)."""
        self._rt_init()
        import cv2
        im = f[2].copy()
        mode = os.environ.get("KCI_LIVE_BOXES", "tracked")
        if mode in ("1", "legacy"):                                              # student overlay (comparison only)
            meta = self._live_meta
            boxes = live_boxes_for(f[1], meta)
            if boxes:
                self._paint(im, boxes)
            elif meta and meta.get("boxes"):
                age = time.time() - meta["t_drawn"]
                if age <= 2.5:
                    self._paint(im, meta["boxes"])
                    cv2.putText(im, f"LEGACY boxes from {age:.1f}s ago (frame {meta['frame_id']})", (8, 470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1)
            return im
        same = live_boxes_for(f[1], self._live_meta)                       # this live frame IS the scene's source frame
        if same:
            self._paint(im, same)
            return im
        with self._seed_lock:
            live = [dict(t) for t in self.rt_tracker.live_on(f[1])]
            sem = dict(self.sem_by_track)
        boxes, note = LR.live_overlay(f[1], live, sem, time.monotonic(), mode=mode)
        LR.paint(im, boxes)
        cv2.putText(im, f"LIVE frame {f[1]} | {len(boxes)} tracked (geometry only)", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return im

    def _live_loop(self, src):
        """Renders the newest frame the OWNER tracked (so its boxes are verified on that exact frame). Never tracks."""
        self._rt_init()
        import cv2
        last = None
        while src.alive and not N.STOP[0]:
            self._trk_event.wait(0.5)
            self._trk_event.clear()
            t0 = time.time()
            try:
                with self._seed_lock:
                    cur = self._trk_latest
                if cur is None:                                                  # owner has not tracked yet: newest raw
                    with src.lock:                                               # frame, drawn with same-frame boxes only
                        f = getattr(src, "latest", None)
                    cur = None if f is None else (dict(frame_id=int(f[1]), header_stamp=float(f[0])), f[2])
                if cur is not None and cur[0]["frame_id"] != last:
                    last = cur[0]["frame_id"]
                    im = self.render_live((cur[0]["header_stamp"], cur[0]["frame_id"], cur[1]))
                    if self.a.ui_scale != 1.0:
                        im = cv2.resize(im, None, fx=self.a.ui_scale, fy=self.a.ui_scale, interpolation=cv2.INTER_AREA)
                    N.atomic_write(N.SHM / "frame.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
            except Exception as e:                                               # 사진 쓰기 실패가 판단을 멈추면 안 됨
                print(f"live frame: {type(e).__name__}: {e}", flush=True)
                time.sleep(1.0)
            time.sleep(max(0.0, LIVE_S - (time.time() - t0)))                    # UI render rate limit

    def on_generation_change(self, why):
        super().on_generation_change(why)
        vlm = getattr(self.frontend, "vlm", None) if self.frontend is not None else None
        inner = getattr(vlm, "c", vlm)                                           # BudgetedSiglip -> cache wrapper
        if inner is not None and hasattr(inner, "sched"):
            inner.sched.set_revision(odom_generation=self.scene_generation)      # cached results of the old session unusable
        with self._seed_lock:
            for tr in self.rt_tracker.tracks.values():
                tr.setdefault("meta", {})["odom_generation_changed"] = why

    # ------------------------------------------------------------------ E5RT2 observation hand-off (planner + reader)
    def start_event(self, stop, stamp, pose, bgr, dimg, latest_frame=None):
        f = super().start_event(stop, stamp, pose, bgr, dimg, latest_frame=latest_frame)
        try:
            oc = self.build_observation(f, stop)
            f.observation, f.observation_id = oc, oc.observation_id
            f.observation_stop = oc                                              # STOP-time hand-off (kept as record)
            self.store.set_fields(f.key, observation_id=oc.observation_id, observation=oc.to_dict(), observation_stop_id=oc.observation_id,
                                  payload_sha256=oc.payload_sha256, payload_revision=oc.payload_revision)
            self.fsm_log("OBSERVATION", "handoff_built", observation_id=oc.observation_id, geometry_confirmed=oc.geometry_confirmed,
                         geometry_reasons=oc.geometry_reasons, latest_frame_id=oc.latest_frame_id, source_frame_id=oc.source_frame_id)
        except Exception as e:                                                   # missing hand-off = no physical action
            f.observation, f.observation_id = None, None
            self.fault("OBSERVATION_HANDOFF_FAILED", repr(e))
            self.store.set_fields(f.key, observation_id=None, observation_null_reason=f"handoff failed: {type(e).__name__}: {e}")
        return f

    def build_observation(self, f, stop, at="stop", view=None):
        """at='stop': from the STOP proposal. at='aim': rebuilt when the aim is decided, from the CURRENT state of the same
        track (latest verified source frame + its depth, accumulated verifications, live ROI on the newest tracked frame,
        view from the current pose). The aim-time hand-off is the one used by BOTH the planner request and the reader."""
        self._rt_init()
        spot = stop.get("spot")
        if at == "aim" and stop.get("stage") == "P2" and spot in self.p2.tracks:
            t2 = self.p2.tracks[spot]
            stop = dict(stop, box=t2.get("box", stop.get("box")), tile=t2.get("tile", stop.get("tile")),
                        z=t2.get("z_last", stop.get("z")), z_tile=t2.get("z_last", stop.get("z")), sim=t2.get("sim", stop.get("sim")),
                        box_frame=t2.get("box_frame") or stop.get("box_frame"),
                        now=({k: view.get(k) for k in ("range", "bearing", "obliquity", "pan", "tilt", "reachable")} if view else stop.get("now")))
        with self._seed_lock:
            tr = self.rt_tracker.tracks.get(spot)
            tr = None if tr is None else {k: v for k, v in tr.items() if k != "pts"}
            latest = dict(self._trk_latest[0]) if self._trk_latest is not None else None
            sem_t = (self.sem_by_track.get(spot) or {}).get("t_mono")
        bf = stop.get("box_frame") or {}
        depth_rec = self.scene_depth.get(bf.get("frame_id")) if bf.get("frame_id") is not None else None
        p2tr = self.p2.tracks.get(spot) if stop.get("stage") == "P2" else None
        plane = (p2tr or {}).get("plane_last") if p2tr is not None else ({k: v for k, v in (stop.get("plane") or {}).items() if k != "normal"} or None)
        n_verify = (p2tr or {}).get("n_verify") if p2tr is not None else stop.get("nscene")
        plane_ok = bool((p2tr or {}).get("n_xy") is not None) if p2tr is not None else bool((stop.get("plane") or {}).get("ok"))
        reg = next((r for r in ((getattr(self, "last_scene", None) or {}).get("regions") or []) if r.get("spot") == spot), None)
        rel = None
        if stop.get("now"):
            rel = dict(stop["now"], source="P2.view_from(world track, latest odom pose at decision)", predicted=stop.get("predicted"))
        rec = self.store.get(f.key) or {}
        img_size = None
        if getattr(self, "last_bgr", None) is not None:
            img_size = [int(self.last_bgr.shape[1]), int(self.last_bgr.shape[0])]
        src = getattr(self, "_src_ref", None)
        return OBS.build_observation(self.run_id, f"{N.C.BOOT_ID}:{self.scene_generation}", f.key, stop, tr, latest, depth_rec,
                                     sem_extra=dict(adv=(reg or {}).get("adv"), board=(reg or {}).get("board")) if reg else None,
                                     boot_id=N.C.BOOT_ID, odom_session=getattr(src, "odom_session", None),
                                     trial_id=rec.get("trial_id"), relative_target=rel, image_size=img_size, n_verify=n_verify,
                                     plane_ok=plane_ok, calibration_version=os.environ.get("SCI_PANTILT_CALIB") or None,
                                     semantic_t_mono=sem_t, geometry_t_mono=time.monotonic(), plane=plane,
                                     prev=getattr(f, "observation", None))

    def _aim_capture_with_plan(self, f, plan, b, u, v, z, src, now_stamp, rt=None, pose2=None, view=None):
        """E5RT2: the planner request carries the event's observation id; physical aim needs a hand-off with
        CONFIRMED geometry (fail-closed). Dry-run planning still runs and is recorded."""
        try:                                                                     # aim-time hand-off (current geometry)
            oc = self.build_observation(f, f.stop, at="aim", view=view)
            f.observation, f.observation_id = oc, oc.observation_id
            self.store.set_fields(f.key, observation_id=oc.observation_id, observation=oc.to_dict(),
                                  payload_sha256=oc.payload_sha256, payload_revision=oc.payload_revision)
            self.fsm_log("OBSERVATION", "handoff_rebuilt_at_aim", payload_sha256=oc.payload_sha256, payload_revision=oc.payload_revision, observation_id=oc.observation_id, geometry_confirmed=oc.geometry_confirmed,
                         geometry_reasons=oc.geometry_reasons, source_frame_id=oc.source_frame_id, latest_frame_id=oc.latest_frame_id)
        except Exception as e:                                                   # noqa: BLE001  no hand-off = no aim
            self.fault("OBSERVATION_HANDOFF_FAILED", repr(e))
            f.observation, f.observation_id = None, None
        oc = getattr(f, "observation", None)
        plan["observation_id"] = getattr(f, "observation_id", None)
        plan["payload_sha256"] = getattr(oc, "payload_sha256", None) if oc is not None else None
        plan["payload_revision"] = getattr(oc, "payload_revision", None) if oc is not None else None
        plan["observation_geometry"] = None if oc is None else dict(confirmed=oc.geometry_confirmed, reasons=oc.geometry_reasons,
                                                                     roi_latest=oc.roi_box_latest, latest_frame_id=oc.latest_frame_id,
                                                                     source_frame_id=oc.source_frame_id)
        self.store.set_fields(f.key, aim_request=dict(observation_id=plan["observation_id"], payload_sha256=plan["payload_sha256"],
                                                      payload_revision=plan["payload_revision"], reachable=plan.get("reachable"),
                                                      pan_cmd=plan.get("pan_cmd"), tilt_cmd=plan.get("tilt_cmd"),
                                                      geometry=plan["observation_geometry"]))
        if self.planner.execute and plan.get("reachable") and E5RT2_REQUIRE_CONFIRMED_GEOMETRY and (oc is None or not oc.geometry_confirmed):
            f.plan = plan
            self._ev_to(f, "AIM", "image_stable")
            why = "observation_missing" if oc is None else "geometry_unconfirmed:" + ",".join(oc.geometry_reasons)
            f.result["aim"] = dict(mode="REFUSED", ok=False, why=why)
            self._finish_observation(f, N.C.CaptureStatus.NO_CAPTURE, "geometry_unconfirmed_no_aim")
            return
        return super()._aim_capture_with_plan(f, plan, b, u, v, z, src, now_stamp, rt=rt, pose2=pose2, view=view)

    def reader_job_extra(self, f):
        oc = getattr(f, "observation", None)
        return dict(observation_id=getattr(f, "observation_id", None), observation=None if oc is None else oc.to_dict(),
                    payload_sha256=None if oc is None else oc.payload_sha256, payload_revision=None if oc is None else oc.payload_revision,
                    observation_null_reason=None if oc is not None else "no hand-off for this event")

    # ------------------------------------------------------------------ status (single builder at a time, monotonic _rev)
    def publish_status(self, src, rec, force=False):
        self._rt_init()
        with self._status_lock:
            g = gate_full()
            self.status["kci_gate"] = dict(active=g["observe"], observe=g["observe"], arm=g["arm"], heartbeat_ok=g["heartbeat_ok"],
                                           heartbeat_age_s=g["heartbeat_age_s"], mode=g["mode"], why=g["why"])
            self.status["e5rt_gate"] = {k: g[k] for k in ("observe_enabled", "request_enabled", "actuator_armed", "operator_link_alive",
                                                          "trial_logging_active", "global_match_state", "trial_state", "reasons", "legacy_schema")}
            st = self.rt_tracker.stats
            cpu = sorted(st["cpu_ms"][-100:])
            ts = self.track_stats
            self.status["e5rt_tracker"] = dict(updates=st["updates"], lost=st["lost"], live=len([t for t in self.rt_tracker.tracks.values() if t["status"] == "LIVE"]),
                                               cpu_ms_p50=cpu[len(cpu) // 2] if cpu else None, frames=ts["frames"],
                                               seeded=ts["seeded"], seed_rejected=ts["seed_rejected"], caught_up_frames=ts["caught_up_frames"],
                                               lost_in_catch_up=ts["lost_in_catch_up"], expired_no_semantic=ts["expired_no_semantic"],
                                               owner="e5 owner thread (same as P2 tick)",
                                               note="geometry tracking of scene candidates; not semantic inference")
            vlm = getattr(self.frontend, "vlm", None) if self.frontend is not None else None
            cache = getattr(vlm, "c", None) if vlm is not None and hasattr(vlm, "max_crops") else vlm
            if cache is not None and hasattr(cache, "sched"):
                self.status["e5rt_siglip_cache"] = dict(cache.sched.counters, last=(cache.tele or [None])[-1])
            if vlm is not None and hasattr(vlm, "max_crops"):
                self.status["e5rt2_crop_budget"] = dict(vlm.counters, max_crops=vlm.max_crops, last=vlm.last_budget)
            if getattr(self, "scene_worker", None) is not None:
                self.status["e5rt2_scene_worker"] = self.scene_worker.stats()
            tl = list(getattr(self, "tick_log", []))[-50:]
            gaps = [b[0] - a[0] for a, b in zip(tl, tl[1:])]
            self.status["e5rt2_p2_tick"] = dict(n=len(getattr(self, "tick_log", [])), last_gap_max_s=max(gaps) if gaps else None,
                                                scene_mode=getattr(self, "scene_mode", None))
            self.status["home_last"] = self.home_log[-1] if self.home_log else None
            if getattr(self, "would_stop", None) is not None:
                self.status["sem1_would_stop_private"] = self.would_stop.status()
            self.status["_rev"] = self._rev.take()
            return super().publish_status(src, rec, force)


class SharedRawUnavailable(RuntimeError):
    pass


def _source(a):
    if a.source == "replay":
        return SRC.ReplaySource(a.dataset, realtime=a.realtime, t_start=a.t_start, t_end=a.t_end)
    if a.source == "ros_readonly":
        # SEM1: explicit approved read-only receiver (allowlist, graph check, BEST_EFFORT/KEEP_LAST1, no publisher).
        # Visual-only when /Odometry is not published: identity pose FLAGGED so scenes run; P2 STOP stays deferred.
        from ros_readonly import RosReadonlySource
        return RosReadonlySource(odom_topic="/Odometry", visual_identity_pose=os.environ.get("SCI_SEM1_VISUAL_ONLY_IDENTITY", "1") == "1")
    if a.source == "shared_raw":
        # SEM1: only a verified read-only shared producer (schema file + live heartbeat) is acceptable; there is none on this
        # robot. Never falls back to ROS.
        raise SharedRawUnavailable("NO_VERIFIED_SHARED_RAW_PRODUCER: no adapter/producer schema present; refusing (no ROS fallback)")
    return BaseOdomRosSource(odom_topic=a.odom_topic)


N.PanTiltPlanner = Planner              # Node.__init__ 이 이 이름으로 만든다
N.ReplaySource_or_Ros = _source
N.Node = GatedNode                      # main() 이 이 이름으로 만든다

if __name__ == "__main__":
    print(f"e5_onboard: 오돔 {'몸체 기준(δ %g°)' % DELTA if ODOM_BASE else '원본 그대로'} · 조준 대기 {AIM_TIMEOUT:g}s · "
          f"찾기 스위치 {'웹 실험 상태(걷는 중만)' if GATE_ON else '없음(항상)'} · SCI {N.__file__}", flush=True)
    N.main()
