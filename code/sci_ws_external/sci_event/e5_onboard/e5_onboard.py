#!/usr/bin/env python3
"""선배 SCI E5(e5_node, 로봇에서 SigLIP·판단·FSM)에 우리 것을 얹어 띄운다(2026-10-15). SCI 파일은 하나도 고치지 않는다.

  - 오돔: FAST-LIO /Odometry 를 로봇 몸체 기준으로(odom_base.py). 환경변수 SCI_ODOM_BASE=1, SCI_ODOM_DELTA_DEG(기본 45.3).
    SCI e5_node 는 오돔을 변환 없이 써서, /Odometry 를 그대로 주면 표지 월드 좌표가 거리의 약 0.77 배 틀어지고
    멈춘 뒤 다시 겨누는 방향이 45° 어긋난다(region_world·P2 predict_pose·view_from·plan_point 모두 같은 자세를 씀).
  - 조준: 콘솔 /aim 뒤 사진(job 증가)을 SCI_AIM_TIMEOUT_S(기본 60) 초까지 기다리고(SCI 25 초), 그 사이 콘솔 로그에
    '조준 안 함/조준 불가/조준 작업 오류' 가 새로 뜨면 바로 실패로 끝낸다(우리 PC 실행기 live_event_runner 와 같은 식).
  - 나머지(후보·P2 정지 정책·FSM·상태 파일·UI·판독 작업 큐)는 SCI 그대로.

  쓰임(로봇, SCI 파이썬): start_e5_onboard.sh 가 부른다. 인자는 e5_node.py 와 같다.
"""
import os
import sys
import time
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCI_SRC = Path(os.environ.get("SCI_WS", str(Path.home() / "sci_ws"))) / "src"
sys.path.insert(0, str(SCI_SRC / "sci_visual_observation"))
sys.path.insert(0, str(HERE))

import e5_node as N                      # noqa: E402  SCI 원본. import 때 sci_sensor_bridge·sci_pantilt 경로를 넣는다
import sources as SRC                    # noqa: E402
import planner as PL                     # noqa: E402
from odom_base import fastlio_to_base_2d  # noqa: E402

ODOM_BASE = os.environ.get("SCI_ODOM_BASE", "0") == "1"
DELTA = float(os.environ.get("SCI_ODOM_DELTA_DEG", "45.3"))
AIM_TIMEOUT = float(os.environ.get("SCI_AIM_TIMEOUT_S", "60"))
BAD = ("조준 안 함", "조준 불가", "조준 작업 오류")
HOME_AFTER = os.environ.get("SCI_HOME_AFTER", "1") == "1"   # 촬영 뒤 팬틸트 정면 복귀


class BaseOdomRosSource(SRC.RosSource):
    """SCI RosSource 와 같고 오돔 자세만 몸체 기준으로 바꿔 쌓는다. 웹 실시간 사진용으로 가장 최근 컬러 한 장을 따로 잡아 둔다
    (poll() 이 frames 를 비우므로 거기서는 못 읽음)."""
    latest = None
    new_color = None                     # threading.Event — 실시간 사진 스레드를 새 컬러가 올 때 바로 깨움

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
        with self.lock:
            self.n_odom += 1
            self.odom.append((self._st(m.header), x, y, yaw))
            self.last_odom_wall = time.time()

    def health(self):
        h = super().health()
        h["odom_frame"] = f"base(δ{DELTA:g}°)" if ODOM_BASE else "raw"
        return h


class Planner(PL.PanTiltPlanner):
    """SCI 계획기와 같고, 기다리는 시간과 콘솔 거절 처리만 다르다."""

    def _state_full(self, timeout=1.5):
        try:
            return json.loads(urllib.request.urlopen(self.console + "/state", timeout=timeout).read())
        except Exception as e:
            return dict(error=f"{type(e).__name__}: {e}")

    def aim(self, plan, poll_s=0.5, timeout_s=None):
        timeout_s = AIM_TIMEOUT if timeout_s is None else timeout_s
        if not plan["reachable"]:
            return dict(mode="REFUSED", ok=False, why="target_out_of_pantilt_range", plan=plan)
        if not self.execute:
            return dict(mode="DRY_RUN", ok=False, why="PENDING_APPROVAL (SCI_PANTILT_APPROVED!=1 or execute=False)", plan=plan)
        st0 = self._state_full()
        if "job" not in st0:
            return dict(mode="EXECUTE", ok=False, why="console_unreachable", detail=st0, plan=plan)
        job0, log0 = int(st0.get("job", -1)), list(st0.get("log", []))
        t0 = time.time()
        reply = urllib.request.urlopen(plan["url"], timeout=5).read().decode()
        if "이미 작업 중" in reply:          # 10-15 밤: 전에는 앞 조준이 끝나 작업 번호가 오르면 그걸 내 결과로 받았다(흐린 남의 사진)
            return dict(mode="EXECUTE", ok=False, why="console_busy", reply=reply, sec=time.time() - t0, plan=plan)
        while time.time() - t0 < timeout_s:
            time.sleep(poll_s)
            st = self._state_full()
            if "job" not in st:
                continue
            if int(st.get("job", -1)) > job0:
                return dict(mode="EXECUTE", ok=True, reply=reply, sec=time.time() - t0,
                            console=dict(reachable=True, status=st.get("status"), job=st.get("job")), plan=plan)
            bad = [x for x in st.get("log", []) if x not in log0 and any(w in x for w in BAD)]
            if bad:
                return dict(mode="EXECUTE", ok=False, why="console_refused: " + bad[-1], reply=reply, sec=time.time() - t0, plan=plan)
        return dict(mode="EXECUTE", ok=False, why="aim_timeout", reply=reply, sec=time.time() - t0, plan=plan)


# ---------- 찾기 스위치(2026-10-15): 웹(실시간 위치 보기)이 실험 상태를 /dev/shm/sci_e5/kci_gate.json 에 쓴다 ----------
#   걷는 중(active) 에만 SigLIP·후보 만들기·P2 정지 판단을 돌리고, [시작] 전·계산 중에는 쉰다. 진행 중인 이벤트(멈춤→조준→촬영)는
#   스위치와 상관없이 끝까지 간다. 파일이 없거나 5초 넘게 안 바뀌면 쉰다. SCI_KCI_GATE=0 이면 스위치 없이 SCI 처럼 항상 돈다.
GATE = Path(os.environ.get("SCI_SHM", "/dev/shm/sci_e5")) / "kci_gate.json"
GATE_ON = os.environ.get("SCI_KCI_GATE", "1") == "1"
_gate_cache = [0.0, (False, "시작 전")]


def gate():
    if not GATE_ON:
        return True, "스위치 끔(항상 찾기)"
    now = time.time()
    if now - _gate_cache[0] < 0.5:
        return _gate_cache[1]
    try:
        age = now - GATE.stat().st_mtime
        d = json.loads(GATE.read_text())
        v = (bool(d.get("active")), str(d.get("why", ""))) if age <= 5.0 else (False, "웹 상태가 오래됨")
    except (OSError, ValueError):
        v = (False, "웹 상태 없음")
    _gate_cache[:] = [now, v]
    return v


class GatedFrontend:
    """SCI Frontend 를 감싼다. 쉬는 동안 prefetch(후보 만들기)·scene(SigLIP)을 부르지 않고 '후보 없음' 장면을 돌려준다."""

    def __init__(self, fe):
        self._fe, self._thr = fe, 0.0

    def __getattr__(self, k):
        return getattr(self._fe, k)

    def prefetch(self, t_next):
        if gate()[0]:
            return self._fe.prefetch(t_next)

    def scene(self, t, dimg, pose, allow_stop=True):
        on, why = gate()
        if on:
            r = self._fe.scene(t, dimg, pose, allow_stop)
            self._thr = r.get("thr", self._thr)
            return r
        import cv2
        import numpy as np
        fr = self._fe.buf.newest()
        bgr = fr["bgr"].copy() if fr is not None else np.zeros((480, 640, 3), np.uint8)
        cv2.putText(bgr, "SEARCH PAUSED", (8, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return dict(t=t, base=dict(bgr=bgr), regs=[], vp=None, thr=self._thr, legacy_ev=None,
                    timing=dict(paused=1.0, scene_ms=0.0), t_siglip_req=time.time(), regions=[])


SIGN_DISP = ("VERIFIED", "VERIFYING", "STOP_REQUEST", "ALREADY_OBSERVED")   # SigLIP 이 표지판으로 본 것(그 밖 후보는 안 그림)
LIVE_S = float(os.environ.get("SCI_LIVE_FRAME_S", "0.125"))                     # 웹 실시간 사진 주기(0 = 끄고 원래대로 장면마다)


class GatedNode(N.Node):
    """2026-10-15 밤: 웹 사진을 장면 처리와 떼어 냄. 원래는 장면(1초)마다 모든 후보 상자를 그려 frame.jpg 를 썼고, 이벤트 중·찾기 쉼에는
    장면을 안 돌려 사진이 멈췄다. 이제 frame.jpg = 최신 RealSense 사진(LIVE_S 마다) + 최근 장면(2초 안)에서 표지판으로 본 상자만 주황.
    last_stop.jpg = 멈춤을 낸 장면 + 표지판 상자만. 판단·기록(scenes.jsonl, events/)은 원래 그대로."""
    _live_boxes, _live_t, _live_on = [], 0.0, False

    def loop(self, src):
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

    def draw(self, bgr, res, dec, t, act):
        import cv2
        boxes = self._sign_boxes(res, dec, act)
        self._live_boxes, self._live_t = boxes, time.time()
        if act:
            im = self._paint(bgr.copy(), boxes)
            N.atomic_write(N.SHM / "last_stop.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        if not self._live_on:                                                    # 실시간 스레드를 끈 경우만 장면마다
            im = self._paint(bgr.copy(), boxes)
            if self.a.ui_scale != 1.0:
                im = cv2.resize(im, None, fx=self.a.ui_scale, fy=self.a.ui_scale, interpolation=cv2.INTER_AREA)
            N.atomic_write(N.SHM / "frame.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())

    def _live_loop(self, src):
        import cv2
        last = None
        while src.alive and not N.STOP[0]:
            t0 = time.time()
            try:
                with src.lock:
                    f = src.latest
                if f is not None and f[1] != last:
                    last = f[1]
                    im = f[2].copy()
                    if time.time() - self._live_t < 2.0:                         # 장면은 1초마다라 상자는 최대 약 1~2초 전 위치
                        self._paint(im, self._live_boxes)
                    if self.a.ui_scale != 1.0:
                        im = cv2.resize(im, None, fx=self.a.ui_scale, fy=self.a.ui_scale, interpolation=cv2.INTER_AREA)
                    N.atomic_write(N.SHM / "frame.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
            except Exception as e:                                               # 사진 쓰기 실패가 판단을 멈추면 안 됨
                print(f"live frame: {type(e).__name__}: {e}", flush=True)
                time.sleep(1.0)
            if getattr(src, "new_color", None) is not None:                     # 새 컬러가 오면 바로(최소 LIVE_S 간격, 기다림 최대 0.5초)
                time.sleep(max(0.0, LIVE_S - (time.time() - t0)))
                src.new_color.wait(0.5)
                src.new_color.clear()
            else:
                time.sleep(max(0.02, LIVE_S - (time.time() - t0)))

    def _aim_capture_with_plan(self, f, plan, *a, **k):
        try:
            return super()._aim_capture_with_plan(f, plan, *a, **k)
        finally:
            if getattr(self.planner, "execute", False) and HOME_AFTER:
                import threading
                threading.Thread(target=self._go_home, args=(getattr(f, "id", None),), daemon=True).start()

    def _go_home(self, ev):
        st = dict(ev=ev, state="RETURNING", t=time.time())
        self.status["kci_home"] = st
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                r = urllib.request.urlopen(self.planner.console + "/home", timeout=3).read().decode()
            except Exception as e:
                r = f"{type(e).__name__}"
            if "정면" in r:
                self.status["kci_home"] = dict(st, state="DONE", sec=round(time.time() - t0, 1), t=time.time())
                return
            time.sleep(1.0)                                   # 콘솔이 아직 초점 찾는 중이면 끝날 때까지
        self.status["kci_home"] = dict(st, state="FAILED", why=r, t=time.time())

    def _start_frontend(self):
        super()._start_frontend()
        if self.frontend is not None and not isinstance(self.frontend, GatedFrontend):
            self.frontend = GatedFrontend(self.frontend)

    def p2_tick(self, src, stamp, rel):
        if gate()[0]:
            return super().p2_tick(src, stamp, rel)

    def publish_status(self, src, rec, force=False):
        on, why = gate()
        self.status["kci_gate"] = dict(active=on, why=why)
        return super().publish_status(src, rec, force)


def _source(a):
    if a.source == "replay":
        return SRC.ReplaySource(a.dataset, realtime=a.realtime, t_start=a.t_start, t_end=a.t_end)
    return BaseOdomRosSource(odom_topic=a.odom_topic)


N.PanTiltPlanner = Planner              # Node.__init__ 이 이 이름으로 만든다
N.ReplaySource_or_Ros = _source
N.Node = GatedNode                      # main() 이 이 이름으로 만든다

if __name__ == "__main__":
    print(f"e5_onboard: 오돔 {'몸체 기준(δ %g°)' % DELTA if ODOM_BASE else '원본 그대로'} · 조준 대기 {AIM_TIMEOUT:g}s · "
          f"찾기 스위치 {'웹 실험 상태(걷는 중만)' if GATE_ON else '없음(항상)'} · SCI {N.__file__}", flush=True)
    N.main()
