#!/usr/bin/env python3
"""plate_site.py — 태블릿용 "명판 재촬영 실험" 사이트 (학생 스테이징, 2026-11-13). 기본 포트 8791.

무엇을 하나
  * SigLIP(E5)이 고정 카메라에서 잡은 표지 영역을 보여 주고, 표지 하나를 골라(자동 또는 화면을 눌러) 그 위치를 FAST-LIO 누적 위치에 기억한다.
  * 기억한 위치까지 얼마나 더 걸어야 하는지 알려 준다(소리 포함).
      케이스 1  머리는 진행 방향 그대로, 팬틸트만 돌려 촬영  -> 팬틸트가 닿는 가장 가까운 지점까지 안내
      케이스 2  표지 옆까지 가서 표지 쪽으로 돌고 머리를 든 뒤 촬영 -> 옆까지 전진 / 회전 / 머리 들기 순서 안내
  * 운영자가 [조준 + 촬영] 을 누르면 사용자 조준 콘솔(aim_console.py, 수정 없음)에 조준을 맡기고 사진을 받아 저장한다.
    조준 뒤에는 화살표로 조금씩 옮겨 다시 찍을 수 있다(매번 사진이 남는다).
어디서 센서를 얻나
  기존 화면(8770)에서 실험을 시작하면 도는 센서 허브·SigLIP·FAST-LIO 를 읽기만 한다.
  명판 촬영 방식(운영폴더 runtime/plate_case1.on)에서는 그 실험을 '이번 실험에서 팬틸트 카메라 사용' 을 체크하고 시작해야 팬틸트를 쓴다(운영판 E6 는 뜨지 않는다).
  스위치가 없으면 예전 규칙: 끄고 시작해야 한다(켜면 기존 자동 촬영이 팬틸트를 쥔다 — 그때는 이 사이트가 팬틸트 연결을 거부한다).
건드리지 않는 것
  운영 폴더에는 쓰지 않는다. 로봇을 움직이지 않는다. 팬틸트는 운영자가 버튼을 누를 때만, 로봇이 멈춰 있을 때만 움직인다.
     python3 plate_site.py [--port 8791] [--arm-s 21600] [--pan-lim 20 180] [--tilt-lim 100 180]
기록: <staging>/_runs/plate_site/<실험ID 또는 bench_날짜>/events.jsonl, photos/*.jpg"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aimgeo                            # noqa: E402
import e6_tracker as T                   # noqa: E402  aim_up (계산만)
G = aimgeo.G

C = dict(O=Path(os.environ.get("PLATE_SITE_O") or (Path.home() / "sci_ws/repairs/livefield2_20261018_151606")),
         S=Path(os.environ.get("PLATE_SITE_S") or (Path.home() / "sci_ws_student_staging_20261109")),
         hub=Path(os.environ.get("PLATE_SITE_HUB") or "/dev/shm/live1_hub"),
         store=Path(os.environ.get("PLATE_SITE_STORE") or "/dev/shm/live2_store"),      # 운영판 구조점 저장소(읽기만): E2·E3 가 쓰는 자세가 odom_link.csv 에 있다
         pt_port=int(os.environ.get("PLATE_SITE_PT_PORT") or 8089), fake_console=os.environ.get("PLATE_SITE_FAKE_CONSOLE") == "1")
ACTIVE = ("RUNNING", "PARTIAL", "PAUSED")
# 연결 전 온도 확인: 로봇 컴퓨터(Jetson) 내부 센서(CPU·GPU·SoC) 중 가장 높은 값. 팬틸트 카메라의 온도가 아니다.
# 운영판은 실험을 시작할 때 72 도 미만을 요구하고 80 도에서 실험을 중단한다. 실험이 도는 동안 본체는 72~76 도에 머문다
# (X20261113_103935: 시작 3분 만에 72 도, 최고 76.3 도). 시작 기준 72 를 그대로 쓰면 카메라가 끊긴 뒤 다시 연결할 수 없어서
# (그 실험에서 74·75·76 도로 3번 거부) 운영판이 실험을 중단하는 온도(80 도) 바로 아래까지 허용한다. 80 도 이상이면 운영판이
# SigLIP 을 멈추고 60초 이어지면 실험을 끝내므로 그때는 연결해도 쓸 수 없다. 보드 자체의 속도 제한은 99 도, 강제 종료는 104 도.
TEMP_CONNECT_C = 80.0
PORT_DEV = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AC00BLLM-if00-port0"
CAM_DEV = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_IMX477_HQ_Camera_UC517-video-index0"
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "REJECTED_BUSY")
STOP_MARGIN, STOP_ZONE = 0.10, (-0.10, 0.15)          # 케이스 1: 정지 지점에서 -0.10~+0.15 m 안이면 "여기서 정지". 정지 지점은 표지를 ST["ahead"] m 앞에 둔 곳
# (11-13 첫 실험: 표지 0.22 m 앞까지 가니 팬이 77도로 끝에 붙었다 -> 사용자 결정: 완전히 나란히 가지 말고 1 m 쯤 여유를 둔다. 팬틸트가 닿는 한계보다 가까워지지는 않는다)
ABREAST_FWD, FACE_DEG, RAISE_DEG = 0.45, 12.0, 8.0     # 케이스 2: 표지가 머리 카메라보다 0.45 m 이내 앞 / 표지 방향 12도 이내 / 몸 8도 이상 들림

LOCK = threading.RLock()
ST = dict(case=1, auto=True, ahead=1.0, target=None, tid=0, done=[], sel=None, phase2="A", exp=dict(state=None, id=None, consent=False), sens={}, pose=None,
          samples=[], hist=[], K=None, scene=dict(stamp=None, regions=[], thr=None), frames={}, frame_order=[], rs_seq=None, pitch_ref=None, pitch_up=None,
          guide={}, photo=None, n_photos=0, log=[], evseq=0, cand=None, auto_scene=None, e5_requests=[], e5req_seen=set(), started=time.time(), siglip=None)
GEO = None
SIM = None                               # 시뮬 모드의 재생기(sim_player.Player). 실제 실험 때는 None


def p_exp():
    return C["O"] / "runtime" / "ctl" / "experiment.json"


def p_e5():
    return C["O"] / "runtime" / "e5_shm" / "status.json"


def p_mat():
    return C["O"] / "runtime" / "fastlio" / "Log" / "mat_pre.txt"


def p_out():
    return C["S"] / "_runs" / "plate_site"


def jread(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def last_line(p, n=600):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            ls = f.read().decode("utf-8", "replace").strip().splitlines()
        return ls[-1] if ls else None
    except OSError:
        return None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose_at(samples, t):
    """FAST-LIO 위치 기록 [(t, x, y, yaw)] 에서 시각 t 의 위치(선형 보간). 기록 범위를 0.5 초 넘게 벗어나면 None."""
    if not samples:
        return None
    if t <= samples[0][0]:
        return samples[0] if samples[0][0] - t <= 0.5 else None
    if t >= samples[-1][0]:
        return samples[-1] if t - samples[-1][0] <= 0.5 else None
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    if b[0] - a[0] > 0.6:
        return None
    k = (t - a[0]) / max(1e-6, b[0] - a[0])
    return (t, a[1] + k * (b[1] - a[1]), a[2] + k * (b[2] - a[2]), a[3] + k * wrap(b[3] - a[3]))


STORE_ROW_MAX_AGE_S, STORE_HIST_S, STORE_MAX_SHIFT_M, STORE_MAX_DYAW = 6.0, 30.0, 5.0, math.radians(20.0)


def store_tail(n=12):
    """운영판 구조점 저장소(odom_link.csv)의 마지막 n 줄 [(t, x, y, yaw)] — 읽기만. 잘린 첫 줄과 쓰는 중인 마지막 줄은 버린다."""
    try:
        with open(C["store"] / "odom_link.csv", "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").split("\n")
    if lines and lines[-1] != "":
        lines = lines[:-1]                       # 줄바꿈으로 끝나지 않은 마지막 줄 = 쓰는 중
    if size > 4096:
        lines = lines[1:]                        # 파일 중간에서 읽기 시작했으면 첫 줄은 잘려 있다
    out = []
    for l in lines:
        q = l.split(",")
        try:
            out.append((float(q[1]), float(q[2]), float(q[3]), float(q[4])))
        except (IndexError, ValueError):
            continue                             # 머리줄, 빈 줄
    return out[-n:]


def store_frame(hist):
    """FAST-LIO 원래 값으로 적은 좌표를 운영판 저장소의 좌표로 옮기는 변환(촬영 순간에 한 번 구해 사진 기록에 둔다).
    운영판의 E2·E3 는 저장소(odom_link.csv)의 자세로 후보를 만들고 옮긴다. 저장소의 자세가 FAST-LIO 원래 값과 다르면(예: 표본 자세를 실시간으로
    보정하는 e2_feed_posefix — 11-13~15 실험 22건에서 끝 시점의 차이 18~138 cm) 원래 값으로 적은 표지 위치는 후보의 지도 변환에 넣었을 때
    그만큼 어긋난다. 저장소의 가장 최근 줄과 같은 시각의 원래 자세를 짝지어 그 자리의 강체 변환(회전 = 방향 차, 이동 = 위치 차)을 구한다.
    저장소가 원래 값을 그대로 쓰면 변환은 0 이다. hist = [(t, x, y, yaw)] FAST-LIO 원래 값. -> dict(ok, why, row_t, age_s, r, c, shift_m, dyaw_deg)"""
    rows = store_tail()
    if not rows:
        return dict(ok=False, why="NO_STORE")
    if not hist:
        return dict(ok=False, why="NO_ODOMETRY")
    t_now = hist[-1][0]
    for t, cx, cy, cyaw in reversed(rows):
        if t_now - t > STORE_ROW_MAX_AGE_S:
            return dict(ok=False, why="STORE_ROW_STALE", age_s=round(t_now - t, 2))
        p = pose_at(hist, t)
        if p is None:
            continue
        rx, ry, ryaw = -p[1], p[2], -p[3]         # FAST-LIO 원래 값 -> 저장소의 관례 (e2_feed: (-x, y, -yaw))
        shift, dyaw = math.hypot(cx - rx, cy - ry), wrap(cyaw - ryaw)
        if shift > STORE_MAX_SHIFT_M or abs(dyaw) > STORE_MAX_DYAW:
            return dict(ok=False, why="IMPLAUSIBLE", shift_m=round(shift, 3), dyaw_deg=round(math.degrees(dyaw), 2), age_s=round(t_now - t, 2))
        return dict(ok=True, why=None, row_t=t, age_s=round(t_now - t, 2), r=[rx, ry, ryaw], c=[cx, cy, cyaw],
                    shift_m=round(shift, 3), dyaw_deg=round(math.degrees(dyaw), 2))
    return dict(ok=False, why="NO_ODOMETRY_AT_STORE_ROW")


def to_store(x, y, sf):
    """odom_link 관례의 점 (x, y) [FAST-LIO 원래 값 기준] -> 저장소 좌표. sf = store_frame() 의 결과(ok)."""
    (rx, ry, ryaw), (cx, cy, cyaw) = sf["r"], sf["c"]
    a = cyaw - ryaw
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = x - rx, y - ry
    return cx + ca * dx - sa * dy, cy + sa * dx + ca * dy


def hub_newest():
    """(frame meta, color BGR, depth uint16 mm) of the newest hub frame, or None (같은 읽기 방식: tools/field1112/scene_rec.py)"""
    hub = C["hub"]
    try:
        fr = json.loads((hub / "index.json").read_text())["frames"][-1]
        w, h, s = int(fr["w"]), int(fr["h"]), int(fr["slot"])
        c = np.fromfile(hub / ("slot_%02d.color" % s), dtype=np.uint8)
        d = np.fromfile(hub / ("slot_%02d.depth" % s), dtype=np.uint16)
        if c.size != w * h * 3 or d.size != w * h:
            return None
        again = [f for f in json.loads((hub / "index.json").read_text())["frames"] if f["slot"] == s]
        if not again or again[-1]["seq"] != fr["seq"]:
            return None
        return fr, c.reshape(h, w, 3), d.reshape(h, w)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def session_dir():
    xid = ST["exp"].get("id") if ST["exp"].get("state") in ACTIVE else None
    d = p_out() / (xid or time.strftime("bench_%Y%m%d"))
    (d / "photos").mkdir(parents=True, exist_ok=True)
    return d


def target_brief():
    tg = ST.get("target")
    return None if tg is None else {k: tg.get(k) for k in ("id", "src", "sim", "z0", "up", "uv", "left0")}


def ev(kind, msg=None, **kw):
    """기록 한 줄(events.jsonl) + 화면 기록 줄"""
    with LOCK:
        g = ST.get("guide") or {}
        rec = dict(wall=time.time(), time=time.strftime("%H:%M:%S"), kind=kind, case=ST["case"], exp=ST["exp"].get("id"), target=target_brief(),
                   pose=ST.get("pose"), guide={k: g.get(k) for k in ("phase", "fwd", "left", "range", "bearing", "remaining", "stop_at")},
                   pt_cmd=PTC.cmd, pitch_up=ST.get("pitch_up"), stopped=(ST.get("sens") or {}).get("stopped"))
        rec.update(kw)
        rec["msg"] = msg                                      # 11-13 첫 실험: 조준 실패·연결 실패의 사유가 기록에 남지 않아 원인을 콘솔 로그에서 찾아야 했다
        if msg:
            ST["log"] = (ST["log"] + [time.strftime("%H:%M:%S ") + msg])[-10:]
        ST["evseq"] += 1
        try:
            with open(session_dir() / "events.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


def save_frame(tag):
    """지금 고정 카메라 장면(컬러 jpg + 깊이 16비트 png)을 그 실험 폴더의 frames/ 에 남긴다 -> 파일 이름(없으면 None).
    나중에 '기억할 때·조준할 때 표지가 고정 카메라 어디에 있었나'를 다시 볼 수 있게 하기 위한 것이다."""
    with LOCK:
        fr = ST["frames"].get(ST["rs_seq"]) if ST.get("rs_seq") is not None else None
        if fr is None:
            return None
        ST["n_frames"] = ST.get("n_frames", 0) + 1
        name = "%03d_%s_seq%d" % (ST["n_frames"], tag, ST["rs_seq"])
        d = session_dir() / "frames"
        jpg, depth, stamp = fr["jpg"], fr["depth"], fr["stamp"]
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / (name + "_color.jpg")).write_bytes(jpg)
        cv2.imwrite(str(d / (name + "_depth.png")), depth)
        return dict(name=name, stamp=stamp)
    except OSError:
        return None


# ------------------------------------------------------------------ 표지 위치 기억
def region_point(reg, stamp, samples, K, od=None):
    """SigLIP 영역(상자 + 거리) -> 평면 좌표의 표지 위치. 선배 E6 와 같은 식(e6_core.target_in_body, e6_tracker.aim_up)."""
    od = od if od is not None else pose_at(samples, float(stamp))
    if od is None:
        return None, "그 장면 시각의 위치 기록이 없습니다"
    fwd, left, up0 = G.target_in_body(reg["box"], float(reg["z"]), K)
    up, basis = T.aim_up(reg["box"], float(reg["z"]), K, up0)
    pose = G.raw_to_planar(od[1], od[2], od[3])
    tu, tv = G.body_to_planar(pose, fwd, left)
    return dict(uv=[float(tu), float(tv)], up=float(up), up_basis=basis, body=[float(fwd), float(left), float(up0)], pose_det=[float(v) for v in pose],
                stamp=float(stamp)), None


def remember(src, reg, stamp, od=None):
    """od = 그 장면 시각의 FAST-LIO 원래 값 (stamp, x, y, yaw). E5 정지 요청에는 이 값이 들어 있다 — 없으면 허브의 최근 3초 기록에서 찾는다."""
    with LOCK:
        if not ST["K"] or not ST["samples"]:
            return False, "위치·카메라 정보가 없습니다 (실험이 돌고 있어야 합니다)"
        pt, why = region_point(reg, stamp, ST["samples"], ST["K"], od)
        if pt is None:
            return False, why
        ST["tid"] += 1
        tg = dict(id=ST["tid"], src=src, sim=reg.get("sim"), z0=float(reg["z"]), box=[float(v) for v in reg["box"]], wall=time.time(), left0=pt["body"][1],
                  best_rng=math.hypot(pt["body"][0], pt["body"][1]), n_refine=0, refined_scene=float(stamp))
        tg.update(pt)
        tg["exp"] = ST["exp"].get("id")                       # 이 표지 위치는 이 실험의 FAST-LIO 기준에서만 뜻이 있다
        ST.update(target=tg, sel=None, phase2="A", cand=None)
        ev("remember", "표지 기억: %s · 거리 %.1f m · %s %.1f m · 높이 %.2f m" % (src, tg["z0"], "왼쪽" if tg["left0"] >= 0 else "오른쪽", abs(tg["left0"]), tg["up"]),
           region=reg, pitch_at_detection=ST.get("pitch_up"), frame=save_frame("remember"))
    return True, None


def refine_target(now):
    """기억한 표지가 SigLIP 에 다시 잡히면, 더 가까이서 본 값으로 기억 위치를 바꾼다.
    깊이 오차는 거리의 제곱에 비례해 커지므로(4.5 m 에서 본 값보다 3 m 에서 본 값이 낫다) 가장 가까이서 본 값만 남긴다.
    표지는 약 3 m 앞에서 고정 카메라 화면을 벗어난다 — 그 뒤 나란한 지점까지는 이 마지막 값과 FAST-LIO 누적으로 간다.
    몸을 든 자세(케이스 2 의 뒤 단계)에서는 위치 식이 수평 자세를 가정하므로 갱신하지 않는다."""
    tg, sc = ST["target"], ST["scene"]
    if tg is None or sc.get("stamp") is None or sc["stamp"] == tg.get("refined_scene") or not ST["K"] or not ST["samples"]:
        return
    tg["refined_scene"] = sc["stamp"]
    if abs(ST.get("pitch_up") or 0.0) > 4.0 or (ST["case"] == 2 and ST["phase2"] in ("C", "D")):
        return
    best = None
    for g in sc["regions"]:
        if g.get("state") != "VERIFIED" or not (0.8 <= g["z"] <= 8.0):     # 11-13: 기각·탐색 중 영역으로 갱신해 0.4~0.56 m 씩 튄 일이 있었다
            continue
        pt, _ = region_point(g, sc["stamp"], ST["samples"], ST["K"])
        if pt is None or abs(pt["up"] - tg["up"]) > 0.35:
            continue
        d = math.hypot(pt["uv"][0] - tg["uv"][0], pt["uv"][1] - tg["uv"][1])
        if d <= 0.5 and (best is None or d < best[0]):
            best = (d, g, pt)
    if best is None:
        return
    d, g, pt = best
    rng = math.hypot(pt["body"][0], pt["body"][1])
    if rng < tg.get("best_rng", 99.0) - 0.15:
        old_uv = list(tg["uv"])
        tg.update(uv=pt["uv"], best_rng=rng, n_refine=tg.get("n_refine", 0) + 1)
        ev("refine", "기억 위치 갱신: %.1f m 에서 다시 봄 (%.2f m 옮김)" % (rng, d), region=g, old_uv=old_uv, new_uv=pt["uv"], moved_m=round(d, 3), seen_range_m=round(rng, 2),
           frame=save_frame("refine"))


def auto_remember(now):
    """자동 기억: (1) E5 가 낸 정지 요청 (2) 점수가 기준 이상인 같은 영역이 두 장면 연속으로 같은 자리에 보일 때"""
    if not ST["auto"] or not ST["K"] or not ST["samples"]:
        return
    if ST["target"] is not None:
        # 기억한 표지가 있으면 새로 잡지 않는다. 다만 그 표지를 이미 지나쳐 조준할 수 없게 됐다면(안내 "지나쳤습니다") 새 표지를 받을 수 있게 비운다
        # (끝내기를 누르지 않고 다음 표지로 걸어간 경우 — 11-13~15 기록에서 뒤 표지의 정지 요청이 모두 무시됐다).
        if (ST.get("guide") or {}).get("phase") != "passed":
            return
        old_tg = ST["target"]
        fresh = [r for r in (ST.get("e5_requests") or []) if r.get("observation_id") and r["observation_id"] not in ST["e5req_seen"]
                 and float(r.get("wall") or 0.0) >= max(ST["started"], old_tg["wall"]) and now - float(r.get("wall") or 0.0) <= 20.0]
        if not fresh:
            return
        ST["done"].append(list(old_tg["uv"])); ST.update(target=None, sel=None, phase2="A", cand=None)
        ev("forget", "지나친 표지 #%d 를 비움 — 새 정지 요청을 받음" % old_tg["id"], auto=True)
    for r in ST.get("e5_requests") or []:
        oid = r.get("observation_id")
        if not oid or oid in ST["e5req_seen"]:
            continue
        ST["e5req_seen"].add(oid)
        w = float(r.get("wall") or 0.0)
        if w < ST["started"]:
            continue                                           # 이 도구가 뜨기 전의 요청
        why = ("요청이 %.0f초 전 것" % (now - w)) if now - w > 20.0 else ("상자·깊이·장면 시각이 없음" if (not r.get("bbox") or not r.get("depth_m") or r.get("detection_timestamp") is None) else None)
        if why is None:
            p = r.get("odom_pose_at_detection") or {}
            od = None
            if p.get("status") in ("INTERPOLATED", "NEAREST") and all(p.get(k) is not None for k in ("x", "y", "yaw")):
                od = (float(r["detection_timestamp"]), float(p["x"]), float(p["y"]), float(p["yaw"]))      # E5 가 그 장면 시각으로 보간해 둔 위치
            ok, why = remember("E5 정지 요청", dict(box=r["bbox"], z=float(r["depth_m"]), sim=r.get("sim")), float(r["detection_timestamp"]), od)
            if ok:
                return
        ev("skip", "정지 요청 %s 을 표지로 기억하지 못함: %s" % (str(oid)[-4:], why), request=oid, why=why)
    sc = ST["scene"]
    if sc.get("stamp") is None or sc["stamp"] == ST.get("auto_scene"):
        return
    ST["auto_scene"] = sc["stamp"]
    best = None
    for g in sc["regions"]:
        # 11-13 첫 실험: 상태가 SEARCH·TRACK 인 영역(E5 가 아직 표지로 확인하지 않은 것)까지 받았더니 자동 기억 9회 중 6회가 표지가 아니었다.
        # E5 상태값: VERIFIED = 표지로 확인, SEARCH = 살펴보는 중, TRACK = 각도 등으로 보류, REJECT = 기각
        if g.get("state") != "VERIFIED" or not (0.8 <= g["z"] <= 8.0):
            continue
        pt, _ = region_point(g, sc["stamp"], ST["samples"], ST["K"])
        if pt is None or not (1.2 <= pt["up"] <= 2.3):
            continue
        if any(math.hypot(pt["uv"][0] - d[0], pt["uv"][1] - d[1]) < 1.0 for d in ST["done"]):
            continue
        if best is None or g["sim"] > best[0]["sim"]:
            best = (g, pt)
    prev = ST.get("cand")
    ST["cand"] = None if best is None else best[1]["uv"]
    if best is not None and prev is not None and math.hypot(best[1]["uv"][0] - prev[0], best[1]["uv"][1] - prev[1]) < 0.7:
        remember("SigLIP 자동", best[0], sc["stamp"])


# ------------------------------------------------------------------ 안내
def guide(now):
    s = ST["sens"]
    out = dict(phase=None, title="", sub="", tone=None, can_aim=False, need=None)
    if not s.get("active"):
        out.update(title="실험이 꺼져 있습니다", sub=("운용 탭에서 '팬틸트 카메라 사용'을 체크하고 실험을 시작하세요. 팬틸트 수동 이동은 지금도 됩니다." if plate_mode() else
                                                    "기존 화면(8770)에서 실험을 시작하세요 — '이번 실험에서 팬틸트 카메라 사용'은 끄고. 팬틸트 수동 이동은 지금도 됩니다."))
        return out
    if not s.get("odom_ok"):
        out.update(title="위치 정보 대기 중", sub="FAST-LIO 가 준비될 때까지 움직이지 마세요.")
        return out
    tg = ST["target"]
    if tg is None:
        out.update(title="표지를 찾는 중", sub=("걸어가면 SigLIP 이 잡은 표지를 자동으로 기억합니다." if ST["auto"] else "화면에서 표지 상자를 누르세요.")
                   + (" SigLIP: %s" % ST.get("siglip") if ST.get("siglip") not in (None, "UP", "OK") else ""), phase="search")
        return out
    fwd, left = G.planar_to_body(ST["pose"], tg["uv"][0], tg["uv"][1])
    rng, bearing = math.hypot(fwd, left), math.degrees(math.atan2(left, fwd))
    side = "왼쪽" if left >= 0 else "오른쪽"
    out.update(fwd=round(fwd, 2), left=round(left, 2), range=round(rng, 2), bearing=round(bearing, 1))
    pt_ready = PTC.connected and not PTC.busy
    if ST["case"] == 1:
        try:
            need = GEO.servo(fwd, left, tg["up"])
            ok, why = GEO.check(need["pan_cmd"], need["tilt_cmd"])
        except ValueError as ex:
            need, ok, why = None, False, str(ex)
        fmin = GEO.closest_fwd(left, tg["up"])
        out["need"] = None if need is None else dict(pan=round(need["pan_cmd"], 1), tilt=round(need["tilt_cmd"], 1), ok=ok, why=why,
                                                     side_deg=round(abs(need["pan_deg"])), up_deg=round(need["tilt_deg"]), dist=round(need["dist"], 2))
        if fmin is None:
            out.update(phase="unreachable", title="이 표지는 조준할 수 없습니다", sub=why or "", tone="passed")
            return out
        stop_at = max(float(ST["ahead"]), fmin + STOP_MARGIN)
        rem = fwd - stop_at
        out.update(remaining=round(rem, 2), stop_at=round(stop_at, 2))
        if rem > STOP_ZONE[1]:
            if ok:
                out.update(phase="walk", title="앞으로 %.1f m" % rem, tone="walk",
                           sub="표지는 %s %.1f m. 표지를 %.1f m 앞에 두고 멈춥니다. 지금 멈춰도 조준은 됩니다." % (side, abs(left), stop_at))
            else:
                out.update(phase="turn", title="몸 방향을 표지 쪽으로 조금 돌리세요", tone="walk", sub="표지가 %s %.0f° 에 있어 팬 범위 밖입니다. (%s)" % (side, abs(bearing), why))
        elif rem >= STOP_ZONE[0] and ok:
            out.update(phase="stop", tone="stop", title="여기서 정지" if not s.get("stopped") else "조준 가능 — [조준 + 촬영]",
                       sub="표지는 %s %.1f m, 앞으로 %.1f m, 거리 %.1f m. 팬틸트는 옆으로 %.0f° 돕니다." % (side, abs(left), fwd, rng, abs(need["pan_deg"])))
        elif ok:
            out.update(phase="over", tone="over", title="정지 지점을 %.1f m 지났습니다 — 여기서도 조준은 됩니다" % (-rem),
                       sub="팬틸트가 옆으로 %.0f° 돕니다(끝은 왼쪽 74°·오른쪽 62°). %.1f m 더 가면 범위를 벗어납니다." % (abs(need["pan_deg"]), max(0.0, fwd - fmin)))
        else:
            out.update(phase="passed", tone="passed", title="지나쳤습니다 — 뒤로 %.1f m" % max(0.1, -rem), sub="표지가 팬틸트 범위 뒤로 넘어갔습니다. (%s)" % (why or ""))
        out["can_aim"] = bool(ok and s.get("stopped") and pt_ready)
        return out
    # ---- 케이스 2: 표지 옆까지 -> 표지 쪽으로 회전 -> 머리 들기 -> 조준
    pu = ST.get("pitch_up") or 0.0
    facing = abs(bearing) <= FACE_DEG and fwd > 0.3
    ph = ST["phase2"]
    if facing and rng <= max(3.5, abs(tg.get("left0") or 0.0) + 1.5) and ph in ("A", "B"):
        ph = "C"
    if ph == "A" and fwd <= ABREAST_FWD:
        ph = "B"
    if ph == "B":
        if facing:
            ph = "C"
        elif fwd > 1.0 and abs(bearing) < 40.0:
            ph = "A"
    if ph == "C":
        if abs(bearing) > 20.0:
            ph = "B"
        elif pu >= RAISE_DEG:
            ph = "D"
    if ph == "D" and pu < RAISE_DEG - 3.0:
        ph = "C"
    ST["phase2"] = ph
    out.update(phase=ph, pitch_up=round(pu, 1))
    try:
        need = GEO.servo(fwd, left, tg["up"], pu)
        ok, why = GEO.check(need["pan_cmd"], need["tilt_cmd"])
        out["need"] = dict(pan=round(need["pan_cmd"], 1), tilt=round(need["tilt_cmd"], 1), ok=ok, why=why, side_deg=round(abs(need["pan_deg"])),
                           up_deg=round(need["tilt_deg"]), dist=round(need["dist"], 2))
    except ValueError:
        pass
    seen = None                                    # 머리를 들었을 때(아직이면 18도로 가정) 표지가 고정 카메라 화면 안에 들어오는가
    if ST["K"]:
        pp = GEO.pixel(fwd, left, tg["up"], ST["K"], pu if pu >= RAISE_DEG else 18.0)
        seen = bool(pp and 0 <= pp[0] <= 640 and 20 <= pp[1] <= 480)
    out["seen_when_raised"] = seen
    far = "" if seen or seen is None else " 이 거리(%.1f m)에서는 머리를 들어도 표지가 고정 카메라 위로 벗어납니다 — 팬틸트는 기억한 위치로 조준합니다." % rng
    if ph == "A":
        out.update(title="앞으로 %.1f m (표지 옆까지)" % max(0.0, fwd - 0.2), remaining=round(fwd - 0.2, 2), tone="walk", sub="표지는 %s %.1f m." % (side, abs(left)))
    elif ph == "B":
        out.update(title="%s으로 %.0f° 도세요 (표지 쪽으로)" % (side, abs(bearing)), tone="turn", sub="표지까지 %.1f m. 정면이 되면 소리가 납니다." % rng)
    elif ph == "C":
        out.update(title="select 누르고 머리 들기 (지금 %.0f°)" % pu, tone="raise",
                   sub="머리를 들지 않고 찍으려면 멈춘 뒤 바로 [조준 + 촬영]. 표지까지 %.1f m.%s" % (rng, far))
    else:
        out.update(title="조준 가능 — 표지를 누르거나 [조준 + 촬영]" if s.get("stopped") else "움직임이 멈추면 조준할 수 있습니다", tone="ready",
                   sub="머리 %.0f° 들림. 화면에서 표지를 누르면 그 점을 조준합니다.%s" % (pu, far))
    out["can_aim"] = bool(ph in ("C", "D") and s.get("stopped") and pt_ready)
    return out


# ------------------------------------------------------------------ 팬틸트 사용 동의 (명판 촬영 방식)
NO_CONSENT = "이 실험은 '팬틸트 카메라 사용'을 체크하지 않고 시작했습니다 — 팬틸트를 쓰려면 체크하고 다시 시작하세요"


def plate_mode():
    """명판 촬영(케이스 1) 방식 스위치: 운영폴더 runtime/plate_case1.on 이 있으면 켜짐.
    켜져 있으면 '팬틸트 카메라 사용'을 체크하고 시작한 실험에서 운영판 E6 자동 촬영은 뜨지 않고(콘솔 쪽 스위치), 팬틸트는 이 사이트가 쓴다.
    그때는 동의가 있어야만 팬틸트를 움직인다. 꺼져 있으면 예전 규칙(동의로 시작한 실험은 E6 가 팬틸트를 쥐므로 여기서는 못 씀)."""
    return (C["O"] / "runtime" / "plate_case1.on").exists()


def consent_block(exp=None):
    """진행 중인 실험에서 팬틸트를 쓰면 안 되는 이유(없으면 None)."""
    e = ST["exp"] if exp is None else exp
    if e.get("state") not in ACTIVE + ("STARTING", "STOPPING"):
        return None                                        # 실험이 없을 때(수동 점검)는 예전처럼 쓸 수 있다
    if e.get("plate"):
        return None if e.get("consent") else NO_CONSENT
    return "이 실험은 '팬틸트 카메라 사용'으로 시작되어 기존 자동 촬영이 팬틸트를 씁니다" if e.get("consent") else None


# ------------------------------------------------------------------ 센서 읽기 (읽기만)
def poll_once(now=None):
    now = now or time.time()
    e = jread(p_exp()) or {}
    exp = dict(state=e.get("state"), id=e.get("exp_id"), consent=bool(e.get("pantilt_consent")), plate=plate_mode(),
               e6=((e.get("modules") or {}).get("e6") or {}).get("state"))
    exp["block"] = consent_block(exp)
    active = exp["state"] in ACTIVE
    idx = (jread(C["hub"] / "index.json") or {}) if active else {}
    samples = [tuple(float(v) for v in r[:4]) for r in (idx.get("odom") or [])]
    odom_age = (now - float(idx.get("odom_last_rx_wall") or 0.0)) if samples else None
    odom_ok = bool(samples) and odom_age is not None and odom_age < 1.5
    K = (idx.get("caminfo") or {}).get("k")
    stp = G.stop_state(samples, samples[-1][0], wall_now=now) if odom_ok else dict(stopped=False, why="NO_ODOMETRY")
    pose = [float(v) for v in G.raw_to_planar(samples[-1][1], samples[-1][2], samples[-1][3])] if odom_ok else None
    raw = None
    try:
        fresh = now - p_mat().stat().st_mtime < 2.5
    except OSError:
        fresh = False
    if fresh and odom_ok:
        p = (last_line(p_mat()) or "").split()
        if len(p) >= 4:
            try:
                raw = -float(p[2])                 # FAST-LIO 상태 기록의 두 번째 회전 값 B: 머리를 들면 줄어든다(11-13 시범) -> 든 쪽이 +
            except ValueError:
                raw = None
    s5 = jread(p_e5()) if active else None
    fr = hub_newest() if active else None
    with LOCK:
        dropped = None
        if exp.get("id") != (ST.get("exp") or {}).get("id"):
            # 실험이 바뀌면 FAST-LIO 가 새로 시작해 위치의 기준(원점·방향)이 달라진다. 앞 실험에서 기억한 표지와 끝낸 자리는 새 기준에서는
            # 다른 곳을 가리키므로 모두 버린다. 버리지 않으면 (1) 새 표지를 기억하지 않고 (2) 엉뚱한 방향을 안내·조준한다
            # (11-13~15: 실험 32건 중 뒤 실험의 정지 요청이 모두 무시됐고, 11-14 17:02 에는 앞 실험의 표지로 조준해 팬 160° 로 돌았다).
            if ST.get("target") is not None:
                dropped = (ST["target"]["id"], ST["target"].get("exp"))
            ST.update(target=None, sel=None, phase2="A", cand=None, auto_scene=None, done=[], photo=None, sent=[], n_used=0, n_photos=0, hist=[])
            ST["e5req_seen"] = set()
        ST["exp"] = exp
        ST["samples"], ST["K"], ST["pose"] = samples, K, pose
        if samples:                                # 허브는 최근 3초만 준다. 저장소 줄(1.5초 간격, 처리 지연 포함)의 시각을 덮도록 30초치를 이어 붙여 둔다
            h = ST["hist"]
            if h and samples[-1][0] < h[-1][0] - 1.0:
                h = []                             # 시각이 뒤로 갔다 = 다른 기록
            last = h[-1][0] if h else -1e18
            h = h + [q for q in samples if q[0] > last]
            ST["hist"] = [q for q in h if q[0] >= h[-1][0] - STORE_HIST_S]
        if dropped is not None and exp.get("state") in ACTIVE + ("STARTING",):
            ev("forget", "새 실험 %s — 앞 실험에서 기억한 표지 #%d 를 버림" % (str(exp.get("id"))[-6:], dropped[0]), auto=True, from_exp=dropped[1])
        if raw is None:
            ST["pitch_up"] = None
        else:
            if ST["pitch_ref"] is None:
                ST["pitch_ref"] = raw
            dev = raw - ST["pitch_ref"]
            if abs(dev) < 4.0 and stp.get("stopped"):      # 멈춰 선 자세를 기준으로 삼는다(걷는 동안·머리 든 동안은 기준을 옮기지 않는다)
                ST["pitch_ref"] += 0.15 * dev
            ST["pitch_up"] = round(raw - ST["pitch_ref"], 1)
        if s5 is not None:
            sc = s5.get("scene") or {}
            regs = []
            for g in sc.get("regions") or []:
                b = g.get("box") or []
                if len(b) >= 4 and g.get("z"):
                    regs.append(dict(box=[float(v) for v in b[:4]], sim=None if g.get("sim") is None else float(g["sim"]), z=float(g["z"]),
                                     state=g.get("state"), display=g.get("display")))
            ST["scene"] = dict(stamp=sc.get("stamp"), regions=regs, thr=sc.get("thr"))
            ST["siglip"] = s5.get("siglip")
            ST["e5_requests"] = list(((s5.get("live2_stop_request") or {}).get("last")) or [])
        elif not active:
            ST["scene"] = dict(stamp=None, regions=[], thr=None)
        if fr is not None and fr[0]["seq"] != ST["rs_seq"]:
            ok, jpg = cv2.imencode(".jpg", fr[1], [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                seq = int(fr[0]["seq"])
                ST["frames"][seq] = dict(jpg=jpg.tobytes(), depth=fr[2], stamp=float(fr[0]["stamp"]))
                ST["frame_order"].append(seq)
                while len(ST["frame_order"]) > 5:
                    ST["frames"].pop(ST["frame_order"].pop(0), None)
                ST["rs_seq"] = seq
        elif not active:
            ST["rs_seq"] = None
        sc_age = None if ST["scene"].get("stamp") is None else round(now - float(ST["scene"]["stamp"]), 1)
        ST["sens"] = dict(active=active, odom_ok=odom_ok, odom_age=None if odom_age is None else round(odom_age, 2), stopped=bool(stp.get("stopped")),
                          stop_why=stp.get("why"), speed=stp.get("speed_mps"), k_ok=bool(K), scene_age=sc_age, siglip=ST.get("siglip"))
        if odom_ok:
            auto_remember(now)
            refine_target(now)
        ST["guide"] = guide(now)


def poll_loop():
    while True:
        try:
            poll_once()
        except Exception as ex:                                # noqa: BLE001  읽기 실패가 사이트를 멈추면 안 된다
            with LOCK:
                ST["poll_error"] = "%s: %s" % (type(ex).__name__, ex)
        time.sleep(0.25)


# ------------------------------------------------------------------ 팬틸트 (사용자 조준 콘솔에 맡긴다)
class PanTilt:
    def __init__(self):
        self.url = "http://127.0.0.1:%d" % C["pt_port"]
        self.proc, self.pid = None, None
        self.connected, self.starting, self.busy = False, False, False
        self.job, self.err, self.cmd, self.device_state, self.n = None, None, None, None, 0
        self.last_dist = None                              # 마지막으로 조준한 목표까지의 거리(팬축 기준). 미세 조정 때 같은 거리의 점을 쓴다

    def get(self, path, timeout=4.0):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def view(self):
        return dict(connected=self.connected, starting=self.starting, busy=self.busy, job=self.job, err=self.err,
                    cmd=None if self.cmd is None else [round(self.cmd[0], 1), round(self.cmd[1], 1)], device_state=self.device_state,
                    home=[round(GEO.home[0], 1), round(GEO.home[1], 1)], lim=dict(pan=list(GEO.pan_lim), tilt=list(GEO.tilt_lim), lash=GEO.lash))

    # ---- 시작 전 확인: 다른 주인이 있으면 열지 않는다(같은 포트를 둘이 열면 나노가 리셋된다)
    def guard(self):
        why = []
        if consent_block():
            why.append(consent_block())
        pidf = jread_text(C["O"] / "runtime" / "owned" / "e6_aim_console.pid")
        if pidf and pidf.split():
            # 운영판은 이 파일을 지우지 않는다(끝난 뒤에도 옛 번호가 남는다). 재부팅 뒤 그 번호를 다른 프로세스가 쓰고 있을 수 있으므로
            # 번호가 살아 있는지만 보지 않고 그 프로세스가 정말 조준 콘솔인지 명령줄로 확인한다.
            cmd = (jread_text("/proc/%s/cmdline" % pidf.split()[0]) or "").replace("\0", " ")
            if "aim_console" in cmd:
                why.append("기존 자동 촬영의 조준 콘솔이 떠 있습니다 (pid %s)" % pidf.split()[0])
        for dev, name in ((PORT_DEV, "팬틸트 직렬 포트"), (CAM_DEV, "팬틸트 카메라")):
            if not os.path.exists(dev):
                why.append("%s 가 보이지 않습니다" % name)
                continue
            try:
                r = subprocess.run(["fuser", os.path.realpath(dev)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5)
                if r.stdout.split():
                    why.append("%s 를 다른 프로세스가 쓰고 있습니다 (pid %s)" % (name, " ".join(r.stdout.split())))
            except (OSError, subprocess.TimeoutExpired):
                why.append("%s 점유 확인 실패" % name)
        for port in (8088, C["pt_port"]):
            try:
                import socket
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    why.append("포트 %d 를 이미 누가 쓰고 있습니다" % port)
            except OSError:
                pass
        t = max_temp()
        m = mem_available_mb()
        if t is not None and t >= TEMP_CONNECT_C:
            why.append("로봇 컴퓨터 온도 %.0f°C (%.0f 미만이어야 함 — 팬틸트 카메라의 온도가 아님)" % (t, TEMP_CONNECT_C))
        if m is not None and m < 4300:
            why.append("남은 메모리 %d MB (4300 이상이어야 함)" % m)
        return why

    def connect(self):
        if self.connected or self.starting:
            return False, "이미 연결됨"
        self.starting, self.err = True, None
        threading.Thread(target=self._connect, daemon=True).start()
        return True, None

    def _connect(self):
        try:
            if C["fake_console"]:
                st = self.get("/state", 2.0)
            else:
                why = self.guard()
                if why:
                    raise RuntimeError(" / ".join(why))
                out = session_dir() / "console" / time.strftime("%Y%m%d_%H%M%S")      # 그 실험 폴더 아래: 실험 하나의 기록이 한 폴더에 모인다
                out.mkdir(parents=True, exist_ok=True)
                log = open(out / "aim_console.log", "w")
                kc = C["O"] / "candidate" / "external" / "sci_event" / "code"
                args = ["--config", str(GEO.calib), "--no-eeprom-sync", "--out-root", str(out), "--port", str(C["pt_port"]), "--view-width", "800",
                        "--pan-lim", "%g" % GEO.pan_lim[0], "%g" % GEO.pan_lim[1], "--tilt-lim", "%g" % GEO.tilt_lim[0], "%g" % GEO.tilt_lim[1],
                        "--lash-deg", "%g" % GEO.lash]
                cmd = ["bash", "-c", "set +u; source /opt/ros/foxy/setup.bash >/dev/null 2>&1; source $HOME/unitree_ros2/setup.sh >/dev/null 2>&1; "
                       "cd %s && exec taskset -c 2,4,5 /usr/bin/python3 -B %s %s" % (kc, HERE / "pt_console.py", " ".join(args))]
                ev("pt_connect", "팬틸트 연결 시작 (나노 리셋 -> 정면)", out=str(out))
                self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PLATE_SITE_KC=str(kc)))
                self.pid = self.proc.pid
                (p_out() / "pt_console.pid").write_text("%d %s\n" % (self.pid, out))
                st = None
                for _ in range(60):
                    try:
                        st = self.get("/state", 1.0)
                        break
                    except Exception:                          # noqa: BLE001
                        if self.proc.poll() is not None:
                            raise RuntimeError("조준 콘솔이 종료됨 (rc %s) — %s/aim_console.log" % (self.proc.returncode, out))
                        time.sleep(1.0)
                if st is None:
                    raise RuntimeError("조준 콘솔이 60초 안에 응답하지 않음")
            self.device_state = st.get("device_state")
            self.cmd = tuple(GEO.home)                         # 콘솔은 시작하자마자 정면으로 보낸다
            self.connected = True
            ev("pt_connected", "팬틸트 연결됨 (범위 팬 %g~%g, 틸트 %g~%g)" % (GEO.pan_lim + GEO.tilt_lim))
        except Exception as ex:                                # noqa: BLE001
            self.err = "%s" % ex
            ev("pt_connect_failed", "팬틸트 연결 실패: %s" % ex, error="%s: %s" % (type(ex).__name__, ex))
        finally:
            self.starting = False

    def disconnect(self):
        if not self.connected:
            return False, "연결되어 있지 않음"
        if self.busy:
            return False, "팬틸트가 동작 중"
        self.busy, self.job = True, "연결 끊는 중"
        threading.Thread(target=self._disconnect, daemon=True).start()
        return True, None

    def _disconnect(self):
        try:
            try:
                self._home()
            except Exception:                                  # noqa: BLE001  정면 복귀 실패여도 끊는다
                pass
            if not C["fake_console"] and self.pid:
                for _ in range(2):
                    try:
                        os.killpg(os.getpgid(self.pid), signal.SIGINT)      # 사용자 콘솔의 종료 경로(초점 모터 정지 F0)
                    except OSError:
                        break
                    for _ in range(16):
                        if not os.path.exists("/proc/%d" % self.pid) or (self.proc is not None and self.proc.poll() is not None):
                            break
                        time.sleep(0.5)
                    else:
                        continue
                    break
            alive = bool(self.pid) and os.path.exists("/proc/%d" % self.pid) and not (self.proc is not None and self.proc.poll() is not None)
            if alive and not C["fake_console"]:               # 카메라 읽기에 묶여 SIGINT 로 안 끝나는 경우(11-13: 영상이 끊긴 콘솔) -> SIGTERM
                try:
                    os.killpg(os.getpgid(self.pid), signal.SIGTERM)
                except OSError:
                    pass
                for _ in range(10):
                    if not os.path.exists("/proc/%d" % self.pid) or (self.proc is not None and self.proc.poll() is not None):
                        alive = False
                        break
                    time.sleep(0.5)
            ev("pt_disconnected", "팬틸트 연결 끊음" + (" (콘솔 프로세스가 아직 남아 있음 — 잠시 뒤 다시 연결)" if alive else ""), console_still_alive=alive)
        finally:
            self.connected, self.busy, self.job, self.proc, self.pid, self.cmd = False, False, None, None, None, None

    # ---- 동작: 한 번에 하나. 실험 중에는 로봇이 멈춰 있을 때만
    def motion_ok(self):
        if not self.connected:
            return False, "팬틸트가 연결되어 있지 않습니다"
        if self.busy:
            return False, "팬틸트가 동작 중입니다"
        s = ST["sens"]
        if s.get("active"):
            if not s.get("odom_ok"):
                return False, "위치 정보가 끊겼습니다"
            if not s.get("stopped"):
                return False, "로봇이 멈춘 뒤에 누르세요 (%s)" % (s.get("stop_why") or "")
        return True, None

    def submit(self, label, fn):
        ok, why = self.motion_ok()
        if not ok:
            return False, why
        self.busy, self.job, self.err = True, label, None

        def run():
            try:
                fn()
            except Exception as ex:                            # noqa: BLE001
                self.err = "%s" % ex
                ev("pt_error", "팬틸트 실패: %s" % ex, job=label, error="%s: %s" % (type(ex).__name__, ex))
            finally:
                self.busy, self.job = False, None
        threading.Thread(target=run, daemon=True).start()
        return True, None

    def _wait(self, rid, timeout_s):
        t0 = time.time()
        rec = {}
        while time.time() - t0 < timeout_s:
            rec = self.get("/request?id=" + urllib.parse.quote(rid))
            if rec.get("status") in TERMINAL:
                return rec
            time.sleep(0.25)
        raise RuntimeError("콘솔이 %.0f초 안에 끝내지 못함 (상태 %s)" % (timeout_s, rec.get("status")))

    def _home(self):
        self.n += 1
        rid = "ps-home-%d-%d" % (int(time.time()), self.n)
        r = self.get("/home?req=%s&owner=plate_site" % rid)
        if r.get("status") == "REJECTED_BUSY":
            raise RuntimeError("콘솔이 바쁨 (%s)" % r.get("reason_code"))
        rec = self._wait(rid, 10.0)
        self.device_state = rec.get("device_state")
        if rec.get("status") != "COMPLETED":
            raise RuntimeError("정면 복귀 실패 (%s)" % (rec.get("reason_code") or rec.get("status")))
        self.cmd = tuple(GEO.home)

    def home(self):
        def fn():
            self._home()
            ev("pt_home", "정면으로 복귀")
        return self.submit("정면 복귀", fn)

    def aim_servo(self, pan, tilt, tag, basis, extra=None, dist=None):
        ok, why = GEO.check(pan, tilt)
        if not ok:
            return False, "가동 범위 밖: " + why
        if dist is not None:
            self.last_dist = float(dist)
        try:
            u, v, z, err = GEO.virtual_pixel(pan, tilt, self.last_dist)
        except ValueError as ex:
            return False, str(ex)
        if err > 0.3:
            return False, "조준 변환 오차 %.2f° — 보내지 않음" % err
        return self.submit("조준 + 촬영", lambda: self._aim(u, v, z, tag, basis, dict(extra or {}, want=[round(pan, 2), round(tilt, 2)])))

    def aim_pixel(self, u, v, z, tag, basis, extra=None):
        try:
            pan, tilt = GEO.pixel_servo(u, v, z)
        except ValueError as ex:
            return False, str(ex)
        ok, why = GEO.check(pan, tilt)
        if not ok:
            return False, "가동 범위 밖: " + why
        self.last_dist = float(np.linalg.norm(GEO.A.pixel_to_point(float(u), float(v), float(z))))
        return self.submit("조준 + 촬영", lambda: self._aim(int(round(u)), int(round(v)), float(z), tag, basis, dict(extra or {}, want=[round(pan, 2), round(tilt, 2)])))

    def _aim(self, u, v, z, tag, basis, extra):
        ok, why = self.motion_ok_running()
        if not ok:
            raise RuntimeError(why)
        self.n += 1
        rid = "ps-%s-%d-%d" % (tag, int(time.time()), self.n)
        t0 = time.time()
        extra = dict(extra, frame=save_frame(tag))               # 조준을 보내는 순간의 고정 카메라 장면
        r = self.get("/aim?u=%d&v=%d&z=%.3f&focus=0&strict=1&req=%s&owner=plate_site" % (u, v, z, rid))
        if r.get("status") == "REJECTED_BUSY":
            raise RuntimeError("콘솔이 바쁨 (%s)" % r.get("reason_code"))
        rec = self._wait(rid, 25.0)
        self.device_state = rec.get("device_state")
        plan = rec.get("plan") or {}
        if plan.get("pan") is not None and not plan.get("refused"):
            self.cmd = (float(plan["pan"]), float(plan["tilt"]))      # 사진을 못 받았어도 팬틸트는 이미 그쪽으로 돌았다
        if rec.get("status") != "COMPLETED":
            why = str(rec.get("reason_code") or rec.get("status"))
            if "fresh" in why.lower() or "capture" in why.lower():
                raise RuntimeError("팬틸트 카메라 영상이 안 들어옵니다 (%s) — [연결 끊기] 후 다시 [팬틸트 연결]. 케이블이 당겨졌는지 확인하세요" % why)
            raise RuntimeError("조준 실패 (%s)" % why)
        if plan.get("pan") is not None:
            self.cmd = (float(plan["pan"]), float(plan["tilt"]))
        cap = rec.get("capture") or {}
        src = cap.get("path")
        if not src or not os.path.exists(src):
            raise RuntimeError("사진 파일이 없음 (%s)" % rec.get("reason_code"))
        with LOCK:
            have = [int(f.name[:3]) for f in (session_dir() / "photos").glob("[0-9][0-9][0-9]_*.jpg")]
            ST["n_photos"] = max([ST["n_photos"]] + have) + 1      # 같은 실험 폴더에 이미 있는 사진 다음 번호 (도구가 도중에 다시 떠도 덮어쓰지 않게)
            name = "%03d_c%d_%s.jpg" % (ST["n_photos"], ST["case"], tag)
            dst = session_dir() / "photos" / name
        shutil.copyfile(src, dst)
        sign = sign_check(str(dst))
        with LOCK:
            smp = ST["samples"][-1] if ST.get("samples") else None          # 촬영한 자리: 그때의 FAST-LIO 원래 값 (stamp, x, y, yaw)
            ST["photo"] = dict(name=name, wall=time.time(), tag=tag, basis=basis, pan=round(self.cmd[0], 1), tilt=round(self.cmd[1], 1), sign=sign,
                               session=session_dir().name, sec=round(time.time() - t0, 1), path=str(dst), capture_id=cap.get("capture_id"),
                               target_id=(ST.get("target") or {}).get("id"), exp=ST["exp"].get("id"), stopped_wall=t0,
                               stop=None if smp is None else dict(t=float(smp[0]), x=float(smp[1]), y=float(smp[2]), yaw=float(smp[3])),
                               store=store_frame(list(ST.get("hist") or [])))
            ev("shot", "촬영 %s · %s · 팬 %.1f 틸트 %.1f · 표지 %s" % (name, basis, self.cmd[0], self.cmd[1], SIGN_KO.get(sign.get("state"), sign.get("state"))),
               photo=name, basis=basis, tag=tag, console_request=rid, plan=plan, capture={k: cap.get(k) for k in ("capture_id", "sha256", "width", "height")},
               sign=sign, aim_pixel=[u, v, round(z, 3)], sec=round(time.time() - t0, 1), **extra)

    def motion_ok_running(self):
        """조준을 보내기 직전에 한 번 더: 그 사이 로봇이 움직였으면 보내지 않는다"""
        s = ST["sens"]
        if s.get("active") and not (s.get("odom_ok") and s.get("stopped")):
            return False, "보내기 직전에 로봇이 움직였습니다 — 조준 취소"
        return True, None


SIGN_KO = dict(CENTERED="가운데", OFF_CENTER="치우침", NOT_FOUND="못 찾음", CHECK_UNAVAILABLE="판정 불가")


def sign_check(path):
    """팬틸트 사진에서 표지 위치(선배 E6 의 target_check 그대로). 실패해도 촬영 기록은 남는다."""
    try:
        import e6_pipeline as P6
        tc = P6.target_check(cv2.imread(path))
        od = tc.get("offset_deg_est") or {}
        return dict(state=tc.get("state"), right=od.get("right"), down=od.get("down"), sharp=tc.get("sharpness"), roi=tc.get("sign_roi_1280"))
    except Exception as ex:                                    # noqa: BLE001
        return dict(state="CHECK_UNAVAILABLE", why="%s: %s" % (type(ex).__name__, ex))


def jread_text(p):
    try:
        return Path(p).read_text()
    except OSError:
        return None


def max_temp():
    v = []
    for z in Path("/sys/devices/virtual/thermal").glob("thermal_zone*/temp"):
        try:
            v.append(int(z.read_text()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(v) if v else None


def mem_available_mb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return None


PTC = PanTilt()


# ------------------------------------------------------------------ 운영자 동작
def op_pick(u, v, seq):
    """고정 카메라 화면을 누름: 상자 안이면 그 SigLIP 영역, 밖이면 그 점의 깊이. 케이스 2 에서 표지 쪽을 본 뒤에는 '조준점 선택', 그 밖에는 '표지 기억'."""
    with LOCK:
        sc = ST["scene"]
        hit = None
        for g in sc.get("regions") or []:
            b = g["box"]
            if b[0] <= u <= b[2] and b[1] <= v <= b[3]:
                if hit is None or (b[2] - b[0]) * (b[3] - b[1]) < (hit["box"][2] - hit["box"][0]) * (hit["box"][3] - hit["box"][1]):
                    hit = g
        if hit is not None and sc.get("stamp") is not None:
            reg, stamp, src = hit, float(sc["stamp"]), "화면 선택(SigLIP 영역)"
            pu, pv = (hit["box"][0] + hit["box"][2]) / 2.0, (hit["box"][1] + hit["box"][3]) / 2.0
        else:
            fr = ST["frames"].get(int(seq)) if seq is not None else None
            if fr is None:
                return False, "그 화면의 깊이가 남아 있지 않습니다 — 다시 눌러 주세요"
            d = fr["depth"]
            h, w = d.shape[:2]
            ui, vi = int(round(u)), int(round(v))
            win = d[max(0, vi - 6):min(h, vi + 7), max(0, ui - 6):min(w, ui + 7)]
            val = win[win > 0]
            if val.size < 8:
                return False, "그 점에는 깊이 값이 없습니다 (유리·너무 가까움) — 표지의 다른 부분을 눌러 보세요"
            z = float(np.median(val)) / 1000.0
            reg, stamp, src = dict(box=[u - 15.0, v - 15.0, u + 15.0, v + 15.0], z=z, sim=None), float(fr["stamp"]), "화면 선택(점)"
            pu, pv = float(u), float(v)
        select_mode = ST["case"] == 2 and ST["target"] is not None and ST["phase2"] in ("C", "D")
        if select_mode:
            ST["sel"] = dict(u=round(pu, 1), v=round(pv, 1), z=round(float(reg["z"]), 3), src=src, sim=reg.get("sim"), wall=time.time())
            ev("select", "조준점 선택: %s · 거리 %.2f m" % (src, reg["z"]), sel=ST["sel"])
            return True, "조준점을 골랐습니다 — [조준 + 촬영]"
    ok, why = remember(src, reg, stamp)
    return ok, why or "표지를 기억했습니다"


def auto_select(fwd, left, up):
    """케이스 2: 지금 장면에서 예측 위치에 가장 가까운 SigLIP 영역(없으면 None)"""
    sc = ST["scene"]
    if sc.get("stamp") is None or time.time() - float(sc["stamp"]) > 3.0 or not ST["K"]:
        return None
    pp = GEO.pixel(fwd, left, up, ST["K"], ST.get("pitch_up") or 0.0)
    best = None
    for g in sc["regions"]:
        if (g.get("sim") or 0.0) < 0.05:
            continue
        cu, cv_ = (g["box"][0] + g["box"][2]) / 2.0, (g["box"][1] + g["box"][3]) / 2.0
        d = math.hypot(cu - pp[0], cv_ - pp[1]) if pp else -float(g["sim"])
        if pp and d > 160.0:
            continue
        if best is None or d < best[0]:
            best = (d, dict(u=round(cu, 1), v=round(cv_, 1), z=float(g["z"]), src="SigLIP 영역(자동, 예측에서 %.0f px)" % d if pp else "SigLIP 영역(자동)", sim=g.get("sim")))
    return None if best is None else best[1]


def op_aim():
    with LOCK:
        tg = ST["target"]
        if tg is None:
            return False, "기억한 표지가 없습니다"
        if tg.get("exp") != ST["exp"].get("id"):               # 실험이 바뀌면 위치의 기준이 달라진다 (poll_once 가 비우지만 한 번 더 막는다)
            return False, "앞 실험에서 기억한 표지입니다 — [기억 지우기] 후 다시 기억하세요"
        if ST["pose"] is None:
            return False, "위치 정보가 없습니다"
        fwd, left = G.planar_to_body(ST["pose"], tg["uv"][0], tg["uv"][1])
        extra = dict(target_body=[round(fwd, 3), round(left, 3), round(tg["up"], 3)])
        if ST["case"] == 1:
            try:
                s = GEO.servo(fwd, left, tg["up"])
            except ValueError as ex:
                return False, str(ex)
            return PTC.aim_servo(s["pan_cmd"], s["tilt_cmd"], "aim", "기억한 위치", extra, dist=s["dist"])
        if ST["phase2"] not in ("C", "D"):
            return False, "표지 쪽으로 돈 뒤에 누르세요"
        sel = ST["sel"] or auto_select(fwd, left, tg["up"])
        if sel is not None:
            return PTC.aim_pixel(sel["u"], sel["v"], sel["z"], "aim", sel["src"], dict(extra, sel=sel))
        try:
            s = GEO.servo(fwd, left, tg["up"], ST.get("pitch_up") or 0.0)
        except ValueError as ex:
            return False, str(ex)
        return PTC.aim_servo(s["pan_cmd"], s["tilt_cmd"], "aim", "기억한 위치(몸 기울기 %.0f° 반영, 영역 못 찾음)" % (ST.get("pitch_up") or 0.0), extra, dist=s["dist"])


def op_nudge(dx_deg, dy_deg, tag="nudge", basis=None):
    """지금 명령에서 보는 방향을 오른쪽 dx 도, 위쪽 dy 도 옮겨 다시 찍는다"""
    if PTC.cmd is None:
        return False, "먼저 조준하세요"
    pan = PTC.cmd[0] - GEO.A.pan_gain * dx_deg              # 팬 명령은 왼쪽이 +
    tilt = PTC.cmd[1] + GEO.A.tilt_gain * dy_deg            # 틸트 명령은 위쪽이 -(tilt_gain 이 음수)
    return PTC.aim_servo(pan, tilt, tag, basis or ("미세 조정 %s%.0f° %s%.0f°" % ("오른쪽 " if dx_deg >= 0 else "왼쪽 ", abs(dx_deg), "위 " if dy_deg >= 0 else "아래 ", abs(dy_deg))),
                         dict(nudge=[dx_deg, dy_deg]))


def op_center():
    ph = ST.get("photo") or {}
    sg = ph.get("sign") or {}
    if sg.get("state") not in ("OFF_CENTER", "CENTERED") or sg.get("right") is None:
        return False, "마지막 사진에서 표지를 찾지 못했습니다 — 화살표로 옮겨 보세요"
    return op_nudge(float(sg["right"]), -float(sg["down"]), "center", "표지 가운데로 (오른쪽 %.1f°, 아래 %.1f°)" % (sg["right"], sg["down"]))


def op_done(label=None):
    with LOCK:
        tg = ST["target"]
        if tg is not None:
            ST["done"].append(list(tg["uv"]))
        ev("done", "이 표지 끝%s" % ("" if not label else " (%s)" % label), label=label)
        ST.update(target=None, sel=None, phase2="A", cand=None)
    if PTC.connected and not PTC.busy and PTC.cmd is not None and tuple(PTC.cmd) != tuple(GEO.home):
        PTC.home()
    return True, "다음 표지를 찾습니다"


def pt_op(body):
    """태블릿의 팬틸트 버튼 하나 -> (ok, 안내 문구)"""
    op = body.get("op")
    if op not in ("connect", "disconnect", "home") and consent_block():     # 정면 복귀·연결 끊기는 언제나 된다
        return False, consent_block()
    if op == "connect":
        return PTC.connect()
    if op == "disconnect":
        return PTC.disconnect()
    if op == "home":
        return PTC.home()
    if op == "aim":
        return op_aim()
    if op == "nudge":
        return op_nudge(float(body.get("dx") or 0.0), float(body.get("dy") or 0.0))
    if op == "center":
        return op_center()
    if op == "reshoot":
        if PTC.cmd is None:
            return False, "먼저 조준하세요"
        return PTC.aim_servo(PTC.cmd[0], PTC.cmd[1], "reshoot", "같은 방향 다시 촬영")
    if op == "goto":
        pan, tilt = float(body["pan"]), float(body["tilt"])
        return PTC.aim_servo(pan, tilt, "manual", "수동 이동 (명령 팬 %.1f 틸트 %.1f)" % (pan, tilt))
    return False, "모르는 동작 %s" % op


def sim_changed(kind):
    """시뮬에서 기록을 바꾸거나 건너뛰었을 때: 앞 장면에 묶인 것들을 비운다"""
    with LOCK:
        ST["frames"].clear()
        ST["frame_order"][:] = []
        ST.update(rs_seq=None, cand=None, auto_scene=None, pitch_ref=None, sel=None)
        if kind == "dataset":
            ST.update(target=None, phase2="A", done=[], photo=None)
        ev("sim_" + kind, "시뮬: %s" % ("기록 바꿈" if kind == "dataset" else "재생 위치 옮김"))


def sim_op(body):
    op = body.get("op")
    if op == "play":
        SIM.play(True)
    elif op == "pause":
        SIM.play(False)
    elif op == "seek":
        SIM.seek(float(body.get("t") or 0.0))
    elif op == "speed":
        SIM.speed = min(8.0, max(0.25, float(body.get("v") or 1.0)))
    elif op == "dataset":
        if not SIM.select(str(body.get("id"))):
            return False, "그 기록이 없습니다"
    elif op == "next":
        if not SIM.next_request():
            return False, "뒤에 남은 정지 요청이 없습니다"
    else:
        return False, "모르는 동작"
    return True, None if not SIM.blocked else SIM.blocked


# ------------------------------------------------------------------ 사진 -> 운영판의 판독·위치추정 (e34 안의 E8 쌍 판정 -> E7)
# 운영판은 E6 가 사진을 찍으면 runtime/state/e6_precision_results.jsonl 에 한 줄을 쓰고, e34 가 그 줄과 사진 파일만 읽어 OCR·VLM·쌍 판정·E7 을 한다
# (e6e7_link._new_results -> e8_pair.on_precision). 명판 촬영 방식에서는 E6 가 꺼져 있으므로, 운영자가 [이 사진으로 위치추정] 을 누르면 같은 형식의 줄을 쓴다.
# 표지 위치는 선배 E6 와 같은 식(e6_core: 평면 좌표 -> odom_link)으로 낸다. 팬·틸트 각은 쓰이지 않는다.
def p_state():
    return C["O"] / "runtime" / "state"


def use_block():
    """마지막 사진을 위치추정에 넘길 수 없는 이유(없으면 None)."""
    e = ST["exp"]
    ph, tg = ST.get("photo"), ST.get("target")
    if not e.get("plate"):
        return "명판 촬영 방식 스위치가 꺼져 있습니다"
    if e.get("state") not in ACTIVE:
        return "진행 중인 실험이 없습니다"
    if not e.get("consent"):
        return NO_CONSENT
    if not ph:
        return "먼저 촬영하세요"
    if ph.get("used"):
        return "이미 넘긴 사진입니다 (%s)" % ph["used"]
    if ph.get("exp") != e.get("id"):
        return "이 사진은 지금 실험에서 찍은 것이 아닙니다"
    if not tg or ph.get("target_id") != tg.get("id"):
        return "이 사진은 지금 기억한 표지를 찍은 것이 아닙니다 (표지 위치를 알아야 지도와 맞춰 볼 수 있습니다)"
    if not ph.get("stop"):
        return "촬영한 자리의 위치 기록이 없습니다"
    if ph.get("tag") == "manual":
        return "수동 이동으로 찍은 사진은 넘기지 않습니다 (기억한 표지를 조준한 사진만)"
    return None


def make_record(oid, exp, ph, tg, capture_file, sha256, resolution):
    """운영판 e6_precision_results.jsonl 한 줄(dict). ph = 촬영 기록(name, wall, stopped_wall, stop{t,x,y,yaw}, sign, pan, tilt, basis, capture_id),
    tg = 기억한 표지(uv 평면 좌표, up, id, n_refine). e8_pair.on_precision 이 읽는 칸은 observation_id, status, image_quality.ok, capture_file,
    image_sha256, target_odom_link_xy, odom_pose_at_stop, stopped_wall, capture_wall, experiment_id 이다. home_confirmed 는 적지 않는다(촬영 뒤 정면 복귀를 확인하지 않으므로)."""
    sg = ph.get("sign") or {}
    found = sg.get("state") in ("CENTERED", "OFF_CENTER")
    xr, yr = G.odom_link_point(tg["uv"][0], tg["uv"][1])      # FAST-LIO 원래 값 기준(선배 E6 와 같은 식)
    sf = ph.get("store") or dict(ok=False, why="NOT_COMPUTED")
    xl, yl = to_store(xr, yr, sf) if sf.get("ok") else (xr, yr)   # 운영판 후보(E3)의 지도 변환은 저장소 자세 기준이므로 그 좌표로 옮겨 적는다
    st = ph["stop"]
    return dict(kind="LIVE2_E6_PRECISION", observation_id=oid, source="OPERATOR_ASSISTED", capture_tool="plate_site_case1", experiment_id=exp,
                status="CAPTURE_STORED", reading="DEFERRED_TO_E8_PAIR", why="operator sent this photo from the plate site (case 1)", ocr_lines=[], vlm=None,
                capture_file=capture_file, capture_id=ph.get("capture_id") or oid, image_sha256=sha256, resolution=resolution,
                image_quality=dict(ok=True, basis="PLATE_SITE_OPERATOR_SENT", roi_laplacian_var=sg.get("sharp"), roi=sg.get("roi"),
                                   roi_source="SIGN_CHECK" if (found and sg.get("roi")) else "NONE", flags=[] if found else ["TARGET_ROI_NOT_DETECTED"], sign_state=sg.get("state")),
                target_odom_link_xy=[round(float(xl), 3), round(float(yl), 3)],
                target_odom_link_xy_fastlio=[round(float(xr), 3), round(float(yr), 3)],
                store_frame=dict(applied=bool(sf.get("ok")), why=sf.get("why"), row_t=sf.get("row_t"), row_age_s=sf.get("age_s"), shift_m=sf.get("shift_m"),
                                 dyaw_deg=sf.get("dyaw_deg"), note="target_odom_link_xy is in the frame of the structure store (odom_link.csv) used by E2/E3; "
                                                                   "odom_pose_at_stop and target_odom_link_xy_fastlio stay FAST-LIO raw"),
                target_pose_in_odom=dict(planar_uv=[round(float(tg["uv"][0]), 3), round(float(tg["uv"][1]), 3)], up_m=round(float(tg["up"]), 3),
                                         frame="planar (e6_core) from the plate site's remembered sign, refined %d time(s)" % int(tg.get("n_refine") or 0)),
                odom_pose_at_stop=dict(t=st["t"], x=st["x"], y=st["y"], yaw=st["yaw"]), stopped_wall=float(ph["stopped_wall"]), capture_wall=float(ph["wall"]), wall=time.time(),
                pan_command=ph.get("pan"), tilt_command=ph.get("tilt"), aim_basis=ph.get("basis"), photo=ph.get("name"), target_id=tg.get("id"))


def op_use():
    """마지막 사진을 운영판의 판독·위치추정에 넘긴다: 사진을 e6_captures 에 두고 e6_precision_results.jsonl 에 한 줄을 덧붙인다."""
    import hashlib
    with LOCK:
        why = use_block()
        if why:
            return False, why
        ph, tg, exp = dict(ST["photo"]), dict(ST["target"]), ST["exp"]["id"]
        ST["n_used"] = int(ST.get("n_used") or 0) + 1
        oid = "plate_%s_P%03d" % (exp, ST["n_used"])
    try:
        data = Path(ph["path"]).read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False, "사진 파일을 읽을 수 없습니다"
        caps = p_state() / "e6_captures"; caps.mkdir(parents=True, exist_ok=True)
        dst = caps / (oid + ".jpg"); tmp = caps / (oid + ".jpg.tmp")
        tmp.write_bytes(data); os.replace(str(tmp), str(dst))                    # 사진을 먼저 다 쓴 뒤에 줄을 덧붙인다
        rec = make_record(oid, exp, ph, tg, str(dst), hashlib.sha256(data).hexdigest(), [int(img.shape[1]), int(img.shape[0])])
        line = (json.dumps(finite(rec), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        fd = os.open(str(p_state() / "e6_precision_results.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
        try:
            os.write(fd, line)                                                   # 한 번의 write 로 한 줄
        finally:
            os.close(fd)
    except OSError as ex:
        return False, "넘기지 못했습니다: %s" % ex
    with LOCK:
        if ST.get("photo") and ST["photo"].get("name") == ph.get("name"):
            ST["photo"]["used"] = oid
        ST["sent"] = (ST.get("sent") or []) + [dict(id=oid, photo=ph.get("name"), wall=time.time(), target_id=tg.get("id"))]
        sfr = rec.get("store_frame") or {}
        note = (" · 저장소 좌표로 옮김 %d cm" % round(100 * (sfr.get("shift_m") or 0.0))) if sfr.get("applied") else \
            ("" if sfr.get("why") == "NO_STORE" else " · 저장소 좌표로 못 옮김(%s)" % sfr.get("why"))
        ev("use", "위치추정에 넘김: %s (%s)%s" % (ph.get("name"), oid, note), observation_id=oid, record=rec)
    return True, "위치추정에 넘겼습니다 — 판독 결과는 잠시 뒤에 나옵니다"


def pair_view():
    """운영판이 넘긴 사진을 어떻게 읽고 판정했는지(e8_pair_state.json, 읽기만) — 이 실험에서 넘긴 사진만 추린다."""
    sent = [x for x in (ST.get("sent") or [])]
    if not sent:
        return None
    d = jread(p_state() / "e8_pair_state.json") or {}
    obs = {o.get("observation_id"): o for o in ((d.get("observations") or []) + ((d.get("last_closed") or {}).get("observations") or []))}
    rows = []
    for x in sent[-6:]:
        o = obs.get(x["id"]) or {}
        lines = [[str(l[0]), round(float(l[1] or 0), 2)] for l in (o.get("ocr_lines") or []) if isinstance(l, (list, tuple)) and len(l) >= 2][:4]
        vlm = ((o.get("reading") or {}).get("vlm") or o.get("vlm") or {})
        rows.append(dict(id=x["id"], photo=x["photo"], seen=bool(o), read=bool(o.get("read")), ocr=lines, vlm=str(vlm.get("raw") or "")[:160] if isinstance(vlm, dict) else None))
    ev_ = d.get("evaluation") or {}
    e7 = d.get("e7") or {}
    return dict(rows=rows, state=d.get("state"), why=d.get("why"), verdict=ev_.get("verdict"), floors=ev_.get("floors_explaining"),
                e7=e7.get("global_decision"), e7_why=e7.get("why"), age=None if not d.get("written_wall") else round(time.time() - float(d["written_wall"]), 1))


def finite(o):
    """화면으로 보내는 값에서 NaN·Infinity 를 None 으로 바꾼다. 브라우저는 이 값이 든 JSON 을 통째로 거부한다
    (E5 기준값이 아직 정해지지 않은 실험 초반에 thr = Infinity 가 와서 화면이 "연결 끊김"으로 멈췄다)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def state_view():
    sim = SIM.view() if SIM is not None else None
    with LOCK:
        sc = ST["scene"]
        tg = ST["target"]
        return dict(now=time.time(), use=dict(block=use_block(), pair=pair_view()), exp=ST["exp"], sens=ST["sens"], case=ST["case"], auto=ST["auto"], ahead=ST["ahead"],
                    rs=dict(seq=ST["rs_seq"], regions=sc.get("regions") or [], thr=sc.get("thr"), scene_age=ST["sens"].get("scene_age")),
                    target=None if tg is None else dict(id=tg["id"], src=tg["src"], sim=tg.get("sim"), z0=round(tg["z0"], 2), up=round(tg["up"], 2),
                                                        left0=round(tg["left0"], 2), age=round(time.time() - tg["wall"]), n_refine=tg.get("n_refine", 0),
                                                        best_rng=round(tg.get("best_rng", 0.0), 2)),
                    guide=ST["guide"], sel=ST["sel"], pt=PTC.view(), photo=ST["photo"], n_photos=ST["n_photos"], log=ST["log"], evseq=ST["evseq"],
                    pitch_up=ST.get("pitch_up"), poll_error=ST.get("poll_error"), sim=sim)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def allowed(self):
        ip = self.client_address[0]
        return ip.startswith(("127.", "192.168.")) or (ip.startswith("100.") and 64 <= int(ip.split(".")[1]) <= 127)     # 이 기기, 같은 공유기, Tailscale

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def js(self, ok, msg=None, **kw):
        self.send(200, json.dumps(finite(dict(ok=bool(ok), msg=msg, **kw)), ensure_ascii=False, default=str, allow_nan=False))

    def do_GET(self):
        if not self.allowed():
            return self.send(403, "{}")
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/":
            return self.send(200, (HERE / "page.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self.send(200, json.dumps(finite(state_view()), ensure_ascii=False, default=str, allow_nan=False))
        if url.path == "/rs.jpg":
            with LOCK:
                fr = ST["frames"].get(int((q.get("seq") or ["-1"])[0])) or (ST["frames"].get(ST["rs_seq"]) if ST["rs_seq"] is not None else None)
            return self.send(200, fr["jpg"], "image/jpeg") if fr else self.send(404, "{}")
        if url.path.startswith("/photo/"):
            parts = url.path.split("/")
            if len(parts) == 4 and all(p and "/" not in p and ".." not in p for p in parts[2:]):
                f = p_out() / parts[2] / "photos" / parts[3]
                if f.is_file():
                    return self.send(200, f.read_bytes(), "image/jpeg")
            return self.send(404, "{}")
        if url.path == "/pt.mjpg":
            return self.stream()
        self.send(404, "{}")

    def stream(self):
        """조준 콘솔의 팬틸트 실시간 화면을 그대로 전달 (콘솔은 127.0.0.1 에만 열려 있다)"""
        if not PTC.connected:
            return self.send(503, "{}")
        try:
            r = urllib.request.urlopen(PTC.url + "/pt.mjpg", timeout=5.0)
        except Exception:                                      # noqa: BLE001
            return self.send(503, "{}")
        self.send_response(200)
        self.send_header("Content-Type", r.headers.get("Content-Type") or "multipart/x-mixed-replace; boundary=f")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while PTC.connected:
                b = r.read1(65536)
                if not b:
                    break
                self.wfile.write(b)
        except Exception:                                      # noqa: BLE001  보는 쪽이 닫으면 끝
            pass
        finally:
            r.close()

    def do_POST(self):
        if not self.allowed():
            return self.send(403, "{}")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            p = urllib.parse.urlparse(self.path).path
            if p == "/api/case":
                with LOCK:
                    ST.update(case=2 if int(body.get("case")) == 2 else 1, sel=None, phase2="A")
                    ev("case", "케이스 %d" % ST["case"])
                return self.js(True)
            if p == "/api/ahead":
                with LOCK:
                    ST["ahead"] = min(2.5, max(0.0, float(body.get("m"))))
                    ev("ahead", "케이스 1 정지 지점: 표지 %.1f m 앞" % ST["ahead"], ahead=ST["ahead"])
                return self.js(True)
            if p == "/api/auto":
                with LOCK:
                    ST["auto"] = bool(body.get("on"))
                    ev("auto", "자동 기억 %s" % ("켬" if ST["auto"] else "끔"))
                return self.js(True)
            if p == "/api/pick":
                return self.js(*op_pick(float(body["u"]), float(body["v"]), body.get("seq")))
            if p == "/api/forget":
                with LOCK:
                    ev("forget", "기억 지움 (이 자리는 자동으로 다시 잡지 않습니다)")
                    if ST["target"] is not None:
                        ST["done"].append(list(ST["target"]["uv"]))
                    ST.update(target=None, sel=None, phase2="A", cand=None)
                return self.js(True)
            if p == "/api/label":
                ev("label", "표시: %s" % str(body.get("label"))[:20], label=str(body.get("label"))[:40], photo=(ST.get("photo") or {}).get("name"))
                return self.js(True)
            if p == "/api/done":
                return self.js(*op_done(body.get("label")))
            if p == "/api/sim":
                return self.js(*sim_op(body)) if SIM is not None else self.js(False, "시뮬 모드가 아닙니다")
            if p == "/api/use":
                ok, msg = op_use()
                if not ok:
                    ev("refused", "거부(위치추정에 넘기기): %s" % msg, op="use", why=msg)
                return self.js(ok, msg)
            if p == "/api/pt":
                ok, msg = pt_op(body)
                if not ok:
                    ev("refused", "거부(%s): %s" % (body.get("op"), msg), op=body.get("op"), why=msg)
                return self.js(ok, msg)
            self.send(404, "{}")
        except Exception as ex:                                # noqa: BLE001
            self.js(False, "%s: %s" % (type(ex).__name__, ex))


def shutdown(*_):
    if PTC.connected and not C["fake_console"]:
        PTC._disconnect()
    os._exit(0)


def main():
    global GEO, SIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--arm-s", type=float, default=21600.0)
    ap.add_argument("--pan-lim", nargs=2, type=float, default=list(aimgeo.PAN_LIM))
    ap.add_argument("--tilt-lim", nargs=2, type=float, default=list(aimgeo.TILT_LIM))
    ap.add_argument("--sim", action="store_true", help="시뮬 모드: 지난 실험 기록을 재생하고 팬틸트는 가짜 콘솔로 대신한다(장치·운영 폴더를 건드리지 않는다)")
    ap.add_argument("--sim-root", action="append", default=[], help="실험 폴더들이 들어 있는 곳 (여러 번 줄 수 있다)")
    ap.add_argument("--scene-rec-root", action="append", default=[], help="장면 기록기 폴더(깊이 영상)가 들어 있는 곳")
    ap.add_argument("--sim-out", default=None, help="시뮬 기록을 둘 곳 (실제 기록과 섞이지 않게 따로)")
    a = ap.parse_args()
    os.nice(10)
    if a.sim:
        import tempfile
        import fake_console as FC
        import sim_player as SP
        ds = SP.find_datasets(a.sim_root, a.scene_rec_root)
        if not ds:
            sys.exit("시뮬에 쓸 기록을 찾지 못했습니다 (--sim-root)")
        real = C["O"] / "runtime" / "ctl" / "experiment.json"          # 로봇에서 돌릴 때만 있다: 진짜 실험이 시작되면 재생을 멈춘다
        world = Path(tempfile.mkdtemp(prefix="plate_site_sim_"))
        out = Path(a.sim_out or (Path.home() / "sci_r42_work" / "_local_runs" / "plate_site_sim"))
        C.update(O=world / "O", hub=world / "hub", S=out, fake_console=True)
        SIM = SP.Player(ds, world, real_exp=real if real.exists() else None)
        SIM.on_change = sim_changed
        SIM.write()
    p_out().mkdir(parents=True, exist_ok=True)
    GEO = aimgeo.Geo(pan_lim=a.pan_lim, tilt_lim=a.tilt_lim)
    if a.sim:
        FC.start(C["pt_port"], p_out() / "console_sim", a.pan_lim, a.tilt_lim, render=SIM.render)
        threading.Thread(target=SIM.loop, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=poll_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    threading.Timer(a.arm_s, shutdown).start()
    with open(p_out() / "site.log", "a") as f:
        f.write("%s serving on :%d pid %d for %.0f s, limits pan %s tilt %s%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), a.port, os.getpid(), a.arm_s, a.pan_lim, a.tilt_lim,
                                                                                   " SIM %d datasets" % len(SIM.list) if SIM else ""))
    srv.serve_forever()


if __name__ == "__main__":
    main()
