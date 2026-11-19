#!/usr/bin/env python3
"""sim_player.py — 지난 실험 기록을 '지금 돌고 있는 실험'처럼 다시 흘려 주는 재생기 (명판 재촬영 사이트의 시뮬 모드, 2026-11-13).

사이트는 실험 중에 파일 네 가지를 읽는다(실험 상태, 센서 허브, SigLIP 상태, FAST-LIO 상태 기록). 재생기는 지난 실험의 기록으로 그 파일들을
임시 폴더에 같은 형식으로 써 준다. 사이트 코드는 실제 실험 때와 같은 경로로 돈다.
쓰는 기록(실험 폴더 archive/observe_run): scenes.jsonl(장면마다 SigLIP 영역·점수·거리), events/scene_*.jpg(그 장면의 고정 카메라 사진, 1초마다),
  live2_detection_pose.jsonl(그 장면 시각의 FAST-LIO 위치), live2_stop_requests.jsonl(E5 정지 요청), fastlio_Log/mat_pre.txt(몸 기울기).
  장면 기록기(scene_rec)가 켜져 있던 실험은 깊이 영상도 쓴다(없으면 깊이 0 — 상자 밖을 눌러 고르는 것은 안 된다).
실제와 다른 점
  * 위치는 1초 간격 기록 사이를 직선으로 이은 것이다.
  * 일시정지하면 로봇이 그 자리에 멈춘 것으로 친다(기록에서 실제로 멈춘 곳이 아니어도 '여기서 멈췄다면'을 볼 수 있다).
  * 팬틸트 사진은 고정 카메라 사진을 조준 방향으로 잘라 확대한 것이다(render). 고정 카메라 시야 밖은 보여 줄 수 없다.
  * 영역의 통과 여부는 기록의 fired·like 로 흉내 낸다(실제 상태 파일의 state 값과 같다는 보장은 없다)."""
import bisect
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

K0 = [606.7977294921875, 0.0, 310.4879150390625, 0.0, 606.7450561523438, 240.51687622070312, 0.0, 0.0, 1.0]
PT_HALF_FOV_DEG = 12.15            # 팬틸트 카메라 가로 반시야(선배 추정 0.01266 도/px x 1920 px / 2)


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for ln in open(p):
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def jread(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def jwrite(p, obj):
    """읽는 쪽이 반쯤 쓰인 파일을 보지 않게 임시 파일에 쓴 뒤 바꿔치기"""
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Dataset:
    def __init__(self, exp_dir, scene_rec=None):
        self.dir = Path(exp_dir)
        self.id = self.dir.name
        A = self.dir / "archive" / "observe_run"
        poses = {r.get("scene_t"): r["odom_raw"] for r in jl(A / "live2_detection_pose.jsonl") if (r.get("odom_raw") or {}).get("x") is not None}
        self.scenes = []
        for s in jl(A / "scenes.jsonl"):
            od, img = poses.get(s.get("t")), A / "events" / ("scene_%07.1f.jpg" % float(s.get("t") or 0))
            if od is None or not img.exists():
                continue
            regs = [dict(box=[float(v) for v in g["box"][:4]], sim=float(g.get("sim") or 0.0), z=float(g["z"]),
                         state="VERIFIED" if (g.get("fired") and g.get("like")) else "SEARCH", display=None)
                    for g in (s.get("regions") or []) if len(g.get("box") or []) >= 4 and g.get("z")]
            self.scenes.append(dict(t=float((s.get("frame") or {}).get("header_stamp") or s["stamp"]), x=float(od["x"]), y=float(od["y"]), yaw=float(od["yaw"]),
                                    thr=s.get("thr"), regions=regs, img=str(img)))
        self.scenes.sort(key=lambda r: r["t"])
        self.ts = [r["t"] for r in self.scenes]
        self.ok = len(self.scenes) >= 10
        self.t0, self.t1 = (self.ts[0], self.ts[-1]) if self.ok else (0.0, 0.0)
        self.requests = sorted((dict(t=float(r["detection_timestamp"]), box=r.get("bbox"), z=r.get("depth_m"), sim=r.get("sim"))
                                for r in jl(A / "live2_stop_requests.jsonl") if r.get("detection_timestamp") and r.get("bbox") and r.get("depth_m")), key=lambda r: r["t"])
        self.K = ((jread(self.dir / "archive" / "hub_index.json" / "index.json") or {}).get("caminfo") or {}).get("k") or K0
        self.pt, self.pb = [], []                      # 몸 기울기: (시각, FAST-LIO 상태 기록의 B)
        first = None
        for f in sorted(self.dir.glob("fastlio_*.out")):
            m = re.search(r"\[FLIO-06\] timestamp field.*?min=([0-9.]+)", f.read_text(errors="replace"))
            if m:
                first = float(m.group(1))
                break
        mp = self.dir / "archive" / "fastlio_Log" / "mat_pre.txt"
        if first is not None and mp.exists():
            for ln in open(mp):
                p = ln.split()
                if len(p) >= 4:
                    try:
                        self.pt.append(first + float(p[0]))
                        self.pb.append(float(p[2]))
                    except ValueError:
                        pass
        self.depth = []                                # 장면 기록기의 깊이: (시각, 파일)
        if scene_rec and (Path(scene_rec) / "frames.jsonl").exists():
            self.depth = sorted((float(r["stamp"]), str(Path(scene_rec) / ("%06d_depth.png" % r["n"]))) for r in jl(Path(scene_rec) / "frames.jsonl"))
        self.dts = [d[0] for d in self.depth]

    def label(self):
        hot = sum(1 for s in self.scenes if any(g["state"] == "VERIFIED" for g in s["regions"]))
        return "%s · %d초 · 정지 요청 %d회 · 표지 후보 장면 %d%s" % (self.id[1:], self.t1 - self.t0, len(self.requests), hot, " · 깊이 있음" if self.depth else "")

    def scene_index(self, t):
        return max(0, bisect.bisect_right(self.ts, t) - 1)

    def pose(self, t):
        i = self.scene_index(t)
        a = self.scenes[i]
        if t <= a["t"] or i + 1 >= len(self.scenes):
            return a["x"], a["y"], a["yaw"]
        b = self.scenes[i + 1]
        k = min(1.0, (t - a["t"]) / max(1e-6, b["t"] - a["t"]))
        return a["x"] + k * (b["x"] - a["x"]), a["y"] + k * (b["y"] - a["y"]), a["yaw"] + k * wrap(b["yaw"] - a["yaw"])

    def pitch_b(self, t):
        if not self.pt:
            return 0.0
        i = min(len(self.pt) - 1, max(0, bisect.bisect_left(self.pt, t)))
        return self.pb[i]

    def depth_at(self, t):
        if not self.depth:
            return None
        i = min(len(self.dts) - 1, max(0, bisect.bisect_left(self.dts, t)))
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.dts)), key=lambda j: abs(self.dts[j] - t))
        if abs(self.dts[best] - t) > 0.7:
            return None
        d = cv2.imread(self.depth[best][1], cv2.IMREAD_UNCHANGED)
        return d if d is not None and d.dtype == np.uint16 and d.shape[:2] == (480, 640) else None


def find_datasets(roots, scene_rec_roots=()):
    """roots 아래의 실험 폴더(X*/archive/observe_run/scenes.jsonl 이 있는 것)를 찾는다. 최근 것부터."""
    out, seen = [], set()
    for r in roots:
        for sc in sorted(Path(r).glob("**/archive/observe_run/scenes.jsonl")):
            d = sc.parents[2]
            if d.name in seen:
                continue
            rec = next((Path(q) / d.name for q in scene_rec_roots if (Path(q) / d.name / "frames.jsonl").exists()), None)
            ds = Dataset(d, rec)
            if ds.ok:
                out.append(ds)
                seen.add(d.name)
    return sorted(out, key=lambda d: d.id, reverse=True)


class Player:
    def __init__(self, datasets, world, real_exp=None):
        self.list, self.ds = datasets, datasets[0]
        self.w = Path(world)
        for d in ("O/runtime/ctl", "O/runtime/e5_shm", "O/runtime/fastlio/Log", "hub"):
            (self.w / d).mkdir(parents=True, exist_ok=True)
        self.t, self.playing, self.speed = self.ds.t0, False, 1.0
        self.lock = threading.RLock()
        self.seq, self.gen, self.scene_i, self.frame = 1000, 0, None, None
        self.real_exp, self.blocked = real_exp, None
        self.on_change = None                           # 기록을 바꾸거나 건너뛸 때 사이트가 기억을 정리하도록 부르는 함수

    # ---- 조작
    def select(self, ds_id):
        with self.lock:
            for d in self.list:
                if d.id == ds_id:
                    self.ds, self.t, self.playing, self.scene_i, self.gen = d, d.t0, False, None, self.gen + 1
                    if self.on_change:
                        self.on_change("dataset")
                    return True
        return False

    def seek(self, t_rel):
        with self.lock:
            self.t = min(self.ds.t1, max(self.ds.t0, self.ds.t0 + float(t_rel)))
            self.scene_i, self.gen = None, self.gen + 1
            if not self.playing:
                self.t = self.ds.ts[self.ds.scene_index(self.t)]
            if self.on_change:
                self.on_change("seek")

    def play(self, on):
        with self.lock:
            self.playing = bool(on) and self.blocked is None
            if not self.playing:                        # 멈출 때는 그 장면을 찍은 시각으로 맞춘다: 화면·영역·위치가 같은 순간이 되게
                self.t = self.ds.ts[self.ds.scene_index(self.t)]

    def next_request(self):
        with self.lock:
            nxt = [r["t"] for r in self.ds.requests if r["t"] > self.t + 7.0]
            if not nxt:
                return False
            self.seek(nxt[0] - 6.0 - self.ds.t0)
            self.play(True)
            return True

    def view(self):
        with self.lock:
            ds = self.ds
            return dict(id=ds.id, t=round(self.t - ds.t0, 1), dur=round(ds.t1 - ds.t0, 1), playing=self.playing, speed=self.speed, blocked=self.blocked,
                        scene=self.scene_i, n_scenes=len(ds.scenes), depth=bool(ds.depth), requests=[round(r["t"] - ds.t0, 1) for r in ds.requests],
                        datasets=[dict(id=d.id, label=d.label()) for d in self.list])

    # ---- 세계 쓰기
    def write(self):
        now = time.time()
        with self.lock:
            ds, t = self.ds, self.t
            rate = self.speed if self.playing else 0.0
            rows = []
            for i in range(61):                          # 최근 6초, 10 Hz. 멈춰 있으면 같은 위치가 이어진다
                back = (60 - i) * 0.1
                x, y, yaw = ds.pose(t - back * rate)
                rows.append([now - back, x, y, yaw])
            i = ds.scene_index(t)
            sc = ds.scenes[i]
            sc_wall = now - ((t - sc["t"]) / rate if rate > 0 else 0.0)
            if i != self.scene_i:
                img = cv2.imread(sc["img"])
                if img is None or img.shape[:2] != (480, 640):
                    img = np.zeros((480, 640, 3), np.uint8) if img is None else cv2.resize(img, (640, 480))
                depth = ds.depth_at(sc["t"])
                if depth is None:
                    depth = np.zeros((480, 640), np.uint16)
                (self.w / "hub" / "slot_00.color").write_bytes(img.tobytes())
                (self.w / "hub" / "slot_00.depth").write_bytes(depth.tobytes())
                self.scene_i, self.frame, self.seq = i, img, self.seq + 1
            jwrite(self.w / "hub" / "index.json", dict(odom=rows, odom_last_rx_wall=now, caminfo=dict(k=ds.K, w=640, h=480),
                                                      frames=[dict(seq=self.seq, stamp=sc_wall, slot=0, w=640, h=480, depth_stamp=sc_wall)]))
            reqs = []
            for r in ds.requests:
                if t - 20.0 < r["t"] <= t:
                    w = now - ((t - r["t"]) / rate if rate > 0 else 0.0)
                    reqs.append(dict(observation_id="%s_%d_g%d" % (ds.id, r["t"], self.gen), wall=w, detection_timestamp=w, bbox=r["box"], depth_m=r["z"], sim=r["sim"]))
            jwrite(self.w / "O/runtime/e5_shm/status.json", dict(siglip="UP", scene=dict(stamp=sc_wall, thr=sc["thr"], regions=sc["regions"]),
                                                                  live2_stop_request=dict(last=reqs)))
            (self.w / "O/runtime/fastlio/Log/mat_pre.txt").write_text("0 0 %.3f 0 0 0 0\n" % ds.pitch_b(t))
            jwrite(self.w / "O/runtime/ctl/experiment.json", dict(state="RUNNING", exp_id="SIM_" + ds.id))

    def loop(self, dt=0.2):
        last = time.time()
        while True:
            now = time.time()
            with self.lock:
                if self.real_exp is not None:            # 로봇에서 돌릴 때: 진짜 실험이 시작되면 재생을 멈춘다
                    st = (jread(self.real_exp) or {}).get("state")
                    self.blocked = "진짜 실험이 도는 중이라 시뮬을 멈췄습니다" if st in ("RUNNING", "PARTIAL", "PAUSED", "STARTING", "STOPPING") else None
                    if self.blocked:
                        self.playing = False
                if self.playing:
                    self.t += (now - last) * self.speed
                    if self.t >= self.ds.t1:
                        self.t, self.playing = self.ds.t1, False
            last = now
            try:
                self.write()
            except Exception as ex:                       # noqa: BLE001
                self.error = "%s: %s" % (type(ex).__name__, ex)
            time.sleep(dt)

    # ---- 가짜 팬틸트 카메라: 고정 카메라 사진을 조준 방향으로 잘라 확대
    def render(self, u, v, z, pan, tilt, width=1920):
        with self.lock:
            img = self.frame if self.frame is not None else np.zeros((480, 640, 3), np.uint8)
            fx = self.ds.K[0]
        w = 2.0 * fx * math.tan(math.radians(PT_HALF_FOV_DEG))
        h = w * 9.0 / 16.0
        s = width / w
        M = np.float32([[s, 0, -(u - w / 2.0) * s], [0, s, -(v - h / 2.0) * s]])
        out = cv2.warpAffine(img, M, (width, int(round(width * 9 / 16))), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 70, 70))
        k = width / 1920.0
        inside = -w / 2 < u < 640 + w / 2 and -h / 2 < v < 480 + h / 2
        txt = ("SIM  pan %.1f  tilt %.1f  (crop of the fixed camera)" % (pan, tilt)) if inside else ("SIM  pan %.1f  tilt %.1f  — outside the fixed camera view" % (pan, tilt))
        cv2.putText(out, txt, (int(30 * k), int(out.shape[0] - 30 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, (0, 0, 0), max(1, int(6 * k)), cv2.LINE_AA)
        cv2.putText(out, txt, (int(30 * k), int(out.shape[0] - 30 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, (0, 230, 255), max(1, int(2 * k)), cv2.LINE_AA)
        return out
