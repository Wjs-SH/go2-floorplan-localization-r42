#!/usr/bin/env python3
"""aimgeo.py — 명판 재촬영 사이트의 조준 기하 (계산만, 장치에 아무것도 보내지 않는다. 학생 스테이징 2026-11-13).

조준식은 사용자 보정 코드 pantilt_aim.Aim 의 3차원 식(point_to_angles)을 그대로 쓴다. LIVE2 의 e6_core.body_to_pixel -> calibrated_aim 과 같은 값이며
(test_site.py 에서 확인), 표지가 고정 카메라 옆으로 가서 화소로 표현할 수 없는 방향도 계산할 수 있다.
가동 범위 기본값은 2026-09-19 실측(팬 20~180, 틸트 100~180; PANTILT_FIELD_GUIDE_20260919.md 4절)이다. 조준 콘솔은 목표로 가기 전에
두 축을 lash(5도)만큼 물렀다가 다가가므로(aim_console.move_to) 그 예비 위치도 범위 안이어야 한다."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

S = Path(os.environ.get("PLATE_SITE_CODE") or (Path.home() / "sci_ws_student_staging_20261109"))     # candidate/ 가 들어 있는 폴더 (노트북 시뮬은 사본 폴더)
for _p in (S / "candidate" / "live2", S / "candidate" / "external" / "sci_event" / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import e6_core as G                      # noqa: E402  선배 E6 기하(읽기만): 몸체<->평면 좌표, 정지 판정
import pantilt_aim as PA                 # noqa: E402  사용자 보정 식(읽기만)

PAN_LIM, TILT_LIM, LASH = (20.0, 180.0), (100.0, 180.0), 5.0


class Geo:
    def __init__(self, calib=None, pan_lim=PAN_LIM, tilt_lim=TILT_LIM, lash=LASH):
        self.calib = Path(calib or G.PANTILT_CALIB)
        self.A = PA.Aim(json.loads(self.calib.read_text(encoding="utf-8")), fill_missing=True)
        self.pan_lim, self.tilt_lim, self.lash = (float(pan_lim[0]), float(pan_lim[1])), (float(tilt_lim[0]), float(tilt_lim[1])), float(lash)
        self.ext = G.CAMERA_EXTRINSICS
        self.home = (float(self.A.pan_zero), float(self.A.tilt_zero))
        self._cache = {}

    # ---- 몸체 기준 점 -> 고정 카메라 축 기준 벡터 (e6_core.body_to_pixel 과 같은 회전). pitch_up_deg: 몸이 그만큼 들린 자세(케이스 2)
    def rs_vec(self, fwd, left, up, pitch_up_deg=0.0):
        e = self.ext
        f, l, u = fwd - e["lever_fwd_m"], left - e["lever_left_m"], up - e["height_m"]
        if pitch_up_deg:
            t = math.radians(pitch_up_deg)
            f, u = f * math.cos(t) + u * math.sin(t), -f * math.sin(t) + u * math.cos(t)
        w = math.radians(e["yaw_left_deg"])
        f, l = f * math.cos(w) + l * math.sin(w), -f * math.sin(w) + l * math.cos(w)
        p = math.radians(e["pitch_up_deg"])
        f, u = f * math.cos(p) + u * math.sin(p), -f * math.sin(p) + u * math.cos(p)
        return np.array([f, l, u])

    def servo(self, fwd, left, up, pitch_up_deg=0.0):
        """몸체 기준 점(앞, 왼쪽, 바닥에서 높이 m) -> 서보 명령과 실제 각도. 목표가 팬축에 너무 가까우면 ValueError."""
        pt = self.A.r_mount @ self.rs_vec(fwd, left, up, pitch_up_deg) - self.A.t_pan
        pan, tilt = self.A.point_to_angles(pt)
        sp, st = self.A.angles_to_servo(pan, tilt)
        return dict(pan_cmd=float(sp), tilt_cmd=float(st), pan_deg=math.degrees(pan), tilt_deg=math.degrees(tilt), dist=float(np.linalg.norm(pt)))

    def pixel_servo(self, u, v, z):
        """고정 카메라 화소 + 깊이 -> 서보 명령 (보정의 원래 쓰임새. 몸 기울기와 무관하다: 카메라와 팬틸트가 같은 몸체에 붙어 있다)"""
        sp, st = self.A.aim(float(u), float(v), float(z))
        return float(sp), float(st)

    def pixel(self, fwd, left, up, K, pitch_up_deg=0.0):
        """몸체 기준 점이 고정 카메라 화면 어디에 보일지 (u, v, 깊이) — 화면 뒤쪽이면 None"""
        f, l, u = self.rs_vec(fwd, left, up, pitch_up_deg)
        if f <= 0.05:
            return None
        return K[2] + K[0] * (-l) / f, K[5] + K[4] * (-u) / f, float(f)

    def check(self, pan_cmd, tilt_cmd):
        """-> (ok, why). 목표와 예비 위치(두 축 모두 목표 - lash) 가 가동 범위 안인가"""
        (p0, p1), (t0, t1) = self.pan_lim, self.tilt_lim
        for name, (p, t) in (("목표", (pan_cmd, tilt_cmd)), ("예비 위치", (pan_cmd - self.lash, tilt_cmd - self.lash))):
            if not (p0 <= p <= p1):
                return False, "%s 팬 %.1f 이 범위 %g~%g 밖" % (name, p, p0, p1)
            if not (t0 <= t <= t1):
                return False, "%s 틸트 %.1f 이 범위 %g~%g 밖" % (name, t, t0, t1)
        return True, None

    def virtual_pixel(self, pan_cmd, tilt_cmd, dist=None, far_m=200.0):
        """콘솔의 /aim 은 고정 카메라 화소 + 거리만 받는다. 이 서보 명령이 나오는 (u, v, z) 를 만든다: 팬틸트가 볼 방향 위의 한 점.
        dist(팬축에서 목표까지 거리)를 주면 그 거리의 점을 먼저 써 본다 — 그러면 화소가 목표가 고정 카메라에 실제로 보이는 자리가 된다.
        그 점이 고정 카메라 옆·뒤라 화소로 못 쓰면 먼 점을 쓴다(먼 점은 팬축·틸트축 간격의 영향도 없앤다).
        방향이 고정 카메라 옆 90도에 가까우면 ValueError. 되돌려 계산한 오차(도)도 준다."""
        A = self.A
        pan = math.radians((pan_cmd - A.pan_zero) / A.pan_gain)
        tilt = math.radians((tilt_cmd - A.tilt_zero) / A.tilt_gain)
        d = np.array([math.cos(tilt) * math.cos(pan), math.cos(tilt) * math.sin(pan), math.sin(tilt)])
        for s_, near in ([(float(dist), True)] if dist and dist > 0.3 else []) + [(far_m, False)]:
            p_rs = A.r_mount.T @ (s_ * d + A.t_pan)
            f, l, u = float(p_rs[0]), float(p_rs[1]), float(p_rs[2])
            if f < (0.3 if near else 1.0):
                continue
            uu, vv = int(round(A.cx - A.fx * l / f)), int(round(A.cy - A.fy * u / f))
            try:
                bp, bt = A.aim(uu, vv, f)
            except ValueError:
                continue
            err = max(abs(bp - pan_cmd), abs(bt - tilt_cmd))
            if err <= 0.15 or not near:
                return uu, vv, f, float(err)
        raise ValueError("고정 카메라 기준 옆 90도에 가까운 방향은 보낼 수 없다")

    def closest_fwd(self, left, up):
        """똑바로 걸어갈 때(표지가 옆으로 left m) 조준이 허용되는 가장 가까운 '앞 거리'. 없으면 None. 0.05 m 간격, 6 m 부터 줄여 가며 찾는다."""
        key = (round(left, 2), round(up, 2))
        if key in self._cache:
            return self._cache[key]
        best = None
        for i in range(161):
            fwd = 6.0 - i * 0.05
            try:
                s = self.servo(fwd, left, up)
            except ValueError:
                break
            if self.check(s["pan_cmd"], s["tilt_cmd"])[0]:
                best = fwd
            elif best is not None:
                break
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = best
        return best
