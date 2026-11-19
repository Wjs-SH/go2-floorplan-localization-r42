#!/usr/bin/env python3
"""e2_feed_posefix.py — LIVE2 구조점(e2_feed.py)은 선배 규칙 그대로 두고, 표본 자세만 실시간으로 보정한다(2026-11-13, esy).
보정 = scripts/rt_pose_refine2.py(2026-10-09)의 증분판: 최근 8초 구조점(최근 2.4초 제외)으로 만든 5 cm 칸 지도에 이번 표본 구조점을 2D ICP 로 맞추고,
한 번에 최대 25 cm·2°, 그중 60 % 만 반영한다. 보정은 다음 표본까지 유지된다(누적). 오류가 나면 그 표본은 FAST-LIO 자세 그대로.
odom_link.csv 에 보정된 자세가 들어간다(탐색·추적이 같은 자세를 쓴다). FAST-LIO 원래 자세는 lio_state·허브에 그대로 남는다."""
import collections
import math
import time

import numpy as np

import e2_feed as EF

BETA = math.radians(228.0)
WIN, MAXT, MAXR, GAIN, ITERS, CORR, TRIM, VOX, GAP = 8.0, 0.25, math.radians(2.0), 0.6, 8, 0.35, 0.8, 0.05, 2.4
KEY = 1 << 21


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseRefiner:
    def __init__(self):
        self.vox = {}
        self.pend = collections.deque()
        self.tree = None
        self.M = None
        self.raw = None
        self.cor = None
        self.stats = collections.Counter()
        self.last = None

    @staticmethod
    def world(p, x, y, u):
        th = u + BETA
        c, s = math.cos(th), math.sin(th)
        return np.c_[x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]]

    def step(self, t, x, y, u, p):
        from scipy.spatial import cKDTree
        if self.raw is None:
            cx, cy, cu = x, y, u
        else:
            rx, ry, ru = self.raw
            cx, cy, cu = self.cor
            a = cu - ru
            ca, sa = math.cos(a), math.sin(a)
            dx, dy = x - rx, y - ry
            cx += ca * dx - sa * dy
            cy += sa * dx + ca * dy
            cu += _wrap(u - ru)
        gx = gy = gu = 0.0
        if self.tree is not None and len(self.M) > 200 and len(p) >= 40:
            px, py, pu = cx, cy, cu
            for _ in range(ITERS):
                Wp = self.world(p, px, py, pu)
                d, i = self.tree.query(Wp, distance_upper_bound=CORR)
                ok = np.isfinite(d)
                if ok.sum() < 40:
                    break
                dd = d[ok]
                thr = max(np.quantile(dd, TRIM), 0.05)
                o2 = ok.copy(); o2[ok] = dd <= thr
                if o2.sum() < 40:
                    break
                A, Bm = Wp[o2], self.M[i[o2]]
                ma, mb = A.mean(0), Bm.mean(0)
                H = (A - ma).T @ (Bm - mb)
                U_, _, Vt = np.linalg.svd(H)
                Rm = Vt.T @ U_.T
                if np.linalg.det(Rm) < 0:
                    Vt[1] *= -1; Rm = Vt.T @ U_.T
                dth = math.atan2(Rm[1, 0], Rm[0, 0])
                tr = mb - Rm @ ma
                nx, ny = Rm @ np.array([px, py]) + tr
                nu = pu + dth
                if abs(nu - cu) > MAXR or math.hypot(nx - cx, ny - cy) > MAXT:
                    break
                conv = math.hypot(nx - px, ny - py) < 1e-4 and abs(dth) < 1e-5
                px, py, pu = nx, ny, nu
                if conv:
                    break
            gx, gy, gu = px - cx, py - cy, pu - cu
            sc = min(1.0, MAXT / max(math.hypot(gx, gy), 1e-9), MAXR / max(abs(gu), 1e-9)) * GAIN
            gx, gy, gu = gx * sc, gy * sc, gu * sc
            cx += gx; cy += gy; cu += gu
            self.stats["icp"] += 1
        Wc = self.world(p, cx, cy, cu)
        self.pend.append((t, Wc))
        while self.pend and t - self.pend[0][0] >= GAP:
            tm, P = self.pend.popleft()
            ij = np.floor(P / VOX).astype(np.int64)
            for (a_, b_), (X, Y) in zip(map(tuple, ij), P):
                self.vox[a_ * KEY + b_] = (X, Y, tm)
        if len(self.vox) > 400:
            for k in [k for k, v in self.vox.items() if t - v[2] > WIN]:
                del self.vox[k]
        self.M = np.array([(v[0], v[1]) for v in self.vox.values()]) if self.vox else None
        self.tree = cKDTree(self.M) if self.M is not None and len(self.M) > 200 else None
        self.raw, self.cor = (x, y, u), (cx, cy, cu)
        self.last = dict(dx_cm=round(100 * math.hypot(gx, gy), 2), dyaw_deg=round(math.degrees(gu), 3),
                         total_cm=round(100 * math.hypot(cx - x, cy - y), 1), total_deg=round(math.degrees(_wrap(cu - u)), 2))
        return cx, cy, _wrap(cu)


class PoseFixE2Feed(EF.E2Feed):
    """E2Feed 와 입출력이 같다. _sample 만 바꿔 store.add 에 보정된 자세를 넣는다(구조점·먼 점은 선배 규칙 그대로)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ref = PoseRefiner()
        self.pose_ms = collections.deque(maxlen=200)
        self.pose_fail = 0

    def new_session(self, why, first_stamp_ns):
        super().new_session(why, first_stamp_ns)
        self.ref = PoseRefiner()
        self.session["e2_posefix"] = dict(module="e2_feed_posefix.py (esy 2026-11-13)", win_s=WIN, gap_s=GAP, max_cm=100 * MAXT, max_deg=math.degrees(MAXR), gain=GAIN)
        import json
        (self.root / "session.json").write_text(json.dumps(self.session))

    def _sample(self, scans, poses, store):
        t0 = time.perf_counter()
        tgt_ns = scans[-1][0]
        if any(p is None for p in poses):
            self.stats['no_pose'] += 1
            return None
        p_t, q_t = poses[-1]
        R_t = EF.quat_to_R(q_t)
        pw = np.vstack([xyz for _, xyz in scans]).astype(np.float64)
        p_imu = (pw - p_t) @ R_t
        p_lidar = (p_imu - self.T_ext) @ self.R_ext
        pb = p_lidar @ self.R_bl.T + self.t_bl
        cells, far, st = EF.structure_proxy(pb)
        if len(cells) < 20:
            self.stats['too_few'] += 1
            return None
        x, y, yaw = EF.odom_link(p_t, q_t)
        t1 = time.perf_counter()
        try:
            x, y, yaw = self.ref.step(tgt_ns / 1e9, x, y, yaw, cells.astype(np.float64))
        except Exception as e:                    # 보정이 넘어져도 표본은 FAST-LIO 자세로 낸다
            self.pose_fail += 1
            self.pose_last_error = repr(e)
        self.pose_ms.append(1000 * (time.perf_counter() - t1))
        with self.lock:
            if store is not self.store:
                return None
            sid = store.add(tgt_ns / 1e9, x, y, yaw, cells, np.zeros(len(cells), np.int8), far)
        ms = 1000 * (time.perf_counter() - t0)
        self.stats['sample_ms'].append(ms)
        if sid is not None:
            self.stats['samples'] += 1
            self.stats['last'] = dict(sid=sid, stamp=tgt_ns / 1e9, wall=self.clock(), ms=round(ms, 1), **st)
        return sid

    def summary(self):
        s = super().summary()
        pm = list(self.pose_ms)
        s["posefix"] = dict(n=len(pm), ms_p50=round(float(np.median(pm)), 1) if pm else None, ms_p95=round(float(np.percentile(pm, 95)), 1) if pm else None,
                            fail=self.pose_fail, icp=int(self.ref.stats.get("icp", 0)), last=self.ref.last, last_error=getattr(self, "pose_last_error", None))
        return s
