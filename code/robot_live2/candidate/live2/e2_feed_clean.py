#!/usr/bin/env python3
"""e2_feed_clean.py — LIVE2 구조점(e2_feed.py)에 정제를 얹는 덧붙임 모듈 (2026-11-13, esy).

선배 e2_feed.py 는 고치지 않는다. CleanE2Feed 는 E2Feed 를 이어받아, 표본을 만들 때(1.5초에 한 번) 점 → 칸 규칙만 바꾼다.
  1) 높이 띠를 기울어지는 몸체 z 대신 '그 표본의 점에서 맞춘 바닥면 위 높이'로 자르고, 맞춘 천장면 아래 0.25 m 안은 버린다  (바닥·천장 원)
  2) 먼 점(9~18 m)도 16 cm 칸 안에서 높이 차 0.30 m 이상, 곧 세로로 선 것만                                        (바닥·천장 원)
  3) 8 cm 칸의 중심 대신 칸에 든 점들의 평균 자리                                                                (선 두께)
  4) 따라오는 사람·움직이는 물체: a131_clean_live.LiveCleaner 의 3)·4) 규칙과 숫자 그대로, 표본 단위로             (사람)
전부 그 표본과 과거 자세·과거 표본만 쓴다. 정제 중 오류가 나면 그 표본은 선배 규칙 그대로 낸다(clean_stats["fallback"]).
쓰는 법: feed = CleanE2Feed(store_root, extrinsic, ...)   # E2Feed 와 같은 인자. person=False 면 4) 끔, move=False 면 움직이는 물체만 끔
이 파일의 함수들은 시험대 tb.py 에서 08_배포모듈_만들기.py 가 글자 그대로 옮긴 것이다(직접 고치지 말 것).
"""
import collections
import json
import math
import threading
import time

import numpy as np

import e2_feed as EF

FLOOR_Z0 = -0.32          # 선배 띠(몸체 z -0.15~1.6 m)를 '바닥 위 높이'로 옮길 때의 기준: 서 있을 때 바닥의 몸체 z
CEIL_GAP = 0.25           # 천장면에서 이만큼 아래까지는 버린다
FAR_CELL, FAR_SPAN = 0.16, 0.3   # 먼 점: 이 크기 칸 안에서 높이 차가 이만큼 나야 세로로 선 것(벽)으로 본다


def fit_plane(p, z_lo, z_hi, r_near=3.0, r_max=8.0, min_n=300, max_tilt=0.15):
    """z = a x + b y + c 평면(몸체 좌표). 가까운 점의 높이 봉우리에서 시작해 잔차로 두 번 다듬는다. 못 맞추면 None."""
    r = np.hypot(p[:, 0], p[:, 1])
    m = (p[:, 2] > z_lo) & (p[:, 2] < z_hi) & (r < r_max)
    q, rq = p[m], r[m]
    if len(q) < min_n:
        return None
    if len(q) > 30000:
        k = len(q) // 30000
        q, rq = q[::k], rq[::k]
    zn = q[rq < r_near, 2]
    if len(zn) < min_n // 3:
        zn = q[:, 2]
    h, e = np.histogram(zn, bins=np.arange(z_lo, z_hi + 0.02, 0.02))
    c0 = 0.5 * (e[int(np.argmax(h))] + e[int(np.argmax(h)) + 1])
    coef = np.array([0.0, 0.0, c0])
    for tol, rr in ((0.12, r_near), (0.06, r_max), (0.04, r_max)):
        res = q[:, 2] - (q[:, 0] * coef[0] + q[:, 1] * coef[1] + coef[2])
        k = (np.abs(res) < tol) & (rq < rr)
        if k.sum() < min_n // 3:
            return None
        A = np.c_[q[k, 0], q[k, 1], np.ones(int(k.sum()))]
        coef = np.linalg.lstsq(A, q[k, 2], rcond=None)[0]
    if abs(coef[0]) > max_tilt or abs(coef[1]) > max_tilt:
        return None
    return coef


def heights(pb):
    """바닥 위 높이, 천장면 아래 여부, 바닥면 계수. 바닥을 못 맞추면 선배 띠와 같아지게 몸체 z 를 그대로 쓴다."""
    fl = fit_plane(pb, -0.9, 0.05)
    if fl is None:
        return pb[:, 2] - FLOOR_Z0, np.ones(len(pb), bool), None
    hgt = (pb[:, 2] - (pb[:, 0] * fl[0] + pb[:, 1] * fl[1] + fl[2])) / np.sqrt(1 + fl[0] ** 2 + fl[1] ** 2)
    ce = fit_plane(pb, fl[2] + 1.9, fl[2] + 3.4, r_near=15.0, r_max=15.0)
    ok = np.ones(len(pb), bool) if ce is None else ((pb[:, 0] * ce[0] + pb[:, 1] * ce[1] + ce[2]) - pb[:, 2] > CEIL_GAP)
    return hgt, ok, fl


def proxy_h(pb, hgt, ok_h, r=None, keep=None):
    """V3 규칙(바닥 위 높이 띠 + 먼 점은 세로로 선 것만 + 칸 중심 대신 점 평균). keep = 남길 점 표시(없으면 전부)."""
    CELL_M, MIN_HITS, MIN_SPAN, NEAR_M = EF.CELL_M, EF.MIN_HITS, EF.MIN_SPAN, EF.NEAR_M
    if r is None:
        r = np.hypot(pb[:, 0], pb[:, 1])
    if keep is not None:
        ok_h = ok_h & keep
    m = (hgt > EF.BAND_Z[0] - FLOOR_Z0) & (hgt < EF.BAND_Z[1] - FLOOR_Z0) & (r >= NEAR_M[0]) & (r < NEAR_M[1]) & ok_h
    q, qh = pb[m], hgt[m]
    cells = np.zeros((0, 2), np.float32)
    if len(q):
        ij = np.floor(q[:, :2] / CELL_M).astype(np.int64)
        key = ij[:, 0] * 2_000_003 + ij[:, 1]
        order = np.argsort(key, kind="stable")
        key, z, qs = key[order], qh[order], q[order]
        u, start, cnt = np.unique(key, return_index=True, return_counts=True)
        ok = (cnt >= MIN_HITS) & (np.maximum.reduceat(z, start) - np.minimum.reduceat(z, start) >= MIN_SPAN)
        cells = (np.c_[np.add.reduceat(qs[:, 0], start), np.add.reduceat(qs[:, 1], start)] / cnt[:, None])[ok].astype(np.float32)
    fm = (r >= EF.FAR_NEAR) & (r < EF.FAR_HI) & (hgt > EF.FAR_Z[0] - FLOOR_Z0) & (hgt < EF.FAR_Z[1] - FLOOR_Z0) & ok_h
    fp, fh = pb[fm], hgt[fm]
    if len(fp):
        ij = np.floor(fp[:, :2] / FAR_CELL).astype(np.int64)
        u, inv = np.unique(ij[:, 0] * 2_000_003 + ij[:, 1], return_inverse=True)
        zmax = np.full(len(u), -np.inf); zmin = np.full(len(u), np.inf)
        np.maximum.at(zmax, inv, fh); np.minimum.at(zmin, inv, fh)
        fp = fp[(zmax - zmin >= FAR_SPAN)[inv]]
    far = fp[:, :2].astype(np.float32)
    if len(far) > EF.FAR_CAP:
        far = far[np.linspace(0, len(far) - 1, EF.FAR_CAP).astype(int)]
    if len(far) < EF.FAR_KEEP:
        far = np.zeros((0, 2), np.float32)
    return cells, far


class LightPerson:
    """따라오는 사람(3)·움직이는 물체(4)를 표본 단위로. 과거 자세와 과거 표본만 쓴다(인과적).
    add_pose(t, x, y)        : 오돔이 올 때마다(자세만, 점은 안 만진다)
    apply(t, pose, pb, hgt)  : 표본 하나의 점(몸체 좌표, 3스캔 묶음)과 바닥 위 높이 → 지울 점 표시"""
    NBIN, GR, RES, HALF = 720, 0.10, 0.05, 9.6

    def __init__(self, merge=3, win=12.0, r_seed=0.15, r_grow=0.35, path_max=0.6, move=True, move_min_pts=20):
        from scipy import ndimage
        self.nd = ndimage
        self.merge, self.win, self.r_seed, self.r_grow, self.path_max = int(merge), win, r_seed, r_grow, path_max
        self.move, self.move_min_pts = move, move_min_pts
        self.poses = collections.deque()         # (t, x, y)
        self.hist = collections.deque()          # (t, x, y, yaw, 방위별 첫 장애물 거리)
        self.occ = collections.deque()           # (t, 찍힌 10 cm 칸 키)
        self.stats = collections.Counter()

    def add_pose(self, t, x, y):
        self.poses.append((t, x, y))
        while self.poses and t - self.poses[0][0] > self.win + 1.0:
            self.poses.popleft()

    def _path_dist(self, t, px, py, pw, poses=None):
        """과거 win 초 경로까지의 거리(몸체 좌표 5 cm 격자, 경로 둘레 path_max 만큼만). 지금 몸체 0.6 m 안의 경로는 뺀다.
        반환 (거리 격자, x0, y0) 또는 None."""
        sel = [(xx, yy) for (tt, xx, yy) in (self.poses if poses is None else poses) if t - self.win <= tt <= t - 0.3]
        if len(sel) < 2:
            return None
        P = np.array(sel)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
        if s[-1] < 1e-3:
            return None
        g = np.arange(0, s[-1] + 1e-9, 0.05)
        P = np.c_[np.interp(g, s, P[:, 0]), np.interp(g, s, P[:, 1])]
        P = P[np.hypot(P[:, 0] - px, P[:, 1] - py) > 0.6]
        if len(P) == 0:
            return None
        c, sn = math.cos(pw), math.sin(pw)
        dx, dy = P[:, 0] - px, P[:, 1] - py
        B = np.c_[c * dx + sn * dy, -sn * dx + c * dy]
        B = B[(np.abs(B) < self.HALF).all(1)]
        if len(B) == 0:
            return None
        pad = self.path_max + 2 * self.RES
        x0, y0 = B[:, 0].min() - pad, B[:, 1].min() - pad
        nx = int(math.ceil((B[:, 0].max() + pad - x0) / self.RES)) + 1
        ny = int(math.ceil((B[:, 1].max() + pad - y0) / self.RES)) + 1
        G = np.ones((nx, ny), bool)
        ij = np.floor((B - (x0, y0)) / self.RES).astype(int)
        G[ij[:, 0], ij[:, 1]] = False
        return self.nd.distance_transform_edt(G) * self.RES, x0, y0

    def apply(self, t, pose, pb, hgt, poses=None):
        px, py, pw = pose
        N = len(pb)
        r = np.hypot(pb[:, 0], pb[:, 1])
        kill = np.zeros(N, bool)
        # 3) 따라오는 사람: 과거 경로 위(0.15 m) 바닥+0.08~0.55 m 씨앗 → 0.35 m 키우되 경로 0.6 m 안만
        pd = self._path_dist(t, px, py, pw, poses)
        if pd is not None:
            dist, x0, y0 = pd
            nx, ny = dist.shape
            cand = np.flatnonzero((hgt > 0.08) & (hgt < 2.2) & (pb[:, 0] >= x0) & (pb[:, 1] >= y0)
                                  & (pb[:, 0] < x0 + nx * self.RES) & (pb[:, 1] < y0 + ny * self.RES))
            if cand.size:
                ij = np.floor((pb[cand, :2] - (x0, y0)) / self.RES).astype(int)
                d = dist[np.minimum(ij[:, 0], nx - 1), np.minimum(ij[:, 1], ny - 1)]
                hc = hgt[cand]
                seed = cand[(hc < 0.55) & (r[cand] > 0.5) & (d < self.r_seed)]
                if seed.size >= 8 * self.merge:
                    from scipy.spatial import cKDTree
                    SP = pb[seed, :2]
                    cnt = cKDTree(SP).query_ball_point(SP, 0.4, return_length=True)
                    strong = SP[cnt >= 8 * self.merge]
                    if len(strong):
                        gc = cand[(hc > 0.10) & (d < self.path_max)]
                        d2, _ = cKDTree(strong).query(pb[gc, :2], distance_upper_bound=self.r_grow)
                        kill[gc[d2 < self.r_grow]] = True
                        self.stats["person_pts"] += int((d2 < self.r_grow).sum()); self.stats["samples_with_person"] += 1
        if not self.move:
            self.stats["samples"] += 1
            return kill
        # 4) 움직이는 물체: 덩어리(1 m 이하)가 앞선 표본에서는 그 방위가 비어 있었고, 2~5초 전에도 그 자리가 비었으면 제거
        band = (hgt > 0.15) & (hgt < 2.0) & (r > 0.7) & (r < 12)
        minr = np.full(self.NBIN, np.inf)
        if band.any():
            np.minimum.at(minr, ((np.degrees(np.arctan2(pb[band, 1], pb[band, 0])) % 360) / 0.5).astype(int) % self.NBIN, r[band])
        cw, sw = math.cos(pw), math.sin(pw)
        if self.move and self.hist:
            lags = [e for e in self.hist if 1.0 <= t - e[0] <= 3.5]
            occ = [k for (tt, k) in self.occ if 2.0 <= t - tt <= 5.0]
            occ = np.unique(np.concatenate(occ)) if occ else np.zeros(0, np.int64)
            cidx = np.flatnonzero(band & ~kill & (r < 8))
            if cidx.size and lags:
                Q = pb[cidx, :2]
                ij = np.floor((Q + 8) / 0.15).astype(int)
                okq = (ij >= 0).all(1) & (ij < 107).all(1)
                Gm = np.zeros((107, 107), bool); Gm[ij[okq, 0], ij[okq, 1]] = True
                lab, nl = self.nd.label(Gm, structure=np.ones((3, 3)))
                pl = np.zeros(len(Q), int); pl[okq] = lab[ij[okq, 0], ij[okq, 1]]
                order = np.argsort(pl, kind="stable")
                bnd = np.searchsorted(pl[order], np.arange(1, nl + 2))
                for cl in range(1, nl + 1):
                    sel = order[bnd[cl - 1]:bnd[cl]]
                    if len(sel) < self.move_min_pts * self.merge:
                        continue
                    P2 = Q[sel]
                    if max(np.ptp(P2[:, 0]), np.ptp(P2[:, 1])) > 1.0:
                        continue
                    Wx = px + cw * P2[:, 0] - sw * P2[:, 1]; Wy = py + sw * P2[:, 0] + cw * P2[:, 1]
                    anyfree = False
                    for (te, xe, ye, we, minre) in lags:
                        ce, se = math.cos(we), math.sin(we)
                        dx, dy = Wx - xe, Wy - ye
                        ex, ey = ce * dx + se * dy, -se * dx + ce * dy
                        prev = minre[((np.degrees(np.arctan2(ey, ex)) % 360) / 0.5).astype(int) % self.NBIN]
                        known = np.isfinite(prev)
                        if known.sum() >= 10 * self.merge and np.mean(prev[known] > np.hypot(ex, ey)[known] + 0.5) >= 0.6:
                            anyfree = True
                            break
                    if not anyfree:
                        continue
                    u, v = np.floor(Wx / self.GR).astype(np.int64), np.floor(Wy / self.GR).astype(np.int64)
                    hit = np.zeros(len(u), bool)
                    for du in (-1, 0, 1):
                        for dv in (-1, 0, 1):
                            hit |= np.isin((u + du) * 2_000_003 + (v + dv), occ)
                    if hit.mean() >= 0.3:
                        self.stats["moving_static_skip"] += 1
                        continue
                    kill[cidx[sel]] = True
                    self.stats["moving_pts"] += len(sel); self.stats["moving_blobs"] += 1
        # 이력(이 표본): 방위별 첫 장애물 거리 + 찍힌 칸(셋 중 하나만 뽑아도 충분)
        self.hist.append((t, px, py, pw, minr))
        while self.hist and t - self.hist[0][0] > 4.0:
            self.hist.popleft()
        b = np.flatnonzero(band)[::3]
        Wx = px + cw * pb[b, 0] - sw * pb[b, 1]; Wy = py + sw * pb[b, 0] + cw * pb[b, 1]
        self.occ.append((t, np.unique(np.floor(Wx / self.GR).astype(np.int64) * 2_000_003 + np.floor(Wy / self.GR).astype(np.int64))))
        while self.occ and t - self.occ[0][0] > 5.5:
            self.occ.popleft()
        self.stats["samples"] += 1
        return kill


class CleanE2Feed(EF.E2Feed):
    """E2Feed 와 입출력이 같다. 표본을 만들 때 부르는 structure_proxy 자리에 정제 규칙을 끼운다."""

    def __init__(self, *a, person=True, move=True, **k):
        super().__init__(*a, **k)
        self._person, self._move = bool(person), bool(move)
        self.lp = LightPerson(merge=self.merge, move=self._move) if self._person else None
        self.T0inv = None
        self._ctx = None
        self.clean_stats = collections.Counter()
        self._orig_proxy = EF.structure_proxy
        EF.structure_proxy = self._proxy          # E2Feed._sample 이 부르는 자리(이 프로세스에 feed 는 하나)

    def new_session(self, why, first_stamp_ns):
        super().new_session(why, first_stamp_ns)
        self.T0inv = None
        if self._person:
            self.lp = LightPerson(merge=self.merge, move=self._move)
        self.session["e2_clean"] = dict(module="e2_feed_clean.py (esy 2026-11-13)", person=self._person, move=self._move)
        (self.root / "session.json").write_text(json.dumps(self.session))      # 이 실험이 정제를 켜고 돌았다는 표시

    def summary(self):
        s = super().summary()
        s["clean"] = dict(self.clean_stats, last_ms=round(getattr(self, "clean_ms", 0.0), 1), person=self._person, move=self._move,
                          last_error=getattr(self, "clean_last_error", None))
        return s

    def _level(self, p, q):
        """첫 자세의 몸체 좌표(서 있을 때 z 가 위)에서 본 몸체의 (x, y, yaw). FAST-LIO 세계 z 는 위가 아닐 수 있어 그대로 못 쓴다."""
        Rt = EF.quat_to_R(q)
        T = np.eye(4)
        T[:3, :3] = Rt @ self.R_ext @ self.R_bl.T
        T[:3, 3] = np.asarray(p, float) + Rt @ (self.T_ext - self.R_ext @ self.R_bl.T @ self.t_bl)
        if self.T0inv is None:
            self.T0inv = np.linalg.inv(T)
        Tr = self.T0inv @ T
        return float(Tr[0, 3]), float(Tr[1, 3]), float(math.atan2(Tr[1, 0], Tr[0, 0]))

    def on_odom(self, stamp_ns, p, q, session_changed=False):
        super().on_odom(stamp_ns, p, q, session_changed)
        if self.lp is not None:
            with self.lock:
                x, y, _ = self._level(np.asarray(p, float), np.asarray(q, float))
                self.lp.add_pose(int(stamp_ns) / 1e9, x, y)

    def _sample(self, scans, poses, store):
        self._ctx = None if poses[-1] is None else (scans[-1][0], poses[-1])
        return super()._sample(scans, poses, store)

    def _proxy(self, pb):
        t0 = time.perf_counter()
        try:
            pb = pb[np.isfinite(pb).all(1)]
            hgt, okc, fl = heights(pb)
            keep, removed = None, 0
            if self.lp is not None and self._ctx is not None:
                ns, (p, q) = self._ctx
                with self.lock:
                    pose = self._level(p, q)
                    hist = list(self.lp.poses)
                kill = self.lp.apply(ns / 1e9, pose, pb, hgt, poses=hist)
                keep, removed = ~kill, int(kill.sum())
            cells, far = proxy_h(pb, hgt, okc, keep=keep)
            self.clean_stats["samples"] += 1; self.clean_stats["removed_pts"] += removed
            self.clean_stats["floor_fit_fail"] += int(fl is None)
            self.clean_ms = 1000 * (time.perf_counter() - t0)
            return cells, far, dict(n_in=int(len(pb)), n_band=-1, n_cells=int(len(cells)), n_far=int(len(far)), clean="ON",
                                    clean_removed=removed, clean_floor_fit=fl is not None)
        except Exception as e:                   # 정제가 넘어져도 구조점은 끊기지 않게: 이 표본은 선배 규칙 그대로
            self.clean_stats["fallback"] += 1
            self.clean_last_error = repr(e)
            c, f, st = self._orig_proxy(pb)
            return c, f, dict(st, clean="FALLBACK")
