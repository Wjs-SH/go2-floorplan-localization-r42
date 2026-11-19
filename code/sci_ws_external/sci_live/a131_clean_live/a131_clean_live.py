#!/usr/bin/env python3
"""정제(이중반사·따라오는 사람·움직이는 물체)를 들어온 스캔만으로 하는 라이브판 (2026-10-08 성환이 요청).

녹화본 정제(`build_a131_clean_bag_v7.py`)와 같은 자리에 들어간다. 지운 점은 xyz=NaN 으로 바꿔 돌려주고,
점 개수·순서·ring·timestamp 는 그대로 둔다(A1 의 ring×column 격자가 유지돼야 한다).

옮긴 단계와 쓰는 데이터(전부 과거만 본다)
  2)  이중반사(near-echo)   : 스캔 하나   — 같은 방위 두 리턴이 gcut 이상 벌어지면 먼 쪽
  2b) 면 위 리턴 살리기     : 스캔 하나   — 먼 리턴이라도 같은 스캔 다른 점 8 cm 안이면 진짜 면으로 보고 유지(바닥 아래는 계속 제거)
  3)  따라오는 사람         : 과거 win 초 경로(자세만)
  4)  움직이는 물체         : 과거 1·2·3초 스캔의 방위별 최소거리 + 2~5초 전 점유 격자
옮기지 않은 것
  - tficp(스캔-국소지도 ICP): 점을 미는 양이 층마다 중앙 0.35~0.62 cm 로 작아 뺐다(이웃 스캔 어긋남 2.0→1.9 cm).
  - v6 거울상 규칙: 전달본은 전부 꺼 둔 상태로 만들었다(mirror_pts=0).
  - 유령점 평행선 마스크(apply_ghost_mask): 녹화 전체 통계 + 사람이 고른 선이라 자동 실시간판이 아니다.

쓰는 법
    from a131_clean_live import LiveCleaner
    C = LiveCleaner(static_matrix=T_base_lidar)
    C.observe(scan_id, x, y, z, ring, t_sec, interp)                 # 건너뛰는 스캔(이력만, 수 ms)
    x, y, z, i, ring, t = C.clean(scan_id, x, y, z, intensity, ring, t_sec, interp)   # 쓸 스캔(지운 점 = NaN)
  interp 는 a131_live 가 쓰는 PoseInterpolator(matrix_at(ns) → 4x4) 또는 pose=(x, y, yaw) 를 직접 줘도 된다.
"""
from __future__ import annotations

import collections
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

NBIN = 720          # 방위 0.5도 칸
GR = 0.10           # 점유 격자 칸 크기(m)
LAGS_SEC = (1.0, 2.0, 3.0)


class LiveCleaner:
    def __init__(self, static_matrix=None, gcut=0.5, no_person=False, no_move=False, no_support=False,
                 win=12.0, r_seed=0.15, r_grow=0.35, path_max=0.6, rings=16,
                 support_stride=1, move_min_pts=20, floor_default=-0.5):
        """static_matrix: 4x4 (base_link ← lidar). 없으면 우리 Go2 장착값(+90° yaw, (0.145,0,0.110))."""
        if static_matrix is None:
            static_matrix = np.array([[0., -1., 0., 0.145], [1., 0., 0., 0.], [0., 0., 1., 0.110], [0., 0., 0., 1.]])
        M = np.asarray(static_matrix, dtype=np.float64)
        self.R = M[:3, :3].copy(); self.t = M[:3, 3].copy()
        self.gcut = float(gcut); self.rings = int(rings)
        self.no_person = bool(no_person); self.no_move = bool(no_move); self.no_support = bool(no_support)
        self.win = float(win); self.r_seed = float(r_seed); self.r_grow = float(r_grow); self.path_max = float(path_max)
        self.support_stride = max(1, int(support_stride)); self.move_min_pts = int(move_min_pts)
        self.floor_default = float(floor_default)
        self.pose_hist = collections.deque()         # (t, x, y, yaw)  경로용
        self.hist = collections.deque()              # (t, x, y, yaw, minr[NBIN])  움직이는 물체용
        self.pend = collections.deque()              # (t, cells) 2초 대기열
        self.winq = collections.deque()              # (t, cells) 2~5초 창
        self.count = {}                              # 셀 → 2~5초 창 안 관측 횟수
        self.stats = collections.Counter()
        self._last_scan_id = None

    # ---------- 공통 ----------
    def _pose(self, t_sec, interp=None, pose=None):
        if pose is not None:
            return float(pose[0]), float(pose[1]), float(pose[2])
        if interp is None:
            raise ValueError('interp 또는 pose 가 필요하다')
        T = np.asarray(interp.matrix_at(int(round(t_sec * 1e9))), dtype=np.float64)
        return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))

    def _base(self, x, y, z):
        P = np.c_[x, y, z] @ self.R.T + self.t
        return P[:, 0], P[:, 1], P[:, 2]

    def _floor_z(self, bz, hr, fin0):
        fc = fin0 & (hr > 1.0) & (hr < 5.0) & (bz < -0.25) & (bz > -0.9)
        if fc.sum() > 200:
            return float(np.median(bz[fc]))
        self.stats['floor_fallback'] += 1
        return self.floor_default

    def _minr(self, bx, by, bz, hr, floor_z, finite):
        """방위 0.5도 칸별 첫 장애물 거리(움직이는 물체 판정용)."""
        hh = bz - floor_z
        band = finite & (hh > 0.15) & (hh < 2.0) & (hr > 0.7) & (hr < 12)
        minr = np.full(NBIN, np.inf)
        if band.any():
            azs = np.degrees(np.arctan2(by[band], bx[band])) % 360
            np.minimum.at(minr, (azs / 0.5).astype(int) % NBIN, hr[band])
        return minr, band

    def _push_hist(self, t, px, py, pw, minr, cells):
        self.pose_hist.append((t, px, py, pw))
        while self.pose_hist and t - self.pose_hist[0][0] > self.win + 1.0:
            self.pose_hist.popleft()
        self.hist.append((t, px, py, pw, minr))
        while self.hist and t - self.hist[0][0] > max(LAGS_SEC) + 0.5:
            self.hist.popleft()
        self.pend.append((t, cells))
        while self.pend and t - self.pend[0][0] >= 2.0:
            tt, cc = self.pend.popleft()
            for c in cc: self.count[c] = self.count.get(c, 0) + 1
            self.winq.append((tt, cc))
        while self.winq and t - self.winq[0][0] > 5.0:
            tt, cc = self.winq.popleft()
            for c in cc:
                n = self.count.get(c, 0) - 1
                if n <= 0: self.count.pop(c, None)
                else: self.count[c] = n

    def _cells(self, bx, by, px, py, pw, band):
        if not band.any():
            return ()
        cw, sw = math.cos(pw), math.sin(pw)
        WX = px + cw * bx[band] - sw * by[band]; WY = py + sw * bx[band] + cw * by[band]
        ij = np.floor(np.c_[WX, WY] / GR).astype(np.int64)
        return tuple(map(tuple, np.unique(ij, axis=0)))

    # ---------- 이력만 넣기(건너뛰는 스캔) ----------
    def observe(self, scan_id, x, y, z, ring, t_sec, interp=None, pose=None):
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); z = np.asarray(z, dtype=np.float64)
        bx, by, bz = self._base(x, y, z)
        hr = np.hypot(bx, by); fin = np.isfinite(bx) & np.isfinite(hr)
        floor_z = self._floor_z(bz, hr, fin)
        px, py, pw = self._pose(t_sec, interp, pose)
        minr, band = self._minr(bx, by, bz, hr, floor_z, fin)
        self._push_hist(float(t_sec), px, py, pw, minr, self._cells(bx, by, px, py, pw, band))
        self.stats['observed'] += 1

    # ---------- 정제 ----------
    def clean(self, scan_id, x, y, z, intensity, ring, t_sec, interp=None, pose=None, return_mask=False):
        x = np.asarray(x, dtype=np.float64).copy(); y = np.asarray(y, dtype=np.float64).copy(); z = np.asarray(z, dtype=np.float64).copy()
        ring = np.asarray(ring)
        npt = x.size
        r = np.sqrt(x * x + y * y + z * z)
        kill = np.zeros(npt, bool)
        R16 = self.rings

        # 2) 이중반사: 열 2k/2k+1 = 같은 방위 두 리턴
        ghost = np.zeros(npt, bool)
        col = npt // R16 if R16 else 0
        if R16 and npt % (2 * R16) == 0 and (ring.reshape(col, R16) == np.arange(R16)).all():
            az = np.degrees(np.arctan2(y, x)).reshape(col, R16)
            rr = r.reshape(col, R16)
            both = np.isfinite(rr[0::2]) & np.isfinite(rr[1::2])
            daz = np.abs(((az[0::2] - az[1::2]) + 180.0) % 360.0 - 180.0)[both]
            if daz.size == 0 or np.median(daz) < 0.05:
                g = np.zeros((col, R16), bool)
                g[0::2] = both & (rr[0::2] - rr[1::2] > self.gcut)
                g[1::2] = both & (rr[1::2] - rr[0::2] > self.gcut)
                ghost = g.ravel()
            else:
                self.stats['pair_check_fail'] += 1
        else:
            self.stats['pair_check_fail'] += 1

        bx, by, bz = self._base(x, y, z)
        hr = np.hypot(bx, by)
        fin0 = np.isfinite(r) & ~ghost
        floor_z = self._floor_z(bz, hr, fin0)

        # 2b) 먼 리턴이 같은 스캔 다른 점 8 cm 안이면 진짜 면 → 유지(바닥 아래는 계속 제거)
        if not self.no_support and ghost.any():
            cand = np.flatnonzero(ghost)
            below = bz[cand] < floor_z - 0.10
            P3 = np.c_[bx, by, bz]
            ref = P3[fin0][::self.support_stride]
            dd0, _ = cKDTree(ref).query(P3[cand], distance_upper_bound=0.08)
            sup = (dd0 < 0.08) & ~below
            ghost[cand[sup]] = False
            self.stats['ghost_supported_kept'] += int(sup.sum())
        kill |= ghost
        self.stats['ghost'] += int(ghost.sum())

        px, py, pw = self._pose(t_sec, interp, pose)
        t = float(t_sec)

        # 3) 따라오는 사람: 과거 win 초 경로 위 바닥+0.08~0.55 m 씨앗 → 키워서 제거
        n_person = 0
        if not self.no_person:
            path = self._past_path(t, px, py, pw)
            if path is not None:
                fin = np.isfinite(r) & ~kill
                h = bz - floor_z
                kd_path = cKDTree(path)
                sc = np.flatnonzero(fin & (h > 0.08) & (h < 0.55) & (hr > 0.5))
                if sc.size:
                    dd, _ = kd_path.query(np.c_[bx[sc], by[sc]], distance_upper_bound=self.r_seed)
                    seed = sc[dd < self.r_seed]
                    if seed.size >= 8:
                        SP = np.c_[bx[seed], by[seed]]
                        cnt = cKDTree(SP).query_ball_point(SP, 0.4, return_length=True)
                        strong = SP[cnt >= 8]
                        if len(strong):
                            gc = np.flatnonzero(fin & (h > 0.10) & (h < 2.2))
                            G = np.c_[bx[gc], by[gc]]
                            d2, _ = cKDTree(strong).query(G, distance_upper_bound=self.r_grow)
                            dp, _ = kd_path.query(G, distance_upper_bound=self.path_max)
                            per = gc[(d2 < self.r_grow) & (dp < self.path_max)]
                            kill[per] = True
                            n_person = int(per.size)
            self.stats['person'] += n_person
            self.stats['scans_with_person'] += int(n_person > 0)

        # 4) 움직이는 물체: 1·2·3초 전 그 방위가 비어 있었고, 2~5초 전에도 그 자리가 비었으면 제거
        n_move = 0
        minr, band = self._minr(bx, by, bz, hr, floor_z, np.isfinite(r) & ~ghost)
        cells = self._cells(bx, by, px, py, pw, band)
        if not self.no_move and len(self.hist) >= 2:
            cw, sw = math.cos(pw), math.sin(pw)
            cidx = np.flatnonzero(band & ~kill & (hr < 8))
            if cidx.size:
                Q = np.c_[bx[cidx], by[cidx]]
                ij = np.floor((Q + 8) / 0.15).astype(int)
                okq = (ij >= 0).all(1) & (ij < 107).all(1)
                Gm = np.zeros((107, 107), bool); Gm[ij[okq, 0], ij[okq, 1]] = True
                lab, nl = ndimage.label(Gm, structure=np.ones((3, 3)))
                pl = np.zeros(len(Q), int); pl[okq] = lab[ij[okq, 0], ij[okq, 1]]
                for cl in range(1, nl + 1):
                    mm = pl == cl
                    if mm.sum() < self.move_min_pts: continue
                    P2 = Q[mm]
                    if max(np.ptp(P2[:, 0]), np.ptp(P2[:, 1])) > 1.0: continue
                    Wx = px + cw * P2[:, 0] - sw * P2[:, 1]; Wy = py + sw * P2[:, 0] + cw * P2[:, 1]
                    anyfree = False
                    for lg in LAGS_SEC:
                        e = self._hist_at(t - lg)
                        if e is None: continue
                        te_, xe_, ye_, we_, minre = e
                        ce, se = math.cos(we_), math.sin(we_)
                        dx_, dy_ = Wx - xe_, Wy - ye_
                        ex_, ey_ = ce * dx_ + se * dy_, -se * dx_ + ce * dy_
                        prev = minre[((np.degrees(np.arctan2(ey_, ex_)) % 360) / 0.5).astype(int) % NBIN]
                        known = np.isfinite(prev)
                        if known.sum() >= 10 and np.mean(prev[known] > np.hypot(ex_, ey_)[known] + 0.5) >= 0.6:
                            anyfree = True; break
                    if not anyfree: continue
                    ij2 = np.floor(np.c_[Wx, Wy] / GR).astype(np.int64)
                    occ = np.array([max(self.count.get((int(u + du), int(v + dv)), 0) for du in (-1, 0, 1) for dv in (-1, 0, 1)) > 0
                                    for u, v in ij2])
                    if occ.mean() >= 0.3:
                        self.stats['moving_static_skip'] += 1; continue
                    kill[cidx[mm]] = True
                    n_move += int(mm.sum()); self.stats['moving_blobs'] += 1
            self.stats['moving_pts'] += n_move

        self._push_hist(t, px, py, pw, minr, cells)
        self.stats['scans'] += 1; self.stats['points'] += npt
        x[kill] = np.nan; y[kill] = np.nan; z[kill] = np.nan
        if return_mask:
            return x, y, z, intensity, ring, t_sec, kill
        return x, y, z, intensity, ring, t_sec

    def _hist_at(self, t_target):
        """t_target 에 가장 가까운 과거 이력(0.15초 안)."""
        best = None; bd = 0.15
        for e in self.hist:
            d = abs(e[0] - t_target)
            if d < bd: bd = d; best = e
        return best

    def _past_path(self, t, px, py, pw):
        """시각 t 기준 과거 경로(0.05 m 간격), 현재 몸체 0.6 m 이내 제외, base 프레임."""
        sel = [(tt, xx, yy) for (tt, xx, yy, _) in self.pose_hist if t - self.win <= tt <= t - 0.3]
        if len(sel) < 2: return None
        X = np.array([s[1] for s in sel]); Y = np.array([s[2] for s in sel])
        seg = np.hypot(np.diff(X), np.diff(Y))
        pts = [np.c_[X[:1], Y[:1]]]
        for k in np.flatnonzero(seg > 1e-4):
            n = max(1, int(math.ceil(seg[k] / 0.05)))
            f = np.arange(1, n + 1) / n
            pts.append(np.c_[X[k] + f * (X[k + 1] - X[k]), Y[k] + f * (Y[k + 1] - Y[k])])
        P = np.vstack(pts)
        P = P[np.hypot(P[:, 0] - px, P[:, 1] - py) > 0.6]
        if len(P) == 0: return None
        c, s = math.cos(pw), math.sin(pw)
        dx, dy = P[:, 0] - px, P[:, 1] - py
        return np.c_[c * dx + s * dy, -s * dx + c * dy]
