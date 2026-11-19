"""구간별로 쌓은 구조점끼리 몇 도 돌아가 있나(FAST-LIO yaw 드리프트). 거울상 켬, δ 는 셋째 인자(기본 45.3°).
    python yaw_drift.py <저장소> [구간초 30] [δ 45.3]"""
import csv, sys, numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
D = Path(sys.argv[1]); SEG = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 45.3
rows = list(csv.DictReader(open(D / 'odom_link.csv')))
P = []
for r in rows:
    try:
        with np.load(D / 'samples' / r['sample_id'] / 'data_short.npz') as f:
            p = f['a1_3_1_all_candidate_xy'][f['a1_3_1_all_candidate_role'] != 4].astype(float)
    except OSError:
        continue
    x, y, yaw = float(r['fastlio_x']), float(r['fastlio_y']), float(r['fastlio_yaw_rad'])
    yw = yaw + np.deg2rad(DELTA); c, s = np.cos(yw), np.sin(yw)
    P.append((float(r['t_sec']), np.c_[-x + c * p[:, 0] - s * p[:, 1], -y + s * p[:, 0] + c * p[:, 1]], (-x, -y)))
t0 = P[0][0]; segs = {}
for t, a, pos in P:
    segs.setdefault(int((t - t0) // SEG), []).append((a, pos))
keys = sorted(segs)
ref = np.vstack([a for a, _ in segs[keys[0]]]); lo = ref.min(0) - 3; R = 0.05
def field(a):
    ij = np.floor((a - lo) / R).astype(int); sh = ij.max(0) + 80
    g = np.ones(sh, bool); ok = (ij >= 0).all(1); g[ij[ok, 0], ij[ok, 1]] = False
    return distance_transform_edt(g) * R
F = field(ref)
def cost(a):
    ij = np.floor((a - lo) / R).astype(int); ok = (ij >= 0).all(1) & (ij[:, 0] < F.shape[0]) & (ij[:, 1] < F.shape[1])
    d = np.full(len(a), 1.0); d[ok] = np.minimum(F[ij[ok, 0], ij[ok, 1]], 1.0); return float(np.mean(np.sort(d)[:int(.6 * len(d))]))
print(f'표본 {len(P)}개 · 구간 {SEG:.0f}초 · δ {DELTA}° · 기준 = 첫 구간({len(segs[keys[0]])}표본)')
for k in keys[1:]:
    a = np.vstack([q for q, _ in segs[k]]); c0 = np.mean([p for _, p in segs[k]], axis=0)
    best = (cost(a), 0.0, 0.0, 0.0)
    for th in np.arange(-20, 20.1, 1.0):
        c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)); b = (a - c0) @ np.array([[c, s], [-s, c]]) + c0
        for dx in np.arange(-1.0, 1.01, 0.25):
            for dy in np.arange(-1.0, 1.01, 0.25):
                v = cost(b + [dx, dy])
                if v < best[0]: best = (v, th, dx, dy)
    print(f'  {k*SEG:4.0f}~{(k+1)*SEG:4.0f}초 ({len(segs[k])}표본): 그대로 {cost(a):.3f} m → 회전 {best[1]:+.0f}°·이동 ({best[2]:+.2f},{best[3]:+.2f}) m 일 때 {best[0]:.3f} m')
