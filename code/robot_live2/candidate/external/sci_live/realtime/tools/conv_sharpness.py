"""누적 선명도로 축 규약 찾기: 거울상 켬/끔 × δ 를 바꿔 a1_3_1 구조점(몸체 기준)을 FAST-LIO 자세로 쌓고, 차지한 5 cm 칸 수(적을수록 선명)."""
import csv, sys, numpy as np
from pathlib import Path
D = Path(sys.argv[1]); W = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
rows = list(csv.DictReader(open(D / 'odom_link.csv')))
T = np.array([float(r['t_sec']) for r in rows]); keep = T >= T[-1] - W
rows = [r for r, k in zip(rows, keep) if k]
P = []
for r in rows:
    try:
        with np.load(D / 'samples' / r['sample_id'] / 'data_short.npz') as f:
            p = f['a1_3_1_all_candidate_xy'][f['a1_3_1_all_candidate_role'] != 4].astype(float)
    except OSError:
        continue
    P.append((float(r['fastlio_x']), float(r['fastlio_y']), float(r['fastlio_yaw_rad']), p))
def occ(mirror, delta):
    pts = []
    for x, y, yaw, p in P:
        if mirror: px, py, yw = -x, -y, yaw + np.deg2rad(delta)
        else:      px, py, yw = -x, y, -yaw + np.deg2rad(delta)
        c, s = np.cos(yw), np.sin(yw)
        pts.append(np.c_[px + c * p[:, 0] - s * p[:, 1], py + s * p[:, 0] + c * p[:, 1]])
    a = np.vstack(pts); return len(np.unique(np.floor(a / 0.05).astype(np.int64), axis=0))
print(f'표본 {len(P)}개, 점 {sum(len(p[3]) for p in P)}')
for mirror in (True, False):
    res = sorted((occ(mirror, d), d) for d in range(-180, 180, 2))
    best = res[0][1]; fine = sorted((occ(mirror, d), d) for d in np.arange(best - 4, best + 4.1, 0.5))
    print(f"거울상 {'켬' if mirror else '끔'}: 가장 선명 δ {fine[0][1]:.1f}° (칸 {fine[0][0]}) · δ 45.3° 칸 {occ(mirror, 45.3)} · δ -90.8° 칸 {occ(mirror, -90.8)}")
