#!/usr/bin/env python3
"""Go2 에서 실행: 시험 데이터를 순서대로 LiveCleaner 에 먹여 기준 마스크와 대조한다.
사용: check_testdata.py testdata/clean_live_test_3F.npz [--stride 2]"""
import sys, time, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a131_clean_live import LiveCleaner
Z = np.load(sys.argv[1]); ST = int(sys.argv[sys.argv.index('--stride')+1]) if '--stride' in sys.argv else 2
PT, PX, PY, PW = Z['pose_t'], Z['pose_x'], Z['pose_y'], Z['pose_yaw']
n = int(Z['n']); ev = int(Z['eval_from'])
C = LiveCleaner(support_stride=ST)
same = extra = miss = 0; ts = []
for j in range(n):
    t = float(Z[f's{j}_t']); x = Z[f's{j}_x'].astype(np.float64); y = Z[f's{j}_y'].astype(np.float64); z = Z[f's{j}_z'].astype(np.float64)
    pose = (float(np.interp(t, PT, PX)), float(np.interp(t, PT, PY)), float(np.interp(t, PT, PW)))
    t0 = time.perf_counter()
    _, _, _, _, _, _, kill = C.clean(j, x, y, z, Z[f's{j}_i'], Z[f's{j}_ring'], t, pose=pose, return_mask=True)
    ts.append(time.perf_counter() - t0)
    if j < ev: continue
    ref = Z[f's{j}_killref']; fin = np.isfinite(x)
    same += int((kill & ref & fin).sum()); extra += int((kill & ~ref & fin).sum()); miss += int((~kill & ref & fin).sum())
ts = np.array(ts[ev:])
print(f'평가 스캔 {n-ev}개')
print(f'  둘 다 지움 {same:,} | 실시간만 {extra:,} | 오프라인만(놓침) {miss:,}')
print(f'  오프라인이 지운 점 재현율 {100*same/max(same+miss,1):.2f}%  | 전체 일치 {100*same/max(same+extra+miss,1):.2f}%')
print(f'  스캔당 중앙 {1000*np.median(ts):.1f} ms, p90 {1000*np.percentile(ts,90):.1f} ms')
