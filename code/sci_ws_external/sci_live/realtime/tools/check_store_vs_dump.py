# -*- coding: utf-8 -*-
"""check_store_vs_dump.py — 노드가 저장소에 쓴 표본을 a131_live 시험 데이터의 기준 결과와 대조한다.

    python check_store_vs_dump.py /dev/shm/rt_live/data ../../a131_live/testdata/S6_chunk0_55.npz

같은 target(스캔 끝 시각)의 표본끼리 구조점 좌표·역할이 비트 단위로 같은지, 다르면 몇 칸 다른지 본다.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np


def main():
    store, dump = Path(sys.argv[1]), np.load(sys.argv[2])
    refs = {}
    for k in dump.files:
        m = re.match(r'target(\d+)_group$', k)
        if m:
            st = int(m.group(1))
            refs[st] = (dump[f'target{st}_ref_xy'], dump[f'target{st}_ref_role'])
    rows = list(csv.DictReader(open(store / 'odom_link.csv')))
    n = same = 0
    for r in rows:
        ts = int(round(float(r['t_sec']) * 1e9))
        key = min(refs, key=lambda k: abs(k - ts))
        if abs(key - ts) > 2000:                    # 2 µs 안(시각 표기 반올림)이 아니면 기준 없음
            continue
        with np.load(store / 'samples' / r['sample_id'] / 'data_short.npz') as f:
            xy, role = f['a1_3_1_all_candidate_xy'], f['a1_3_1_all_candidate_role']
        rx, rr = refs[key]
        n += 1
        ok = np.array_equal(xy, rx) and np.array_equal(role, rr)
        same += ok
        note = ''
        if not ok:
            A = {tuple(v): q for v, q in zip(map(tuple, xy), role)}
            B = {tuple(v): q for v, q in zip(map(tuple, rx), rr)}
            note = f' (칸 집합 차이 {len(set(A) ^ set(B))}, 같은 칸 역할 차이 {sum(1 for k in set(A) & set(B) if A[k] != B[k])})'
        print(f"{r['sample_id']} t={float(r['t_sec']):.3f} 칸 {len(role)} 기준과 {'같음' if ok else '다름'}{note} "
              f"자세 x={float(r['fastlio_x']):.3f} y={float(r['fastlio_y']):.3f} yaw={float(r['fastlio_yaw_rad']):.3f}")
    print(f'\n저장 표본 {len(rows)}개 중 기준이 있는 것 {n}개, 그중 같음 {same}개')


if __name__ == '__main__':
    main()
