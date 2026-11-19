# -*- coding: utf-8 -*-
"""compare_live_offline.py — 실시간 입력으로 층 판별해도 오프라인 입력과 같은 성적인지 잰다.

    python compare_live_offline.py --building sw --data <S1폴더> <S6폴더> --truth 1 6 [--sample-dt 1.5] [--out cmp.json]

시점은 floor_id.py 와 같다(70초부터 60초 간격). 시점마다 8개 층을 두 입력으로 맞춘다:
  오프라인 : 주행 폴더 전체(0.1초 간격 표본, 미래 표본 포함 — floor_id.py 와 같은 입력)
  실시간   : 그 시점까지 LiveStore 가 들고 있을 것 — 첫 표본 + 그 시점에서 sample_dt 격자로 거슬러 뽑은 표본(최근 180초)
정합은 실시간 작업자(loc_worker.Loc)와 같은 코드다. 바뀌는 것은 입력뿐이다.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'floor_id'))


def build_live_store(src, root, t_now, dt, keep_s):
    """t_now 시점의 LiveStore 내용: 첫 표본 + t_now 에서 dt 간격으로 거슬러 가장 가까운 표본들(keep_s 안)."""
    import numpy as np
    from live_store import LiveStore
    rows = list(csv.DictReader(open(src / 'odom_link.csv')))
    T = np.array([float(r['t_sec']) for r in rows])
    far = np.load(src / 'far_points.npz')
    grid = t_now - np.arange(0.0, t_now - T[0] + 1e-9, dt)
    idx = sorted({int(np.argmin(np.abs(T - g))) for g in grid if g >= t_now - keep_s - 1e-6} | {0})
    st = LiveStore(root, keep_s=keep_s, reset=True)
    for i in idx:
        r = rows[i]
        with np.load(src / 'samples' / r['sample_id'] / 'data_short.npz') as d:
            xy, role = d['a1_3_1_all_candidate_xy'], d['a1_3_1_all_candidate_role']
        fx = far[r['sample_id']] if r['sample_id'] in far.files else np.zeros((0, 2), np.float32)
        st.add(T[i], float(r['fastlio_x']), float(r['fastlio_y']), float(r['fastlio_yaw_rad']), xy, role, fx)
    return T[0], len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--building', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--truth', type=int, nargs='+', required=True)
    ap.add_argument('--sample-dt', type=float, default=1.5)
    ap.add_argument('--keep-s', type=float, default=900.0)
    ap.add_argument('--root', default='/dev/shm/cmp_live_offline')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from loc_worker import Loc
    root = Path(a.root)
    rows_out = []
    for data, truth in zip(map(Path, a.data), a.truth):
        T = [float(r['t_sec']) for r in csv.DictReader(open(data / 'odom_link.csv'))]
        span = T[-1] - T[0]
        import numpy as np
        TS = [float(t) for t in np.arange(70.0, span - 2.0, 60.0)]
        ns = argparse.Namespace(out=str(root), store=str(data), building=a.building, role='cmp')
        root.mkdir(parents=True, exist_ok=True)
        C = Loc(ns)
        n_maps = int(C.b['n_maps'])
        t1 = time.time()
        OFF = {m: C.open_floor(m) for m in range(1, n_maps + 1)}       # 오프라인: 주행 폴더 전체
        LIVE = None
        print(f'== {data.name} (정답 {truth}층) · 시점 {TS} · 지도 준비 {time.time() - t1:.0f}s', flush=True)
        for ts in TS:
            res = {}
            C.store = str(data)
            res['off'] = {m: C.fit(OFF[m], m, T[0] + ts) for m in OFF}
            live_dir = root / f'{data.name}_live'
            t0, n = build_live_store(data, live_dir, T[0] + ts, a.sample_dt, a.keep_s)
            C.store = str(live_dir.resolve())
            if LIVE is None:
                LIVE = {m: C.open_floor(m) for m in range(1, n_maps + 1)}
            for Z in LIVE.values():                                     # 저장소가 통째로 바뀌었으니 장면 캐시를 비운다
                Z.L.dir = C.store
                Z.L._lscn_cache = {}
            t_live = float(list(csv.DictReader(open(live_dir / 'odom_link.csv')))[-1]['t_sec'])   # 작업자처럼 마지막 표본 시각
            res['live'] = {m: C.fit(LIVE[m], m, t_live) for m in LIVE}
            line = dict(data=data.name, truth=truth, ts=ts, live_samples=n)
            for k in ('off', 'live'):
                order = sorted(res[k], key=lambda m: -res[k][m]['twoway'])
                line[k] = dict(verdict=order[0], ranking=order, true_rank=order.index(truth) + 1,
                               twoway={m: round(res[k][m]['twoway'], 4) for m in order},
                               true_pos=res[k][truth]['pos'], true_wall_cm=round(100 * res[k][truth]['wall_cost_m'], 1))
            rows_out.append(line)
            print(f"  {ts:5.0f}s  오프라인 판정 {line['off']['verdict']}층·정답 {line['off']['true_rank']}위  |  "
                  f"실시간 판정 {line['live']['verdict']}층·정답 {line['live']['true_rank']}위  (표본 {n}개)", flush=True)
    for k, name in (('off', '오프라인'), ('live', '실시간')):
        r = [x[k]['true_rank'] for x in rows_out]
        print(f'{name}: 1위 {sum(v == 1 for v in r)}/{len(r)} · 상위3 {sum(v <= 3 for v in r)}/{len(r)} · 정답 순위 {r}')
    same = sum(x['off']['verdict'] == x['live']['verdict'] for x in rows_out)
    print(f'판정 일치 {same}/{len(rows_out)} · 설정: 창 {C.W}s · dt {C.DT}s · 표본 간격 {a.sample_dt}s · 보관 {a.keep_s}s')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(sample_dt=a.sample_dt, keep_s=a.keep_s, W=C.W, DT=C.DT, rows=rows_out),
                                          ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
