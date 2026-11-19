# -*- coding: utf-8 -*-
"""replay_samples.py — 센서 없이 실시간 경로를 시험한다: 이미 뽑아 둔 주행 데이터(예: testdata/S6)를
실제 시간 흐름대로 LiveStore 에 흘려 넣고, 층 판별·위치 작업자가 그것을 따라가는지 본다.

    python replay_samples.py --building sw --data ../testdata/S6 --sample-dt 1.5 [--speed 1.0] [--until 400]

라이브 노드와 다른 점은 구조점을 새로 뽑지 않고 저장된 표본을 쓴다는 것 하나다(구조점 라이브판은 a131_live 의
demo_from_dump.py 로 따로 검증). 표본 간격은 라이브와 같게 --sample-dt 로 솎는다(저장본은 0.1초 간격).
초당 10번, 그때의 정합 결과로 주행거리 위치를 지도 위치로 옮겨 trajectory.csv 에 남긴다(라이브 노드의 발행과 같은 계산).
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from live_store import LiveStore  # noqa: E402
from pose_tools import map_pose_px, read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--building', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--root', default='/dev/shm/rt_live_replay')
    ap.add_argument('--sample-dt', type=float, default=1.5)
    ap.add_argument('--speed', type=float, default=1.0, help='재생 배속(1 = 실제 시간)')
    ap.add_argument('--until', type=float, default=None, help='데이터 시작 뒤 이 초까지만')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--no-pose-worker', action='store_true')
    ap.add_argument('--floor-procs', type=int, default=3, help='층 판별을 나눠 맡을 프로세스 수')
    a = ap.parse_args()

    data = Path(a.data)
    rows = list(csv.DictReader(open(data / 'odom_link.csv')))
    T = np.array([float(r['t_sec']) for r in rows])
    far = np.load(data / 'far_points.npz')
    root = Path(a.root)
    store = LiveStore(root / 'data', reset=True)
    out = root / 'out'
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)

    procs = []
    import os
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')   # 라이브와 같게(구조점 3코어 몫을 남김)
    for role in (['floor'] if a.no_pose_worker else ['floor', 'pose']):
        lf = open(out / f'{role}.stdout', 'w')
        extra = ['--procs', str(a.floor_procs)] if role == 'floor' else []
        procs.append(subprocess.Popen([a.python, str(HERE / 'loc_worker.py'), '--role', role, '--building', a.building,
                                       '--store', str(root / 'data'), '--out', str(out)] + extra, stdout=lf,
                                      stderr=subprocess.STDOUT, env=env))

    end = T[-1] if a.until is None else min(T[-1], T[0] + a.until)
    wall0 = time.time()
    next_sample = T[0]
    traj = open(out / 'trajectory.csv', 'w')
    traj.write('t_rel,x,y,yaw,map,px,py,heading_rad,fit_t_rel,fit_age_s\n')
    i = 0
    try:
        while i < len(T) and T[i] <= end:
            # 실제 시간에 맞춰 기다린다
            due = wall0 + (T[i] - T[0]) / a.speed
            dt = due - time.time()
            if dt > 0:
                time.sleep(dt)
            r = rows[i]
            if T[i] >= next_sample - 0.05:                       # 라이브 노드처럼 T0 기준 sample_dt 격자에 맞춘다
                with np.load(data / 'samples' / r['sample_id'] / 'data_short.npz') as d:
                    xy, role = d['a1_3_1_all_candidate_xy'], d['a1_3_1_all_candidate_role']
                fx = far[r['sample_id']] if r['sample_id'] in far.files else np.zeros((0, 2), np.float32)
                store.add(T[i], float(r['fastlio_x']), float(r['fastlio_y']), float(r['fastlio_yaw_rad']), xy, role, fx)
                next_sample = T[0] + (round((T[i] - T[0]) / a.sample_dt) + 1) * a.sample_dt
            fit = read_json(out / 'pose_fit.json')
            if fit:
                px, py, hd = map_pose_px(fit, float(r['fastlio_x']), float(r['fastlio_y']), float(r['fastlio_yaw_rad']))
                traj.write(f"{T[i] - T[0]:.2f},{r['fastlio_x']},{r['fastlio_y']},{r['fastlio_yaw_rad']},{fit['map']},"
                           f"{px:.1f},{py:.1f},{hd:.4f},{fit['t'] - T[0]:.2f},{T[i] - fit['t']:.2f}\n")
            i += 1
        # 데이터가 끝난 뒤에도 진행 중인 바퀴가 끝나기를 잠시 기다린다
        t_wait = time.time() + 600
        while time.time() < t_wait:
            fl = read_json(out / 'floor.json')
            if fl and fl['t_now'] >= store.rows[-1]['t_sec'] - 1e-6:
                break
            time.sleep(5)
    finally:
        traj.close()
        for p in procs:
            p.terminate()
    print(json.dumps(dict(samples=store.n, span=store.span(), wall_sec=round(time.time() - wall0, 1)), ensure_ascii=False))


if __name__ == '__main__':
    main()
