# -*- coding: utf-8 -*-
"""fastlio_info.py — sensors_go2.sh 가 logs/live_<월일>/ 에 남긴 FAST-LIO 실행 정보를 읽는다(실험 기록용, 2026-10-09).

    fastlio.cfg          "FAST-LIO 설정: <yaml 경로>" 한 줄
    fastlio.env          impl=go2ws|esy · fast_lio_prefix=… · started=<epoch> · FASTLIO_*=…
    imu_ref_calib.json   esy 입력 브리지의 IMU 시각 보정(esy 일 때만)

노드(run_config.json, 노드 시작 때)와 화면 서버(실험 보관 때) 둘 다 이걸 부른다.
10-09 실험에서 [초기화]가 FAST-LIO 를 다른 구현으로 다시 띄웠는데 기록은 노드 시작 때 것이 남았다 → 보관 때 다시 읽는다.
"""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def live_dir(t=None):
    return ROOT / 'logs' / time.strftime('live_%m%d', time.localtime(t))


def read(ldir=None):
    ldir = Path(ldir) if ldir else live_dir()
    cfg = {}
    try:
        fl_cfg = (ldir / 'fastlio.cfg').read_text().split(':', 1)[1].strip()
        txt = Path(fl_cfg).read_text() if Path(fl_cfg).exists() else ''
        cfg['fastlio_config'] = fl_cfg
        m = re.search(r'extrinsic_R:\s*\[([^\]]+)\]', txt)
        cfg['fastlio_extrinsic_R'] = [float(v) for v in m.group(1).replace('\n', ' ').split(',')] if m else None
        for k in ('acc_cov', 'gyr_cov', 'max_iteration', 'filter_size_surf'):
            mm = re.search(rf'{k}:\s*([0-9.eE+-]+)', txt)
            cfg[f'fastlio_{k}'] = float(mm.group(1)) if mm else None
    except Exception as e:
        cfg['fastlio_config'] = f'알 수 없음({type(e).__name__})'
    try:
        env = dict(l.split('=', 1) for l in (ldir / 'fastlio.env').read_text().splitlines() if '=' in l)
        cfg['fastlio_impl'] = env.pop('impl', '')
        cfg['fastlio_prefix'] = env.pop('fast_lio_prefix', '')
        st = env.pop('started', '')
        cfg['fastlio_started'] = float(st) if st else None
        cfg['fastlio_env'] = env
    except OSError:
        cfg['fastlio_impl'] = 'go2ws(기록 없음)'
    try:
        cfg['imu_ref_calib'] = json.loads((ldir / 'imu_ref_calib.json').read_text()) if cfg.get('fastlio_impl') == 'esy' else None
    except (OSError, ValueError):
        cfg['imu_ref_calib'] = None
    return cfg
