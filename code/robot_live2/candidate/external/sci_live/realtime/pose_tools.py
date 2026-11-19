# -*- coding: utf-8 -*-
"""pose_tools.py — 정합 결과 하나로 주행거리 위치(odom_link 규약)를 지도 위치로 옮긴다.

정합(metric_fit_v2.fit_window_v2)은 '창 마지막 표본의 주행거리 위치(anchor)'를 지도 픽셀 pos 에 놓는 회전(phi)·축척(ppm)을 준다.
그 사이 시각의 위치는 같은 변환을 그대로 적용하면 된다 — rt_localizer.RTLocalizer.scan_to_map 과 같은 식(점 대신 로봇 원점).
  지도 px  = pos + ppm · A(phi) · (xy − anchor)
  지도 방향 = phi + yaw + beta     (A 는 이미지 y 가 아래인 규약. 미터 지도(y 위)에서 방향각은 같은 값이다)
"""
import json
from pathlib import Path

import numpy as np


def _A(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [-s, -c]])


def map_pose_px(fit, x, y, yaw):
    """(px, py, heading_rad). fit 은 loc_worker 가 쓴 pose_fit.json 내용."""
    phi = np.deg2rad(fit['phi'])
    off = fit['ppm'] * (_A(phi) @ (np.array([x, y], float) - np.asarray(fit['anchor'], float)))
    p = np.asarray(fit['pos'], float) + off
    hd = phi + yaw + fit['beta']
    return float(p[0]), float(p[1]), float((hd + np.pi) % (2 * np.pi) - np.pi)


def load_map_meta(maps_root, floor):
    """그 층 base_map.yaml(해상도·원점)과 이미지 높이. px → m 변환에 쓴다."""
    import yaml
    from PIL import Image
    d = Path(maps_root) / f'{floor}F' / 'maps'
    y = yaml.safe_load((d / 'base_map.yaml').read_text())
    h = Image.open(d / 'base_map.png').size[1]
    return dict(resolution=float(y['resolution']), origin=[float(v) for v in y['origin'][:2]], height=h)


def px_to_m(meta, px, py):
    """map_server 규약: 왼쪽 아래가 origin, y 위. px(열), py(행)."""
    r = meta['resolution']
    return meta['origin'][0] + px * r, meta['origin'][1] + (meta['height'] - py) * r


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
