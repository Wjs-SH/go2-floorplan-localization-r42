# -*- coding: utf-8 -*-
"""live_store.py — 실시간 표본을 위치추정기가 읽는 주행 데이터 형식으로 메모리 디스크에 쓴다.

위치추정기(floor_id/localizer)는 폴더 하나(odom_link.csv · samples/<id>/data_short.npz · far_points.npz)를 읽는다.
알고리즘을 건드리지 않으려고, 라이브 표본도 **같은 형식 그대로** /dev/shm 에 쓰고 위치추정기의 기존 읽기 함수로 다시 읽힌다.

- 파일은 임시 이름으로 쓴 뒤 rename 한다(다른 프로세스가 반쯤 쓴 파일을 읽지 않게).
- 표본 파일을 먼저 쓰고 odom_link.csv 에 행을 더한다(행에 있는 표본은 늘 파일이 있다).
- 오래된 표본은 지우되 **첫 표본은 남긴다**. 위치추정기는 첫 표본 시각(T[0])을 시간 원점으로 쓰고(장면 캐시 열쇠 등),
  원점이 움직이면 캐시가 어긋난다.
"""
import os
import time
from pathlib import Path

import numpy as np

COLUMNS = ['sample_id', 't_sec', 'fastlio_x', 'fastlio_y', 'fastlio_yaw_rad', 'delta_s_m', 'delta_yaw_deg']


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays):
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class LiveStore:
    # 보관 900초: 로봇에서 층 판별 한 바퀴가 2분 넘게 걸리고(2026-10-07 127초), 그동안 창(60초) 안 표본 파일이 지워지면
    # 그 바퀴 정합이 실패한다(180초일 때 실제로 남). 보관 길이는 정합 결과에 영향이 없다 — 저장소 전체로 계산되는
    # 주축(AX0)·중심(ctr)을 바꿔도 fit_window_v2 결과가 같았다. 표본당 ~40 KB → 900초 ≈ 25 MB.
    def __init__(self, root, keep_s=900.0, reset=True):
        self.root = Path(root)
        self.samples = self.root / 'samples'
        if reset and self.root.exists():
            import shutil
            shutil.rmtree(self.root)
        self.samples.mkdir(parents=True, exist_ok=True)
        self.keep_s = float(keep_s)
        self.rows = []          # dict(sample_id, t_sec, x, y, yaw)
        self.far = {}
        self.n = 0
        self._yaw_prev = None

    def add(self, t_sec, x, y, yaw, xy, role, far_xy):
        """표본 하나를 더한다. t_sec 는 앞 표본보다 커야 한다(위치추정기가 엄격 증가를 요구)."""
        if self.rows and t_sec <= self.rows[-1]['t_sec']:
            return None
        # yaw 는 전달본처럼 이어진 값(펼친 값)으로 쓴다.
        if self._yaw_prev is not None:
            yaw = self._yaw_prev + (yaw - self._yaw_prev + np.pi) % (2 * np.pi) - np.pi
        self._yaw_prev = yaw
        sid = f'LIVE_{self.n:06d}'
        self.n += 1
        d = self.samples / sid
        d.mkdir(exist_ok=True)
        _atomic_savez(d / 'data_short.npz',
                      a1_3_1_all_candidate_xy=np.asarray(xy, np.float32).reshape(-1, 2),
                      a1_3_1_all_candidate_role=np.asarray(role, np.int8).ravel())
        self.rows.append(dict(sample_id=sid, t_sec=float(t_sec), x=float(x), y=float(y), yaw=float(yaw)))
        self.far[sid] = np.asarray(far_xy, np.float32).reshape(-1, 2)
        self._prune()
        self._write_index()
        return sid

    def _prune(self):
        t_last = self.rows[-1]['t_sec']
        keep = [self.rows[0]] + [r for r in self.rows[1:] if r['t_sec'] >= t_last - self.keep_s]
        gone = {r['sample_id'] for r in self.rows} - {r['sample_id'] for r in keep}
        self.rows = keep
        for sid in gone:
            self.far.pop(sid, None)
            p = self.samples / sid / 'data_short.npz'
            try:
                p.unlink()
                (self.samples / sid).rmdir()
            except OSError:
                pass

    def _write_index(self):
        lines = [','.join(COLUMNS)]
        prev = None
        for r in self.rows:
            ds = 0.0 if prev is None else float(np.hypot(r['x'] - prev['x'], r['y'] - prev['y']))
            dy = 0.0 if prev is None else float(np.degrees(r['yaw'] - prev['yaw']))
            lines.append(f"{r['sample_id']},{r['t_sec']:.6f},{r['x']:.6f},{r['y']:.6f},{r['yaw']:.6f},{ds:.4f},{dy:.3f}")
            prev = r
        _atomic_savez(self.root / 'far_points.npz', **{k: v for k, v in self.far.items()})
        _atomic_write_bytes(self.root / 'odom_link.csv', ('\n'.join(lines) + '\n').encode())
        _atomic_write_bytes(self.root / 'UPDATED', f'{time.time():.3f} {self.rows[-1]["t_sec"]:.6f} {len(self.rows)}\n'.encode())

    def span(self):
        return self.rows[-1]['t_sec'] - self.rows[0]['t_sec'] if self.rows else 0.0


def read_span(root):
    """다른 프로세스에서: (첫 표본 시각, 마지막 표본 시각, 표본 수). 아직 없으면 None."""
    p = Path(root) / 'UPDATED'
    o = Path(root) / 'odom_link.csv'
    if not p.exists() or not o.exists():
        return None
    lines = o.read_text().splitlines()
    if len(lines) < 2:
        return None
    t0 = float(lines[1].split(',')[1])
    t1 = float(lines[-1].split(',')[1])
    return t0, t1, len(lines) - 1
