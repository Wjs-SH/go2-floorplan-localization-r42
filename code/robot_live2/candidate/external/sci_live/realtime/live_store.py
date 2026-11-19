# -*- coding: utf-8 -*-
"""live_store.py — 실시간 표본을 위치추정기가 읽는 주행 데이터 형식으로 메모리 디스크에 쓴다.

위치추정기(floor_id/localizer)는 폴더 하나(odom_link.csv · samples/<id>/data_short.npz · far_points.npz)를 읽는다.
알고리즘을 건드리지 않으려고, 라이브 표본도 **같은 형식 그대로** /dev/shm 에 쓰고 위치추정기의 기존 읽기 함수로 다시 읽힌다.

- 파일은 임시 이름으로 쓴 뒤 rename 한다(다른 프로세스가 반쯤 쓴 파일을 읽지 않게).
- 표본 파일을 먼저 쓰고 odom_link.csv 에 행을 더한다(행에 있는 표본은 늘 파일이 있다).
- 오래된 표본은 지우되 **첫 표본은 남긴다**. 위치추정기는 첫 표본 시각(T[0])을 시간 원점으로 쓰고(장면 캐시 열쇠 등),
  원점이 움직이면 캐시가 어긋난다.
"""
import errno
import json
import os
import shutil
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
    def __init__(self, root, keep_s=900.0, reset=True, archive=None):
        self.root = Path(root)
        self.archive = archive          # R1: RunEvidenceArchive(영구 연구 증거) — 짧은 실시간 저장소(keep_s)와 따로
        self.archive_status = None if archive is None else 'OK'
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
        if self.archive is not None:
            self.archive_status = self.archive.persist_sample(sid, self.rows[-1], d / 'data_short.npz', self.far[sid])
        self._prune()
        self._write_index()
        return sid

    def _prune(self):
        t_last = self.rows[-1]['t_sec']
        pinned = pinned_ranges(self.root)          # R1: 진행 중 계산이 잡아 둔 구간은 keep_s 를 넘어도 지우지 않는다
        keep = [self.rows[0]] + [r for r in self.rows[1:] if r['t_sec'] >= t_last - self.keep_s
                                 or any(a - 1e-6 <= r['t_sec'] <= b + 1e-6 for a, b in pinned)]
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


# ---------------------------------------------------------------- R1 (2026-10-16): 구간 고정(pin)과 영구 증거 보관
def _pins_dir(root):
    return Path(root) / 'pins'


def pin(root, owner, t0, t1, ttl_s=600.0):
    """다른 프로세스(loc_worker 층 판별 바퀴 등)가 쓰는 동안 [t0, t1] 표본을 지우지 말라는 표시. ttl 이 지나면 무시된다(무한 보존 아님)."""
    d = _pins_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f'.{owner}.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(dict(owner=owner, t0=float(t0), t1=float(t1), expires=time.time() + float(ttl_s))))
    os.replace(tmp, d / f'{owner}.json')


def unpin(root, owner):
    try:
        os.remove(_pins_dir(root) / f'{owner}.json')
    except OSError:
        pass


def pinned_ranges(root, now=None):
    now = time.time() if now is None else now
    out = []
    for p in _pins_dir(root).glob('*.json') if _pins_dir(root).exists() else []:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get('expires', 0) >= now:
            out.append((float(d['t0']), float(d['t1'])))
    return out


class RunEvidenceArchive:
    """실험(run)별 표본 원본을 영구 디스크에 남긴다(연구 증거). 실시간 저장소의 keep_s·60 s 창과 무관.
    용량 상한(max_bytes)을 넘으면 RESOURCE_LIMITED, 쓰기 실패(디스크 가득 등)는 IO_FAILURE 를 돌려주고 manifest 에 남긴다 —
    성공 파일을 지어내지 않는다. RAM 에 쌓지 않는다(파일 복사만)."""

    def __init__(self, root, max_bytes=2 * 1024 ** 3):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.status = 'OK'
        self.n = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def _manifest(self, rec):
        try:
            with open(self.root / 'manifest.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except OSError:
            pass

    def persist_sample(self, sid, row, npz_path, far_xy):
        if self.status == 'RESOURCE_LIMITED':
            return self.status
        try:
            size = Path(npz_path).stat().st_size + int(np.asarray(far_xy).nbytes)
            if self.bytes + size > self.max_bytes:
                self.status = 'RESOURCE_LIMITED'
                self._manifest(dict(t=time.time(), sid=sid, status=self.status, bytes=self.bytes, max_bytes=self.max_bytes))
                return self.status
            d = self.root / 'samples' / sid
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(npz_path, d / 'data_short.npz')
            np.save(d / 'far_xy.npy', np.asarray(far_xy, np.float32))
            with open(self.root / 'odom_link_full.csv', 'a') as f:
                f.write(f"{sid},{row['t_sec']:.6f},{row['x']:.6f},{row['y']:.6f},{row['yaw']:.6f}\n")
            self.bytes += size
            self.n += 1
            self.status = 'OK'
            return 'OK'
        except OSError as e:
            self.status = 'IO_FAILURE'
            self._manifest(dict(t=time.time(), sid=sid, status=self.status, errno=e.errno,
                                enospc=e.errno == errno.ENOSPC, error=f'{type(e).__name__}: {e}'))
            return self.status
