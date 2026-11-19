# -*- coding: utf-8 -*-
"""loc_worker.py — 실시간 저장소(LiveStore)를 읽어 층 판별·지도 정합을 계속 돌리는 작업자 프로세스.

    python loc_worker.py --role floor --building sw --store /dev/shm/rt_live/data --out /dev/shm/rt_live/out [--procs 3]
    python loc_worker.py --role pose  --building sw --store ... --out ...

알고리즘은 floor_id.py 와 **같은 함수를 같은 설정으로** 부른다(RTLocalizer · fit_window_v2 · twoway). 다른 점은 데이터를 읽는 때뿐이다:
정합 직전마다 위치추정기의 기존 읽기 함수(_load_odom)로 저장소를 다시 읽힌다.

  floor : 8개 층(건물 설정의 n_maps)을 **같은 창(같은 t_now)** 으로 맞춘 뒤 twoway 점수로 순위를 매긴다(floor_id.py 와 같다).
          층을 --procs 개 자식 프로세스에 나눠 맡긴다(지도 준비·정합이 동시에 돈다). 바퀴마다 이 프로세스가 t_now 를 정해 보내고
          결과를 모은다. 한 바퀴 결과를 floor.json 에, 1위 층 정합을 pose_fit.json 에도 쓴다. 끝나면 바로 다음 바퀴(그때의 최신 창).
          자식은 spawn(새 인터프리터)으로 띄운다: 벽정합이 torch CUDA 를 쓰는데, CUDA 를 초기화한 프로세스를 fork 하면
          자식에서 CUDA 를 못 쓴다(2026-10-07 노트북에서 전 층 정합 실패로 확인).
          바퀴 판정(verdict)과 별도로 최근 --vote-n 바퀴 다수결(vote)을 낸다 — 한 구간 판정은 믿을 수 없다(1위 21%).
  pose  : floor.json 의 다수결 층 하나만 계속 다시 맞춰 pose_fit.json 을 갱신한다(층이 바뀌면 그 층 지도를 새로 연다).
"""
import argparse
import json
from collections import Counter
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'floor_id'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKG))


def wall_passes(wall, track_px, ppm, ratio=3.0, extra_m=2.0, max_runs=200):
    """2026-10-13 esy: 경로가 벽을 '반대편으로' 넘어간 횟수.
    벽 안 구간마다 들어가기 직전 칸 a·나온 직후 칸 b 를 잡고, 벽을 안 지나는(문 자리는 벽 아님) 8방향 걸음으로
    a→b 가 max(ratio·직선, 직선+extra_m) 안에 닿는지 본다. 걸음 수는 실제 보행 거리의 하한이라 '확실히 멀 때'만 센다.
    같은 쪽으로 들어갔다 나온 스침·모서리 깎기는 바로 닿아서 안 센다. 돌려주는 값: [(직선 m, 걸음 하한 m 또는 None=못 닿음), ...]"""
    import numpy as np
    from collections import deque
    H, W = wall.shape
    t = np.asarray(track_px, float)
    if len(t) < 2:
        return []
    seg = np.diff(t, axis=0); n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int), 1)
    P = np.vstack([t[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))] + [t[-1:]])
    P = np.rint(P).astype(int)
    P = P[(P[:, 0] >= 0) & (P[:, 0] < W) & (P[:, 1] >= 0) & (P[:, 1] < H)]
    inw = wall[P[:, 1], P[:, 0]]
    ev, i, k, N = [], 0, 0, len(P)
    while i < N and k < max_runs:
        if not inw[i]:
            i += 1; continue
        j = i
        while j < N and inw[j]:
            j += 1
        if i > 0 and j < N:
            k += 1
            (ax, ay), (bx, by) = P[i - 1], P[j]
            eu = float(np.hypot(ax - bx, ay - by))
            lim = int(np.ceil(max(ratio * eu, eu + extra_m * ppm)))
            x0, y0 = max(0, min(ax, bx) - lim), max(0, min(ay, by) - lim)
            x1, y1 = min(W, max(ax, bx) + lim + 1), min(H, max(ay, by) + lim + 1)
            free = ~wall[y0:y1, x0:x1]
            seen = np.zeros_like(free); q = deque([(ay - y0, ax - x0, 0)]); seen[ay - y0, ax - x0] = True; got = None
            while q:
                y, x, d = q.popleft()
                if y == by - y0 and x == bx - x0:
                    got = d; break
                if d >= lim:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if (dy or dx) and 0 <= yy < free.shape[0] and 0 <= xx < free.shape[1] and free[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True; q.append((yy, xx, d + 1))
            if got is None:
                ev.append((round(eu / ppm, 2), None))
        i = j
    return ev


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _die_with_parent():
    """부모가 어떻게 죽든(SIGKILL 포함) 이 프로세스도 SIGTERM 을 받게 한다(PR_SET_PDEATHSIG, 리눅스)."""
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)
    except Exception:
        pass


def _on_term():
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # terminate 를 정상 종료 경로로(자식 정리)


def log(out, role, msg):
    line = f'{time.strftime("%H:%M:%S")} [{role}] {msg}'
    print(line, flush=True)
    with open(Path(out) / f'{role}.log', 'a') as f:
        f.write(line + '\n')


def wait_span(store, need):
    from live_store import read_span
    while True:
        s = read_span(store)
        if s and s[1] - s[0] >= need:
            return s
        time.sleep(1.0)


class _Far(dict):
    """np.load 결과(NpzFile)처럼 .files 를 가진 사전 — _scan 이 FAR.files 로 표본 유무를 본다."""
    @property
    def files(self):
        return list(self.keys())


class Loc:
    """한 프로세스 안의 위치추정기 묶음(층별 RTLocalizer). floor_id.py 와 같은 설정·같은 함수."""

    def __init__(self, a):
        self.a, self.out, self.store = a, Path(a.out), str(Path(a.store).resolve())
        import floor_id as F
        self.cfg = F.load_cfg()
        self.b = F.setup_env(self.cfg, a.building)
        import numpy as np
        import rt_localizer as RT, run_floor_id as R, twoway as TW
        import metric_fit_v2 as V
        self.np, self.RT, self.R, self.TW, self.V = np, RT, R, TW, V
        self.W = float(self.cfg['consts']['window_s'])
        self.DT = float(self.cfg['consts']['dt_s'])
        self.SC = F.map_scales(self.b, R, int(self.b['n_maps']))
        self.booted = set()

    def log(self, msg):
        log(self.out, self.a.role, msg)

    def refresh(self, Z):
        """저장소를 위치추정기의 기존 읽기 함수로 다시 읽힌다(알고리즘 불변).
        장면 라벨 캐시(초 k → 라벨)는 T0+k-1.5 ~ T0+k 시각의 최근접 표본으로 만든다. 그 시각이 이전 마지막 표본보다
        뒤였던 라벨은 새 표본이 더 가까울 수 있으므로 지운다(그 앞은 이전 마지막 표본이 늘 더 가깝다).
        이렇게 하면 캐시가 있어도 '지금 있는 데이터로 처음부터 계산한 값' 과 같다(2026-10-07 85·100초 대조 같음)."""
        L = Z.L
        t_old = float(L.T[-1]) if getattr(L, 'T', None) is not None and len(L.T) else None
        t0_old = float(L.T[0]) if t_old is not None else None
        L._load_odom(self.store)
        with self.np.load(f'{self.store}/far_points.npz') as f:
            L.FAR = _Far({k: f[k] for k in f.files})
        c = getattr(L, '_lscn_cache', None)
        if c and t0_old is not None and abs(float(L.T[0]) - t0_old) > 1e-6:
            c.clear()                                   # 초기화로 시간 원점(T[0])이 바뀌었다 → 장면 캐시 열쇠가 모두 어긋난다
        elif c and t_old is not None:
            for k in [k for k in c if L.T[0] + k > t_old - 1e-6]:
                del c[k]

    def ensure_boot(self, Z, m):
        """floor_id.py 처럼 지도를 연 뒤 시동 정렬을 한 번. fit_window_v2 는 β·시동을 창 안에서 다시 구하고 끝나면
        상태를 되돌리므로 이 값은 정합 결과에 쓰이지 않는다(2026-10-07 실측: 있든 없든 100·250초 결과 같음).
        15분(보관 기간) 뒤 층이 바뀌어 지도를 새로 열면 첫 15초 표본이 지워져 있을 수 있어, 실패해도 넘어간다."""
        if m not in self.booted:
            try:
                Z.bootstrap(Z.L.T[0], boot_s=15.0)
            except Exception as e:
                self.log(f'{m}층 시동 정렬 건너뜀({type(e).__name__}: {e}) — 정합에는 영향 없음')
            self.booted.add(m)

    def _cache_key(self, m):
        """2026-10-13 esy: 지도 준비 저장본 열쇠 — 그 층 지도·의미지도 파일, 위치추정 코드, SEMLOC 환경, 생성 인자의 md5.
        하나라도 바뀌면 열쇠가 달라져 처음부터 다시 만든다."""
        import hashlib, glob
        h = hashlib.md5()
        h.update(repr((self.a.building, m, float(self.R.MID), 0.0, 3.0, sys.version, self.np.__version__,
                       sorted((k, v) for k, v in os.environ.items() if k.startswith('SEMLOC_')))).encode())
        files = []
        for k in ('SEMLOC_MAPS_ROOT', 'SEMLOC_SEMMAPS_ROOT'):
            root = os.environ.get(k)
            if root:
                files += sorted(glob.glob(f'{root}/{m}F/**/*', recursive=True))
        files += sorted(glob.glob(str(PKG / 'localizer' / '*.py'))) + [__file__]
        for fp in files:
            if os.path.isfile(fp):
                h.update(fp.encode()); h.update(open(fp, 'rb').read())
        return h.hexdigest()[:16]

    def open_floor(self, m):
        """지도 준비는 주행 데이터를 쓰지 않는다 — 표본이 하나만 있어도 연다(걷는 동안 미리).
        2026-10-13 esy: 준비 결과(RTLocalizer 생성 + metric_fit_v2._maps)는 지도·코드로만 정해지므로 처음 한 번 디스크에 저장하고
        다음부터 읽는다(로봇 9개 층 약 76초 → 수 초). 주행 데이터에 기대는 값(오돔·FAR·저장소 경로)은 읽은 뒤 refresh 로 새로 채운다.
        생성자의 방향 복구는 꺼져 있고(recover_directions=False) 장면 캐시는 비어 있는 채 저장된다. --map-cache 0 이면 예전처럼 매번 만든다."""
        import pickle, gzip
        t1 = time.time(); Z = None; src = '새로 만듦'
        cdir = Path(self.a.map_cache_dir); f = None
        if self.a.map_cache:
            try:
                key = self._cache_key(m); f = cdir / f'{self.a.building}_{m}F_{key}.pkl.gz'
                if f.exists():
                    with gzip.open(f, 'rb') as fh:
                        Z = pickle.load(fh)
                    Z.L.dir = self.store; src = '저장본'
            except Exception as e:
                Z = None; self.log(f'{m}층 지도 저장본 못 읽음({type(e).__name__}: {e}) — 새로 만든다')
        if Z is None:
            Z = self.RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=self.R.MID, data_dir=self.store, bag=None)
            self.V._maps(Z, m)                                  # 지도로만 정해지는 정합용 판(벽 거리장·건물 밖 거리·경로 벽)
            if self.a.map_cache and f is not None:
                try:
                    cdir.mkdir(parents=True, exist_ok=True)
                    far, Z.L.FAR = Z.L.FAR, None                # 열린 npz 는 저장 못 함 — refresh 가 다시 채운다
                    tmp = f.with_name(f'.{f.name}.{os.getpid()}.tmp')
                    with gzip.open(tmp, 'wb', compresslevel=1) as fh:       # 층당 약 365→100 MB(9개 층 약 0.9 GB), 풀기 약 1.4 s(PC)
                        pickle.dump(Z, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp, f); Z.L.FAR = far
                    for old in cdir.glob(f'{self.a.building}_{m}F_*.pkl*'):   # 같은 층 옛 열쇠 저장본 정리
                        if old != f:
                            old.unlink(missing_ok=True)
                except Exception as e:
                    self.log(f'{m}층 지도 저장본 못 씀({type(e).__name__}: {e}) — 이번엔 그냥 쓴다')
        self.refresh(Z)
        self.booted.discard(m)
        self.log(f'{m}층 지도 준비 {time.time() - t1:.1f}s ({src})')
        return Z

    def fit(self, Z, m, t_now):
        np, L = self.np, Z.L
        t1 = time.time()
        self.refresh(Z)
        self.ensure_boot(Z, m)
        r = self.V.fit_window_v2(Z, t_now, self.SC[m], floor=m, W=self.W, dt=self.DT)
        pts, yaw = L._scan(t_now)
        d_obs = self.TW.sector_obs(np.asarray(pts, float) * [1, r['mirror']], np.deg2rad(r['phi']) + yaw + r['beta'])
        tw = float(self.TW.twoway_score(getattr(L, 'wall_real', L.wall), *r['pos'], r['ppm'], d_obs)[0])
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            # Jetson 은 GPU 메모리가 곧 시스템 RAM 이다. 정합 중 torch 가 잡아 둔 임시 메모리(프로세스당 ~340 MB)를
            # 정합 사이에 돌려준다(계산 결과와 무관, 2026-10-07 로봇 실측 예약 410→74 MB).
            torch.cuda.empty_cache()
        # 2026-10-13 esy: 고른 자세로 창 경로를 지도에 올려 '벽 반대편 통과' 횟수(층 순위에서 불가능한 층을 거르는 데 쓴다)
        try:
            import csv as _csv, pose_tools as _PT
            _rows = [q for q in _csv.DictReader(open(f'{self.store}/odom_link.csv')) if float(r['t']) - self.W <= float(q['t_sec']) <= float(r['t'])]
            _tr = [_PT.map_pose_px(r, float(q['fastlio_x']), float(q['fastlio_y']), float(q['fastlio_yaw_rad']))[:2] for q in _rows]
            _ev = wall_passes(self.V._maps(Z, m)['wall_tr'], _tr, float(r['ppm']))
        except Exception as _e:
            _ev = None; self.log(f'{m}층 벽 통과 판정 실패: {_e}')
        return dict(map=m, t=float(r['t']), t_now=float(t_now), twoway=tw, ppm=float(r['ppm']), phi=float(r['phi']),
                    beta=float(r['beta']), mirror=int(r['mirror']), pos=[float(v) for v in r['pos']],
                    anchor=[float(v) for v in r['anchor']], wall_cost_m=float(r['wall_cost_m']),
                    scene=r['scene'], sec=round(time.time() - t1, 1),
                    X_cross_m=float(r.get('X_cross_m') or 0.0), wall_pass=(None if _ev is None else len(_ev)), wall_pass_ev=(_ev or [])[:5])


def floor_part(a, part, conn):
    """floor 역할의 자식(spawn): 맡은 층 지도를 열고, t_now 를 받을 때마다 그 층들을 맞춰 하나씩 돌려준다."""
    _on_term()
    _die_with_parent()
    try:
        C = Loc(a)
        FL = {m: C.open_floor(m) for m in part}
        conn.send(('ready', part))
        while True:
            t_now = conn.recv()
            if t_now is None:
                return
            for m in part:
                try:
                    conn.send(('one', m, C.fit(FL[m], m, t_now)))
                except Exception as e:
                    conn.send(('one', m, dict(error=f'{type(e).__name__}: {e}')))
            conn.send(('done', part))
    except (EOFError, KeyboardInterrupt):
        return


def run_floor(a):
    import multiprocessing as mp
    from multiprocessing.connection import wait as mp_wait
    from live_store import read_span
    import floor_id as F
    cfg = F.load_cfg()
    b = F.setup_env(cfg, a.building)
    out, store = Path(a.out), str(Path(a.store).resolve())
    W = float(cfg['consts']['window_s'])
    floors = a.floors or list(range(1, int(b['n_maps']) + 1))
    min_span = a.min_span if a.min_span is not None else W + 10.0
    K = max(1, min(a.procs, len(floors)))
    parts = [floors[j::K] for j in range(K)]
    log(out, a.role, f'건물 {b["name"]} · 층 {floors} 를 프로세스 {K}개에 나눔 {parts} · 창 {W}s · 첫 정합까지 데이터 {min_span:.0f}s 기다림')
    wait_span(store, 0.0)                              # 표본 하나(위치추정기가 읽을 파일)가 생기면 지도부터 연다

    ctx = mp.get_context('spawn')
    kids = []
    for part in parts:
        pc, cc = ctx.Pipe()
        pr = ctx.Process(target=floor_part, args=(a, part, cc), daemon=True)
        pr.start()
        kids.append((pr, pc))
    t1 = time.time()
    for pr, pc in kids:
        pc.recv()
    log(out, a.role, f'지도 준비 끝 {time.time() - t1:.0f}s')
    trig = Path(a.trigger) if a.trigger else None
    # 실험 모드(신호 파일): 신호(파일 내용 = 창 끝 시각, 초)가 새로 바뀔 때마다 그 시각까지의 창으로 **한 바퀴만** 돈다.
    # 웹 화면의 [시작] 버튼이 60초 뒤 신호를 쓴다. 신호 사이에는 돌지 않는다(서 있는 동안의 바퀴가 결과를 덮지 않게).
    last_sig = None
    if trig:
        log(out, a.role, f'실험 신호({trig})를 기다린다 — 신호마다 한 바퀴')
    else:
        wait_span(store, min_span)
    k = 0
    recent = []                                        # 최근 바퀴 판정(다수결용)
    vote_n = 1 if trig else a.vote_n                   # 실험마다 장소가 달라 실험끼리는 다수결하지 않는다
    while True:
        txt = None
        if trig:
            while True:
                try:
                    txt = trig.read_text().strip()
                except OSError:
                    txt = None
                if txt and txt != last_sig:
                    break
                time.sleep(0.2)
            last_sig = txt
        t_pass = time.time()
        t_now = read_span(store)[1]                    # 이번 바퀴의 창 끝(모든 층 같은 시각, 저장소의 마지막 표본)
        if trig:
            import csv as _csv
            ts = [float(r['t_sec']) for r in _csv.DictReader(open(Path(store) / 'odom_link.csv'))]
            t_now = max([t for t in ts if t <= float(txt) + 1e-6] or [t_now])   # 신호 시각 이하의 마지막 표본
            log(out, a.role, f'실험 바퀴: 창 끝 {t_now:.2f} (신호 {txt})')
        for pr, pc in kids:
            pc.send(t_now)
        win, pending = {}, len(kids)
        while pending:
            for pc in mp_wait([pc for _, pc in kids]):
                try:
                    msg = pc.recv()
                except EOFError:
                    log(out, a.role, '층 정합 프로세스가 죽었다 — 끝낸다')
                    sys.exit(1)
                if msg[0] == 'done':
                    pending -= 1
                elif 'error' in msg[2]:
                    log(out, a.role, f'{msg[1]}층 정합 실패: {msg[2]["error"]}')
                else:
                    win[msg[1]] = msg[2]
                    atomic_json(out / 'floor_progress.json', dict(t_now=t_now, done=sorted(win), total=len(floors),
                                                                  cands=[dict(map=m, twoway=round(win[m]['twoway'], 3),
                                                                              wall_cm=round(100 * win[m]['wall_cost_m'], 1), wall_pass=win[m].get('wall_pass'))
                                                                         for m in sorted(win, key=lambda m: -win[m]['twoway'])]))
        if not win:
            time.sleep(2.0)
            continue
        # 2026-10-13 esy: 고른 자세가 벽을 반대편으로 한 번이라도 넘은 층은 순위 뒤로(물리적으로 불가능). 모든 층이 넘었으면 그대로.
        _tw = lambda m: -win[m]['twoway']
        _bad = [m for m in win if a.wall_pass_rank and (win[m].get('wall_pass') or 0) > 0]
        _ok = [m for m in win if m not in _bad]
        order = (sorted(_ok, key=_tw) + sorted(_bad, key=_tw)) if _ok else sorted(win, key=_tw)
        best = order[0]
        # 다수결: 한 구간 판정은 1위 정답률이 21% 라 그대로 쓰지 않는다(floor_id/README). 최근 vote_n 바퀴의 최빈 층,
        # 동률이면 가장 최근에 나온 층. 바퀴 판정(verdict)은 그대로 남긴다.
        recent = (recent + [best])[-vote_n:]
        cnt = Counter(recent)
        vote = next(m for m in reversed(recent) if cnt[m] == max(cnt.values()))
        res = dict(t_now=t_now, verdict=best, ranking=order, vote=vote, vote_counts={str(m): c for m, c in cnt.items()},
                   vote_n=len(recent), cands=[win[m] for m in order],
                   pass_sec=round(time.time() - t_pass, 1), pass_index=k, wall_time=time.time(), procs=K,
                   signal=txt)
        atomic_json(out / 'floor.json', res)
        # 지도 위치는 다수결 층으로. 위치 작업자가 같은 층을 더 최근 창으로 맞춰 두었으면 덮어쓰지 않는다(층이 바뀌었으면 쓴다).
        use = vote if vote in win else best
        prev = _read(out / 'pose_fit.json')
        if not prev or prev.get('map') != use or prev.get('t', -1) < win[use]['t']:
            atomic_json(out / 'pose_fit.json', dict(win[use], source='floor', wall_time=time.time()))
        with open(out / 'floor_history.jsonl', 'a') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
        log(out, a.role, f'바퀴 {k}: 판정 {best}층 (2위 {order[1] if len(order) > 1 else "-"}층) · 다수결 {vote}층 {dict(cnt)} · '
                          f'twoway {win[best]["twoway"]:.3f} · 벽비용 {100 * win[best]["wall_cost_m"]:.1f}cm · '
                          f'벽 통과로 뒤로 보낸 층 {sorted(_bad) if _ok else "(모든 층 통과 — 적용 안 함)"} · '
                          f'{res["pass_sec"]:.0f}s')
        k += 1


def run_pose(a):
    out = Path(a.out)
    C = Loc(a)
    C.log(f'건물 {C.b["name"]} · 판정 층 하나를 계속 다시 맞춘다 · 창 {C.W}s')
    wait_span(C.store, 0.0)
    Z, cur = None, None
    while True:
        fl = _read(out / 'floor.json')
        if not fl:
            time.sleep(2.0)
            continue
        m = int(fl.get('vote', fl['verdict']))       # 다수결 층을 따라간다
        if m != cur:
            Z, cur = C.open_floor(m), m
        try:
            C.refresh(Z)
            r = C.fit(Z, m, float(Z.L.T[-1]))
        except Exception as e:
            C.log(f'{m}층 정합 실패: {type(e).__name__}: {e}')
            time.sleep(2.0)
            continue
        # 층 판별이 더 최근 창으로 같은 층을 맞췄으면 그것을 덮어쓰지 않는다.
        prev = _read(out / 'pose_fit.json')
        if not prev or prev.get('map') != m or prev.get('t', -1) < r['t']:
            atomic_json(out / 'pose_fit.json', dict(r, source='pose', wall_time=time.time()))
        C.log(f'{m}층 위치 보정 · px {r["pos"][0]:.0f},{r["pos"][1]:.0f} · 벽비용 {100 * r["wall_cost_m"]:.1f}cm · {r["sec"]:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=('floor', 'pose'), required=True)
    ap.add_argument('--building', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--floors', type=int, nargs='*', default=None, help='floor 역할: 볼 층(기본 전부)')
    ap.add_argument('--min-span', type=float, default=None, help='첫 정합 전에 필요한 데이터 길이(초). 기본 창 길이+10')
    ap.add_argument('--vote-n', type=int, default=5, help='floor 역할: 다수결에 쓸 최근 바퀴 수')
    ap.add_argument('--wall-pass-rank', type=int, default=1, help='floor 역할: 1=벽을 반대편으로 넘은 층을 순위 뒤로(기본), 0=끔 (2026-10-13 esy)')
    ap.add_argument('--trigger', default=None, help='floor 역할: 이 파일이 생기면 첫 바퀴(파일 내용 = 창 끝 시각, 초)')
    ap.add_argument('--map-cache', type=int, default=1, help='1=지도 준비 결과를 디스크에 저장해 다음 실행에 읽음(지도·코드 md5 가 같을 때만, 2026-10-13 esy), 0=매번 새로')
    ap.add_argument('--map-cache-dir', default=str(HERE / 'cache' / 'map_prep'), help='지도 준비 저장본 폴더')
    ap.add_argument('--procs', type=int, default=3, help='floor 역할: 층을 나눠 맡을 프로세스 수(로봇 8코어 중 구조점·FAST-LIO 몫을 남긴다)')
    a = ap.parse_args()
    _on_term()
    _die_with_parent()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (run_floor if a.role == 'floor' else run_pose)(a)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
