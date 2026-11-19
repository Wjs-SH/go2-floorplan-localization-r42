# -*- coding: utf-8 -*-
"""live_viewer.py — 실시간 노드 결과를 브라우저로 보는 가벼운 웹 화면(ROS 불필요, 파일만 읽는다).

    python live_viewer.py --root /dev/shm/rt_live --building kci [--port 8765]
    → 노트북 브라우저에서 http://<로봇 IP>:8765

  왼쪽  : FAST-LIO 좌표 — 최근 창(60초) 누적 구조점 · 이동 경로 · 지금 로봇 위치·방향(초당 5번 갱신)
  오른쪽: 다수결 층 안내도 — 같은 누적점을 가장 최근 정합으로 옮긴 것 · 추정 위치·방향 · 층 순위
읽기만 한다. 노드(rt_live_node.py)가 쓰는 store(data/) 와 out/(live_odom.json · pose_fit.json · floor.json · status.jsonl)을 본다.
"""
import argparse
import csv
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from pose_tools import map_pose_px  # noqa: E402
import r1_trial as T  # noqa: E402   R1: 실험 세대 — 다른 실험의 결과를 '지금 결과'로 그리지 않는다
import r1_labels as LB  # noqa: E402  R1: 단계별 평가 라벨


def _low_mask(hmax, rng, h0=0.37, gnd=0.15, ceil=1.70, thr=0.45):
    """2026-10-13 esy: 칸 최고 높이가 그 거리에서 보이는 한계 높이의 thr 배도 못 미치면 낮은 것(책상·의자 등)."""
    vis = np.clip(np.minimum(h0 + rng * np.tan(np.radians(15.0)), ceil), gnd + 0.05, None)
    return (hmax - gnd) / (vis - gnd) <= thr


def _A(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [-s, -c]])


def _R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


class State:
    def __init__(self, root, maps_root, window, per, trigger=None, archive=None):
        self.root, self.maps_root, self.window, self.per = Path(root), Path(maps_root), window, per
        self.archive = Path(archive) if archive else None
        self.meta_path = (self.archive.parent / 'viewer_meta.json') if self.archive else None
        self.meta = (_json(self.meta_path) if self.meta_path else None) or dict(truth=None, note='')   # 정답 층·메모(다음 실험에도 유지)
        self.trigger = Path(trigger) if trigger else None
        self.cache = {}            # sid → (pts_body(role!=4), local_pts, local_pose)
        self.lock = threading.Lock()
        self.exp_path = self.root / 'out' / 'experiment.json'
        self.exp = _json(self.exp_path) or dict(state='idle')
        # E5RT2: explicit, expiring operator grant for NEW physical actions (aim/focus/HOME), separate from the trial state.
        #   [시작] (walking) alone never arms. The grant is revoked when the trial leaves walking or on [동작 중지].
        self.e5_grant, self.e5_abort = None, False
        # 2026-10-14 esy: [시작]/[멈춤]에 로스백 녹화를 묶는다 — 노드를 켤 때 환경변수 RT_RECORD_CHAPTER=<record_chapter.sh 경로>를 주면 켜짐.
        #   [시작] → 'start exp_<시분초>', [멈춤] → RT_RECORD_TAIL 초(기본 5) 뒤 'stop'(정지 상태로 끝나게). 이름은 실험 보관 폴더와 같은 앞부분.
        self.rec_script = os.environ.get('RT_RECORD_CHAPTER', '')
        self.rec_tail = float(os.environ.get('RT_RECORD_TAIL', '5'))
        # 2026-10-16: 실험과 따로 켜고 끄는 [녹화] 버튼(10-14 웹 초안의 '녹화 버튼'). 스크립트는 같은 record_chapter.sh(한 번에 하나만).
        self.man_script = self.rec_script or str(HERE.parent / 'field_tools' / 'record_chapter.sh')
        self.man_rec = None
        self.recs = {}             # 녹화 이름 → dict(started, stopping, done)
        self.pending_marks = []    # [시작] 전에 찍은 정답 위치(출발 위치) — 다음 [시작] 때 그 실험에 붙인다
        # 정답 지점: 바닥 표시 지점에 세우고 누르면 그 순간 시각·자세를 남긴다(로스백 없이도 채점할 점 정답)
        self.marks_path = (self.archive.parent / 'checkpoints.jsonl') if self.archive else (self.root / 'out' / 'checkpoints.jsonl')
        self.persist = self.archive.parent if self.archive else T.default_persist_dir()
        self.tp = T.TrialPointer(self.root / 'out', self.persist)    # R1: out/trial_current.json (실험 모드에서만 생긴다)

    # ---------- 실험(시작·멈춤) ----------
    def _save_exp(self):
        try:
            tmp = self.exp_path.with_name('.experiment.json.tmp')
            tmp.write_text(json.dumps(self.exp))
            os.replace(tmp, self.exp_path)
        except OSError:
            pass

    def start(self):
        import time
        with self.lock:
            if self.trigger is None or self.exp.get('state') in ('walking', 'waiting', 'computing'):
                return self.exp
            self.exp = dict(state='walking', start=time.time())
            self.e5_clear_t = self.exp['start']                            # KCI 10-17: 새 실험 = 지난 이벤트 지움
            tr = self.tp.start(time.strftime('exp_%H%M%S', time.localtime(self.exp['start'])))   # R1: 새 세대 + 이전 결과 치움
            self.exp.update(trial_id=tr['trial_id'], generation_id=tr['generation_id'])
            if self.pending_marks:
                self.exp['marks'], self.pending_marks = self.pending_marks, []
            if self.rec_script:
                name = time.strftime('exp_%H%M%S', time.localtime(self.exp['start']))
                self.exp['record'] = dict(name=name, state='켜는 중')
                self.recs[name] = dict(started=None, stopping=False, done=False)
                threading.Thread(target=self._rec_start, args=(name,), daemon=True).start()
            self._save_exp()
            return self.exp

    # 녹화(2026-10-15 정리): 녹화마다 self.recs[name] = 켜기 결과(started)·끄기 예약(stopping)·끝(done)을 따로 둔다.
    #   - 끄기는 그 녹화를 우리가 켰을 때만(켜기 실패 = 손으로 켜 둔 녹화가 있었던 경우 등 → 그 녹화는 안 건드림).
    #   - [멈춤] 직후 [시작]이면 새 녹화는 앞 녹화가 꺼진 뒤에 켠다(녹화 스크립트는 한 번에 하나만).
    #   - 화면용 글은 지금 실험의 exp['record'] 에만 쓰고, 보관이 먼저 끝났으면 보관본 experiment.json 도 고친다.
    def _rec_run(self, cmd, name):
        """녹화 스크립트를 돌리고 (성공, 요약 줄, 로스백 폴더). 환경 스크립트 줄(Setup unitree …)은 거른다."""
        import subprocess
        try:
            r = subprocess.run(['bash', self.rec_script, cmd] + ([name] if cmd == 'start' else []),
                               capture_output=True, text=True, timeout=60)
        except Exception as ex:
            return False, f'{type(ex).__name__}: {ex}', None
        msg = [m.strip() for m in ((r.stdout or '') + (r.stderr or '')).splitlines()
               if m.strip() and not m.startswith('Setup unitree')]
        bag = None
        if cmd == 'start':
            hit = [m for m in msg if m.startswith('녹화 시작:')]
            last = hit[0] if hit else (msg[-1] if msg else '')
            if hit:
                bag = hit[0].split(':', 1)[1].strip().split()[0]
        else:
            last = ' / '.join(m for m in msg if 'Hz' in m or '없음' in m or '녹화 끝' in m or '길이' in m or '강제' in m)[:400] \
                or (msg[-1] if msg else '')
        return r.returncode == 0, last, bag

    def _rec_show(self, name, **kw):
        """지금 실험의 녹화 글을 고친다(다른 실험의 늦은 결과면 무시). 보관이 이미 됐으면 보관본도."""
        with self.lock:
            rec = self.exp.get('record')
            if rec is None or rec.get('name') != name:
                return
            rec.update(kw)
            self._save_exp()
            arch, rec = self.exp.get('archived'), dict(rec)
        if arch:
            try:
                p = Path(arch) / 'experiment.json'
                e = json.loads(p.read_text())
                e['record'] = rec
                p.write_text(json.dumps(e, ensure_ascii=False, indent=1))
            except (OSError, ValueError):
                pass

    def _rec_start(self, name):
        import time
        t_end = time.time() + 30
        while time.time() < t_end:                      # 앞 녹화를 끄는 중이면 기다린다
            with self.lock:
                busy = [n for n, r in self.recs.items() if n != name and r['stopping'] and not r['done']]
            if not busy:
                break
            self._rec_show(name, state=f'앞 녹화({busy[0]}) 끝나기 기다리는 중')
            time.sleep(0.5)
        ok, last, bag = self._rec_run('start', name)
        with self.lock:
            r = self.recs[name]
            r['started'] = ok
            stopping = r['stopping']
        self._rec_show(name, state=('녹화 중' + (' · 곧 정지' if stopping else '')) if ok else '녹화 실패',
                       msg=last, **({'bag': bag} if bag else {}))

    def _rec_stop_later(self, name, tail):
        import time
        time.sleep(tail)
        t_end = time.time() + 30
        while time.time() < t_end:                      # 켜는 중이면 켜기가 끝날 때까지
            with self.lock:
                started = self.recs[name]['started']
            if started is not None:
                break
            time.sleep(0.5)
        if started:
            self._rec_show(name, state='끄는 중')
            ok, last, _ = self._rec_run('stop', name)
            self._rec_show(name, state='녹화 끝' if ok and '강제' not in last else '끄기 실패', msg=last)
        with self.lock:
            self.recs[name]['done'] = True

    def _rec_schedule_stop(self, tail):
        """지금 실험의 녹화를 tail 초 뒤 끄도록 예약(lock 안에서 부른다). 우리가 켠 녹화가 아니면 아무것도 안 한다."""
        rec = self.exp.get('record')
        r = self.recs.get(rec['name']) if rec else None
        if not r or r['stopping'] or r['started'] is False:
            return False
        r['stopping'] = True
        threading.Thread(target=self._rec_stop_later, args=(rec['name'], tail), daemon=True).start()
        return True

    def record(self, on, camera=False):
        """[녹화] 버튼: on=True 면 'start man_<시분초> [--camera]', False 면 'stop'. 실험 녹화와 같은 스크립트라 동시에 둘은 안 된다."""
        import time
        with self.lock:
            cur = self.man_rec
            if on:
                if cur and cur.get('state') in ('켜는 중', '녹화 중', '끄는 중'):
                    return dict(cur)
                if not Path(self.man_script).is_file():
                    self.man_rec = dict(state='녹화 실패', msg=f'녹화 스크립트 없음: {self.man_script}')
                    return dict(self.man_rec)
                name = time.strftime('man_%H%M%S')
                self.man_rec = dict(name=name, state='켜는 중', camera=bool(camera), t=time.time())
            else:
                if not cur or cur.get('state') not in ('녹화 중', '켜는 중'):
                    return dict(cur or dict(state='녹화 중 아님'))
                cur['state'] = '끄는 중'
                name = cur['name']
        threading.Thread(target=self._man_rec_run, args=(on, name, camera), daemon=True).start()
        return dict(self.man_rec)

    def _man_rec_run(self, on, name, camera):
        import subprocess
        cmd = ['bash', self.man_script] + (['start', name] + (['--camera'] if camera else []) if on else ['stop'])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            msg = [m.strip() for m in ((r.stdout or '') + (r.stderr or '')).splitlines() if m.strip() and not m.startswith('Setup unitree')]
            ok = r.returncode == 0
        except Exception as ex:
            msg, ok = [f'{type(ex).__name__}: {ex}'], False
        with self.lock:
            if self.man_rec is None or self.man_rec.get('name') != name:
                return
            if on:
                hit = [m for m in msg if m.startswith('녹화 시작:')]
                self.man_rec.update(state='녹화 중' if ok else '녹화 실패', msg=(hit[0] if hit else (msg[-1] if msg else ''))[:300])
            else:
                self.man_rec.update(state='녹화 끝' if ok else '끄기 실패',
                                    msg=(' / '.join(m for m in msg if 'Hz' in m or '녹화 끝' in m or '길이' in m or '강제' in m) or (msg[-1] if msg else ''))[:400])

    def mark(self, name, floor=None, px=None, py=None, heading=None, role=None):
        """[정답 위치]: 안내도에서 로봇이 실제로 서 있는 곳을 클릭한다(2026-10-15, 로스백 없는 위치 정답).
        저장: 시각·층·클릭 픽셀·그때 FAST-LIO 자세·추정 위치와 오차 → logs/live_<월일>/checkpoints.jsonl 한 줄,
        실험 중이면(결과가 나온 뒤 다음 [시작] 전까지 포함) 실험 기록 marks 에도. 결과가 이미 보관됐으면 보관본에도."""
        import time
        rows = self._rows()
        live = _json(self.root / 'out' / 'live_odom.json')
        rec = dict(t=time.time(), name=str(name or '')[:40], live_odom=live, last_sample=(rows[-1] if rows else None),
                   role=role if role in ('start', 'checkpoint', 'end') else None, trial_id=(self._pointer() or {}).get('trial_id'))
        try:
            rec['live_odom_age_s'] = round(time.time() - (self.root / 'out' / 'live_odom.json').stat().st_mtime, 2)
        except OSError:
            rec['live_odom_age_s'] = None
        if floor is not None and px is not None and py is not None:
            rec.update(floor=int(floor), px=round(float(px), 1), py=round(float(py), 1))
            if heading is not None:                      # 2026-10-15: 로봇이 보는 방향(안내도 그림 좌표: x 오른쪽, y 아래, 라디안)
                rec['heading'] = round(float(heading), 4)
        with self.lock:
            exp = dict(self.exp)
        rec.update(exp_state=exp.get('state'), exp_start=exp.get('start'))
        if 'floor' in rec:
            rec.update(self._truth_eval(rec, rows, live, exp))
        with self.lock:
            arch = None
            if self.exp.get('state') in ('walking', 'waiting', 'computing', 'done') and self.exp.get('start') == exp.get('start'):
                if not rec['name']:
                    rec['name'] = f"P{len(self.exp.get('marks', [])) + 1}"
                self.exp.setdefault('marks', []).append(self._mark_brief(rec))
                self._save_exp()
                if self.exp.get('state') == 'done':          # 결과 뒤에 찍은 것은 다음 [시작] 의 출발 위치로도(마지막 것 하나만)
                    nb = {k: v for k, v in self._mark_brief(rec).items() if k not in ('est', 'err_px', 'err_m', 'basis')}
                    self.pending_marks = [m for m in self.pending_marks if not m.get('from_prev_end')] + [dict(nb, before_start=True, from_prev_end=True)]
                arch, marks = self.exp.get('archived'), list(self.exp['marks'])
            elif self.exp.get('state') in ('idle', None):
                if not rec['name']:
                    rec['name'] = f"P{len(self.pending_marks) + 1}"
                self.pending_marks.append(dict(self._mark_brief(rec), before_start=True))
            try:
                self.marks_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.marks_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                rec['saved'] = str(self.marks_path)
            except OSError as ex:
                rec['error'] = f'{type(ex).__name__}: {ex}'
        if arch:                                        # 결과가 이미 보관됐으면 보관본에도
            try:
                p = Path(arch) / 'experiment.json'
                e = json.loads(p.read_text())
                e['marks'] = marks
                p.write_text(json.dumps(e, ensure_ascii=False, indent=1))
            except (OSError, ValueError):
                pass
        return dict(self._mark_brief(rec), saved=rec.get('saved'), error=rec.get('error'))

    def unmark(self):
        """2026-10-15 밤: 마지막 정답 위치 하나 지우기. 시작 전이면 출발 위치 목록에서, 그 밖이면 이 실험 marks 에서(결과 뒤라 다음 출발로
        넘긴 사본도 같이). checkpoints.jsonl 은 덧붙이기만 하므로 지운 줄(unmark)을 남긴다."""
        import time
        with self.lock:
            gone, arch, marks = None, None, None
            if self.exp.get('state') in ('idle', None):
                if self.pending_marks:
                    gone = self.pending_marks.pop()
            elif self.exp.get('marks'):
                gone = self.exp['marks'].pop()
                if any(m.get('t') == gone.get('t') for m in self.pending_marks):   # 다음 출발로 넘긴 사본이었으면 그 앞(결과 뒤에 찍은 것)으로 되돌림
                    self.pending_marks = [m for m in self.pending_marks if not m.get('from_prev_end')]
                    end = self.exp.get('end') or float('inf')
                    prev = [m for m in self.exp['marks'] if float(m.get('t') or 0) > end]
                    if prev:
                        nb = {k: v for k, v in prev[-1].items() if k not in ('est', 'err_px', 'err_m', 'basis')}
                        self.pending_marks.append(dict(nb, before_start=True, from_prev_end=True))
                self._save_exp()
                arch, marks = self.exp.get('archived'), list(self.exp['marks'])
            if gone is None:
                return dict(error='지울 정답 위치 없음')
            try:
                with open(self.marks_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(t=time.time(), unmark=gone, exp_state=self.exp.get('state'), exp_start=self.exp.get('start')),
                                       ensure_ascii=False) + '\n')
            except OSError:
                pass
        if arch:
            try:
                p = Path(arch) / 'experiment.json'
                e = json.loads(p.read_text())
                e['marks'] = marks
                p.write_text(json.dumps(e, ensure_ascii=False, indent=1))
            except (OSError, ValueError):
                pass
        return dict(removed=gone)

    @staticmethod
    def _mark_brief(rec):
        return {k: rec[k] for k in ('t', 'name', 'floor', 'px', 'py', 'heading', 'est', 'err_px', 'err_m', 'basis', 'role',
                                    'estimate_id', 'reference_time', 'err_null_reason', 'trial_id') if k in rec}

    def _truth_eval(self, rec, rows, live, exp):
        """R1: 찍은 정답 위치와 '같은 실험·같은 순간'의 추정 위치 오차만 낸다.
        필요: 역할(start/checkpoint/end) · 지금 실험(세대)의 정합(estimate_id) · 클릭 순간의 신선한 FAST-LIO 자세(reference_time).
        예전처럼 정합 창 끝(다른 시각)의 자세와 짝짓지 않는다(10-15: 61분 뒤 클릭 ↔ 창 끝 자세 → err 12.33 m).
        m 는 정합 축척(ppm, 축척 탐색 결과)으로 나눈 값이라 참고용(안내도는 미터 지도가 아님)."""
        if rec.get('role') not in ('start', 'checkpoint', 'end'):
            return dict(basis='오차 안 냄: 역할(start/checkpoint/end) 없음', err_null_reason='NO_ROLE')
        out = self.root / 'out'
        fl, pfit, _ = self._filter_current(_json(out / 'floor.json'), _json(out / 'pose_fit.json'))
        f = rec['floor']
        cands = {int(c['map']): c for c in fl['cands']} if fl else {}
        fit = cands.get(f)
        if pfit and int(pfit['map']) == f and (fit is None or pfit.get('t', -1) > fit['t']):
            fit = pfit
        if fit is None:
            return dict(basis='추정 없음(이 실험·이 층 정합 전)', err_null_reason='NO_CURRENT_TRIAL_ESTIMATE')
        age = rec.get('live_odom_age_s')
        if not (live and live.get('odom_link')) or age is None or age > 2.0:
            return dict(basis='추정 없음(클릭 순간 자세가 없거나 오래됨)', err_null_reason='NO_FRESH_ODOM_AT_CLICK')
        ex, ey, _ = map_pose_px(fit, *live['odom_link'])
        d = float(np.hypot(ex - rec['px'], ey - rec['py']))
        return dict(est=[round(ex, 1), round(ey, 1)], err_px=round(d, 1), err_m=round(d / float(fit['ppm']), 2),
                    ppm=round(float(fit['ppm']), 2), basis='클릭 순간 자세를 이 실험의 정합으로 옮김', fit_t=fit.get('t'),
                    estimate_id=f"{fit.get('trial_id')}:{fit.get('source', 'floor')}:{float(fit['t']):.3f}",
                    reference_time=live.get('t'), reference_clock='odom_sensor_s')

    def stop(self):
        import time
        with self.lock:
            if self.exp.get('state') == 'walking':
                self.exp.update(state='waiting', end=time.time())
                self.tp.set_state('STOPPED')
                if self._rec_schedule_stop(self.rec_tail):
                    self.exp['record'].update(state=f'{self.rec_tail:.0f}초 뒤 정지')
                self._save_exp()
            return self.exp

    def set_meta(self, truth, note):
        """정답 층·메모. 결과가 이미 보관된 실험이면 그 기록도 고친다(실험 뒤에 정답을 골라도 된다)."""
        with self.lock:
            self.meta = dict(truth=int(truth) if str(truth).isdigit() else None, note=str(note or '')[:200])
            if self.meta_path:
                try:
                    self.meta_path.parent.mkdir(parents=True, exist_ok=True)
                    self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False))
                except OSError:
                    pass
            arch = self.exp.get('archived')
        if arch:
            try:
                p = Path(arch) / 'experiment.json'
                e = json.loads(p.read_text())
                e.update(self.meta)
                p.write_text(json.dumps(e, ensure_ascii=False, indent=1))
                self._write_index()
            except Exception:
                pass
        return self.meta

    def _write_index(self):
        try:
            import experiments
            experiments.write_index(self.archive.parent.parent)
        except Exception as ex:
            with self.lock:
                self.exp['index_error'] = f'{type(ex).__name__}: {ex}'

    def reset(self):
        """[초기화]: FAST-LIO 끄기 → 노드에 RESET(데이터 비우기) → FAST-LIO 켜기. 로봇은 세워 둬야 한다(켜질 때 IMU 기준)."""
        import time
        with self.lock:
            if self.exp.get('state') == 'resetting':
                return self.exp
            if self.exp.get('state') == 'walking':          # 걷는 중 초기화면 녹화도 바로 끈다
                self._rec_schedule_stop(0)
            self.exp = dict(state='resetting', t=time.time(), step='FAST-LIO 끄는 중')
            self.e5_clear_t = time.time()                                  # KCI 10-17: 아래 E5 칸의 지난 이벤트 지우기
            self.tp.set_state('CANCELLED', why='reset')                   # R1: 진행 중이던 세대의 늦은 결과는 지금 결과가 아니다
            self._save_exp()
        threading.Thread(target=self._do_reset, daemon=True).start()
        return self.exp

    def _do_reset(self):
        import subprocess
        import time
        sh = str(HERE / 'tools' / 'sensors_go2.sh')
        out = self.root / 'out'

        def step(txt):
            with self.lock:
                self.exp['step'] = txt
                self._save_exp()
        try:
            subprocess.run(['bash', sh, 'stop_fastlio'], timeout=40, capture_output=True)
            step('쌓인 데이터 비우는 중')
            done0 = (out / 'RESET_DONE').stat().st_mtime if (out / 'RESET_DONE').exists() else 0
            (out / 'RESET').write_text(str(time.time()))
            t_end = time.time() + 10
            while time.time() < t_end and not ((out / 'RESET_DONE').exists() and (out / 'RESET_DONE').stat().st_mtime > done0):
                time.sleep(0.2)
            with self.lock:
                self.cache.clear()                      # 표본 번호가 LIVE_000000 부터 다시 시작한다
            step('FAST-LIO 켜는 중 — 로봇을 세워 두세요')
            subprocess.run(['bash', sh, 'start_fastlio'], timeout=40, capture_output=True)
            t_end = time.time() + 40
            while time.time() < t_end and not (out / 'live_odom.json').exists():
                time.sleep(0.3)
            step('FAST-LIO 자리 잡는 중 — 로봇을 세워 두세요')
            time.sleep(5)
            with self.lock:
                self.exp = dict(state='idle')
                self._save_exp()
        except Exception as e:
            with self.lock:
                self.exp = dict(state='idle', error=f'초기화 실패: {type(e).__name__}: {e}')
                self._save_exp()

    def pose(self, floor_sel=None):
        """화살표만(가볍게, 초당 여러 번): FAST-LIO 좌표 지금 자세 + 지금 보이는 층 정합으로 옮긴 지도 위치."""
        out = self.root / 'out'
        live = _json(out / 'live_odom.json')
        if not live:
            return {}
        fl_, pf_, _ = self._filter_current(_json(out / 'floor.json'), _json(out / 'pose_fit.json'))
        fit = self._pick_fit(floor_sel, fl_, pf_)
        me = list(map_pose_px(fit, *live['odom_link'])) if fit and live.get('odom_link') else None
        scan = None                                         # 2026-10-15: '지금 스캔'(노드가 초당 5번 씀, 화면 전용)
        try:
            import time
            p = out / 'live_scan.npz'
            if time.time() - p.stat().st_mtime < 1.5:
                with np.load(p) as z:
                    scan = np.round(z['pts'].astype(float), 2).tolist()
        except (OSError, ValueError, KeyError):
            pass
        return dict(base=live.get('base'), me=me, map=int(fit['map']) if fit else None, scan=scan)

    def _pointer(self):
        return self.tp.current()

    def _filter_current(self, fl, pfit):
        """R1: floor.json·pose_fit.json 중 지금 실험(세대)의 성공 결과만 남긴다. 나머지는 HISTORICAL/STALE 정보로만(원래 실험·시각)."""
        ptr, stale = self._pointer(), {}
        for k, rec in (('floor', fl), ('pose_fit', pfit)):
            c, info = T.classify(rec, ptr)
            if c in ('STALE_OTHER_TRIAL', 'CANCELLED', 'FAILED'):
                stale[k] = dict(info, label='HISTORICAL/STALE' if c != 'FAILED' else 'FAILED', status=c,
                                error=T.describe_failure(rec) if c == 'FAILED' else None)
        e = self.exp or {}                                 # 꼬리표 없는 옛 형식(포인터 없음): 이번 실험 신호와 다른 floor.json 도 지금 결과가 아님
        if 'floor' not in stale and fl and e.get('state') in ('computing', 'done') and e.get('signal') \
                and fl.get('signal') != e.get('signal'):
            stale['floor'] = dict(label='HISTORICAL/STALE', status='OTHER_SIGNAL', origin_signal=fl.get('signal'),
                                  origin_t=fl.get('t_now'))
            if pfit and 'pose_fit' not in stale:
                stale['pose_fit'] = dict(label='HISTORICAL/STALE', status='FOLLOWS_STALE_FLOOR', origin_t=pfit.get('t'))
        if 'floor' in stale:
            fl = None
        if 'pose_fit' in stale:
            pfit = None
        return fl, pfit, stale

    @staticmethod
    def _pick_fit(floor_sel, fl, pfit):
        cands = {int(c['map']): c for c in fl['cands']} if fl else {}
        if floor_sel and floor_sel in cands:
            return dict(cands[floor_sel])
        if fl and int(fl['verdict']) in cands:
            fit = dict(cands[int(fl['verdict'])])
            if pfit and int(pfit['map']) == int(fl['verdict']) and pfit.get('t', -1) > fit['t']:
                fit = pfit                                   # 위치 작업자가 같은 층을 더 최근 창으로 다시 맞춘 것
            return fit
        return pfit

    def tick(self):
        """멈춤 뒤: 멈춘 시각 이후 표본이 들어오면(또는 10초 지나면) 신호를 쓰고, 그 신호의 바퀴가 끝나면 결과로."""
        import time
        with self.lock:
            e = self.exp
            if e.get('state') == 'waiting':
                rows = self._rows()
                t_last = float(rows[-1]['t_sec']) if rows else 0.0
                if t_last >= e['end'] or time.time() > e['end'] + 10:
                    sig = f"{e['end']:.3f}"
                    try:                                # 2026-10-16: 층 작업자 '걸은 창'용 실험 시작 시각(신호보다 먼저 써 둔다)
                        Path(str(self.trigger) + '.json').write_text(json.dumps(dict(signal=sig, start=e.get('start'))))
                    except OSError:
                        pass
                    self.tp.set_state('COMPUTING', signal=sig)
                    self.trigger.write_text(sig)
                    e.update(state='computing', signal=sig)
                    self._save_exp()
            elif e.get('state') == 'computing':
                fl = _json(self.root / 'out' / 'floor.json')
                cls, _ = T.classify(fl, self.tp.current())
                mine = bool(fl and fl.get('signal') == e.get('signal') and cls in ('CURRENT', 'LEGACY_UNTRACKED', 'FAILED'))
                if mine and fl.get('status') == 'FAILED':  # R1: 작업자가 남긴 실패 끝 기록 — 실제 오류 종류·층으로 표시
                    e.update(state='done', verdict=None, ranking=[], t_now=fl.get('t_now'), error=T.describe_failure(fl),
                             error_kinds=fl.get('error_kinds'))
                    self.tp.set_state('FAILED', error_kinds=fl.get('error_kinds'))
                    self._save_exp()
                elif mine:
                    e.update(state='done', verdict=fl['verdict'], ranking=fl['ranking'], t_now=fl['t_now'],
                             win_s=fl.get('win_s', self.window), win_note=fl.get('win_note'),
                             fix_status='NOT_CONFIRMED', result_kind='E2_FLOOR_RANKING_TOP1_UNVERIFIED')
                    self.tp.set_state('DONE')
                    self._save_exp()
                    if self.archive is not None:
                        threading.Thread(target=self._archive, args=(dict(e), fl), daemon=True).start()
                else:
                    why = self._floor_failed(e)
                    if why:                             # 2026-10-15: 결과 없이 '계산 중'에 갇히지 않게(그러면 [시작]도 안 눌린다)
                        e.update(state='done', verdict=None, ranking=[], t_now=float(e.get('signal') or e.get('end') or time.time()), error=why)
                        self.tp.set_state('FAILED', why=why)
                        self._save_exp()
                        if self.archive is not None:     # 2026-10-16: 실패도 보관(표본·기록은 남겨야 원인을 본다)
                            threading.Thread(target=self._archive, args=(dict(e), dict(verdict=None, ranking=[], cands=[], t_now=e['t_now'],
                                                                                         signal=e.get('signal'), error=why)), daemon=True).start()

    def _floor_failed(self, e):
        """층 작업자는 모든 층 정합이 실패하면 floor.json 을 안 쓰고 다음 신호를 기다린다. 그 로그(floor.stdout)에서 이번 신호 뒤
        '정합 실패' 가 층 수만큼 나왔으면 실패, 5분 넘게 결과가 없으면 시간 초과로 본다."""
        import re
        import time
        sig = float(e.get('signal') or e.get('end') or 0)
        if not sig:
            return None
        n = len([p for p in Path(self.maps_root).glob('*F') if p.is_dir()]) or 9
        try:
            lines = (self.root / 'out' / 'floor.stdout').read_text(errors='ignore').splitlines()[-300:]
        except OSError:
            lines = []
        day, failed, errs = time.strftime('%Y-%m-%d', time.localtime(sig)), set(), {}
        for line in lines:
            m = re.match(r'(\d\d:\d\d:\d\d) \[floor\] (\d+)층 정합 실패: ?(.*)', line)
            if not m:
                continue
            try:
                t = time.mktime(time.strptime(f'{day} {m.group(1)}', '%Y-%m-%d %H:%M:%S'))
            except ValueError:
                continue
            if t + 1 >= sig:
                failed.add(m.group(2))
                errs.setdefault((m.group(3) or '?').split(':', 1)[0], []).append(m.group(2))
        if len(failed) >= n:                       # R1: 고정 문구 대신 로그에 남은 실제 예외 종류·층
            kinds = ' / '.join(f"{k}: {','.join(sorted(v))}층" for k, v in sorted(errs.items()))
            return f'모든 층({n}개) 정합 실패 — {kinds}'
        if time.time() - sig > 600:                     # KCI 10-16: 걸은 창이 길면(최대 150초) 정합도 길어져 5 → 10분
            return '층 계산 시간 초과(10분)'
        return None

    def _archive(self, e, fl):
        """실험 하나를 따로 보관(논문 비교 그림용): 정합 창의 표본 + 결과. 구조는 plot_live.py 가 읽는 실행 폴더와 같다."""
        import shutil
        import time
        try:
            name = time.strftime('exp_%H%M%S', time.localtime(e['start'])) + (f"_{fl['verdict']}F" if fl.get('verdict') is not None else '_정합실패')
            d = self.archive / name
            (d / 'data' / 'samples').mkdir(parents=True, exist_ok=True)
            (d / 'out').mkdir(parents=True, exist_ok=True)
            rows = self._rows()
            Wn = float(fl.get('win_s') or self.window)
            t_from = min(e.get('start', fl['t_now']), fl['t_now'] - Wn) - 5   # 걸은 전체(창 밖 앞부분 포함) — 논문 그림용
            keep = [r for r in rows if t_from <= float(r['t_sec']) <= e['end'] + 1]
            with open(d / 'data' / 'odom_link.csv', 'w') as f:
                f.write(','.join(rows[0].keys()) + '\n' if rows else '')
                for r in keep:
                    f.write(','.join(r.values()) + '\n')
            for r in keep:
                src = self.root / 'data' / 'samples' / r['sample_id']
                if src.exists():
                    shutil.copytree(src, d / 'data' / 'samples' / r['sample_id'], dirs_exist_ok=True)
            (d / 'out' / 'floor.json').write_text(json.dumps(fl, ensure_ascii=False))
            (d / 'out' / 'floor_history.jsonl').write_text(json.dumps(fl, ensure_ascii=False) + '\n')
            best = [c for c in fl['cands'] if fl.get('verdict') is not None and int(c['map']) == int(fl['verdict'])]
            if best:
                (d / 'out' / 'pose_fit.json').write_text(json.dumps(dict(best[0], source='floor'), ensure_ascii=False))
            xs = [(float(r['fastlio_x']), float(r['fastlio_y'])) for r in keep if float(r['t_sec']) >= fl['t_now'] - Wn]
            walked = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xs, xs[1:])))
            rc = _json(self.root / 'out' / 'run_config.json') or {}
            try:                                        # FAST-LIO 정보는 지금 다시 읽는다 — [초기화]가 FAST-LIO 를 다시 띄웠을 수 있다(10-09)
                sys.path.insert(0, str(HERE))
                import fastlio_info
                fi = fastlio_info.read(fastlio_info.live_dir(e.get('start')))
                if rc.get('fastlio_impl') != fi.get('fastlio_impl') or rc.get('fastlio_config') != fi.get('fastlio_config'):
                    rc['fastlio_at_node_start'] = {k: rc.get(k) for k in ('fastlio_impl', 'fastlio_config', 'fastlio_env')}
                rc.update(fi)
                st = fi.get('fastlio_started')
                rc['fastlio_restarted_during_exp'] = bool(st and e.get('start') and st > e['start'])
            except Exception as ex:
                rc['fastlio_info_error'] = f'{type(ex).__name__}: {ex}'
            with self.lock:
                meta = dict(self.meta)
            (d / 'experiment.json').write_text(json.dumps(dict(e, **meta, window_s=Wn, samples=len(keep),
                                                               walked_m=round(walked, 1), pass_sec=fl.get('pass_sec'),
                                                               twoway={str(c['map']): c['twoway'] for c in fl['cands']},
                                                               wall_cost_cm={str(c['map']): round(100 * c['wall_cost_m'], 1) for c in fl['cands']},
                                                               run_config=rc),
                                                          ensure_ascii=False, indent=1))
            if rc:
                (d / 'out' / 'run_config.json').write_text(json.dumps(rc, ensure_ascii=False, indent=1))
            self._write_index()
            with self.lock:
                self.exp['archived'] = str(d)
                self._save_exp()
                late = {k: self.exp[k] for k in ('record', 'marks') if k in self.exp}   # 보관 중에 바뀐 녹화 글·정답 지점
            if late:
                e2 = json.loads((d / 'experiment.json').read_text())
                e2.update(late)
                (d / 'experiment.json').write_text(json.dumps(e2, ensure_ascii=False, indent=1))
        except Exception as ex:
            with self.lock:
                self.exp['archive_error'] = f'{type(ex).__name__}: {ex}'

    # ---------- 표지판 이벤트(E5, 선배 SCI e5_node 가 /dev/shm/sci_e5/status.json 에 쓴다) ----------
    @staticmethod
    def e5_status(clear_t=0.0):
        """2026-10-15: 위쪽 띠(멈춤/조준 중/출발)용 요약. E5 가 안 돌면 None. 읽기만 한다."""
        import time
        p = Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5')) / 'status.json'
        try:
            age = time.time() - p.stat().st_mtime
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        if age > 20:
            return dict(alive=False, age=round(age))
        f = d.get('fsm') or {}
        evs = d.get('events') or []
        ev = next((e for e in evs if f and e.get('id') == f.get('id')), evs[-1] if evs else None)
        r = (ev or {}).get('reading') or None
        res = f.get('result') or {}
        # KCI 2026-10-17: R1 e5_node 는 단계 이름이 다르고(WAIT_BASE_STOP·BASE_STOPPED·CAPTURE_QUALITY_OK·RELEASE_READY_* …)
        #   시각 t 가 단조 시계(부팅 후 초)다. 웹(아래 JS)이 아는 옛 이름·벽시계로 바꿔 넘긴다. 옛 E5 는 그대로 통과.
        mono2wall = time.time() - time.monotonic()
        def _w(t):
            return None if t is None else (float(t) + mono2wall if float(t) < 1e9 else float(t))
        def _st(x):                                        # R1: CAPTURE(작업 보냄)·AIM(결과) 순서라 이유로 나눈다
            s, r = x.get('state'), str(x.get('reason') or '')
            if s == 'CAPTURE' and (r.startswith('device_job_submitted') or r.startswith('quality_fail_retry')):
                return 'AIM'                               # 아직 조준·초점 중
            if s == 'AIM' and r not in ('image_stable', 'SIMULATED_STOP_POSE', 'ROBOT_OVERSHOT'):
                return 'CAPTURE'                           # 조준·초점 결과 도착 → 촬영
            return State.R1_STATE.get(s, s)
        log = [dict(x, state=_st(x), t=_w(x.get('t'))) for x in (f.get('log') or [])]
        f = dict(f, state=(log[-1]['state'] if log else State.R1_STATE.get(f.get('state'), f.get('state'))))
        if clear_t and log and (log[0].get('t') or 0) < clear_t:        # KCI 10-17: [초기화]/[시작] 전 이벤트는 안 보임
            f, res, log, r, evs = {}, {}, [], None, []
        if (res.get('quality') or {}).get('ok') and res.get('capture') and not any(x['state'] == 'CAPTURED' for x in log):
            k = next((i for i, x in enumerate(log) if x['state'] in ('HOME', 'RELEASE')), None)   # R1 은 촬영 성공을 이 기록에 안 남김
            if k is not None:
                log.insert(k, dict(state='CAPTURED', t=log[k]['t'], reason='quality_ok'))
        if not any(x['state'] == 'CAPTURE' for x in log):     # R1 FSM 기록엔 조준 결과(AIM EXECUTE)가 없다 → 촬영·복귀·끝 시각에 '조준·초점 끝'
            k = next((i for i, x in enumerate(log) if x['state'] in ('CAPTURED', 'HOME', 'RELEASE', 'FAILED')
                      and str(x.get('reason') or '') != 'stop_not_confirmed_timeout'), None)
            if k is not None and any(x['state'] == 'AIM' for x in log[:k]):
                log.insert(k, dict(state='CAPTURE', t=log[k]['t'], reason='aim_focus_done'))
        out = dict(alive=True, age=round(age, 1), run=Path(str(d.get('run') or '')).name, siglip=d.get('siglip'), health=d.get('source_health'),
                   pantilt=bool(d.get('pantilt_execute_enabled')), n_stops=(d.get('n_stops') or {}).get(d.get('active_stage') or 'P2'),
                   id=f.get('id'), state=f.get('state'),
                   why=next((x.get('reason') for x in log if x.get('state') == 'FAILED'), log[-1].get('reason') if log else None),
                   t_state=(log[-1].get('t') if log else None), aim=(res.get('aim') or {}).get('mode'),
                   aim_why=(res.get('aim') or {}).get('why'), capture_source=res.get('capture_source'), has_capture=bool(res.get('capture')))
        if r:
            out['reading'] = State._reading(r)
        out['gate'] = d.get('kci_gate')
        out['run_dir'] = str(d.get('run') or '')
        out['steps'] = [dict(s=x.get('state'), t=x.get('t'), r=x.get('reason')) for x in log][-16:]   # 2026-10-15 밤: 단계 표시용
        w = [t for t in d.get('p2_tracks') or [] if t.get('state') in ('WAIT_READABLE', 'VERIFYING')]   # 2026-10-16: 확인됐고 멈춤 신호 기다리는 표지판
        out['p2_wait'] = dict(n=len(w), near=min(((t.get('now') or {}).get('range') or 99.0) for t in w)) if w else None
        out['home'] = d.get('kci_home')
        if out['home'] is None and f.get('id'):           # KCI 10-17: R1 은 HOME 단계를 FSM 기록에 남긴다
            th = next((x.get('t') for x in log if x.get('state') == 'HOME'), None)
            if th is not None:
                after = [x for x in log if x.get('t') is not None and x['t'] > th]
                out['home'] = dict(ev=f.get('id'), state='DONE' if after else 'RUNNING', t=after[0]['t'] if after else None)
        if f.get('state') in ('AIM', 'CAPTURE'):         # 조준 중이면 콘솔이 지금 뭘 하는지(서보·초점)
            try:
                import urllib.request
                cs = json.loads(urllib.request.urlopen('http://127.0.0.1:8088/state', timeout=0.3).read())
                out['console'] = dict(status=cs.get('status'), last=(cs.get('log') or [''])[-1][-120:])
            except Exception:
                out['console'] = None
        if f.get('id'):                                  # 2026-10-15: 이 이벤트의 E6(판독 → 의미지도 → 층·후보 영역) 결과
            r6 = _json(Path(out['run_dir']) / 'trials' / f"ev{int(f['id']):03d}" / 'e6' / 'e6_result.json')
            if r6:
                out['e6'] = dict(status=r6.get('status'), label=next((h.get('label') for h in r6.get('floor_hypotheses') or []), None),
                                 hyps=[dict(floor=h.get('floor'), p=round(float(h.get('p_floor') or 0), 3), n=len(h.get('components') or []))
                                       for h in r6.get('floor_hypotheses') or []])
        out['events'] = [dict(id=e.get('id'), state=e.get('state'), aim=e.get('aim'), reading=State._reading(e.get('reading')) if e.get('reading') else None)
                         for e in evs[-6:]][::-1]
        return out

    R1_STATE = {'WAIT_BASE_STOP': 'WAIT_STABLE', 'BASE_STOPPED': 'STOP_CONFIRMED', 'WAIT_IMAGE_STABLE': 'STOP_CONFIRMED',
                'RAW_CAPTURE_SAVED': 'CAPTURE', 'CAPTURE_QUALITY_OK': 'CAPTURED', 'CAPTURE_PROXY': 'CAPTURED',
                'CAPTURE_QUALITY_FAIL': 'FAILED', 'NO_CAPTURE': 'FAILED', 'RELEASE_READY_CAPTURE_OK': 'RELEASE',
                'RELEASE_READY_OBSERVATION_FAILED': 'FAILED', 'RELEASE_BLOCKED_ACTUATOR_UNKNOWN': 'FAILED'}   # KCI 10-17: R1 → 웹 이름

    @staticmethod
    def _reading(r):
        if 'semantic_status' in r or 'fusion' in r:      # KCI 10-17: R1 판독기 결과(fusion·ocr.raw_text·vlm.parsed)
            fu, oc, vl = r.get('fusion') or {}, r.get('ocr') or {}, r.get('vlm') or {}
            st = fu.get('status') or r.get('semantic_status') or r.get('status')
            return dict(status=f"{st} ({fu.get('sign_type') or '-'})", sign_type=fu.get('sign_type'), labels=r.get('labels') or [],
                        ocr=(oc.get('raw_text') or [])[:6], vlm=vl.get('parsed') or None, vlm_note=vl.get('status'))
        va = r.get('vlm_answer') or {}
        return dict(status=r.get('status'), sign_type=r.get('sign_type'), labels=r.get('labels'),
                    ocr=(r.get('ocr_text') or [])[:6], vlm=va.get('ours') or va, vlm_note=r.get('vlm_note'))

    def e5_arm(self, body):
        """E5RT2: operator grant (bounded TTL). Refused unless the trial is walking. Clears a previous abort latch."""
        import time
        import uuid
        if self.exp.get('state') != 'walking':
            return dict(ok=False, why='trial_not_walking', state=self.exp.get('state'))
        try:
            ttl = max(10.0, min(600.0, float((body or {}).get('ttl_s') or 120.0)))
        except (TypeError, ValueError):
            ttl = 120.0
        self.e5_grant = dict(id=uuid.uuid4().hex[:12], t=time.time(), expires_t=time.time() + ttl, ttl_s=ttl)
        self.e5_abort = False
        self.write_gate()
        return dict(ok=True, grant=self.e5_grant)

    def e5_disarm(self, abort=False):
        self.e5_grant = None
        if abort:
            self.e5_abort = True                      # latched until a new explicit grant
        self.write_gate()
        return dict(ok=True, abort_latched=self.e5_abort)

    def write_gate(self):
        """2026-10-15: E5 찾기 스위치. 걷는 중에만 켠다([시작] 전·멈춤 뒤·계산 중·결과 뒤·초기화 중에는 끔). e5_onboard 가 0.5 초마다 읽고
        5 초 넘게 안 바뀌면 끈다 — 그래서 바뀌지 않아도 늘 다시 쓴다."""
        import time
        st = self.exp.get('state')
        why = {'walking': '걷는 중', 'waiting': '멈춤 뒤', 'computing': '층 계산 중', 'done': '결과 나옴(다음 [시작] 전)',
               'resetting': '초기화 중'}.get(st, '시작 전')
        d = Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5'))
        try:
            d.mkdir(parents=True, exist_ok=True)
            # E5-RT 2026-10-17: keep 'active' (old E5 readers) and add the explicit trial / global-match state so E5 can keep
            #   SEARCHING while the floor is computed ([멈춤] -> computing) but issue no new STOP until [시작] (walking).
            # E5RT2: gate schema 2 = explicit expiring grant (armed/arm_grant_id/arm_expires_t) + operator_abort latch.
            #   estop is not known to the viewer and is NOT written (E5 treats it as UNKNOWN: no automatic HOME).
            g = self.e5_grant
            if g is not None and (st != 'walking' or time.time() > g['expires_t']):
                self.e5_grant = g = None                   # leaving walking / expiry revokes the grant (no stale re-arm)
            live = bool(g is not None and not self.e5_abort)
            tmp = d / f'.kci_gate.json.{os.getpid()}.{threading.get_ident()}.tmp'
            tmp.write_text(json.dumps(dict(active=st == 'walking', why=why, t=time.time(), trial_state=st or 'idle',
                                           global_match_state='computing' if st == 'computing' else ('done' if st == 'done' else 'idle'),
                                           armed=live, arm_grant_id=g['id'] if live else None,
                                           arm_expires_t=g['expires_t'] if live else None, operator_abort=bool(self.e5_abort),
                                           schema='e5rt-gate-2'), ensure_ascii=False))
            os.replace(tmp, d / 'kci_gate.json')
        except OSError:
            pass

    def _labels_path(self):
        return (self.archive.parent if self.archive else self.root / 'out') / 'e5_labels.jsonl'

    def _labels_v2_path(self):
        return self._labels_path().with_name('e5_labels_v2.jsonl')

    def e5_label(self, body):
        """R1: 단계별 평가 라벨(r1_labels). 예전 O/X 버튼은 sign_true 로 받는다. 단계마다 필요한 증거가 없으면 거절.
        10-15 e5_labels.jsonl 은 고치지 않는다(읽기만, 정답률에서 뺌)."""
        st = _json(Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5')) / 'status.json') or {}
        ev = next((e for e in st.get('events') or [] if e.get('id') == body.get('ev')), None)
        body = dict(body, run=str(body.get('run') or Path(str(st.get('run') or '')).name))
        rec = LB.make_record(body, ev, st.get('run'), self._labels_v2_path())
        if 'error' in rec:
            return rec
        rec['exp_start'] = self.exp.get('start')
        try:
            p = self._labels_v2_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except OSError as ex:
            return dict(error=f'{type(ex).__name__}: {ex}')
        return dict(ok=True, run=rec['run'], ev=rec['ev'], label_type=rec['label_type'], value=rec['value'], revision=rec['revision'],
                    label={True: 'O', False: 'X', None: ''}[rec['value']] if rec['label_type'] == 'sign_true' else None)

    def e5_labels(self):
        """화면 정답률용: v2 의 sign_true 마지막 판만(O/X 로). 예전 무효 기록은 세지 않는다."""
        out = {}
        for (run, ev, lt), r in LB.latest(self._labels_v2_path()).items():
            if lt == 'sign_true' and r.get('value') is not None:
                out[f"{run}#{ev}"] = 'O' if r['value'] else 'X'
        return out

    def _e5_signs(self, fit, rows, T, t0, t1):
        """2026-10-15: E5 멈춤(P2)마다 표지판 자리를 안내도 좌표로. 표지판의 로봇 기준 앞·왼쪽(= E5 의 몸체 기준 오돔·깊이로 잰 세계 좌표 −
        그때 로봇 자세)을, 구조점을 지도에 올리는 식 그대로(그 시각 위치추정 자세 + 정합 φ·β·거울·축척) 옮긴다. 로봇 자리도 같이."""
        import math
        st = _json(Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5')) / 'status.json') or {}
        run = Path(str(st.get('run') or ''))
        try:
            lines = (run / 'stops.jsonl').read_text().splitlines()[-400:]
        except OSError:
            return []
        texts = {}
        for e in st.get('events') or []:
            r = e.get('reading') or {}
            va = (r.get('vlm_answer') or {}).get('ours') or {}
            texts[round(float(e.get('t', -1)), 1)] = (e.get('id'), va.get('text') or '/'.join(x.get('label', '') for x in r.get('labels') or []))
        ox = np.array([float(r['fastlio_x']) for r in rows]); oy = np.array([float(r['fastlio_y']) for r in rows])
        oyaw = np.unwrap(np.array([float(r['fastlio_yaw_rad']) for r in rows]))
        phi, ppm = np.deg2rad(fit['phi']), fit['ppm']
        anc, pos = np.asarray(fit['anchor'], float), np.asarray(fit['pos'], float)
        out, seen = [], set()
        for l in lines:
            try:
                s = json.loads(l)
            except ValueError:
                continue
            if s.get('stage') != 'P2' or not s.get('xyz') or not s.get('pose'):
                continue
            stp = float(s.get('stamp') or 0)
            if not (t0 - 1 <= stp <= t1 + 1) or round(float(s.get('t', -1)), 1) in seen:
                continue
            seen.add(round(float(s.get('t', -1)), 1))
            xb, yb, wb = (float(v) for v in s['pose'][:3])
            dx, dy = float(s['xyz'][0]) - xb, float(s['xyz'][1]) - yb
            fwd, left = math.cos(wb) * dx + math.sin(wb) * dy, -math.sin(wb) * dx + math.cos(wb) * dy
            x, y, yaw = float(np.interp(stp, T, ox)), float(np.interp(stp, T, oy)), float(np.interp(stp, T, oyaw))
            w = np.array([x, y]) - anc + (np.array([fwd, left]) * [1, fit['mirror']]) @ _R(yaw + fit['beta']).T
            m = pos + ppm * (w @ _A(phi).T)
            rx, ry, _ = map_pose_px(fit, x, y, yaw)
            ev, txt = texts.get(round(float(s.get('t', -1)), 1), (None, ''))
            out.append(dict(x=round(float(m[0]), 1), y=round(float(m[1]), 1), rx=round(rx, 1), ry=round(ry, 1),
                            range_m=round(math.hypot(fwd, left), 2), id=ev, text=txt or '',
                            # 화면 영역 반지름: 깊이·방향·위치추정 오차를 다 합친 대략값(0.5 m + 거리의 10%). 표시용일 뿐 계산에는 안 씀
                            r_px=round(float(ppm) * (0.5 + 0.1 * math.hypot(fwd, left)), 1)))
        return out[-12:]

    @staticmethod
    def e6_mask(run_dir, ev, floor):
        """E6 후보 영역 마스크 → 보라 반투명 PNG(안내도와 같은 픽셀 좌표). ~/sci_ws/experiments 안 것만."""
        import cv2
        p = (Path(run_dir) / 'trials' / f'ev{int(ev):03d}' / 'e6' / f'e6_region_{int(floor)}F.png').resolve()
        root = (Path.home() / 'sci_ws' / 'experiments').resolve()
        if root not in p.parents or not p.is_file():
            return None
        m = cv2.imread(str(p), 0)
        if m is None:
            return None
        rgba = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
        rgba[m > 0] = (200, 40, 160, 120)                          # BGRA: 보라
        return cv2.imencode('.png', rgba)[1].tobytes()

    @staticmethod
    def e5_capture():
        """지금(마지막) 이벤트의 사진 파일 — ~/sci_ws/experiments 아래 것만."""
        p = Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5')) / 'status.json'
        try:
            d = json.loads(p.read_text())
            c = Path(((d.get('fsm') or {}).get('result') or {}).get('capture') or '').resolve()
        except (OSError, ValueError):
            return None
        roots = [(Path.home() / 'sci_ws' / 'experiments').resolve()]
        run = Path(str(d.get('run') or '')).resolve()     # KCI 10-17: R1 배포 트리(~/sci_ws/deploy/<판>/experiments/<실행>)
        if d.get('run') and (Path.home() / 'sci_ws').resolve() in run.parents:
            roots.append(run)
        return c if c.is_file() and any(r in c.parents for r in roots) else None

    @staticmethod
    def e5_stop_image():
        """KCI 10-17: 멈춤 순간 사진. SHM last_stop.jpg 가 없으면(R1 은 원본 장면이 남아 있을 때만 씀) 실행 폴더의 이벤트 사진."""
        shm = Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5'))
        try:
            d = json.loads((shm / 'status.json').read_text())
        except (OSError, ValueError):
            d = {}
        fid, run = (d.get('fsm') or {}).get('id'), Path(str(d.get('run') or ''))
        if fid and d.get('run'):
            rd = run / 'trials' / f'ev{int(fid):03d}' / 'runner'
            for n in ('stop_source_frame_with_box.jpg', 'stop_latest_frame_no_box.jpg'):
                if (rd / n).is_file():
                    return rd / n
        return shm / 'last_stop.jpg' if (shm / 'last_stop.jpg').is_file() else None

    # ---------- 데이터 ----------
    def _rows(self):
        p = self.root / 'data' / 'odom_link.csv'
        try:
            return list(csv.DictReader(open(p)))
        except OSError:
            return []

    def _sample(self, sid):
        if sid in self.cache:
            return self.cache[sid]
        d = self.root / 'data' / 'samples' / sid
        try:
            with np.load(d / 'data_short.npz') as f:
                body = f['a1_3_1_all_candidate_xy'].astype(float)[f['a1_3_1_all_candidate_role'] != 4]
        except (OSError, KeyError, ValueError):
            return None
        loc = pose = low = None
        try:
            with np.load(d / 'local.npz') as f:
                loc, pose = f['pts'].astype(float), f['pose'].astype(float)
                if 'hmax' in f.files and 'rng' in f.files and len(f['hmax']) == len(loc):   # 2026-10-13 esy
                    low = _low_mask(f['hmax'].astype(float), f['rng'].astype(float))
        except (OSError, KeyError, ValueError):
            pass
        k = max(1, len(body) // self.per)
        v = (body[::k], None if loc is None else loc[::k], pose, None if low is None else low[::k])
        self.cache[sid] = v
        return v

    def snapshot(self, floor_sel=None, view=None):
        with self.lock:
            return self._snapshot(floor_sel, view)

    def _snapshot(self, floor_sel, view=None):
        import time
        out = self.root / 'out'
        live, pfit, fl = _json(out / 'live_odom.json'), _json(out / 'pose_fit.json'), _json(out / 'floor.json')
        prog = _json(out / 'floor_progress.json')
        fl, pfit, stale = self._filter_current(fl, pfit)      # R1: 다른 실험·실패한 결과는 '지금 결과'가 아니다
        if prog and not T.usable_now(prog, self._pointer()):
            prog = None
        st = None
        try:
            lines = (out / 'status.jsonl').read_text().splitlines()
            st = json.loads(lines[-1]) if lines else None
        except (OSError, ValueError):
            pass
        exp = dict(self.exp)
        if exp.get('state') == 'walking':
            exp['elapsed'] = round(time.time() - exp['start'], 1)
        res = dict(live=live, status=st, floor=None, local=None, map=None, exp=exp, can_start=self.trigger is not None, meta=self.meta,
                   stale=stale, trial=self._pointer(),
                   fix_status='NOT_CONFIRMED (E2 층 순위 1위·정합은 확정이 아님 — E7 미연결)')
        res['e5'] = self.e5_status(getattr(self, 'e5_clear_t', 0.0))
        if res['e5'] and res['e5'].get('alive'):
            res['e5']['labels'] = self.e5_labels()
        res['marks'] = list(exp.get('marks', [])) + (list(getattr(self, 'pending_marks', [])) if exp.get('state') in ('idle', None) else [])
        res['man_rec'] = dict(self.man_rec) if getattr(self, 'man_rec', None) else None
        res['base_floor'] = floor_sel or (self.meta or {}).get('truth') or exp.get('truth')   # 정합 전: 이 층 안내도를 통째로(정답 위치 찍기용)
        if exp.get('state') == 'resetting':
            return res
        # 보기: 멈춘 뒤에는 '결과'(그 실험 창으로 고정) — 계산하는 동안 실시간 누적이 바뀌어도 그림이 안 변하게. 버튼으로 바꿀 수 있다.
        frozen = exp.get('state') in ('waiting', 'computing', 'done') and 'end' in exp
        if view == 'live':
            frozen = False
        elif view == 'result' and 'end' not in exp:
            frozen = False
        res['view'] = 'result' if frozen else 'live'
        prep = exp.get('state') == 'done' and not frozen and type(self).__name__ != 'ArchiveView'   # 결과 뒤 실시간 보기 = 다음 실험 준비: 직전 결과(순위·정합)는
        res['prep_next'] = prep                                                 # 안 보여 준다(위 목록에 보관됨, 결과 보기 버튼으로 다시 봄)
        if prep:
            fl = None
        if fl and exp.get('signal') and str(fl.get('signal')) != str(exp.get('signal')) and exp.get('state') in ('waiting', 'computing', 'done'):
            fl = None                                          # 10-15 밤: 이 실험 신호의 결과가 아니면(정합 실패로 floor.json 이 안 바뀜) 직전 실험 결과를 쓰지 않는다
        cands = {}
        if fl and frozen and exp.get('state') != 'done':
            fl = None                                          # 계산 중: 이전 실험 결과는 보여 주지 않는다
        if fl:
            cands = {int(c['map']): c for c in fl['cands']}
            res['floor'] = dict(vote=fl.get('vote'), verdict=fl['verdict'], ranking=fl['ranking'], pass_index=fl['pass_index'],
                                trial_id=fl.get('trial_id'), result_kind='E2_FLOOR_RANKING_TOP1_UNVERIFIED', fix_status='NOT_CONFIRMED',
                                pass_sec=fl['pass_sec'], t_now=fl['t_now'],
                                cands=[dict(map=int(c['map']), twoway=round(c['twoway'], 3), wall_cm=round(100 * c['wall_cost_m'], 1))
                                       for c in fl['cands']])
        if prog:
            res['progress'] = prog
        rows = self._rows()
        if not rows:
            return res
        T_s = np.array([float(r['t_sec']) for r in rows])
        live_ids = set(r['sample_id'] for r in rows)
        for k in [k for k in self.cache if k not in live_ids]:
            del self.cache[k]
        # 왼쪽: FAST-LIO 좌표 — 실시간이면 지금까지 최근 창, 결과면 멈춘 시각까지의 창(고정)
        t_end = min(T_s[-1], exp['end']) if frozen else T_s[-1]
        if frozen and exp.get('t_now'):
            t_end = exp['t_now']
        Wv = float(exp.get('win_s') or self.window) if frozen else self.window     # 결과 화면은 그 결과의 창(걸은 창이면 더 길다)
        sel = [i for i in range(len(rows)) if t_end - Wv <= T_s[i] <= t_end]
        pre = []                                            # 걷기는 했지만 60초 창 밖(앞부분)인 표본 — 결과 화면과 걷는 중 모두(10-15 밤: 걷는 중에
        if exp.get('start') and (exp.get('state') in ('walking', 'waiting', 'computing') or (exp.get('state') == 'done' and frozen)):   # 60초 넘게 서 있으면 걸어온 곳이 화면에서 사라졌음)
            pre = [i for i in range(len(rows)) if exp['start'] - 1.5 <= T_s[i] < t_end - Wv]
        pts_out, traj_out = [], []
        for i in pre:
            v = self._sample(rows[i]['sample_id'])
            if v is None or v[1] is None:
                continue
            pts_out.append(v[1]); traj_out.append(v[2][:2])
        pts, traj, last, low_pts = [], [], None, []
        for i in sel:
            v = self._sample(rows[i]['sample_id'])
            if v is None or v[1] is None:
                continue
            if v[3] is not None:                                  # 낮은 것(책상·의자)은 따로 — 주황으로 그린다
                pts.append(v[1][~v[3]]); low_pts.append(v[1][v[3]])
            else:
                pts.append(v[1])
            traj.append(v[2][:2]); last = v[1]
        if pts:
            res['local'] = dict(pts=np.round(np.vstack(pts), 2).tolist(), traj=np.round(np.array(traj), 2).tolist(),
                                last=np.round(last, 2).tolist(), n=len(pts), frozen=frozen,
                                pts_out=_thin(np.vstack(pts_out)).tolist() if pts_out else [],
                                traj_out=np.round(np.array(traj_out + traj[:1]), 2).tolist() if traj_out else [], n_out=len(pts_out),
                                low=np.round(np.vstack(low_pts), 2).tolist() if low_pts else [])
        # 오른쪽: 고른 층(없으면 판정 층)의 정합 — **정합에 쓴 창** 을 그대로, 그 뒤 경로와 지금 위치는 같은 정합으로 이어서
        fit = self._pick_fit(floor_sel, fl, pfit)
        if frozen and exp.get('state') != 'done':
            fit = None                                         # 계산 중: 이전 정합으로 그리지 않는다
        if exp.get('state') in ('idle', None) or prep or (exp.get('state') == 'done' and exp.get('error')):
            fit = None                                         # 시작 전·다음 실험 준비: 직전 실험 정합으로 그리지 않는다(정답 층 안내도 통째로, 출발 위치 찍기용)
        if fit:
            phi, ppm = np.deg2rad(fit['phi']), fit['ppm']
            anc, pos = np.asarray(fit['anchor'], float), np.asarray(fit['pos'], float)
            win = [i for i in range(len(rows)) if fit['t'] - float(fit.get('win_s') or self.window) <= T_s[i] <= fit['t']]
            mp, tr = [], []
            for i in win:
                x, y, yaw = float(rows[i]['fastlio_x']), float(rows[i]['fastlio_y']), float(rows[i]['fastlio_yaw_rad'])
                tr.append(map_pose_px(fit, x, y, yaw)[:2])
                v = self._sample(rows[i]['sample_id'])
                if v is None:
                    continue
                w = np.array([x, y]) - anc + (v[0] * [1, fit['mirror']]) @ _R(yaw + fit['beta']).T
                mp.append(pos + ppm * (w @ _A(phi).T))
            after = [map_pose_px(fit, float(rows[i]['fastlio_x']), float(rows[i]['fastlio_y']), float(rows[i]['fastlio_yaw_rad']))[:2]
                     for i in range(len(rows)) if T_s[i] > fit['t']] if not frozen else []
            me = None
            if live and live.get('odom_link') and not frozen:
                me = list(map_pose_px(fit, *live['odom_link']))
            try:
                signs = self._e5_signs(fit, rows, T_s, exp.get('start') or T_s[0], exp.get('end') or T_s[-1] + 5)
            except Exception:
                signs = []
            res['map'] = dict(floor=int(fit['map']), selected=bool(floor_sel), frozen=frozen, pts=np.round(np.vstack(mp), 1).tolist() if mp else [], signs=signs,
                              traj=np.round(np.array(tr), 1).tolist(), after=np.round(np.array(after), 1).tolist() if after else [],
                              me=me, fit_age=round(T_s[-1] - fit['t'], 1), win_n=len(win),
                              wall_cost_cm=round(100 * fit.get('wall_cost_m', float('nan')), 1),
                              twoway=round(fit['twoway'], 3) if 'twoway' in fit else None, source=fit.get('source', 'floor'))
        return res


def _thin(p, cell=0.08, cap=8000):
    """회색(창 밖) 점은 그림용이라 8 cm 칸마다 하나로 줄인다(긴 실험에서 1초마다 보내는 양이 커지지 않게)."""
    if len(p) <= cap:
        return np.round(p, 2)
    _, i = np.unique(np.floor(p[:, :2] / cell).astype(np.int64), axis=0, return_index=True)
    q = p[np.sort(i)]
    return np.round(q[:: max(1, len(q) // cap + 1)] if len(q) > cap else q, 2)


class ArchiveView(State):
    """보관된 실험 하나를 '결과(고정)' 로 다시 보기. 폴더 구조가 실행 폴더와 같다(data/ out/ + experiment.json)."""

    def __init__(self, exp_dir, maps_root, window, per):
        State.__init__(self, exp_dir, maps_root, window, per)
        e = _json(Path(exp_dir) / 'experiment.json') or {}
        self.exp = dict(e, state='done')
        if 'end' not in self.exp and 't_now' in self.exp:
            self.exp['end'] = self.exp['t_now']

    def _pointer(self):
        return None                                      # R1: 보관본은 그 실험 결과 그대로(세대 판정 없음 — 보관 자체가 그 실험의 것)

    def _filter_current(self, fl, pfit):
        return fl, pfit, {}

    def _save_exp(self):
        pass                                             # 지난 실험 보기: 기록은 /mark 처리기가 그 실험의 experiment.json 에 직접 쓴다


def list_archive(archive_root):
    """보관 실험 목록(최근 것 먼저): 이름 · 판정 · 정답 · 시각."""
    out = []
    for p in sorted(Path(archive_root).glob('live_*/experiments/exp_*/experiment.json'), reverse=True):
        e = _json(p) or {}
        if not (p.parent / 'out' / 'floor.json').exists():
            continue                                     # 표본·결과가 없는 기록(중단·유실)은 다시 볼 것이 없다
        import time
        out.append(dict(name=f'{p.parent.parent.parent.name}/{p.parent.name}', verdict=e.get('verdict'), truth=e.get('truth'),
                        start=time.strftime('%m-%d %H:%M:%S', time.localtime(e['start'])) if e.get('start') else ''))
    return out


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>실시간 위치 보기</title><style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--grid:#e6e5e0;--pt:#1baf7a;--tr:#2a78d6;--me:#eb6834;--go:#0ca30c;--stop:#d03b3b}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 "Noto Sans CJK KR","Noto Sans KR",sans-serif}
header{padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0} .st{color:var(--ink2)} .big{font-size:20px;font-weight:700}
button{font:inherit;font-size:16px;font-weight:700;padding:8px 22px;border-radius:8px;border:0;color:#fff;cursor:pointer}
#go{background:var(--go)} #stop{background:var(--stop)} #rst{background:#52514e} #vw{background:#2a78d6} #mkb{background:#c2410c;padding:8px 14px} #e5bar{margin:6px 0;padding:12px 18px;border-radius:10px;font-weight:700;display:flex;gap:16px;align-items:center;flex-wrap:wrap} #e5sec{margin:12px 12px 24px;padding:12px 16px;border:1px solid var(--grid);border-radius:12px} #e5sec h2{margin:0 0 6px} #e5read{font-size:30px;font-weight:800;margin:8px 0} #e5steps{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px;font-size:16px;font-weight:700} #e5steps span{padding:6px 10px;border-radius:8px;background:var(--grid);color:var(--mut)} #e5steps span.ok{background:#dcfce7;color:#166534} #e5steps span.now{background:#f59e0b;color:#111} #e5steps span.bad{background:#7c2d12;color:#fff} #e5e6{font-size:17px;font-weight:700;color:#86198f;margin:2px 0 8px} #e5ox{display:none;align-items:center;gap:12px;margin:6px 0 10px;font-size:18px;font-weight:700} button.ox{font-size:22px;padding:10px 28px} #e5o{background:#16a34a} #e5x{background:#b91c1c} #e5oxr{font-size:18px} #e5acc{font-size:20px;font-weight:800;margin:0 0 10px} #e5acc.ok{color:#15803d} #e5acc.low{color:#b91c1c} #fullb{margin:6px 0 0;padding:6px 12px;background:#475569} #mkdel{background:#78716c;padding:8px 12px} #recb{background:#be123c;padding:8px 12px} #recb.on{background:#7f1d1d;animation:e5blink 1s step-end infinite} .e5grid{grid-template-columns:1fr 1fr 1fr!important} .e5grid{display:grid;grid-template-columns:1fr 1fr;gap:14px} .e5grid figure{margin:0} .e5grid img{width:100%;border-radius:8px;background:#0001;min-height:120px} .e5grid figcaption{color:var(--mut);font-size:13px} #e5hist{margin-top:10px;font-size:14px;color:var(--mut)} #e5hist div{padding:2px 0} #e5alert{position:sticky;top:0;z-index:20;margin:4px 12px;padding:8px 14px;border-radius:8px;font-weight:800;font-size:20px;cursor:pointer} @media (max-width:900px){.e5grid{grid-template-columns:1fr}} #e5bar #e5t{font-size:30px} #e5bar #e5s{font-size:15px;font-weight:600} #e5bar a{color:inherit;font-size:15px} .e5red{background:#dc2626;color:#fff;animation:e5blink .6s step-end infinite} .e5amb{background:#f59e0b;color:#111} .e5grn{background:#16a34a;color:#fff} .e5gry{background:var(--grid);color:var(--mut)} .e5pre{background:#fed7aa;color:#7c2d12} .e5bad{background:#7c2d12;color:#fff} @keyframes e5blink{50%{background:#7f1d1d}} #mkb.armed{background:#7c3aed} #M.armed{cursor:crosshair} #go:disabled,#stop:disabled,#rst:disabled{background:var(--grid);color:var(--mut);cursor:default}
#exp{font-size:16px;font-weight:600} select,input{font:inherit;font-size:15px;padding:6px 8px;border:1px solid var(--grid);border-radius:6px;background:#fff}
main{display:grid;grid-template-columns:1fr 1.4fr;gap:12px;padding:0 16px 16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:8px}
section h2{font-size:14px;margin:2px 4px 6px;color:var(--ink2);font-weight:600}
canvas{width:100%;height:60vh;display:block}
#rank{display:flex;gap:6px;flex-wrap:wrap;margin:6px 4px 0}
.r{padding:3px 10px;border-radius:4px;background:var(--grid);color:var(--ink2);cursor:pointer;border:2px solid transparent;user-select:none}
.r:hover{border-color:var(--mut)} .r.top{background:#0d366b;color:#fff} .r.sel{border-color:var(--me)}
</style></head><body>
<header><h1>실시간 위치 보기</h1><button id="go">시작</button><button id="stop" disabled>멈춤</button><button id="rst">초기화</button><button id="vw">실시간 보기</button>
<select id="truth" title="실제로 있는 층(통계용)"><option value="">정답 층?</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option><option>9</option></select>
<input id="note" placeholder="메모(예: 사람 많음)" size="16">
<input id="mk" placeholder="지점 이름(선택, 예: 417앞)" size="14"><button id="mkb" title="누른 뒤 오른쪽 안내도에서 로봇이 실제로 서 있는 곳을 클릭">정답 위치</button><button id="mkdel" title="가장 최근에 찍은 정답 위치 하나를 지움(기록에는 지운 줄이 남음)" onclick="unmark()">마지막 정답 위치 지우기</button><button id="recb" title="실험과 따로 로스백 녹화(라이다·IMU·오돔 약 16 MB/s, 카메라 포함이면 +23 MB/s)" onclick="recToggle()">녹화 켜기</button><label class="st" title="RealSense 색·깊이도 녹화(무거움)"><input type="checkbox" id="reccam">카메라 포함</label><span class="st" id="mkr"></span><span class="st" id="rec"></span>
<select id="arch" title="보관된 지난 실험 다시 보기"><option value="">지금 실험</option></select><span id="exp"></span>
<span class="big" id="floor"></span><span class="st" id="stat"></span></header>
<div id="e5alert" style="display:none" onclick="document.getElementById('e5sec').scrollIntoView({behavior:'smooth'})"></div>
<main><section><h2>FAST-LIO 좌표 — 최근 60초 누적 구조점 · 경로 · 지금 위치 (실시간)</h2><canvas id="L"></canvas></section>
<section><h2 id="mh">안내도 — 아직 정합 전</h2><canvas id="M"></canvas><div id="rank"></div><button id="fullb" onclick="FULLMAP=!FULLMAP;this.textContent=FULLMAP?'걸은 곳 확대':'안내도 전체 보기';if(S)drawMap()">안내도 전체 보기</button></section></main>
<section id="e5sec" style="display:none"><h2>표지판 이벤트(E5)</h2>
<div id="e5bar"><span id="e5t"></span><span id="e5s"></span><button id="e5arm" onclick="e5grant('/e5/arm')" title="E5RT2: 새 조준·초점·HOME 허용(2분, 걷는 중만)">물리동작 허용(2분)</button><button id="e5dis" onclick="e5grant('/e5/abort')" title="E5RT2: 허용 취소 + 운용자 중단 고정(새 허용 전까지)">물리동작 중지</button><span id="e5g"></span></div>
<div id="e5steps"></div><div id="e5read"></div><div id="e5e6"></div>
<div id="e5ox"><span id="e5oxq">이 멈춤이 표지판이 맞나요?</span><button id="e5o" class="ox">O 맞음</button><button id="e5x" class="ox">X 아님</button><span id="e5oxr"></span></div><div id="e5acc"></div>
<div class="e5grid"><figure><img id="e5frame" alt=""><figcaption>실시간 RealSense(스트림, 초당 약 8장) · 주황 = SigLIP 이 본 표지판(1~2초 전 장면 기준)</figcaption></figure>
<figure><img id="e5stop" alt=""><figcaption id="e5stopcap">멈춤 순간(후보 상자)</figcaption></figure>
<figure><a id="e5img" target="_blank" href="/e5/capture.jpg" style="display:none"><img id="e5cap" alt=""></a><figcaption id="e5capcap">팬틸트 사진(이벤트가 나면)</figcaption></figure></div>
<div id="e5hist"></div></section>
<script>
const imgs={}; let S=null, SEL=null, VIEW=null, ARCH='', busy=false;
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function fit(cv){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;const c=cv.getContext('2d');c.setTransform(d,0,0,d,0,0);return [c,r.width,r.height]}
function bounds(arrs,pad){let x0=1e18,y0=1e18,x1=-1e18,y1=-1e18;for(const a of arrs)for(const p of a){x0=Math.min(x0,p[0]);y0=Math.min(y0,p[1]);x1=Math.max(x1,p[0]);y1=Math.max(y1,p[1])}
 if(x0>x1)return null;return [x0-pad,y0-pad,x1+pad,y1+pad]}
function arrow(c,x,y,h,len,col){c.save();c.translate(x,y);c.rotate(h);c.fillStyle=col;c.strokeStyle='#fff';c.lineWidth=2;c.beginPath();c.moveTo(len,0);c.lineTo(-len*.55,len*.5);c.lineTo(-len*.3,0);c.lineTo(-len*.55,-len*.5);c.closePath();c.fill();c.stroke();c.restore()}
function line(c,pts,X,Y,w,col,dash){if(pts.length<2)return;c.strokeStyle=col;c.lineWidth=w;c.setLineDash(dash||[]);c.beginPath();pts.forEach((p,i)=>i?c.lineTo(X(p[0]),Y(p[1])):c.moveTo(X(p[0]),Y(p[1])));c.stroke();c.setLineDash([])}
function drawLocal(){const cv=document.getElementById('L');const [c,W,H]=fit(cv);c.clearRect(0,0,W,H);if(!S||!S.local){c.fillStyle=css('--mut');c.fillText('표본 대기',12,20);return}
 const L=S.local,me=L.frozen?null:(S.live&&S.live.base);const b=bounds([L.pts,L.traj,L.pts_out||[],me?[me]:[]],1.0);const sc=Math.min(W/(b[2]-b[0]),H/(b[3]-b[1]));
 const X=x=>(x-b[0])*sc+(W-(b[2]-b[0])*sc)/2, Y=y=>H-((y-b[1])*sc+(H-(b[3]-b[1])*sc)/2);
 c.strokeStyle=css('--grid');c.lineWidth=1;const g=Math.pow(10,Math.floor(Math.log10((b[2]-b[0])/4)));for(let v=Math.ceil(b[0]/g)*g;v<b[2];v+=g){c.beginPath();c.moveTo(X(v),0);c.lineTo(X(v),H);c.stroke()}
 for(let v=Math.ceil(b[1]/g)*g;v<b[3];v+=g){c.beginPath();c.moveTo(0,Y(v));c.lineTo(W,Y(v));c.stroke()}
 c.fillStyle=css('--mut');c.fillText(`격자 ${g} m`,8,H-8);
 if(L.pts_out&&L.pts_out.length){c.fillStyle='#b9b7b0';for(const p of L.pts_out)c.fillRect(X(p[0])-1,Y(p[1])-1,2,2);line(c,L.traj_out,X,Y,3,'#b9b7b0',[5,4]);
  c.fillStyle=css('--mut');c.fillText(`회색 = 걸었지만 60초 정합 창 밖(앞부분 ${L.n_out}표본)`,8,16)}
 c.fillStyle=css('--pt');c.globalAlpha=.55;for(const p of L.pts)c.fillRect(X(p[0])-1,Y(p[1])-1,2,2);c.globalAlpha=1;
 if(L.low&&L.low.length){c.fillStyle='#e8743b';c.globalAlpha=.75;for(const p of L.low)c.fillRect(X(p[0])-1,Y(p[1])-1,2,2);c.globalAlpha=1;c.fillStyle='#c25a1c';c.fillText('주황 = 낮은 것(책상·의자 등, 안내도에 없음)',8,30)}
 c.fillStyle='#0b6e4f';for(const p of L.last)c.fillRect(X(p[0])-1.5,Y(p[1])-1.5,3,3);
 if(false&&!L.frozen&&LIVESCAN&&LIVESCAN.length){   // 2026-10-15 사용자: 파란 실시간 스캔은 안 보여도 됨(A1.3.1 표본만)c.fillStyle='#2563eb';c.globalAlpha=.85;for(const p of LIVESCAN)c.fillRect(X(p[0])-1,Y(p[1])-1,2,2);c.globalAlpha=1;
  c.fillStyle='#1d4ed8';c.fillText('파랑 = 지금 스캔(초당 5번, 화면 전용) · 초록 = 위치추정 표본(3초마다)',8,44)}
 line(c,L.traj,X,Y,3,css('--tr'));
 if(L.frozen){const e=L.traj[L.traj.length-1];if(e){c.fillStyle=css('--me');c.beginPath();c.arc(X(e[0]),Y(e[1]),6,0,7);c.fill()}return}   // 결과(고정): 멈춘 시각 위치만
 if(me&&L.traj.length)line(c,[L.traj[L.traj.length-1],me],X,Y,2,css('--me'),[4,4]);   // 표본은 3초 간격·처리 지연만큼 늦다 — 지금 위치까지 점선
 if(me)arrow(c,X(me[0]),Y(me[1]),-me[2],14,css('--me'))}
function drawMap(){const cv=document.getElementById('M');const [c,W,H]=fit(cv);c.clearRect(0,0,W,H);const M=S&&S.map;MT=null;
 const E=(S&&S.exp)||{};let f,b,base=false;
 if(M)f=M.floor;else if(S&&S.base_floor&&E.state!=='computing'){f=+S.base_floor;base=true}   // 정합 전: 고른 층·정답 층 안내도 통째로(정답 위치 찍기용)
 else{c.fillStyle=css('--mut');c.fillText(S&&S.progress&&E.state==='computing'?`층 판별 계산 중 ${S.progress.done.length}/${S.progress.total}`:'아직 정합 전 — "정답 층?"을 고르면 그 층 안내도가 나와 출발 위치를 찍을 수 있음',12,20);return}
 if(!imgs[f]){imgs[f]=new Image();imgs[f].src='/map/'+f+'.png';imgs[f].onload=drawMap}
 const im=imgs[f],iw=im.naturalWidth||0,ih=im.naturalHeight||0;const mk=((S&&S.marks)||[]).filter(k=>k.floor===f);
 let out=false;
 if(base){if(!iw){c.fillStyle=css('--mut');c.fillText('안내도 불러오는 중',12,20);return}b=[0,0,iw,ih]}
 else{b=FULLMAP&&iw?[0,0,iw,ih]:bounds([M.pts,M.traj,mk.map(k=>[k.px,k.py]),(M.signs||[]).map(g=>[g.x,g.y])],120);if(!b){if(!iw)return;b=[0,0,iw,ih]}
  out=iw&&(b[2]<0||b[3]<0||b[0]>iw||b[1]>ih);if(out)b=[Math.min(b[0],0),Math.min(b[1],0),Math.max(b[2],iw),Math.max(b[3],ih)]}
 const sc=Math.min(W/(b[2]-b[0]),H/(b[3]-b[1]));const ox=(W-(b[2]-b[0])*sc)/2,oy=(H-(b[3]-b[1])*sc)/2;const X=x=>(x-b[0])*sc+ox,Y=y=>(y-b[1])*sc+oy;
 MT={f:f,b:b,sc:sc,ox:ox,oy:oy};
 if(im.complete&&iw){c.globalAlpha=.55;c.drawImage(im,b[0],b[1],b[2]-b[0],b[3]-b[1],ox,oy,(b[2]-b[0])*sc,(b[3]-b[1])*sc);c.globalAlpha=1}
 const E6=S&&S.e5&&S.e5.e6;if(E6&&S.e5.id&&(E6.hyps||[]).some(h=>h.floor===`${f}F`)){const k=`${S.e5.id}_${f}`;   // E6 후보 영역(보라)
  if(!E6IMG[k]){E6IMG[k]=new Image();E6IMG[k].src=`/e5/e6mask?ev=${S.e5.id}&floor=${f}`;E6IMG[k].onload=drawMap}
  const mi=E6IMG[k];if(mi.complete&&mi.naturalWidth)c.drawImage(mi,b[0],b[1],b[2]-b[0],b[3]-b[1],ox,oy,(b[2]-b[0])*sc,(b[3]-b[1])*sc)}
 if(!base){const me=M.me;
  c.fillStyle=css('--pt');c.globalAlpha=.6;for(const p of M.pts)c.fillRect(X(p[0])-1,Y(p[1])-1,2,2);c.globalAlpha=1;
  line(c,M.traj,X,Y,3,css('--tr'));line(c,M.after,X,Y,2,css('--tr'),[6,5]);
  if(me&&me[0]>b[0]&&me[0]<b[2]&&me[1]>b[1]&&me[1]<b[3])arrow(c,X(me[0]),Y(me[1]),-me[2],14,css('--me'))}
 if(!base){const SG=M.signs||[];SG.forEach((g,i)=>{const gx=X(g.x),gy=Y(g.y),rx=X(g.rx),ry=Y(g.ry),last=i===SG.length-1;   // 표지판 자리 = 영역(주황)
  const r=Math.max(16,(g.r_px||20)*sc),d=Math.hypot(gx-rx,gy-ry),a=Math.atan2(gy-ry,gx-rx),h=Math.max(Math.asin(Math.min(1,r/Math.max(d,r))),.12);
  c.save();c.globalAlpha=last?1:.45;
  c.fillStyle='rgba(245,158,11,.26)';c.strokeStyle='#d97706';c.lineWidth=1.5;c.setLineDash([5,4]);c.beginPath();c.moveTo(rx,ry);c.arc(rx,ry,d+r,a-h,a+h);c.closePath();c.fill();c.stroke();c.setLineDash([]);   // 로봇에서 본 방향(부채꼴)
  c.fillStyle='#d97706';c.beginPath();c.arc(rx,ry,5,0,2*Math.PI);c.fill();   // 멈춘 자리
  c.fillStyle='rgba(245,158,11,.38)';c.strokeStyle='#d97706';c.lineWidth=3;c.beginPath();c.arc(gx,gy,r,0,2*Math.PI);c.fill();c.stroke();   // 표지판이 있을 곳
  c.fillStyle='#b45309';c.beginPath();c.arc(gx,gy,4,0,2*Math.PI);c.fill();
  const lb=`${g.id?'#'+g.id+' ':''}${g.text||'표지판'} · ${g.range_m}m`;c.font='bold 18px sans-serif';const tw=c.measureText(lb).width;
  const lx=Math.min(gx+r+6,W-tw-10),ly=Math.max(gy-r-6,22);c.fillStyle='rgba(255,255,255,.88)';c.fillRect(lx-4,ly-18,tw+8,24);
  c.fillStyle='#92400e';c.fillText(lb,lx,ly);c.restore()})}
 for(const k of mk){const tx=X(k.px),ty=Y(k.py);                                     // 정답 위치(보라) · 추정 위치(빈 원) · 둘 사이 점선
  if(k.est){const ex=X(k.est[0]),ey=Y(k.est[1]);c.strokeStyle='#7c3aed';c.lineWidth=2;c.setLineDash([5,4]);c.beginPath();c.moveTo(ex,ey);c.lineTo(tx,ty);c.stroke();c.setLineDash([]);
   c.beginPath();c.arc(ex,ey,6,0,7);c.stroke()}
  if(k.heading!=null)arrow(c,tx,ty,k.heading,16,'#7c3aed');
  c.fillStyle='#7c3aed';c.strokeStyle='#fff';c.lineWidth=2;c.beginPath();c.arc(tx,ty,7,0,7);c.fill();c.stroke();
  c.save();c.fillStyle='#4c1d95';c.font='bold 12px sans-serif';c.fillText(`${k.name||''}${k.err_m!=null?' · '+k.err_m+' m':''}`,tx+10,ty-8);c.restore()}
 if(PEND&&PEND.f===f){c.strokeStyle='#7c3aed';c.lineWidth=3;c.beginPath();c.arc(X(PEND.px),Y(PEND.py),9,0,7);c.stroke()}
 if(out){c.save();c.fillStyle=css('--me');c.font='bold 14px sans-serif';c.fillText('정합 창이 지도 밖 — 이 층에는 안 맞음',12,22);c.restore()}}
function text(){const F=S.floor,st=S.status,M=S.map,E=S.exp||{};
 const go=document.getElementById('go'),sp=document.getElementById('stop'),rs=document.getElementById('rst');go.style.display=sp.style.display=rs.style.display=S.can_start?'':'none';
 go.disabled=['walking','waiting','computing','resetting'].includes(E.state);sp.disabled=E.state!=='walking';rs.disabled=['walking','resetting'].includes(E.state);
 const ex={idle:'[시작]을 누르고 걸으세요',walking:(E.elapsed||0)<60?`걷는 중 ${Math.floor(E.elapsed||0)}초 — 60초가 되면 [멈춤]`:`걷는 중 ${Math.floor(E.elapsed)}초 — 지금 [멈춤]! (60초 넘은 앞부분은 정합 창 밖)`,waiting:'멈춘 시각까지 표본 기다리는 중',
  computing:S.progress?`층 판별 계산 중 ${S.progress.done.length}/${S.progress.total}`:'층 판별 계산 중',done:S.prep_next?`다음 실험 준비 — 정답 층 고르고 [정답 위치]로 출발 위치 → [시작] (직전 결과 ${E.verdict??'-'}층은 [결과 보기]·보관 목록에)`:E.error?`결과 없음: ${E.error} — 다시 [시작]`:`결과: ${E.verdict}층 (순위 ${(E.ranking||[]).slice(0,3).join('·')}층)${E.archived?' · 보관됨':''}`,
  resetting:`초기화 중 — ${E.step||''}`}[E.state]||(E.error||'');
 document.getElementById('exp').textContent=S.archive_view?`지난 실험 ${S.archive_view.split('/').pop()} · ${ex}`:(S.can_start?ex:'');
 document.getElementById('floor').textContent=F?`판정 ${F.verdict}층`:'';
 document.getElementById('mh').textContent=M?(M.frozen?`${M.floor}층 안내도${M.selected?' (고른 층)':''} — 이 실험 정합 창 ${M.win_n}표본(고정) · twoway ${M.twoway??'-'} · 벽비용 ${M.wall_cost_cm}cm`
 :`${M.floor}층 안내도${M.selected?' (고른 층)':''} — 정합 창 ${M.win_n}표본(실선)·그 뒤 경로(점선)·지금 위치 · twoway ${M.twoway??'-'} · 벽비용 ${M.wall_cost_cm}cm`):(S.base_floor&&E.state!=='computing'?`${S.base_floor}층 안내도 — 정합 전(정답 위치 찍기용)`:'안내도 — 아직 정합 전');
 document.getElementById('stat').textContent=st?`표본 ${st.samples} · 처리 ${st.last_sample_sec}s · 지연 ${st.last_sample_lag_s}s · 정제 ${st.clean_removed_pct??'-'}%`:'';
 const R=document.getElementById('rank');R.innerHTML='';
 if(E.state==='computing'&&S.progress&&S.progress.cands){const h=document.createElement('span');h.className='st';h.textContent='계산 중 순위(끝난 층만):';R.appendChild(h);
  S.progress.cands.forEach((c,i)=>{const e=document.createElement('span');e.className='r'+(i==0?' top':'');e.textContent=`${i+1}위 ${c.map}층 ${c.twoway}`;R.appendChild(e)});return}
 if(F){const a=document.createElement('span');a.className='r'+(SEL===null?' sel':'');a.textContent='자동(판정 층)';a.onclick=()=>{SEL=null;tick()};R.appendChild(a);
  F.cands.forEach((c,i)=>{const e=document.createElement('span');e.className='r'+(i==0?' top':'')+(SEL===c.map?' sel':'');e.textContent=`${i+1}위 ${c.map}층 ${c.twoway}`;e.title=`벽비용 ${c.wall_cm}cm`;e.onclick=()=>{SEL=c.map;tick()};R.appendChild(e)})}}
function qs(){const p=[];if(SEL)p.push('floor='+SEL);if(VIEW)p.push('view='+VIEW);if(ARCH)p.push('exp='+encodeURIComponent(ARCH));return p.length?'?'+p.join('&'):''}
async function loadArch(){try{const L=await (await fetch('/experiments')).json();const el=document.getElementById('arch');const cur=el.value;el.innerHTML='<option value="">지금 실험</option>';
 L.forEach(x=>{const o=document.createElement('option');o.value=x.name;o.textContent=`${x.start} · 판정 ${x.verdict}층${x.truth?' · 정답 '+x.truth+'층':''}`;el.appendChild(o)});el.value=cur}catch(e){}}
loadArch();setInterval(loadArch,15000);
document.getElementById('arch').onchange=e=>{ARCH=e.target.value;SEL=null;metaSet=false;tick()};
async function tick(){if(busy)return;busy=true;try{S=await (await fetch('/state'+qs())).json();text();drawLocal();drawMap();
 if(!metaSet&&S.meta){document.getElementById('truth').value=S.meta.truth??'';document.getElementById('note').value=S.meta.note||'';metaSet=true}
 const vb=document.getElementById('vw');vb.style.display=S.can_start?'':'none';vb.textContent=S.view==='result'?'실시간 보기':'결과 보기(고정)';
 document.querySelector('main section h2').textContent=S.view==='result'?'FAST-LIO 좌표 — 이 실험 정합 창 누적(고정) · 경로 · 멈춘 위치':'FAST-LIO 좌표 — [시작]부터 누적(초록 = 최근 60초 정합 창, 회색 = 그 앞) · 경로 · 지금 위치 (실시간)'}
 catch(e){document.getElementById('stat').textContent='연결 끊김: '+e}busy=false}
let metaSet=false;
async function sendMeta(){const t=document.getElementById('truth').value,n=document.getElementById('note').value;await fetch('/meta',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({truth:t,note:n,exp:ARCH})});loadArch()}
document.getElementById('truth').onchange=sendMeta;document.getElementById('note').onchange=sendMeta;
document.getElementById('vw').onclick=()=>{VIEW=(S&&S.view==='result')?'live':'result';tick()};
let LIVESCAN=null;
async function ptick(){if(!S||busy||S.view==='result'||ARCH)return;try{const P=await (await fetch('/pose'+(SEL?'?floor='+SEL:''))).json();LIVESCAN=P.scan||null;if(P.base&&S.live)S.live.base=P.base;if(P.me&&S.map&&P.map===S.map.floor)S.map.me=P.me;drawLocal();drawMap()}catch(e){}}
setInterval(ptick,250);
// 2026-10-15 표지판 이벤트 띠: 빨강 = 세우세요(새 이벤트마다 소리), 노랑 = 세워 두세요, 초록 = 출발해도 됨(+ 읽은 글자)
let AC=null, E5LAST=null;
addEventListener('click',()=>{if(!AC){try{AC=new (window.AudioContext||window.webkitAudioContext)()}catch(e){}}},{once:false});
function beep(){if(!AC)return;[0,0.35,0.7].forEach(d=>{const o=AC.createOscillator(),g=AC.createGain();o.frequency.value=880;o.connect(g);g.connect(AC.destination);g.gain.value=0.25;o.start(AC.currentTime+d);o.stop(AC.currentTime+d+0.2)})}
let E5IMGID=null,E5FRAMET=0;
function rtxt(R){if(!R)return'';return (R.vlm&&R.vlm.type!==undefined?`${R.vlm.type} '${R.vlm.text||''}'`:`${R.status||''} ${(R.labels||[]).map(x=>x.label).join('/')}`)+(R.ocr&&R.ocr.length?` · OCR ${R.ocr.slice(0,3).join(' ')}`:'')}
function e5steps(E){const el=document.getElementById('e5steps');if(!E||!E.alive||!E.id){el.innerHTML='';return}   // 단계: 끝(초) / 하는 중(초) / 아직
 const T={};let fail=null;for(const x of E.steps||[]){if(!(x.s in T))T[x.s]=x.t;if(x.s==='FAILED')fail=x}
 const now=Date.now()/1000,R=E.reading,H=E.home||{},pt=E.pantilt,cs=E.console;
 const st=[['발견','STOP_REQUEST',null],['멈춤 확인','STOP_CONFIRMED','조종기로 세우세요'],[pt?'팬틸트 조준·초점':'조준(팬틸트 안 씀)','CAPTURE',cs&&cs.status?cs.status.replace(/^조준 #\d+ /,''):''],
  ['촬영','CAPTURED',''],...(pt?[['정면 복귀','HOME','']]:[]),['판독','READ',R?'':'OCR']];
 const done=k=>k==='READ'?!!R:k==='HOME'?H.ev===E.id&&H.state==='DONE':k in T, tdone=k=>k==='HOME'?H.t:k==='READ'?null:T[k];
 let prev=T['STOP_REQUEST'],html='',cur=false;
 for(const [lab,k,hint] of st){let cls='',txt=lab;
  if(done(k)){const t=tdone(k);cls='ok';txt=`✓ ${lab}`+(t&&prev&&k!=='STOP_REQUEST'?` ${(t-prev).toFixed(1)}초`:'')+(k==='READ'&&R?` → ${rtxt(R)}`:'');if(t)prev=t}
  else if(fail&&!cur){cls='bad';txt=`✗ ${lab} 실패: ${fail.r||''}`;cur=true}
  else if(!cur){cls='now';txt=`▶ ${lab} 하는 중 ${prev?Math.max(0,now-prev).toFixed(0):'?'}초${hint?' · '+hint:''}`;cur=true}
  html+=`<span class="${cls}">${txt.replace(/</g,'&lt;')}</span>`}
 if(pt&&H.ev===E.id&&H.state==='FAILED')html+=`<span class="bad">정면 복귀 실패</span>`;
 el.innerHTML=html}
function e5bar(){const E=S&&S.e5,sec=document.getElementById('e5sec'),al=document.getElementById('e5alert'),bar=document.getElementById('e5bar'),t=document.getElementById('e5t'),
 st=document.getElementById('e5s'),im=document.getElementById('e5img'),rd=document.getElementById('e5read'),hi=document.getElementById('e5hist'),cc=document.getElementById('e5capcap');
 if(!E){sec.style.display='none';al.style.display='none';return}sec.style.display='';
 if(!E.alive){bar.className='e5gry';t.textContent='표지판 이벤트 꺼짐';st.textContent=`(E5 상태 ${E.age}초 전 — E5 를 안 띄웠거나 멈춤)`;al.style.display='none';rd.textContent='';document.getElementById('e5e6').textContent='';return}
 if(E.has_capture&&E5IMGID!==E.id){E5IMGID=E.id;document.getElementById('e5cap').src='/e5/capture.jpg?'+E.id;im.style.display='';
  cc.textContent=`#${E.id} `+(E.capture_source==='IMX477_PANTILT'?'팬틸트 사진(누르면 크게)':'RealSense 대체 사진(팬틸트 안 움직임)')}
 const s=E.state,id=E.id?`#${E.id} `:'',dry=E.pantilt?'':' · 팬틸트 안 움직임(DRY_RUN)',snd=AC?'':' · 소리: 화면 한 번 클릭',g=E.gate||{};
 const ago=E.t_state?Math.max(0,Math.round(Date.now()/1000-E.t_state)):null,R=E.reading;
 const done=['CAPTURED','RELEASE','READ_ASYNC'].includes(s),old=ago!=null&&((done&&ago>60)||(s==='FAILED'&&ago>30));
 const show=(cls,txt)=>{al.className=cls;al.textContent=txt;al.style.display=''};al.style.display='none';
 if(['STOP_REQUEST','WAIT_STABLE'].includes(s)){bar.className='e5red';t.textContent=`멈추세요 — 표지판 ${id}`;st.textContent='조종기로 세우면 자동으로 조준합니다'+dry;
  show('e5red',`멈추세요 — 표지판 ${id}↓`);if(E5LAST!==E.id){E5LAST=E.id;beep()}}
 else if(['STOP_CONFIRMED','AIM','CAPTURE'].includes(s)){const w=s==='AIM'?'조준·초점':s==='CAPTURE'?'촬영':'정지 확인';bar.className='e5amb';t.textContent=`세워 두세요 — ${w} ${id}`;
  st.textContent=(ago!=null?`${ago}초째`:'')+dry;show('e5amb',`세워 두세요 — ${w} ${id}↓`)}
 else if(done&&!old){bar.className='e5grn';t.textContent=`출발해도 됨 ${id}`;st.textContent=E.capture_source&&E.capture_source!=='IMX477_PANTILT'?'사진: RealSense 대체':''}
 else if(s==='FAILED'&&!old){bar.className='e5bad';t.textContent=`이벤트 실패 ${id}— 출발해도 됨`;st.textContent=`${E.why||''}${E.aim_why?' · '+E.aim_why:''}`}
 else if(g.active&&E.p2_wait&&E.p2_wait.n){bar.className='e5pre';t.textContent='표지판 확인됨 — 천천히 걸으세요';st.textContent=`곧 멈춤 신호 · 가장 가까운 것 ${(E.p2_wait.near||0).toFixed(1)} m${dry}`;show('e5pre','표지판 확인됨 — 천천히')}
 else{bar.className='e5gry';t.textContent=g.active?'표지판 찾는 중':`찾기 쉼 — ${g.why||'웹 상태 없음'}`;
  st.textContent=`멈춤 ${E.n_stops??0}번 · 오돔 ${(E.health||{}).odom||'?'} · 카메라 ${(E.health||{}).color||'?'}${dry}${snd}`}
 rd.textContent=R?('읽음: '+rtxt(R)):(done&&!old?'판독 중…':'');
 const e6=document.getElementById('e5e6'),X6=E.e6;e6.textContent=X6?(X6.hyps&&X6.hyps.length?`E6 후보(호실 ${X6.label||'?'}): `+X6.hyps.map(h=>`${h.floor} ${Math.round(h.p*100)}%·영역 ${h.n}`).join(' · ')+' — 오른쪽 안내도 보라 = 로봇이 있을 수 있는 곳':`E6: ${X6.status}`):(R?'E6: 위치 단서로 쓸 호실 번호 없음(OCR 로 확인 안 됨)':'');
 const ox=document.getElementById('e5ox'),L5=E.labels||{},key=`${E.run}#${E.id}`;
 if(E.id){ox.style.display='flex';document.getElementById('e5oxq').textContent=`#${E.id} 이 멈춤이 표지판이 맞나요?`;
  document.getElementById('e5oxr').textContent=L5[key]?`판정: ${L5[key]}`:'아직 판정 안 함';
  const cnt=pre=>{let o=0,x=0;for(const [k,v] of Object.entries(L5)){if(pre&&!k.startsWith(pre))continue;if(v==='O')o++;else if(v==='X')x++}return [o,x]},   // SigLIP 멈춤 정답률(O/X 누른 것만)
   pct=([o,x])=>o+x?`O ${o} · X ${x} = ${Math.round(100*o/(o+x))}%`:'아직 없음',[ro,rx]=cnt(`${E.run}#`),[ao,ax]=cnt(''),acc=document.getElementById('e5acc');
  acc.textContent=`SigLIP 멈춤 정답률 — 이번 실행 ${pct([ro,rx])} · 오늘 전체 ${pct([ao,ax])} (목표 80%${ao+ax<20?', 20번 넘게 모아야 믿을 만함':''})`;
  acc.className=ao+ax?(ao/(ao+ax)>=0.8?'ok':'low'):'';
  document.getElementById('e5o').onclick=()=>sendOX(E.run,E.id,'O');document.getElementById('e5x').onclick=()=>sendOX(E.run,E.id,'X');
  if(E5STOPID!==E.id){E5STOPID=E.id;document.getElementById('e5stop').src='/e5/last_stop.jpg?'+E.id;document.getElementById('e5stopcap').textContent=`#${E.id} 멈춤 순간(주황 상자 = SigLIP 이 본 표지판, 굵은 것 = 멈춤 낸 것)`}}
 else ox.style.display='none';
 hi.textContent='';(E.events||[]).forEach(e=>{const d=document.createElement('div');const lb=L5[`${E.run}#${e.id}`];
  d.textContent=`#${e.id} ${e.state}${e.aim?' · 조준 '+e.aim:''}${e.reading?' · '+rtxt(e.reading):''} · 판정 ${lb||'-'}`;hi.appendChild(d)})}
let E5STOPID=null;
function e5clear(){const c=document.getElementById('e5img'),s=document.getElementById('e5stop');   // KCI 10-17: 지난 이벤트 사진 지우기
 if(E5IMGID!==null){E5IMGID=null;if(c)c.style.display='none';const ci=document.getElementById('e5cap');if(ci)ci.removeAttribute('src')}
 if(E5STOPID!==null){E5STOPID=null;if(s)s.removeAttribute('src');const sc=document.getElementById('e5stopcap');if(sc)sc.textContent='멈춤 순간(후보 상자)'}}
setInterval(()=>{try{const E=S&&S.e5;if(E&&E.alive&&!E.id)e5clear()}catch(e){}},500);
async function sendOX(run,ev,lab){const r=await (await fetch('/e5/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:run,ev:ev,label:lab})})).json();
 document.getElementById('e5oxr').textContent=r.error?('저장 실패: '+r.error):`판정: ${lab} 저장`;tick()}
setInterval(()=>{try{e5bar()}catch(e){}try{e5steps(S&&S.e5)}catch(e){}},500);
function e5grant(u){fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ttl_s:120})}).then(r=>r.json()).then(j=>{const g=document.getElementById('e5g');g.textContent=j.ok?(j.grant?'허용 '+j.grant.id+' (2분)':(j.abort_latched?'중단 고정':'해제')):('거절: '+(j.why||''))}).catch(()=>{})}
setInterval(()=>{const E=S&&S.e5,f=document.getElementById('e5frame'),on=E&&E.alive&&!document.hidden;   // 실시간 사진: 로봇이 새 사진마다 밀어 보냄(MJPEG)
 if(on&&(!E5FRAMET||Date.now()-E5FRAMET>60000)){E5FRAMET=Date.now();f.onerror=()=>{E5FRAMET=0};f.src='/e5/live.mjpg?'+E5FRAMET}
 else if(!on&&E5FRAMET){E5FRAMET=0;f.removeAttribute('src')}},1000);
document.getElementById('rst').onclick=async()=>{if(!confirm('로봇을 세워 두셨나요? FAST-LIO 를 다시 켜고 쌓인 데이터를 비웁니다.'))return;await fetch('/reset',{method:'POST'});tick()};
document.getElementById('go').onclick=async()=>{VIEW=null;await fetch('/start',{method:'POST'});tick()};
document.getElementById('stop').onclick=async()=>{await fetch('/stop',{method:'POST'});tick()};
let ARMED=false, MT=null, PEND=null, FULLMAP=false; const E6IMG={};
async function recToggle(){const R=S&&S.man_rec,on=!(R&&['켜는 중','녹화 중'].includes(R.state));
 if(!on&&!confirm('녹화를 끌까요?'))return;
 await fetch('/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:on,camera:document.getElementById('reccam').checked})});tick()}
async function unmark(){if(ARCH){document.getElementById('mkr').textContent='지난 실험 보기에서는 못 지움 — 지금 실험에서';return}
 const r=await (await fetch('/unmark',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json();
 document.getElementById('mkr').textContent=r.error?r.error:`지움: ${r.removed.name||''} (${r.removed.floor}층)`;tick()}
function arm(on){ARMED=on;PEND=null;const b=document.getElementById('mkb');b.classList.toggle('armed',on);b.textContent=on?'찍기 취소':'정답 위치';
 document.getElementById('M').classList.toggle('armed',on);if(on)document.getElementById('mkr').textContent='① 로봇이 서 있는 곳 클릭 → ② 로봇이 보는 쪽 클릭';if(S)drawMap()}
async function saveMark(p,h){const tr=S&&S.meta&&S.meta.truth;if(tr&&+tr!==p.f&&!confirm(`지금 보는 지도는 ${p.f}층인데 정답 층은 ${tr}층입니다. 그래도 ${p.f}층에 찍을까요?`)){arm(false);return}
 const r=await (await fetch('/mark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:document.getElementById('mk').value.trim(),floor:p.f,px:p.px,py:p.py,heading:h,exp:ARCH})})).json();
 document.getElementById('mkr').textContent=r.error?('정답 위치 저장 실패: '+r.error):(`정답 위치 ${r.name} (${r.floor}층${h==null?', 방향 없음':', 방향 포함'}) 저장`+(r.err_m!=null?` · 추정과 ${r.err_m} m (${r.err_px} px, ${r.basis})`:` · ${r.basis||''}`));
 arm(false);document.getElementById('mk').value='';tick()}
document.getElementById('mkb').onclick=()=>{if(ARMED&&PEND){saveMark(PEND,null);return}arm(!ARMED)};
document.getElementById('M').addEventListener('click',ev=>{if(!ARMED||!MT)return;const r0=ev.currentTarget.getBoundingClientRect();
 const px=(ev.clientX-r0.left-MT.ox)/MT.sc+MT.b[0],py=(ev.clientY-r0.top-MT.oy)/MT.sc+MT.b[1];
 if(!PEND){PEND={px:px,py:py,f:MT.f};document.getElementById('mkr').textContent='② 로봇이 보는 쪽을 클릭(방향 없이 저장하려면 버튼 다시)';drawMap();return}
 if(Math.hypot(px-PEND.px,py-PEND.py)<2)return;saveMark(PEND,Math.atan2(py-PEND.py,px-PEND.px))});
setInterval(()=>{const e=S&&S.exp&&S.exp.record,m=S&&S.man_rec,f=x=>x.msg?' — '+x.msg.replace(/\/\S*\//g,'').replace(/\s+/g,' '):'';
 const t=[];if(e)t.push('실험 녹화 '+e.name+': '+e.state+f(e));if(m)t.push('녹화'+(m.name?' '+m.name:'')+(m.camera?'(카메라)':'')+': '+m.state+f(m));
 document.getElementById('rec').textContent=t.join(' · ');const b=document.getElementById('recb'),on=m&&['켜는 중','녹화 중'].includes(m.state);
 b.textContent=on?'녹화 끄기':'녹화 켜기';b.classList.toggle('on',!!on)},1000);
setInterval(tick,1000);tick();addEventListener('resize',()=>{if(S){drawLocal();drawMap()}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/dev/shm/rt_live')
    ap.add_argument('--building', default='kci')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--window', type=float, default=60.0)
    ap.add_argument('--per', type=int, default=300, help='표본당 보낼 점 수(대략)')
    ap.add_argument('--trigger', default=None, help='실험 신호 파일(노드 floor_trigger 와 같은 경로) — 주면 [시작]/[멈춤] 버튼이 생긴다')
    ap.add_argument('--archive', default=None, help='실험마다 표본·결과를 따로 보관할 폴더(기본: ../logs/live_<월일>/experiments)')
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(HERE.parent / 'floor_id' / 'config' / 'buildings.yaml'))
    b = (cfg.get('buildings') or cfg)[a.building]
    maps_root = HERE.parent / 'floor_id' / b['maps_root']
    import time as _t
    archive = a.archive or str(HERE.parent / 'logs' / _t.strftime('live_%m%d') / 'experiments')
    S = State(a.root, maps_root, a.window, a.per, a.trigger, archive if a.trigger else None)
    logs_root = Path(archive).parent.parent if a.trigger else None

    def _ticker():
        import time
        while True:
            try:
                S.tick()
                S.write_gate()
            except Exception:
                pass
            time.sleep(0.3)
    threading.Thread(target=_ticker, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == '/start':
                return self._send(200, json.dumps(S.start()).encode(), 'application/json')
            if self.path == '/stop':
                return self._send(200, json.dumps(S.stop()).encode(), 'application/json')
            if self.path == '/meta':
                n = int(self.headers.get('Content-Length') or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    body = {}
                ex = str(body.get('exp') or '')
                if ex and logs_root and '..' not in ex:                # 지난 실험을 보는 중이면 그 실험 기록에 넣는다
                    d = Path(logs_root) / ex.split('/')[0] / 'experiments' / ex.split('/')[-1]
                    if (d / 'experiment.json').exists():
                        import experiments
                        e = experiments.set_truth(d, body.get('truth'), body.get('note'))
                        experiments.write_index(logs_root)
                        return self._send(200, json.dumps(dict(truth=e.get('truth'), note=e.get('note')), ensure_ascii=False).encode(), 'application/json')
                return self._send(200, json.dumps(S.set_meta(body.get('truth'), body.get('note')), ensure_ascii=False).encode(), 'application/json')
            if self.path == '/record':
                n = int(self.headers.get('Content-Length') or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    body = {}
                return self._send(200, json.dumps(S.record(bool(body.get('on')), bool(body.get('camera'))), ensure_ascii=False).encode(), 'application/json')
            if self.path == '/reset':
                return self._send(200, json.dumps(S.reset()).encode(), 'application/json')
            if self.path == '/unmark':
                return self._send(200, json.dumps(S.unmark(), ensure_ascii=False).encode(), 'application/json')
            if self.path == '/mark':
                n = int(self.headers.get('Content-Length') or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    body = {}
                fx, px_, py_ = body.get('floor'), body.get('px'), body.get('py')
                ex = str(body.get('exp') or '')
                if ex and logs_root and '..' not in ex:                # 지난 실험을 보는 중이면 그 실험의 정합으로 재고 그 기록에 넣는다
                    d = Path(logs_root) / ex.split('/')[0] / 'experiments' / ex.split('/')[-1]
                    if (d / 'experiment.json').exists():
                        av = ArchiveView(d, maps_root, a.window, a.per)
                        av.marks_path = S.marks_path
                        r = av.mark(body.get('name'), fx, px_, py_, body.get('heading'), body.get('role'))
                        try:
                            e = json.loads((d / 'experiment.json').read_text())
                            e['marks'] = av.exp.get('marks', [])
                            (d / 'experiment.json').write_text(json.dumps(e, ensure_ascii=False, indent=1))
                        except (OSError, ValueError) as err:
                            r['error'] = f'{type(err).__name__}: {err}'
                        return self._send(200, json.dumps(r, ensure_ascii=False).encode(), 'application/json')
                return self._send(200, json.dumps(S.mark(body.get('name'), fx, px_, py_, body.get('heading'), body.get('role')), ensure_ascii=False).encode(), 'application/json')
            if self.path in ('/e5/arm', '/e5/disarm', '/e5/abort'):   # E5RT2: explicit operator grant / abort latch
                n = int(self.headers.get('Content-Length') or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    body = {}
                r = S.e5_arm(body) if self.path == '/e5/arm' else S.e5_disarm(abort=self.path == '/e5/abort')
                return self._send(200, json.dumps(r, ensure_ascii=False).encode(), 'application/json')
            if self.path == '/e5/label':                 # 2026-10-15: 표지판 이벤트 O/X(사람 판정)
                n = int(self.headers.get('Content-Length') or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    body = {}
                return self._send(200, json.dumps(S.e5_label(body), ensure_ascii=False).encode(), 'application/json')
            self._send(404, b'not found', 'text/plain')

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                return self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            if self.path.startswith('/pose'):
                v = self.path.split('?floor=', 1)[1].split('&')[0] if '?floor=' in self.path else ''
                return self._send(200, json.dumps(S.pose(int(v) if v.isdigit() else None)).encode(), 'application/json')
            if self.path.startswith('/experiments'):
                return self._send(200, json.dumps(list_archive(logs_root) if logs_root else [], ensure_ascii=False).encode(), 'application/json')
            if self.path.startswith('/state'):
                from urllib.parse import urlparse, parse_qs, unquote
                q = parse_qs(urlparse(self.path).query)
                fv = (q.get('floor') or [''])[0]
                fs = int(fv) if fv.isdigit() else None
                vw = (q.get('view') or [None])[0]
                ex = unquote((q.get('exp') or [''])[0])
                if ex and logs_root and '..' not in ex:
                    d = Path(logs_root) / ex.split('/')[0] / 'experiments' / ex.split('/')[-1]
                    if (d / 'experiment.json').exists():
                        av = ArchiveView(d, maps_root, a.window, a.per)
                        r = av.snapshot(fs, 'result')
                        r.update(can_start=S.trigger is not None, archive_view=ex, meta=dict(truth=av.exp.get('truth'), note=av.exp.get('note', '')))
                        return self._send(200, json.dumps(r, ensure_ascii=False).encode(), 'application/json')
                return self._send(200, json.dumps(S.snapshot(fs, vw), ensure_ascii=False).encode(), 'application/json')
            if self.path.startswith('/e5/live.mjpg'):                  # 2026-10-15 밤: frame.jpg 가 바뀔 때마다 바로 보냄(10분 뒤 끊음 — 웹이 1분마다 다시 붙음)
                import time as _t
                f = Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5')) / 'frame.jpg'
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=fr')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                last, t_end = None, _t.time() + 600
                try:
                    # 2026-10-16: 사진 바로 뒤에 경계와 '다음 사진 머리'까지 붙인다. 크롬은 다음 사진 머리를 받아야 앞 사진을 그려서,
                    #   경계를 사진 앞에 붙이면 늘 한 장(약 0.13초) 늦게 보였고 사진이 하나뿐이면 아예 안 떴다(화면 없는 크롬 시험으로 확인).
                    self.wfile.write(b'--fr\r\nContent-Type: image/jpeg\r\n\r\n')
                    while _t.time() < t_end:
                        try:
                            m = f.stat().st_mtime_ns
                        except OSError:
                            m = None
                        if m and m != last:
                            last = m
                            b = f.read_bytes()
                            self.wfile.write(b + b'\r\n--fr\r\nContent-Type: image/jpeg\r\n\r\n')
                            self.wfile.flush()
                        _t.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            if self.path.startswith('/e5/frame.jpg') or self.path.startswith('/e5/last_stop.jpg'):
                f = Path(os.environ.get('SCI_SHM', '/dev/shm/sci_e5')) / 'frame.jpg' if 'frame' in self.path else State.e5_stop_image()
                if f is not None and f.is_file():
                    return self._send(200, f.read_bytes(), 'image/jpeg')
                return self._send(404, b'no frame', 'text/plain')
            if self.path.startswith('/e5/e6mask'):
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                e5 = S.e5_status() or {}
                try:
                    b = State.e6_mask(e5.get('run_dir', ''), int(q['ev'][0]), int(q['floor'][0]))
                except (KeyError, ValueError):
                    b = None
                return self._send(200, b, 'image/png') if b else self._send(404, b'no mask', 'text/plain')
            if self.path.startswith('/e5/capture.jpg'):
                c = State.e5_capture()
                if c is not None:
                    return self._send(200, c.read_bytes(), 'image/jpeg')
                return self._send(404, b'no capture', 'text/plain')
            if self.path.startswith('/map/') and self.path.endswith('.png'):
                m = self.path[5:-4]
                if m.isdigit():
                    p = maps_root / f'{int(m)}F' / 'maps' / 'base_map.png'
                    if p.exists():
                        return self._send(200, p.read_bytes(), 'image/png')
            self._send(404, b'not found', 'text/plain')

    srv = ThreadingHTTPServer(('0.0.0.0', a.port), H)
    print(f'실시간 보기: http://<로봇 IP>:{a.port}  (root {a.root}, 건물 {a.building})', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
