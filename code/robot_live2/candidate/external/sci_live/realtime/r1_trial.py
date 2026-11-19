# -*- coding: utf-8 -*-
"""r1_trial.py — R1 (2026-10-16): 실험(trial) 세대 표시와 '지금 결과' 판정. 순수 Python(ROS·GPU·장치 없음).

문제(감사 01/02/09): 실패한 실험에 직전 실험의 floor.json·pose_fit.json 이 '지금 결과'로 쓰였다(화면·pose 작업자·/floorplan_loc/pose).
방법: run 별 불변 결과 + 지금 세대 포인터(out/trial_current.json, r1_contracts.trial_record).
  - 쓰는 쪽(loc_worker): 계산 시작 때 세대 토큰을 잡고, 쓰기 직전 다시 확인한다. 다르면 지금 파일을 덮지 않고 폐기 기록만 남긴다.
  - 읽는 쪽(viewer·노드 발행·pose 작업자): 파일에 붙은 trial_id·generation_id 가 지금 포인터와 같을 때만 '지금 결과'로 쓴다.
    확인과 쓰기 사이에 세대가 바뀌는 경쟁(다른 프로세스)이 있어도, 늦게 쓰인 결과는 꼬리표가 달라 읽는 쪽에서 거부된다.
  - 포인터가 없으면(실험 모드가 아닌 연속 운용) 예전처럼 동작하고 'LEGACY_UNTRACKED' 로 표시한다.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = Path(os.environ.get('SCI_WS', str(_HERE.parents[2]))) / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from sci_common import r1_contracts as C  # noqa: E402

POINTER = C.TRIAL_FILE                     # out/trial_current.json
CURRENT_FILES = ('floor.json', 'pose_fit.json', 'floor_progress.json')
TERMINAL_STATES = ('DONE', 'FAILED', 'CANCELLED')


def default_persist_dir():
    """/dev/shm 밖(재부팅에 남는) 기록 폴더. 노드가 따로 주지 않으면 sci_live/logs/r1_runtime."""
    return Path(os.environ.get('RT_PERSIST_DIR', str(_HERE.parent / 'logs' / 'r1_runtime')))


def pointer_path(out):
    return Path(out) / POINTER


def read_pointer(out):
    return C.read_json(pointer_path(out))


def tag(rec, token):
    """결과에 세대 꼬리표를 붙인다(token 없음 = 포인터 없는 연속 운용)."""
    rec = dict(rec)
    if token:
        rec.update(trial_id=token['trial_id'], generation_id=token['generation_id'])
    else:
        rec.update(trial_id=None, trial_id_null_reason='NO_TRIAL_POINTER(continuous mode)', generation_id=None)
    rec['schema_version'] = C.SCHEMA_VERSION
    return rec


class TrialPointer:
    """웹 화면(viewer)이 [시작]/[멈춤]/결과/[초기화] 때 쓰는 지금 세대 포인터. 이력은 persist/trial_history.jsonl 에 덧붙인다."""

    def __init__(self, out, persist=None):
        self.out, self.persist = Path(out), Path(persist) if persist else None

    def current(self):
        return read_pointer(self.out)

    def start(self, exp_id):
        cur = self.current() or {}
        gen = int(cur.get('generation_id') or 0) + 1
        rec = C.trial_record(trial_id=f'{exp_id}-g{gen}', exp_id=exp_id, generation_id=gen, state='WALKING')
        invalidate_current(self.out, cur, self.persist)            # 이전 결과를 '지금' 자리에서 즉시 치운다(보관 사본으로)
        self._write(rec)
        return rec

    def set_state(self, state, **kw):
        cur = self.current()
        if not cur:
            return None
        rec = dict(cur, state=state, **C.stamp(), **kw)
        self._write(rec)
        return rec

    def _write(self, rec):
        C.atomic_write_json(pointer_path(self.out), rec)
        if self.persist:
            try:
                C.append_jsonl(self.persist / 'trial_history.jsonl', rec)
            except OSError:
                pass


def invalidate_current(out, old_pointer, persist=None):
    """새 세대 시작: out/ 의 지금 결과 파일을 out/history/<옛 trial>/ 로 옮긴다(지우지 않음, 덮어쓰지 않음)."""
    out = Path(out)
    tid = (old_pointer or {}).get('trial_id') or 'untracked'
    dst = out / 'history' / f'{tid}-{time.strftime("%Y%m%dT%H%M%S")}'
    moved = []
    for n in CURRENT_FILES:
        p = out / n
        if p.exists():
            dst.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(p, dst / n)
                moved.append(n)
            except OSError:
                pass
    if moved and persist:
        try:
            C.append_jsonl(Path(persist) / 'invalidated.jsonl', dict(C.stamp(), old_trial=tid, moved=moved, to=str(dst)))
        except OSError:
            pass
    return moved


def classify(rec, pointer):
    """결과 파일 하나가 지금 실험의 것인지.
    돌려줌: (status, info) — CURRENT | LEGACY_UNTRACKED | STALE_OTHER_TRIAL | FAILED | CANCELLED | NONE"""
    if not rec:
        return 'NONE', {}
    info = dict(origin_trial=rec.get('trial_id'), origin_generation=rec.get('generation_id'), origin_t=rec.get('t', rec.get('t_now')))
    if pointer is None:
        if rec.get('status') == 'FAILED':
            return 'FAILED', info
        return 'LEGACY_UNTRACKED', info
    if rec.get('trial_id') != pointer.get('trial_id') or rec.get('generation_id') != pointer.get('generation_id'):
        return 'STALE_OTHER_TRIAL', info
    if pointer.get('state') == 'CANCELLED':
        return 'CANCELLED', info
    if rec.get('status') == 'FAILED':
        return 'FAILED', info
    return 'CURRENT', info


def usable_now(rec, pointer):
    """'지금 결과'로 그리거나 발행해도 되는가."""
    s, _ = classify(rec, pointer)
    return s in ('CURRENT', 'LEGACY_UNTRACKED')


def token_now(out):
    """지금 세대 토큰. 포인터 없음 = None(연속 운용). 취소된 세대도 토큰은 돌려주고 still_current 가 거부한다."""
    p = read_pointer(out)
    if not p:
        return None
    return dict(trial_id=p['trial_id'], generation_id=p['generation_id'])


def still_current(out, token):
    if token is None:                                   # 포인터 없는 연속 운용: 포인터가 새로 생기면 더는 지금 것이 아님
        return read_pointer(out) is None
    return C.GenerationGuard(pointer_path(out)).still_current(token)


def write_result(out, name, rec, token, persist, kind, write_fn=None):
    """세대 꼬리표를 붙인 결과를 persist/trials/<trial>/ 에 불변으로 남기고, 쓰기 직전 세대가 같을 때만 out/<name> 을 바꾼다.
    다르면 폐기 기록(persist/discarded.jsonl)만 남긴다. 돌려줌: True=지금 파일을 바꿈."""
    rec = tag(rec, token)
    tid = (token or {}).get('trial_id') or 'untracked'
    if persist:
        try:
            d = Path(persist) / 'trials' / tid
            d.mkdir(parents=True, exist_ok=True)
            C.atomic_write_json(d / f'{kind}_{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}_{time.monotonic_ns()}.json', rec)
        except OSError:
            pass
    if not still_current(out, token):
        if persist:
            try:
                C.append_jsonl(Path(persist) / 'discarded.jsonl',
                               dict(C.stamp(), kind=kind, name=name, reason='GENERATION_MISMATCH', token=token, now=read_pointer(out)))
            except OSError:
                pass
        return False
    (write_fn or C.atomic_write_json)(Path(out) / name, rec)
    return True


def failure_record(signal, errors, t_now, stage, run=None):
    """모든 층 실패·예외·취소 바퀴의 끝 기록(이전 floor.json 을 계속 쓰지 않게). errors: [{floor, error_type, error, stage}]"""
    kinds = sorted({e.get('reason_code') or e.get('error_type') or '?' for e in errors})
    return dict(status='FAILED', signal=signal, t_now=t_now, verdict=None, vote=None, ranking=[], cands=[], pass_sec=None,
                pass_index=None, stage=stage, run=run, per_floor_errors=errors, error_kinds=kinds, wall_time=time.time(),
                fix_status='NOT_CONFIRMED')


def error_entry(floor, exc_or_msg, stage):
    """예외(또는 자식이 보낸 문자열) → 기록 한 줄. CalibrationUnavailable 이면 reason_code 를 그대로."""
    if isinstance(exc_or_msg, BaseException):
        rc = getattr(exc_or_msg, 'reason_code', None)
        return dict(floor=floor, error_type=type(exc_or_msg).__name__, error=str(exc_or_msg)[:300], stage=stage,
                    reason_code=rc, traceback=traceback.format_exception_only(type(exc_or_msg), exc_or_msg)[-1].strip()[:300])
    d = dict(exc_or_msg) if isinstance(exc_or_msg, dict) else dict(error=str(exc_or_msg))
    d.setdefault('floor', floor); d.setdefault('stage', stage)
    if 'error_type' not in d:
        d['error_type'] = str(d.get('error', '')).split(':', 1)[0] or 'Error'
    return d


def persist_failure(persist, rec):
    if persist:
        try:
            C.append_jsonl(Path(persist) / 'floor_failures.jsonl', dict(C.stamp(), **rec))
        except OSError:
            pass


def describe_failure(rec):
    """화면 문구: 고정 문구('걸은 거리가 너무 짧음') 대신 실제 예외 종류·층·단계."""
    errs = rec.get('per_floor_errors') or []
    if not errs:
        return '층 정합 실패(층별 오류 기록 없음)'
    by = {}
    for e in errs:
        k = e.get('reason_code') or e.get('error_type') or '?'
        by.setdefault(k, []).append(e.get('floor'))
    parts = [f"{k}: {','.join(str(f) for f in sorted(v, key=str))}층" for k, v in sorted(by.items())]
    first = (errs[0].get('error') or '')[:120]
    return f"모든 층({len(errs)}개) 정합 실패 — {' / '.join(parts)} (예: {first})"


def guarded_publish(out, rec, publish_fn, reject_fn=None):
    """ROS 발행 직전 마지막 확인(rt_live_node._publish_pose·on_status 가 부른다). 지금 세대 결과일 때만 publish_fn() 을 부른다.
    돌려줌: (발행했나, 판정)."""
    s, info = classify(rec, read_pointer(out))
    if s in ('CURRENT', 'LEGACY_UNTRACKED'):
        publish_fn()
        return True, s
    if reject_fn:
        reject_fn(s, info)
    return False, s
