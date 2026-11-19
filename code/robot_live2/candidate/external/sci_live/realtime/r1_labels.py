# -*- coding: utf-8 -*-
"""r1_labels.py — R1 (2026-10-16): 평가 전용 라벨. 한 개의 O/X 대신 단계별 라벨과 단계별 필요 증거.

  sign_true          표지판 진위          — 정지 근거 사진(후보를 판단한 원본/정지 사진)만 있으면 OCR 전에도 가능
  stop_position_ok   정지 위치 적합       — 실제 정지 확인(STOP_CONFIRMED 이후)
  aim_ok             조준                 — 팬틸트 촬영본(IMX477) 존재
  focus_ok           초점                 — 팬틸트 촬영본(IMX477) 존재
  text_correct       판독 문자 정확       — 판독 결과 + 그 판독의 입력 이미지
  final_pose_correct 최종 위치            — FIX_CONFIRMED(E7) 필요. 지금은 E7 미연결이라 받지 않는다

기록: e5_labels_v2.jsonl(덧붙이기만, 같은 키의 다시 누름은 revision 을 올려 이력으로). 10-15 의 e5_labels.jsonl(무효 O/X)은
읽기만 하고 고치지 않으며 정답률 계산에서 뺀다. 라벨은 추론 입력으로 쓰이지 않는다(평가 전용).
"""
import json
import time
from pathlib import Path

LABEL_TYPES = ('sign_true', 'stop_position_ok', 'aim_ok', 'focus_ok', 'text_correct', 'final_pose_correct')
NEEDS = {'sign_true': ('stop_image',), 'stop_position_ok': ('stop_confirmed',), 'aim_ok': ('pantilt_capture',),
         'focus_ok': ('pantilt_capture',), 'text_correct': ('reading', 'reading_input_image'), 'final_pose_correct': ('fix_confirmed',)}
AFTER_STOP = ('STOP_CONFIRMED', 'AIM', 'CAPTURE', 'CAPTURED', 'RELEASE', 'READ_ASYNC', 'LOCALIZE_ASYNC', 'FIX_CONFIRMED', 'CLOSED')
PANTILT_CAPTURE_STATUS = ('RAW_CAPTURE_SAVED', 'CAPTURE_QUALITY_OK', 'CAPTURE_QUALITY_FAIL')


def evidence(event, run_dir):
    """E5 상태의 이벤트 한 개(+ run 폴더 파일)로 각 증거가 있는지. 없는 것을 있다고 하지 않는다."""
    e = event or {}
    ev = e.get('id')
    run = Path(run_dir) if run_dir else None
    stop_img = False
    if run is not None and ev is not None:
        try:
            stop_img = (run / 'trials' / f'ev{int(ev):03d}' / 'runner' / 'stop_frame.jpg').is_file() or \
                any((run / 'events').glob(f'stop_{int(ev):03d}_*.jpg'))
        except (OSError, ValueError):
            stop_img = False
    states = set([e.get('state')] + [s.get('state') if isinstance(s, dict) else s for s in (e.get('states') or [])])
    cap_st = e.get('capture_status')
    src = str(e.get('capture_source') or '')
    pantilt_cap = (cap_st in PANTILT_CAPTURE_STATUS and e.get('pantilt_ok') is not False) or \
        (cap_st is None and src.startswith('IMX477'))
    r = e.get('reading') or {}
    r_img = bool(r.get('input_sha256') or (r.get('ocr') or {}).get('input_sha256') or r.get('image') or e.get('capture'))
    return dict(stop_image=stop_img, stop_confirmed=bool(states & set(AFTER_STOP)), pantilt_capture=bool(pantilt_cap),
                reading=bool(r), reading_input_image=bool(r) and r_img, fix_confirmed=e.get('state') == 'FIX_CONFIRMED')


def parse_body(body):
    """요청 → (label_type, value, legacy_ui). 예전 화면의 O/X 는 '표지판 진위(sign_true)' 로 받는다."""
    lt = body.get('label_type')
    if lt is None:
        lab = str(body.get('label') or '').upper()
        if lab not in ('O', 'X', ''):
            return None, None, 'label 은 O, X, 빈칸'
        return 'sign_true', {'O': True, 'X': False, '': None}[lab], 'legacy_ox_ui'
    if lt not in LABEL_TYPES:
        return None, None, f'label_type 은 {LABEL_TYPES} 중 하나'
    v = body.get('value')
    if v not in (True, False, None):
        return None, None, 'value 는 true/false/null'
    return lt, v, None


def make_record(body, event, run_dir, path_v2, now=None):
    lt, value, note = parse_body(body)
    if lt is None:
        return dict(error=note)
    ev_ = evidence(event, run_dir)
    missing = [k for k in NEEDS[lt] if not ev_.get(k)]
    if missing and value is not None:
        return dict(error=f'{lt} 에 필요한 증거 없음: {missing}', label_type=lt, evidence=ev_)
    key = (str(body.get('run') or ''), body.get('ev'), lt)
    rev = 1 + sum(1 for r in read_v2(path_v2) if (r.get('run'), r.get('ev'), r.get('label_type')) == key)
    return dict(t=now or time.time(), schema='e5_label/v2', run=key[0], ev=key[1], label_type=lt, value=value,
                revision=rev, legacy_ui=note, evidence=ev_, event_state=(event or {}).get('state'),
                capture_id=(event or {}).get('capture_id'), reviewer=str(body.get('reviewer') or '')[:40], eval_only=True)


def read_v2(path):
    out = []
    try:
        for l in open(path, encoding='utf-8'):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def latest(path_v2):
    """(run, ev, label_type) 마다 마지막 revision."""
    d = {}
    for r in read_v2(path_v2):
        d[(r.get('run'), r.get('ev'), r.get('label_type'))] = r
    return d


def legacy_excluded_count(path_legacy):
    return len(read_v2(path_legacy))
