# -*- coding: utf-8 -*-
"""experiments.py — 실시간 실험 기록 모으기·통계(논문용).

실험 하나 = logs/live_<월일>/experiments/exp_<시작시각>_<판정층>F/ (웹 [멈춤] 뒤 결과가 나오면 live_viewer 가 자동 보관)
    experiment.json     시작·끝·판정·순위·twoway·정답 층·메모·걸은 거리·조건(run_config)
    data/ out/          그 실험 정합 창의 표본과 결과(plot_live.py 로 그림을 다시 뽑을 수 있다)

    python experiments.py [logs 폴더]           # 목록 CSV(logs/experiments_index.csv)를 다시 만들고 조건별 통계를 찍는다
    python experiments.py set <실험폴더> <정답층> [메모]   # 정답 층을 나중에 넣기(웹 화면에서 골라도 된다)
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / 'logs'

FIELDS = ['id', 'date', 'start', 'end', 'duration_s', 'walked_m', 'building', 'truth', 'verdict', 'true_rank', 'top3',
          'ranking', 'twoway_verdict', 'twoway_truth', 'twoway_gap_1_2', 'pass_sec', 'samples',
          'fastlio_impl', 'fastlio_env', 'imu_shift_s', 'fastlio_config', 'fastlio_extrinsic', 'delta_deg', 'mirror', 'sample_dt', 'clean', 'clean_stride', 'window_s', 'dt_s',
          'scene_source', 'note', 'path']


def extrinsic_label(R):
    if not R:
        return ''
    if [round(v, 3) for v in R] == [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]:
        return 'go2_ws diag(1,-1,-1)'
    if abs(R[0] + 0.747) < 0.01 and abs(R[8] + 0.958) < 0.01:
        return 'esy 로스백 정본'
    return 'R=' + ','.join(f'{v:.3f}' for v in R)


def row_of(exp_dir):
    e = json.loads((exp_dir / 'experiment.json').read_text())
    rc = e.get('run_config') or {}
    tw = {int(k): v for k, v in (e.get('twoway') or {}).items()}
    rk = e.get('ranking') or []
    truth = e.get('truth')
    tr = rk.index(truth) + 1 if truth in rk else ''
    import time
    srt = sorted(tw.values(), reverse=True)
    loc = rc.get('localizer') or {}
    return dict(
        id=exp_dir.name, date=time.strftime('%Y-%m-%d', time.localtime(e['start'])),
        start=time.strftime('%H:%M:%S', time.localtime(e['start'])), end=time.strftime('%H:%M:%S', time.localtime(e['end'])),
        duration_s=round(e['end'] - e['start'], 1), walked_m=e.get('walked_m', ''), building=rc.get('building', ''),
        truth=truth if truth else '', verdict=e.get('verdict'), true_rank=tr, top3=(tr != '' and tr <= 3) if tr != '' else '',
        ranking=' '.join(map(str, rk)), twoway_verdict=tw.get(e.get('verdict'), ''), twoway_truth=tw.get(truth, '') if truth else '',
        twoway_gap_1_2=round(srt[0] - srt[1], 3) if len(srt) > 1 else '', pass_sec=e.get('pass_sec', ''), samples=e.get('samples', ''),
        fastlio_impl=rc.get('fastlio_impl', 'go2ws(기록 없음)'),
        fastlio_env=' '.join(f'{k[8:]}={v}' for k, v in sorted((rc.get('fastlio_env') or {}).items())),   # FASTLIO_ 떼고
        imu_shift_s=((rc.get('imu_ref_calib') or {}).get('shift_s', '')),
        fastlio_config=Path(rc.get('fastlio_config', '')).name, fastlio_extrinsic=extrinsic_label(rc.get('fastlio_extrinsic_R')),
        delta_deg=rc.get('delta_deg', ''), mirror=rc.get('mirror', ''), sample_dt=rc.get('sample_dt', ''), clean=rc.get('clean', ''),
        clean_stride=rc.get('clean_stride', ''), window_s=loc.get('window_s', e.get('window_s', '')), dt_s=loc.get('dt_s', ''),
        scene_source=rc.get('scene_source', ''), note=e.get('note', ''), path=str(exp_dir))


def collect(logs=LOGS):
    rows = []
    for p in sorted(Path(logs).glob('live_*/experiments/exp_*/experiment.json')):
        try:
            rows.append(row_of(p.parent))
        except Exception as ex:
            print(f'건너뜀 {p}: {type(ex).__name__}: {ex}', file=sys.stderr)
    rows.sort(key=lambda r: (r['date'], r['start']))
    return rows


def write_index(logs=LOGS):
    rows = collect(logs)
    out = Path(logs) / 'experiments_index.csv'
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:          # 엑셀에서 한글이 깨지지 않게 BOM
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out, rows


def summary(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        key = (r['building'], r['fastlio_impl'], r['fastlio_env'], r['fastlio_extrinsic'], f"clean={r['clean']}", f"dt={r['sample_dt']}")
        g[key].append(r)
    print(f'실험 {len(rows)}개 (정답 층 있는 것 {sum(1 for r in rows if r["truth"] != "")}개)')
    for key, rs in g.items():
        k = [r for r in rs if r['true_rank'] != '']
        t1 = sum(1 for r in k if r['true_rank'] == 1); t3 = sum(1 for r in k if r['true_rank'] <= 3)
        print(f'  {" · ".join(map(str, key))}: 실험 {len(rs)} · 정답 있는 {len(k)} · 1위 {t1}/{len(k)} · 상위3 {t3}/{len(k)}'
              + (f' · 정답 순위 {[r["true_rank"] for r in k]}' if k else ''))


def set_truth(exp_dir, truth, note=None):
    """나중에 정답 층·메모를 넣는다: python experiments.py set <실험폴더> <정답층> [메모]"""
    p = Path(exp_dir) / 'experiment.json'
    e = json.loads(p.read_text())
    e['truth'] = int(truth) if str(truth).isdigit() else None
    if note is not None:
        e['note'] = note
    p.write_text(json.dumps(e, ensure_ascii=False, indent=1))
    return e


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'set':
        e = set_truth(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        print(f"{sys.argv[2]}: 정답 {e['truth']}층 · 판정 {e.get('verdict')}층 · 메모 {e.get('note', '')}")
        sys.exit(0)
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else LOGS
    out, rows = write_index(logs)
    print(f'목록 {out}')
    summary(rows)
