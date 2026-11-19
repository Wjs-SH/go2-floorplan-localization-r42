# -*- coding: utf-8 -*-
"""plot_live.py — 실시간 노드(또는 replay_samples) 결과 폴더로 그림 4장을 뽑는다(노트북에서).

    python plot_live.py <실행폴더: data/ 와 out/ 가 있는 곳> <출력폴더> --building kci

  L1_층판별_순위.png   바퀴마다 층별 순위(진할수록 위) · 바퀴 판정(테두리) · 다수결
  L2_지도위치.png      다수결 층 안내도 위 경로(가장 최근 정합으로 옮김) · 지금 위치·방향 · 마지막 표본 구조점
  L3_상위3층.png       마지막 바퀴 상위 3개 층에 같은 60초 창을 맞춘 모습(왜 그 층이 이겼나)
  L4_처리상태.png      구조점 표본 처리 시간 · 층 판별 바퀴 시간
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'tools'))
from plot_preflight import style, BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, SURF  # noqa: E402
from pose_tools import map_pose_px  # noqa: E402

RAMP = ['#0d366b', '#184f95', '#256abf', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6']   # 1위→9위 (파랑 한 색 진→연)


def _A(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [-s, -c]])


def _R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def load(run):
    run = Path(run)
    rows = list(csv.DictReader(open(run / 'data' / 'odom_link.csv')))
    od = dict(sid=[r['sample_id'] for r in rows], t=np.array([float(r['t_sec']) for r in rows]),
              x=np.array([float(r['fastlio_x']) for r in rows]), y=np.array([float(r['fastlio_y']) for r in rows]),
              yaw=np.array([float(r['fastlio_yaw_rad']) for r in rows]))
    hist = [json.loads(l) for l in open(run / 'out' / 'floor_history.jsonl')] if (run / 'out' / 'floor_history.jsonl').exists() else []
    fit = json.loads((run / 'out' / 'pose_fit.json').read_text()) if (run / 'out' / 'pose_fit.json').exists() else None
    st = [json.loads(l) for l in open(run / 'out' / 'status.jsonl')] if (run / 'out' / 'status.jsonl').exists() else []
    return od, hist, fit, st


def scan_px(run, od, fit, i):
    """표본 i 의 구조점(role != 4)을 정합 fit 으로 지도 px 에 — 정합 코드와 같은 식(거울상·β 포함)."""
    p = Path(run) / 'data' / 'samples' / od['sid'][i] / 'data_short.npz'
    if not p.exists():
        return np.zeros((0, 2))
    with np.load(p) as d:
        pts = d['a1_3_1_all_candidate_xy'].astype(float)[d['a1_3_1_all_candidate_role'] != 4]
    w = np.array([od['x'][i], od['y'][i]]) - np.asarray(fit['anchor']) + (pts * [1, fit['mirror']]) @ _R(od['yaw'][i] + fit['beta']).T
    return np.asarray(fit['pos']) + fit['ppm'] * (w @ _A(np.deg2rad(fit['phi'])).T)


def window_px(run, od, fit, window, per=400):
    """정합 창(window 초) 안 모든 표본의 구조점을 같은 정합으로 옮겨 쌓는다 = 공간 누적."""
    sel = [i for i in range(len(od['t'])) if fit['t'] - window <= od['t'][i] <= fit['t']]
    out = []
    for i in sel:
        q = scan_px(run, od, fit, i)
        if len(q) > per:
            q = q[np.linspace(0, len(q) - 1, per).astype(int)]
        out.append(q)
    return np.vstack(out) if out else np.zeros((0, 2)), len(sel)


def track_px(od, fit, sel):
    return np.array([map_pose_px(fit, od['x'][i], od['y'][i], od['yaw'][i])[:2] for i in sel])


def map_img(maps_root, m):
    from PIL import Image
    return np.asarray(Image.open(Path(maps_root) / f'{m}F' / 'maps' / 'base_map.png').convert('L'))


def show_map(ax, img, pts_list, pad=180):
    ax.imshow(img, cmap='gray', vmin=-120, vmax=255, interpolation='nearest')   # 지도는 연하게(데이터가 앞에 보이게)
    allp = np.vstack([p for p in pts_list if len(p)]) if any(len(p) for p in pts_list) else np.array([[img.shape[1] / 2, img.shape[0] / 2]])
    x0, y0 = allp.min(0) - pad; x1, y1 = allp.max(0) + pad
    ax.set_xlim(max(0, x0), min(img.shape[1], x1)); ax.set_ylim(min(img.shape[0], y1), max(0, y0))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--building', default='kci')
    ap.add_argument('--maps', default=None, help='층 지도 뿌리(기본 floor_id 설정)')
    ap.add_argument('--window', type=float, default=60.0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import yaml
    style(plt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(HERE.parent / 'floor_id' / 'config' / 'buildings.yaml'))
    b = (cfg.get('buildings') or cfg)[a.building]
    maps_root = Path(a.maps) if a.maps else HERE.parent / 'floor_id' / b['maps_root']
    n_maps = int(b['n_maps'])
    od, hist, fit, st = load(a.run)
    T0 = od['t'][0]
    made = []

    # ---- L1 층별 순위 ----
    if hist:
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(hist) + 2.5), 6))
        for k, h in enumerate(hist):
            for r, m in enumerate(h['ranking']):
                ax.add_patch(Rectangle((k - 0.45, m - 0.45), 0.9, 0.9, color=RAMP[min(r, len(RAMP) - 1)], lw=0))
                ax.text(k, m, f'{r + 1}', ha='center', va='center', fontsize=9, color='white' if r < 4 else INK)
            v = h['verdict']
            ax.add_patch(Rectangle((k - 0.45, v - 0.45), 0.9, 0.9, fill=False, ec=ORANGE, lw=2.5))
            ax.text(k, n_maps + 0.9, f"{h.get('vote', '-')}층", ha='center', va='center', color=INK, fontweight='bold')
        ax.text(-0.75, n_maps + 0.9, '다수결', ha='right', va='center', color=INK2)
        ax.set_xlim(-0.6, len(hist) - 0.4); ax.set_ylim(0.4, n_maps + 1.4)
        ax.set_yticks(range(1, n_maps + 1)); ax.set_yticklabels([f'{m}층' for m in range(1, n_maps + 1)])
        ax.set_xticks(range(len(hist)))
        ax.set_xticklabels([f"{(h['t_now'] - T0) / 60:.1f}분" for h in hist])
        ax.set_xlabel('바퀴(창 끝 시각, 시작부터)'); ax.grid(False)
        ax.set_title('층 판별\n칸 숫자 = 그 바퀴 순위(진할수록 위) · 주황 테두리 = 바퀴 판정')
        fig.tight_layout(); fig.savefig(out / 'L1_층판별_순위.png'); plt.close(fig); made.append('L1')

    # ---- L2 다수결 층 지도 위 경로 ----
    if fit:
        m = int(fit['map'])
        img = map_img(maps_root, m)
        allp = track_px(od, fit, range(len(od['t'])))
        win = [i for i in range(len(od['t'])) if od['t'][i] >= fit['t'] - a.window and od['t'][i] <= fit['t']]
        wp = track_px(od, fit, win)
        sp, nwin = window_px(a.run, od, fit, a.window)
        fig, ax = plt.subplots(figsize=(10, 8))
        show_map(ax, img, [allp])
        if len(sp):
            ax.plot(sp[:, 0], sp[:, 1], '.', ms=1.5, color=AQUA, alpha=0.6, label=f'{a.window:.0f}초 누적 구조점(표본 {nwin}개)')
        ax.plot(allp[:, 0], allp[:, 1], color=BLUE, lw=1.5, alpha=0.55, label='전체 경로')
        if len(wp):
            ax.plot(wp[:, 0], wp[:, 1], color=BLUE, lw=3, label=f'정합 창 {a.window:.0f}초')
        px, py, hd = map_pose_px(fit, od['x'][-1], od['y'][-1], od['yaw'][-1])
        ax.annotate('', xy=(px + 60 * np.cos(hd), py - 60 * np.sin(hd)), xytext=(px, py),
                    arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=2.5, mutation_scale=18))
        ax.plot(px, py, 'o', ms=10, color=ORANGE, mec=SURF, mew=2, label='지금 위치')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, frameon=True)
        age = od['t'][-1] - fit['t']
        ax.set_title(f'{m}층 안내도 위 위치 — 정합은 {age:.0f}초 전 창, 그 뒤는 FAST-LIO 로 이어 감')
        fig.tight_layout(); fig.savefig(out / 'L2_지도위치.png'); plt.close(fig); made.append('L2')

    # ---- L3 상위 3개 층 ----
    if hist:
        h = hist[-1]
        top = h['cands'][:3]
        fig, axs = plt.subplots(1, len(top), figsize=(6 * len(top), 6))
        axs = np.atleast_1d(axs)
        for ax, c in zip(axs, top):
            img = map_img(maps_root, c['map'])
            win = [i for i in range(len(od['t'])) if c['t'] - a.window <= od['t'][i] <= c['t']]
            wp = track_px(od, c, win)
            sp, _ = window_px(a.run, od, c, a.window, per=250)
            show_map(ax, img, [wp, sp[:0]], pad=260)
            if len(sp):
                ax.plot(sp[:, 0], sp[:, 1], '.', ms=1.2, color=AQUA, alpha=0.6)
            if len(wp):
                ax.plot(wp[:, 0], wp[:, 1], color=BLUE, lw=3)
                ax.plot(*wp[-1], 'o', ms=9, color=ORANGE, mec=SURF, mew=2)
            r = h['ranking'].index(c['map']) + 1
            ax.set_title(f"{r}위 {c['map']}층 · twoway {c['twoway']:.3f} · 벽비용 {100 * c['wall_cost_m']:.1f}cm", fontsize=12)
        fig.suptitle(f"마지막 바퀴(창 끝 {(h['t_now'] - T0) / 60:.1f}분) 상위 3개 층 — 파랑 = 60초 경로, 청록 = 60초 누적 구조점",
                     color=INK, fontweight='bold')
        fig.tight_layout(); fig.savefig(out / 'L3_상위3층.png'); plt.close(fig); made.append('L3')

    # ---- L4 처리 상태 ----
    if st or hist:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        if st:
            tt = np.array([s['wall'] for s in st if s.get('last_sample_sec')]); ss = [s['last_sample_sec'] for s in st if s.get('last_sample_sec')]
            if len(tt):
                axs[0].plot((tt - tt[0]) / 60, ss, color=BLUE, lw=2)
            axs[0].axhline(1.5, color=INK2, ls='--', lw=1.2)
            axs[0].annotate('예산 1.5초', (0, 1.5), xytext=(4, 4), textcoords='offset points', color=INK2)
            axs[0].set_xlabel('시간 (분)'); axs[0].set_ylabel('초'); axs[0].set_title('구조점 표본 하나 처리 시간')
            axs[0].set_ylim(bottom=0)
        else:
            axs[0].set_axis_off()
        if hist:
            axs[1].bar(range(len(hist)), [h['pass_sec'] for h in hist], color=BLUE, width=0.6)
            axs[1].set_xticks(range(len(hist))); axs[1].set_xticklabels([str(h['pass_index']) for h in hist])
            axs[1].set_xlabel('바퀴'); axs[1].set_ylabel('초'); axs[1].set_title('층 판별 한 바퀴 시간')
        fig.tight_layout(); fig.savefig(out / 'L4_처리상태.png'); plt.close(fig); made.append('L4')
    print(f'그림 {made} → {out}')


if __name__ == '__main__':
    main()
