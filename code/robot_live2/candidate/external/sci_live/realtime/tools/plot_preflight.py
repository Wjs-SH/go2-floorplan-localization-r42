# -*- coding: utf-8 -*-
"""plot_preflight.py — live_preflight.py motion 녹화(FAST-LIO 자세)로 규약 판정 그림 4장을 뽑는다(노트북에서).

    python plot_preflight.py flio_motion.npz <출력폴더> [--delta 45.3]

  01_궤적.png      위에서 본 경로 + 로봇 앞 방향 화살표(base_link x). δ 가 맞으면 직진 구간에서 화살표가 경로와 나란하다
  02_방향각.png    시간에 따른 yaw. 거울상 켬/끔 두 줄 + 회전 구간 표시. 좌회전(반시계)에서 올라가야 맞다
  03_이동방향.png  직진 창의 이동 방향(몸체 기준, δ=0) 분포. 봉우리 = 실제 δ
  04_속도.png      속도와 직진 판정 문턱(0.25 m/s)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'a131_live'))

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'          # 기준 팔레트 1·2·3번(순서 고정)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e6e5e0', '#fcfcfb'


def style(plt):
    plt.rcParams.update({
        'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'savefig.dpi': 130,
        'legend.frameon': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('out')
    ap.add_argument('--delta', type=float, default=45.3)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as Rot
    from a131_live import fastlio_to_base, estimate_delta_deg
    style(plt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with np.load(a.rec) as f:
        t, p, q = f['t'], f['p'], f['q']
    t = t - t[0]

    pb0, qb0 = fastlio_to_base(p, q, delta_deg=0.0, mirror=True)
    yaw_on = np.unwrap(Rot.from_quat(qb0).as_euler('zyx')[:, 0])
    pbf, qbf = fastlio_to_base(p, q, delta_deg=0.0, mirror=False)
    yaw_off = np.unwrap(Rot.from_quat(qbf).as_euler('zyx')[:, 0])
    d_est, nwin, frac = estimate_delta_deg(t, pb0[:, 0], pb0[:, 1], yaw_on)

    # ---- 01 궤적 + 앞 방향(기본 δ, 거울상 켬) ----
    pb, qb = fastlio_to_base(p, q, delta_deg=a.delta, mirror=True)
    yaw_b = Rot.from_quat(qb).as_euler('zyx')[:, 0]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot(pb[:, 0], pb[:, 1], color=BLUE, lw=2, label='경로')
    step = max(1, int(round(1.0 / max(np.median(np.diff(t)), 1e-3))))
    span = max(np.ptp(pb[:, 0]), np.ptp(pb[:, 1]), 1.0)
    L = 0.1 * span
    ax.quiver(pb[::step, 0], pb[::step, 1], L * np.cos(yaw_b[::step]), L * np.sin(yaw_b[::step]), color=ORANGE,
              angles='xy', scale_units='xy', scale=1, width=0.004, label=f'로봇 앞 방향(δ {a.delta}°)')
    ax.plot(*pb[0, :2], 'o', ms=10, color=INK, mfc=SURF, mew=2); ax.annotate('시작', pb[0, :2], xytext=(8, 8), textcoords='offset points', color=INK2)
    ax.plot(*pb[-1, :2], 's', ms=9, color=INK); ax.annotate('끝', pb[-1, :2], xytext=(8, -14), textcoords='offset points', color=INK2)
    ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('① 위에서 본 경로 — 직진 구간에서 화살표가 경로와 나란하면 δ 가 맞다')
    ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(out / '01_궤적.png'); plt.close(fig)

    # ---- 02 방향각 ----
    rate = np.gradient(yaw_on, t)
    on = np.abs(rate) > np.radians(20)
    segs, j = [], 0
    while j < len(t):
        if on[j]:
            k = j
            while k + 1 < len(t) and on[k + 1]:
                k += 1
            if t[k] - t[j] >= 1.0:
                segs.append((j, k))
            j = k + 1
        else:
            j += 1
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for j, k in segs:
        ax.axvspan(t[j], t[k], color=GRID, lw=0)
        dy = np.degrees(yaw_on[k] - yaw_on[j])
        ax.annotate(f'{dy:+.0f}°', ((t[j] + t[k]) / 2, np.degrees((yaw_on[j] + yaw_on[k]) / 2)), ha='center', va='center',
                    color=INK, fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec=MUTED, lw=0.8))
    ax.plot(t, np.degrees(yaw_on - yaw_on[0]) + np.degrees(yaw_on[0]), color=BLUE, lw=2, label='거울상 켬(기본)')
    ax.plot(t, np.degrees(yaw_off), color=ORANGE, lw=2, ls='--', label='거울상 끔')
    ax.margins(y=0.12); ax.set_xlabel('시간 (초)'); ax.set_ylabel('yaw (도)')
    ax.set_title('② 방향각 — 회전 구간(회색)에서 좌회전이면 파란 선이 올라가야 맞다')
    ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(out / '02_방향각.png'); plt.close(fig)

    # ---- 03 이동 방향 분포(직진 창) ----
    tg = np.arange(t[0], t[-1], 0.1)
    xg, yg, wg = np.interp(tg, t, pb0[:, 0]), np.interp(tg, t, pb0[:, 1]), np.interp(tg, t, yaw_on)
    k = 5
    dx, dy_, dw = xg[k:] - xg[:-k], yg[k:] - yg[:-k], wg[k:] - wg[:-k]
    st = (np.hypot(dx, dy_) / (k * 0.1) > 0.25) & (np.abs(dw) < np.radians(3))
    c, s = np.cos(wg[:-k]), np.sin(wg[:-k])
    ang = np.degrees(np.arctan2(-s * dx + c * dy_, c * dx + s * dy_))[st]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if len(ang):
        ax.hist(ang, bins=np.arange(-180, 181, 5), color=BLUE, edgecolor=SURF, linewidth=1)
    ax.axvline(a.delta, color=INK2, lw=1.5, ls='--')
    ax.annotate(f'기본값 {a.delta}°', (a.delta, ax.get_ylim()[1] * 0.92), xytext=(6, 0), textcoords='offset points', color=INK2)
    if np.isfinite(d_est):
        ax.axvline(d_est, color=ORANGE, lw=2)
        ax.annotate(f'추정 δ {d_est:.1f}°', (d_est, ax.get_ylim()[1] * 0.78), xytext=(6, 0), textcoords='offset points',
                    color=INK, fontweight='bold')
    ax.set_xlim(-180, 180); ax.set_xticks(range(-180, 181, 45))
    ax.set_xlabel('몸체 기준 이동 방향 (도, 거울상 켬·δ=0)'); ax.set_ylabel('직진 창 수 (0.5초 창)')
    ax.set_title(f'③ 직진할 때 이동 방향 — 봉우리가 실제 δ (직진 창 {nwin}개, ±20° 안 {100 * frac:.0f}%)')
    fig.tight_layout(); fig.savefig(out / '03_이동방향.png'); plt.close(fig)

    # ---- 04 속도 ----
    v = np.hypot(np.gradient(pb0[:, 0], t), np.gradient(pb0[:, 1], t))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t, v, color=BLUE, lw=2)
    ax.axhline(0.25, color=INK2, lw=1.2, ls='--')
    ax.annotate('직진 판정 문턱 0.25 m/s', (t[-1], 0.25), xytext=(-4, 5), textcoords='offset points', ha='right', color=INK2)
    for j, k2 in segs:
        ax.axvspan(t[j], t[k2], color=GRID, lw=0)
    ax.set_xlabel('시간 (초)'); ax.set_ylabel('속도 (m/s)'); ax.set_ylim(bottom=0)
    ax.set_title('④ 이동 속도 (회색 = 회전 구간)')
    fig.tight_layout(); fig.savefig(out / '04_속도.png'); plt.close(fig)

    turns = [float(np.degrees(yaw_on[k2] - yaw_on[j])) for j, k2 in segs]
    print(f'녹화 {t[-1]:.0f}초 · 이동 {np.sum(np.linalg.norm(np.diff(pb0[:, :2], axis=0), axis=1)):.1f} m · '
          f'추정 δ {d_est:.1f}° (직진 창 {nwin}) · 회전 구간 {[round(x) for x in turns]} · 그림 {out}')


if __name__ == '__main__':
    main()
