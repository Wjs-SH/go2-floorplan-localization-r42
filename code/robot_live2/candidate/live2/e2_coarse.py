"""e2_coarse.py — LIVE2 E2 COARSE stage: a fast, APPROXIMATE all-floor pre-ranking that runs before the exact pass.

Why: one exact floor fit (student fit_window_v2, CPU) costs 65-110 s on the Orin NX, ~85 % of it in Z.localize — the
wall-fit belief over the whole grid, evaluated 7 scales x 21 frames x 4 headings x 400 points (measured, tools/e2_bench.py).
A building pass is therefore minutes, not the ~60 s the operator needs for a first Top-K.

What changes (and only this): the SAME student functions (fit_window_v2_topk -> Z.localize -> localize_buf, least_squares,
T4, twoway) run on a thinner window:
  * frames every `frame_dt` s instead of 3 s (same 60 s window, same newest frame),
  * `n_pts` points per frame instead of 400 (seeded like the student's own subsample),
  * a subset of the student scale list, `topk` grid seeds per heading instead of 3.
Nothing is precomputed from the run; the map side is the student's cached map prep (pickle), so the "index" is the
offline map cache that already exists.

The result is labelled COARSE everywhere (stage="coarse", verified=False). It is a ranking to show early and to order the
exact pass (best coarse floors first); it never replaces the exact result and never claims the search is complete.
"""
import numpy as np

from e2_core import fit_window_v2_topk as _fit_exact

COARSE_DEFAULT = dict(frame_dt=6.0, n_pts=100, scale_idx=(0, 2, 4, 6), topk=1)


def _thin(Z, n_pts, frame_dt):
    orig = Z.window_scans

    def window_scans(t_now, W=30., dt=3.):
        buf = orig(t_now, W, max(dt, frame_dt))
        for fr in buf:
            p = np.asarray(fr['pts'])
            if len(p) > n_pts:
                rs = np.random.RandomState(int(fr['sample_seed_t']) % 97)
                fr['pts'] = p[rs.choice(len(p), n_pts, replace=False)]
        return buf
    return window_scans


def coarse_floor(Z, t_now, scales, floor, W=60., frame_dt=6.0, n_pts=100, scale_idx=(0, 2, 4, 6), topk=1):
    """Coarse candidates for one floor: same record fields as the exact pass + stage='coarse'."""
    sc = [scales[i] for i in scale_idx if i < len(scales)]
    Z.window_scans = _thin(Z, n_pts, frame_dt)          # instance attribute: the class method stays untouched
    try:
        return _fit_exact(Z, t_now, sc, floor=floor, W=W, dt=frame_dt, topk=topk)
    finally:
        del Z.window_scans
