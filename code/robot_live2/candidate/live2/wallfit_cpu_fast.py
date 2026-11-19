"""wallfit_cpu_fast.py — LIVE2: drop-in for the student's `wallfit_gpu` hook in rt_localizer.localize_buf, CPU only.

Why: on the Orin NX the CPU wall-fit loop of localize_buf is ~90 % of one fit_window_v2 (S6 6F, 7 scales: 304 s) and the
GPU is reserved for SigLIP without a separate approval. This module computes the SAME expression as the CPU branch of
rt_localizer.localize_buf (and wallfit_gpu.wallfit_cpu), bit for bit:

    ex/ey = Pf + R(hd)·pts·ppm  ->  round  ->  int  ->  clip to the image  ->  DFpx lookup  ->  clip [0, 10]
    -> mean of the lowest 70 % (ascending order)  ->  -(v/5)^2

Only the evaluation order and the storage of the looked-up value change, never a value:
  * DFpx is a Euclidean distance transform, i.e. every pixel is sqrt(integer). After the [0, 10] clip only
    sqrt(0..100) and 10 can occur, so the map is stored once as uint8 squared distance (0..100, 101 = "> 10") and
    decoded with the table sqrt(k). _prep() verifies sqrt(d2) == DFpx bit for bit on every pixel <= 10; if that ever
    fails the original float path is used (EXACT_TABLE_UNAVAILABLE) — nothing is approximated.
  * the lookups run point-major (for one scan point, all candidates in grid order) for cache locality, then are
    transposed back; IEEE addition is commutative so ex/ey are the same numbers.
  * rows are sorted as uint8 with numpy's radix sort; sqrt is monotonic and ties are equal values, so the decoded
    first-k row is exactly the first k elements of the original ascending float sort and the per-row mean runs over the
    identical sequence.
Bit equality is asserted by tests/test_e2_core.py on real S6 frames. The student file is not modified: e2_worker
installs this module as sys.modules['wallfit_gpu'] before importing rt_localizer.
"""
import numpy as np

BLOCK = 2048
LUT = np.sqrt(np.arange(102.0))
LUT[101] = 10.0
STATS = dict(exact_table=None, fallback_calls=0, calls=0)


def available():
    return True


def _prep(L):
    c = getattr(L, '_live2_wf', None)
    if c is not None and c['src'] is L.DFpx:
        return c
    DF = np.asarray(L.DFpx, float)
    d2 = np.rint(DF * DF)
    near = DF <= 10.0
    exact = bool((DF >= 0).all() and np.array_equal(np.sqrt(d2[near]), DF[near]))
    q = np.minimum(d2, 101).astype(np.uint8).ravel() if exact else None
    L._live2_wf = c = dict(src=L.DFpx, q=q, exact=exact)
    STATS['exact_table'] = exact
    return c


def _orig(L, pts, hd, ppm):
    c, s = np.cos(hd), np.sin(hd)
    ex = L.Pf[:, 0][:, None] + ((pts[:, 0] * c - pts[:, 1] * s) * ppm)[None]
    ey = L.Pf[:, 1][:, None] + (-(pts[:, 0] * s + pts[:, 1] * c) * ppm)[None]
    ci = np.clip(np.round(ex).astype(int), 0, L.W - 1)
    ri = np.clip(np.round(ey).astype(int), 0, L.H - 1)
    dd = np.clip(L.DFpx[ri, ci], 0, 10)
    return -(np.sort(dd, 1)[:, :int(ex.shape[1] * 0.7)].mean(1) / 5) ** 2


def wallfit(L, pts, hd, ppm):
    STATS['calls'] += 1
    pts = np.asarray(pts, float)
    P = _prep(L)
    if not P['exact'] or not np.isfinite(pts).all():
        STATS['fallback_calls'] += 1
        return _orig(L, pts, hd, ppm)
    c, s = np.cos(hd), np.sin(hd)
    dx = (pts[:, 0] * c - pts[:, 1] * s) * ppm
    dy = -(pts[:, 0] * s + pts[:, 1] * c) * ppm
    Pf, W, H, q = L.Pf, L.W, L.H, P['q']
    n, m = len(Pf), len(pts)
    k = int(m * 0.7)
    out = np.empty(n)
    for a in range(0, n, BLOCK):
        b = min(n, a + BLOCK)
        ci = np.round(Pf[a:b, 0][None, :] + dx[:, None]).astype(int)       # (m, block): point-major
        ri = np.round(Pf[a:b, 1][None, :] + dy[:, None]).astype(int)
        np.maximum(ci, 0, out=ci)
        np.minimum(ci, W - 1, out=ci)
        np.maximum(ri, 0, out=ri)
        np.minimum(ri, H - 1, out=ri)
        ri *= W
        ri += ci
        u = np.ascontiguousarray(q.take(ri).T)                              # (block, m) uint8 squared distance
        u.sort(axis=1, kind='stable')
        out[a:b] = LUT.take(u[:, :k]).mean(1)
    return -(out / 5) ** 2
