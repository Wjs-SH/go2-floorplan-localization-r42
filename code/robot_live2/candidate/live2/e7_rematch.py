"""e7_rematch.py — LIVE2 E7 LiDAR re-registration engine (offline-capable, low priority, numpy/scipy only, no ROS/device).

Independent of E6: input is any set of candidates (each an E2/E3 hypothesis transform = floor + pos/anchor/phi/ppm/beta/
mirror + the E2 light map of that floor) plus the REAL registered LiDAR samples in the LIVE store (odom_link frame).
input_kind must say where the candidates came from: E6_SEMANTIC | E6_OPERATOR_ASSISTED | E2E3_DIAGNOSTIC | STRUCTURE_ONLY
(e6e7_link PATH A: E4 CONFIDENT_GEOMETRY leader + rivals, no semantic reading; 2026-10-30: was missing here, so PATH A always failed).
A GT pose is never an input (there is no parameter for it).

Per candidate:
  accumulated cloud  every store sample in the last `window_s` (default 90 s), up to `max_frames`, projected into the floor
                     plan with the candidate transform (same projection as E3 eval_one / student metric_fit_v2)
  registration       bounded grid (+-1 m, +-4 deg, +-2 % scale) then scipy least_squares (soft_l1) on the student wall
                     distance field F, all frames rigidly together (the transform is re-anchored at the newest frame)
  outputs            alignment_score = trimmed mean (70 %) wall distance [m]; residual = median [m];
                     inlier_ratio = fraction < 0.15 m; convergence (optimizer success, not on a bound);
                     local_ambiguity = another grid minimum > 1 m / 5 deg away within 5 % of the best cost.
Decision (relative, because the FAST-LIO proxy input carries clutter: absolute misfit of the CORRECT pose is ~0.3 m trimmed):
  GEOMETRY_CONSISTENT_SEMANTIC_PENDING  best candidate converged, locally unique, and better than the runner-up on another
                                        floor/place by >= GAP_M; a final pose additionally needs an independent semantic
                                        (E6) match -> never GLOBAL_CONFIRMED from geometry alone
  GLOBAL_CONFIRMED                      only with semantic_ok=True (E6 match on the same floor/place) AND the above
  MORE_OBSERVATION_REQUIRED             no clear separation / not converged / locally ambiguous
  GLOBAL_UNKNOWN                        no candidate could be evaluated
"""
import math
import time

import numpy as np

TRIM, INLIER_M, GAP_M = 0.7, 0.15, 0.05
GRID_XY_M, GRID_DEG, GRID_SCALE = (-1.0, -0.5, 0.0, 0.5, 1.0), (-4.0, -2.0, 0.0, 2.0, 4.0), (-0.02, 0.0, 0.02)
LOCAL_AMBIG_REL, LOCAL_AMBIG_M, LOCAL_AMBIG_DEG = 0.05, 1.0, 5.0


def _A(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [-s, -c]])


LIN_R_M, LIN_MIN_N, LIN_MIN = 0.30, 5, 0.85


def structural_filter(p, r=LIN_R_M, min_n=LIN_MIN_N, lin_min=LIN_MIN):
    """Keep wall-like points: the local 2D neighbourhood (radius r) is line-shaped (PCA linearity
    (l1 - l2) / l1 >= lin_min) and has >= min_n points. The store already keeps only the wall-height band with >= 0.15 m
    vertical span, so what this removes is vertical NON-wall structure (boards, furniture, people, cluttered corners)."""
    from scipy.spatial import cKDTree
    p = np.asarray(p, float)
    if len(p) < min_n:
        return p, 0.0
    tree = cKDTree(p)
    nb = tree.query_ball_point(p, r)
    # vectorised (2026-10-22: the per-point eigvalsh loop was 6.9 s of a 15.3 s E7 call): per-neighbourhood scatter matrix
    # [[sxx, sxy], [sxy, syy]] from sums, eigenvalues in closed form. Same test as before: l1 > 0 and (l1 - l2) / l1 >= lin_min
    cnt = np.fromiter((len(i) for i in nb), int, len(nb))
    keep = np.zeros(len(p), bool)
    ok = cnt >= min_n
    if ok.any():
        idx = np.concatenate([nb[i] for i in np.flatnonzero(ok)]).astype(int)
        starts = np.r_[0, np.cumsum(cnt[ok])[:-1]]
        x, y, n = p[idx, 0], p[idx, 1], cnt[ok].astype(float)
        sx, sy = np.add.reduceat(x, starts), np.add.reduceat(y, starts)
        sxx = np.add.reduceat(x * x, starts) - sx * sx / n
        syy = np.add.reduceat(y * y, starts) - sy * sy / n
        sxy = np.add.reduceat(x * y, starts) - sx * sy / n
        half_tr, disc = (sxx + syy) / 2.0, np.sqrt(np.maximum(((sxx - syy) / 2.0) ** 2 + sxy ** 2, 0.0))
        l1, l2 = half_tr + disc, half_tr - disc
        with np.errstate(divide="ignore", invalid="ignore"):
            keep[ok] = (l1 > 0) & ((l1 - l2) / np.where(l1 > 0, l1, 1.0) >= lin_min)
    return p[keep], float(keep.mean())


def structural_filter_ref(p, r=LIN_R_M, min_n=LIN_MIN_N, lin_min=LIN_MIN):
    """Original loop form (kept for the equivalence test)."""
    from scipy.spatial import cKDTree
    p = np.asarray(p, float)
    if len(p) < min_n:
        return p, 0.0
    nb = cKDTree(p).query_ball_point(p, r)
    keep = np.zeros(len(p), bool)
    for i, idx in enumerate(nb):
        if len(idx) < min_n:
            continue
        q = p[idx] - p[idx].mean(0)
        ev = np.linalg.eigvalsh(q.T @ q)
        keep[i] = ev[1] > 0 and (ev[1] - ev[0]) / ev[1] >= lin_min
    return p[keep], float(keep.mean())


_SF_CACHE, _SF_MAX = {}, 400            # (odom session, sample id) -> filtered points; a sample never changes once stored


def _structural_cached(store, sid, p):
    try:
        sess = (store.session() or {}).get("odom_session_id")
    except Exception:                    # noqa: BLE001  stores without session(): no cache
        return structural_filter(p)[0]
    k = (sess, sid)
    if k not in _SF_CACHE:
        if len(_SF_CACHE) >= _SF_MAX:
            _SF_CACHE.pop(next(iter(_SF_CACHE)))
        _SF_CACHE[k] = structural_filter(p)[0]
    return _SF_CACHE[k]


def frames(store, t_end, window_s=90.0, max_frames=60, pts_per_frame=150, structural=False):
    """[(t, x, y, yaw, pts Nx2 body)] — evenly spaced store samples inside the window, newest last.
    structural=True: wall-like points only (structural_filter on the full scan, before subsampling)."""
    store.refresh()
    rows = [r for r in store.rows if t_end - window_s <= r[1] <= t_end]
    if not rows:
        return []
    pick = rows if len(rows) <= max_frames else [rows[int(i)] for i in np.linspace(0, len(rows) - 1, max_frames)]
    out = []
    for sid, t, x, y, yaw in pick:
        p = store.points(sid)
        if p is None or len(p) < 20:
            continue
        p = np.asarray(p, float)
        if structural:
            p = _structural_cached(store, sid, p)
            if len(p) < 20:
                continue
        if len(p) > pts_per_frame:
            p = p[np.linspace(0, len(p) - 1, pts_per_frame).astype(int)]
        out.append((t, x, y, yaw, p))
    return out


def project(rec, fr, d=(0.0, 0.0, 0.0, 1.0)):
    """All frames into floor-plan px with the candidate transform re-anchored at the newest frame and corrected by
    d = (dx_m, dy_m in map px/ppm units, dphi_rad, scale)."""
    tn, xn, yn, yawn, _ = fr[-1]
    phi0 = math.radians(rec['phi'])
    A0 = _A(phi0)
    pn = np.asarray(rec['pos'], float) + rec['ppm'] * (A0 @ (np.array([xn, yn]) - np.asarray(rec['anchor'], float)))
    phi = phi0 + d[2]
    ppm = rec['ppm'] * d[3]
    A = _A(phi)
    base = pn + np.array([d[0], -d[1]]) * rec['ppm']
    pts = []
    for t, x, y, yaw, p in fr:
        hd = phi + yaw + rec['beta']
        c = base + ppm * (A @ (np.array([x, y]) - np.array([xn, yn])))
        pts.append(c + ppm * ((p * [1, rec['mirror']]) @ _A(hd).T))
    return np.vstack(pts), ppm


FS_MARGIN_M, FS_RAYS_PER_FRAME = 0.30, 40
# Diagnostic only, OFF by default (2026-10-20 offline probe on real runs A/B, truth 4F used only for scoring): the correct
# pose still shows 43-82 % "through-wall" rays (grazing rays along corridor walls touch the rasterised wall before the hit),
# truth rank by this metric was worse than by alignment_score -> never used in the decision until an angle-aware crossing
# test is validated. reports/e7_fs_20261020/.
FS_IN_REGISTER = False


def project_rays(rec, fr, d=(0.0, 0.0, 0.0, 1.0)):
    """Like project(), but per frame also the sensor (robot) position in map px -> [(origin_px (2,), pts_px (N,2))]."""
    tn, xn, yn, yawn, _ = fr[-1]
    phi0 = math.radians(rec['phi'])
    A0 = _A(phi0)
    pn = np.asarray(rec['pos'], float) + rec['ppm'] * (A0 @ (np.array([xn, yn]) - np.asarray(rec['anchor'], float)))
    phi, ppm = phi0 + d[2], rec['ppm'] * d[3]
    A = _A(phi)
    base = pn + np.array([d[0], -d[1]]) * rec['ppm']
    out = []
    for t, x, y, yaw, p in fr:
        c = base + ppm * (A @ (np.array([x, y]) - np.array([xn, yn])))
        out.append((c, c + ppm * ((p * [1, rec['mirror']]) @ _A(phi + yaw + rec['beta']).T)))
    return out, ppm


def free_space_violation(light, rec, fr, d=(0.0, 0.0, 0.0, 1.0), margin_m=FS_MARGIN_M, per_frame=FS_RAYS_PER_FRAME):
    """Fraction of LiDAR rays that pass through a map wall (student through-wall mask wall_tr; outside the map counts)
    before reaching within margin_m of their hit point. A correct pose sees through doors/corridors, not through walls;
    clutter (people, furniture) only shortens rays, so it cannot raise this number. Independent of the wall-distance fit."""
    rays, ppm = project_rays(rec, fr, d)
    W, H, M = light.W, light.H, light.wall_tr
    n_bad = n_all = 0
    m_px = margin_m * ppm
    for o, pts in rays:
        if len(pts) > per_frame:
            pts = pts[np.linspace(0, len(pts) - 1, per_frame).astype(int)]
        v = pts - o
        L = np.hypot(v[:, 0], v[:, 1])
        keep = L > m_px + 1.0
        if not keep.any():
            continue
        v, L = v[keep], L[keep]
        S = int(np.ceil(L.max()))
        f = np.arange(S)[None, :] / L[:, None]                          # 1 px steps along each ray
        valid = f * L[:, None] < (L - m_px)[:, None]
        xs = np.rint(o[0] + f * v[:, 0:1]).astype(int)
        ys = np.rint(o[1] + f * v[:, 1:2]).astype(int)
        inb = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        hit = np.zeros_like(valid)
        hit[inb] = np.asarray(M)[ys[inb], xs[inb]] > 0
        hit |= ~inb
        n_bad += int((hit & valid).any(1).sum())
        n_all += len(L)
    return round(n_bad / n_all, 4) if n_all else None


# ---------------------------------------------------------------- E7 v2 diagnostics (2026-10-20, offline-verified only)
XW_INCIDENCE_DEG, XW_BEYOND_M, XW_MARGIN_M, XW_RAYS_PER_FRAME = 30.0, 0.30, 0.30, 40
SEG_K = 3
COV_TOL_M, COV_RANGE_M, COV_WALL_STRIDE = 0.20, 5.0, 9
V2_IN_REGISTER = False     # + map coverage in register() (diagnostic only; segments are always computed)
# Angle-aware crossing: computed for every candidate (~0.1-0.2 s each) and used ONLY as a conservative veto — the best
# candidate is withheld (never re-ranked, never confirmed) when its crossing exceeds the best other-floor candidate's by
# >= XW_VETO. Real runs A/B (truth 4F used only for scoring): as a ranking term it fixed both alignment misses but lowered
# the truth rank in 3 coarse cases (A +180 s: 1 -> 6), so it is not a ranking term; as a veto it never fired on a
# truth-best case and fired on both wrong-best cases. reports/E7_IMPROVEMENT_20261020.md
XW_IN_REGISTER, XW_VETO = True, 0.10


def _wall_normal(F, x, y, W, H):
    """Unit normal of the nearest wall at free pixel (x, y): gradient of the wall distance field (points away from it)."""
    x0, x1, y0, y1 = max(x - 2, 0), min(x + 2, W - 1), max(y - 2, 0), min(y + 2, H - 1)
    gx, gy = float(F[y, x1]) - float(F[y, x0]), float(F[y1, x]) - float(F[y0, x])
    n = math.hypot(gx, gy)
    return None if n < 1e-6 else (gx / n, gy / n)


def wall_crossing_violation(light, rec, fr, d=(0.0, 0.0, 0.0, 1.0), incidence_deg=XW_INCIDENCE_DEG, beyond_m=XW_BEYOND_M,
                            margin_m=XW_MARGIN_M, per_frame=XW_RAYS_PER_FRAME):
    """Angle-aware version of free_space_violation: a ray counts as 'through wall' only if it ENTERS a wall pixel at an
    incidence above incidence_deg to the wall surface (normal from the distance-field gradient at the last free pixel)
    AND continues >= beyond_m in free, in-map space behind it, all before margin_m of its hit point. Grazing contacts
    along corridor walls (the failure of the first free-space test) and rays leaving the map do not count.
    -> dict(violation fraction, n_rays, outside fraction)."""
    rays, ppm = project_rays(rec, fr, d)
    W, H = light.W, light.H
    M, F = np.asarray(light.wall_tr), light.F
    sin_thr = math.sin(math.radians(incidence_deg))
    m_px, b_px = margin_m * ppm, max(2, int(round(beyond_m * ppm)))
    n_bad = n_all = n_out = 0
    for o, pts in rays:
        if len(pts) > per_frame:
            pts = pts[np.linspace(0, len(pts) - 1, per_frame).astype(int)]
        v = pts - o
        L = np.hypot(v[:, 0], v[:, 1])
        keep = L > m_px + 2.0
        if not keep.any():
            continue
        v, L = v[keep], L[keep]
        S = int(np.ceil(L.max()))
        f = np.arange(S)[None, :] / L[:, None]
        valid = f * L[:, None] < (L - m_px)[:, None]
        xs = np.rint(o[0] + f * v[:, 0:1]).astype(int)
        ys = np.rint(o[1] + f * v[:, 1:2]).astype(int)
        inb = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        wall = np.zeros_like(valid)
        wall[inb] = M[ys[inb], xs[inb]] > 0
        n_all += len(L)
        n_out += int((~inb & valid).any(1).sum())
        for i in np.flatnonzero((wall & valid).any(1)):
            u = v[i] / L[i]
            wi, ok_i, iv = wall[i], inb[i], valid[i]
            j = 1
            while j < S and iv[j]:
                if wi[j] and not wi[j - 1] and ok_i[j - 1]:
                    e = j
                    while e < S and iv[e] and wi[e]:
                        e += 1
                    nrm = _wall_normal(F, xs[i, j - 1], ys[i, j - 1], W, H)
                    steep = nrm is not None and abs(u[0] * nrm[0] + u[1] * nrm[1]) >= sin_thr
                    beyond = e + b_px <= S and iv[e:e + b_px].all() and ok_i[e:e + b_px].all() and not wi[e:e + b_px].any()
                    if steep and beyond:
                        n_bad += 1
                        break
                    j = e
                j += 1
    return dict(violation=round(n_bad / n_all, 4) if n_all else None, n_rays=n_all,
                outside=round(n_out / n_all, 4) if n_all else None)


def segment_costs(light, rec, fr, d, k=SEG_K):
    """Multi-keyframe consistency under ONE joint correction d: frames split into k consecutive time segments, trimmed
    wall-distance cost per segment. A pose that fits only part of the trajectory shows a large spread / worst segment."""
    mp, ppm = project(rec, fr, d)
    e = np.minimum(sample_bilinear(light, mp, float(max(light.H, light.W))) / ppm, 5.0)
    bounds = np.cumsum([0] + [len(f[4]) for f in fr])
    out = []
    for g in np.array_split(np.arange(len(fr)), min(k, len(fr))):
        if len(g) == 0:
            continue
        es = e[bounds[g[0]]:bounds[g[-1] + 1]]
        kk = max(1, int(TRIM * len(es)))
        out.append(round(float(np.mean(np.partition(es, kk - 1)[:kk])), 4))
    return dict(costs=out, spread=round(max(out) - min(out), 4) if out else None, worst=max(out) if out else None)


def _wall_xy(light):
    xy = getattr(light, "_wall_xy_cache", None)
    if xy is None:
        yy, xx = np.nonzero(np.asarray(light.wall_real) > 0)
        xy = np.stack([xx, yy], 1)[::COV_WALL_STRIDE].astype(float)
        try:
            light._wall_xy_cache = xy
        except AttributeError:
            pass
    return xy


def map_coverage(light, rec, fr, d, tol_m=COV_TOL_M, range_m=COV_RANGE_M):
    """point_cov = observed points within tol_m of a map wall; wall_cov = map wall pixels within range_m of the
    trajectory that have an observed point within tol_m (occlusion ignored, so only comparable between candidates)."""
    from scipy.spatial import cKDTree
    rays, ppm = project_rays(rec, fr, d)
    pts = np.vstack([p for _, p in rays])
    org = np.array([o for o, _ in rays])
    e = sample_bilinear(light, pts, float(max(light.H, light.W))) / ppm
    wxy = _wall_xy(light)
    near = cKDTree(org).query(wxy, distance_upper_bound=range_m * ppm)[0] < np.inf if len(wxy) else np.zeros(0, bool)
    wn = wxy[near]
    if len(wn) == 0:
        return dict(point_cov=round(float(np.mean(e < tol_m)), 3), wall_cov=None, n_wall_near=0)
    seen = cKDTree(pts).query(wn, distance_upper_bound=tol_m * ppm)[0] < np.inf
    return dict(point_cov=round(float(np.mean(e < tol_m)), 3), wall_cov=round(float(seen.mean()), 3), n_wall_near=int(len(wn)))


def sample_bilinear(light, px, big):
    """Bilinear wall distance (px) — a smooth cost so the least-squares refinement can move (nearest-pixel lookup is flat)."""
    x, y = px[:, 0], px[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    inb = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < light.W) & (y0 + 1 < light.H)
    v = np.full(len(px), big, float)
    xi, yi, fx, fy = x0[inb], y0[inb], (x - x0)[inb], (y - y0)[inb]
    F = light.F
    v[inb] = ((F[yi, xi] * (1 - fx) + F[yi, xi + 1] * fx) * (1 - fy) + (F[yi + 1, xi] * (1 - fx) + F[yi + 1, xi + 1] * fx) * fy) / 4.0
    return v


def cost(light, rec, fr, d):
    mp, ppm = project(rec, fr, d)
    e = np.minimum(sample_bilinear(light, mp, float(max(light.H, light.W))) / ppm, 5.0)
    k = max(1, int(TRIM * len(e)))
    return float(np.mean(np.partition(e, k - 1)[:k])), e


def register(light, rec, fr):
    t0 = time.perf_counter()
    grid = [(a, b, math.radians(g), 1.0 + s) for a in GRID_XY_M for b in GRID_XY_M for g in GRID_DEG for s in GRID_SCALE]
    costs = np.array([cost(light, rec, fr, d)[0] for d in grid])
    i = int(np.argmin(costs))
    best = grid[i]
    amb = [grid[j] for j in np.argsort(costs)[:12] if costs[j] <= costs[i] * (1 + LOCAL_AMBIG_REL)
           and (math.hypot(grid[j][0] - best[0], grid[j][1] - best[1]) > LOCAL_AMBIG_M
                or abs(math.degrees(grid[j][2] - best[2])) > LOCAL_AMBIG_DEG)]
    from scipy.optimize import least_squares
    lo = np.array([-1.5, -1.5, math.radians(-6), 0.97]); hi = np.array([1.5, 1.5, math.radians(6), 1.03])
    sub = [f for f in fr]

    def res(x):
        return np.minimum(cost(light, rec, sub, x)[1], 2.0)
    r = least_squares(res, np.array(best, float), bounds=(lo, hi), loss="soft_l1", f_scale=0.2,
                      diff_step=[1e-2, 1e-2, 1e-3, 1e-3], max_nfev=40)
    x = r.x
    c, e = cost(light, rec, fr, x)
    on_bound = bool(np.any(np.isclose(x, lo, atol=1e-3)) or np.any(np.isclose(x, hi, atol=1e-3)))
    return dict(correction=dict(dx_m=round(float(x[0]), 3), dy_m=round(float(x[1]), 3), dphi_deg=round(math.degrees(x[2]), 2),
                                scale=round(float(x[3]), 4)),
                alignment_score=round(c, 4), residual=round(float(np.median(e)), 4), inlier_ratio=round(float(np.mean(e < INLIER_M)), 3),
                convergence=bool(r.success) and not on_bound, on_bound=on_bound, n_points=int(len(e)),
                local_ambiguity=dict(ambiguous=bool(amb), n_alternatives=len(amb)), grid_best_cost=round(float(costs[i]), 4),
                free_space_violation=free_space_violation(light, rec, fr, tuple(x)) if FS_IN_REGISTER else None,
                segments=segment_costs(light, rec, fr, tuple(x)), **_v2(light, rec, fr, tuple(x)),
                ms=round(1000 * (time.perf_counter() - t0), 1))


def _v2(light, rec, fr, d):
    t0 = time.perf_counter()
    out = {}
    if XW_IN_REGISTER or V2_IN_REGISTER:
        out["wall_crossing"] = wall_crossing_violation(light, rec, fr, d)
    if V2_IN_REGISTER:
        out["coverage"] = map_coverage(light, rec, fr, d)
    if out:
        out["v2_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return out


def crossing_veto(best, results):
    """-> (veto, detail). Veto when best's angle-aware wall crossing exceeds the best OTHER-FLOOR candidate's by >= XW_VETO."""
    xb = (best.get("wall_crossing") or {}).get("violation")
    xs = [(r.get("wall_crossing") or {}).get("violation") for r in results if r["floor_id"] != best["floor_id"]]
    xs = [x for x in xs if x is not None]
    if xb is None or not xs:
        return False, None
    return xb - min(xs) >= XW_VETO, dict(best=xb, other_floor_min=min(xs), threshold=XW_VETO)


def _segments_overlap(a, b):
    sa, sb = (a.get("segments") or {}).get("costs") or [], (b.get("segments") or {}).get("costs") or []
    return bool(sa and sb) and min(sa) <= max(sb) and min(sb) <= max(sa)


def evaluate(store, candidates, lights, t_end, input_kind, window_s=90.0, semantic_ok=None, structural=False):
    """candidates = [{'candidate_id','floor','rec'}]; lights = {light_dir: LightMap}. -> E7 result dict."""
    assert input_kind in ("E6_SEMANTIC", "E6_OPERATOR_ASSISTED", "E2E3_DIAGNOSTIC", "STRUCTURE_ONLY"), input_kind
    fr = frames(store, t_end, window_s, structural=structural)
    out = dict(kind="LIVE2_E7_RESULT", input_kind=input_kind, t_end=t_end, window_s=window_s, n_frames=len(fr), structural=structural,
               transform_contract="odom_link (E2/E3 frame); map = student floor plan with fitted ppm (scale unverified)",
               map_reference="E2 light map: student metric_fit_v2 wall distance field F (2D floor plan walls)", results=[])
    if len(fr) < 5:
        out.update(final_decision="GLOBAL_UNKNOWN", why=f"only {len(fr)} usable LiDAR frames in {window_s:.0f} s")
        return out
    for c in candidates:
        L = lights.get(c["rec"].get("light"))
        if L is None:
            continue
        r = register(L, c["rec"], fr)
        out["results"].append(dict(candidate_id=c["candidate_id"], floor_id=c["floor"], candidate_global_pose=dict(
            pos_px=c["rec"]["pos"], phi_deg=c["rec"]["phi"], ppm=c["rec"]["ppm"]), **r))
    rs = sorted(out["results"], key=lambda r: r["alignment_score"])
    if not rs:
        out.update(final_decision="GLOBAL_UNKNOWN", why="no candidate had a map")
        return out
    b = rs[0]
    rival = next((r for r in rs[1:] if r["floor_id"] != b["floor_id"]), rs[1] if len(rs) > 1 else None)
    gap = None if rival is None else round(rival["alignment_score"] - b["alignment_score"], 4)
    veto, vdet = crossing_veto(b, rs)
    ok_geom = b["convergence"] and not b["local_ambiguity"]["ambiguous"] and (gap is None or gap >= GAP_M) and not veto
    indist = rival is not None and gap is not None and gap < GAP_M and _segments_overlap(b, rival)
    out.update(best=b["candidate_id"], best_floor=b["floor_id"], runner_up=None if rival is None else rival["candidate_id"], gap_m=gap,
               ambiguity=None if gap is None else gap < GAP_M, structurally_indistinguishable=indist,
               structural_status="STRUCTURALLY_INDISTINGUISHABLE" if indist else None, crossing_veto=veto, crossing=vdet)
    if not ok_geom:
        why = [w for w, bad in (("not converged / on search bound", not b["convergence"]), ("locally ambiguous", b["local_ambiguity"]["ambiguous"]),
                                (f"runner-up within {gap} < {GAP_M} m", gap is not None and gap < GAP_M),
                                ("STRUCTURALLY_INDISTINGUISHABLE (per-segment costs of best and other-floor rival overlap)", indist),
                                (f"angle-aware wall crossing contradicts the best ({vdet})", veto)) if bad]
        out.update(final_decision="MORE_OBSERVATION_REQUIRED", why="; ".join(why))
    elif semantic_ok:
        out.update(final_decision="GLOBAL_CONFIRMED", why="geometry unique + independent semantic match")
    else:
        out.update(final_decision="GEOMETRY_CONSISTENT_SEMANTIC_PENDING",
                   why="geometry unique and separated; final pose needs an independent semantic (E6) match")
    return out
