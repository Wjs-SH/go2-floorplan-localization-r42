"""e2_core.py — LIVE2 E2: keep the student's metric_fit_v2 INTERNAL candidates instead of returning only the best one.

fit_window_v2_topk() is fit_window_v2 (candidate tree external/sci_live/floor_id/localizer/metric_fit_v2.py) with the same
inputs, constants, helper functions and order of operations. The only difference is the return value:
  best        = exactly the dict fit_window_v2 returns (asserted by tests/test_e2_core.py)
  candidates  = every de-duplicated pose fit_window_v2 already scored with T4 (up to N_T4 = 12), ascending merit
The student module is imported, never edited. Scores are the student's costs (merit_v2, lower = better) and the
student's floor score (twoway, higher = better); nothing here is a probability.

pose_in_map() / candidate_record() turn a candidate into the LIVE2 record (floor, px, metric-ish m, yaw, scores,
map_version, odom_session, timestamps). The map frame is the floor-plan image with the FITTED ppm (unverified scale):
it is not a surveyed metric map and never a final pose.
"""
import hashlib
import math
import os
from pathlib import Path

import numpy as np


def fit_window_v2_topk(localizer, t_now, scales, floor, W=60., dt=3., topk=3, beta_provider=None, beta_contract=None, between=None,
                       region=None):
    """between(i, n): optional call before scale i of n (LIVE2 hold / progress point). It only waits; the computation and
    its result are unchanged (field 2026-10-21: a 60-110 s fit could not yield to FAST-LIO or heat until it finished).
    region (2026-10-30, semantic recovery): dict(center_px=[x, y], radius_px=r) — the student's seed grid (Z.allow) is
    intersected with that disk; everything else (scores, least squares, de-duplication, merit) is the student's code."""
    import metric_fit_v2 as V
    from rt_localizer import _A, _R
    from metric_fit import set_scale
    from node_local import GRID_STEP
    Z, L = localizer, localizer.L
    M = V._maps(Z, floor)
    saved = (L.PPM, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis)
    try:
        start = max(L.T[0], t_now - W)
        beta_src = V._window_beta(Z, start, t_now, beta_provider, beta_contract)
        modes, _ = Z.bootstrap(start, boot_s=t_now - start)
        buf = Z.window_scans(t_now, W, dt)
        frames = np.array([fr['t'] for fr in buf])
        anchor = np.asarray(buf[-1]['xy'])
        cloud, trajectory, fidx = [], [], []
        for n, fr in enumerate(buf):
            pts, yaw = fr['pts'], fr['yaw']
            if len(pts) > 400:
                pts = pts[np.random.RandomState(int(fr['sample_seed_t']) % 97).choice(len(pts), 400, replace=False)]
            delta = np.asarray(fr['xy']) - anchor
            cloud.append(delta + (pts * [1, Z.MIRROR]) @ _R(yaw + Z.BETA).T)
            trajectory.append(delta); fidx.append(np.full(len(pts), n, np.int16))
        cloud = np.vstack(cloud); fidx = np.concatenate(fidx); trajectory = np.asarray(trajectory)
        if len(cloud) < 50:
            raise ValueError('Too few scan points for metric fitting')
        sel = np.where((L.T >= start) & (L.T <= frames[-1]))[0]
        keep, last = [], -1e9
        for i in sel:
            if L.T[i] - last >= V.TRAJ_DT:
                keep.append(i); last = L.T[i]
        dense = np.c_[np.asarray(L.X)[keep], np.asarray(L.Y)[keep]] - anchor
        if len(dense) < 2:
            dense = trajectory
        sample = cloud[np.linspace(0, len(cloud) - 1, min(1000, len(cloud))).astype(int)]
        CV = float(max(L.H, L.W))

        def project(x, pts):
            return pts @ _A(x[2]).T * np.exp(x[3]) + x[:2]

        def dist(x, pts):
            return V._sample(M['F'], project(x, pts), CV) / np.exp(x[3])

        def residual(x):
            return np.minimum(dist(x, sample), 5.)

        from scipy.optimize import least_squares
        cands = []
        allow = np.asarray(Z.allow).reshape(-1)
        if region is not None:
            Pf = np.asarray(L.Pf, float).reshape(-1, 2)
            cx, cy = region["center_px"]
            allow = allow & (np.hypot(Pf[:, 0] - float(cx), Pf[:, 1] - float(cy)) <= float(region["radius_px"]))
            if not allow.any():
                raise ValueError('semantic region contains no traversable seed cell')
        for i_sc, ppm in enumerate(scales):
            if between is not None:
                between(i_sc, len(scales))
            set_scale(Z, ppm)
            result = Z.localize(t_now, W=W, dt=dt, phis=modes)
            for phi in modes:
                values = np.where(allow.reshape(L.shp), result['P'][phi], -np.inf).ravel()
                for k in np.argsort(-values)[:topk]:
                    if not np.isfinite(values[k]): continue
                    x0 = np.r_[L.Pf[k], np.deg2rad(phi), np.log(ppm)]
                    shift = 1.5 * GRID_STEP
                    low = x0 + [-shift, -shift, -np.deg2rad(8), np.log(.8)]
                    high = x0 + [shift, shift, np.deg2rad(8), np.log(1.25)]
                    x = least_squares(residual, x0, bounds=(low, high), loss='soft_l1', f_scale=.2,
                                      diff_step=[1e-3, 1e-3, 1e-4, 1e-4], max_nfev=60).x
                    px, py = np.rint(x[:2]).astype(int)
                    if not (0 <= px < L.W and 0 <= py < L.H and Z.ok_px[py, px]):
                        continue
                    s = float(np.exp(x[3]))
                    err = dist(x, cloud)
                    cost = float(np.mean(np.sort(err)[:int(.7 * len(err))]))
                    track = project(x, trajectory)
                    ix, iy = np.rint(track).astype(int).T
                    inside = (ix >= 0) & (ix < L.W) & (iy >= 0) & (iy < L.H)
                    trav = np.zeros(len(ix), bool); trav[inside] = Z.ok_px[iy[inside], ix[inside]]
                    blocked = float(1 - trav.mean())
                    nogo = float(Z.pen_px[py, px])
                    cpx = project(x, cloud)
                    T2 = float(np.mean(err > V.FAR_M))
                    T3 = float(np.mean(V._sample(M['out_d'], cpx, CV) / s > V.OUT_M))
                    dpx = project(x, dense)
                    X = V._crossing_m(M['wall_tr'], dpx, s)
                    part = cost + .5 * blocked + .1 * Z.w_nogo * nogo + V.W_FREE * T2 + V.W_OUT * T3 + V.W_CROSS * min(X, V.CROSS_MAX_M)
                    cands.append(dict(x=x.copy(), err=err, cost=cost, blocked=blocked, nogo=nogo, T2=T2, T3=T3, X=X, part=part,
                                      scene=result['scene'], cpx=cpx, tpx=track, dpx=dpx, start_scale=float(ppm), start_phi=float(phi)))
        if not cands:
            raise ValueError('No traversable metric fit candidates')
        feas = [c for c in cands if c['X'] <= V.CROSS_MAX_M]
        n_feas = len(feas)
        if not feas:
            xmin = min(c['X'] for c in cands)
            feas = [c for c in cands if c['X'] <= xmin + 1e-9]
        feas.sort(key=lambda c: c['part'])
        picked = []
        for c in feas:
            x = c['x']
            if any(np.hypot(*(x[:2] - p['x'][:2])) / np.exp(0.5 * (x[3] + p['x'][3])) < V.DUP_M
                   and abs((np.degrees(x[2] - p['x'][2]) + 180) % 360 - 180) < V.DUP_DEG
                   and abs(x[3] - p['x'][3]) < np.log(V.DUP_SCALE) for p in picked):
                continue
            picked.append(c)
            if len(picked) == V.N_T4: break
        for c in picked:
            s = float(np.exp(c['x'][3]))
            t4, t4_raw, n_hits, n_door, miss = V.unexplained_walls(M['wall'], M['door'], c['cpx'], c['tpx'], s)
            c.update(T4=t4, T4_with_doors=t4_raw, n_wall_hits=n_hits, n_door_rays=n_door, miss_xy=miss,
                     merit=c['part'] + V.W_MISS * t4)
        bi = min(range(len(picked)), key=lambda i: picked[i]['merit'])
        best = picked[bi]
        x = best['x']; e = best['err']
        best_out = dict(pos=x[:2].copy(), phi=float(np.degrees(x[2]) % 360), ppm=float(np.exp(x[3])),
                        beta=float(Z.BETA), beta_source=beta_src.get('source'), beta_verified=beta_src.get('verified'),
                        mirror=int(Z.MIRROR), cost_m=best['merit'], wall_cost_m=best['cost'],
                        blocked_fraction=best['blocked'], nogo=best['nogo'], T2=best['T2'], T3=best['T3'], X_cross_m=best['X'],
                        T4=best['T4'], T4_with_doors=best['T4_with_doors'], n_wall_hits=best['n_wall_hits'],
                        n_door_rays=best['n_door_rays'], rank_in_part=bi, n_candidates=len(cands),
                        n_feasible=n_feas, half_wall_px=M['half_px'], anchor=anchor, frames=frames, t=float(frames[-1]),
                        cloud_px=best['cpx'], trajectory_px=best['dpx'], frame_px=best['tpx'], fidx=fidx, distances_m=e,
                        miss_wall_px=best['miss_xy'], median_m=float(np.median(e)), inlier_10cm=float(np.mean(e < .1)),
                        scene=best['scene'])
        common = dict(beta=float(Z.BETA), beta_source=beta_src.get('source'), beta_verified=beta_src.get('verified'),
                      mirror=int(Z.MIRROR), anchor=anchor.copy(), t=float(frames[-1]), n_frames=len(frames),
                      window_s=float(frames[-1] - frames[0]))
        out = []
        for i, c in enumerate(sorted(picked, key=lambda c: c['merit'])):
            x = c['x']
            out.append(dict(common, pos=[float(x[0]), float(x[1])], phi=float(np.degrees(x[2]) % 360), ppm=float(np.exp(x[3])),
                            merit=float(c['merit']), wall_cost_m=float(c['cost']), blocked=float(c['blocked']), nogo=float(c['nogo']),
                            T2=float(c['T2']), T3=float(c['T3']), X_cross_m=float(c['X']), T4=float(c['T4']),
                            inlier_10cm=float(np.mean(c['err'] < .1)), median_m=float(np.median(c['err'])), scene=c['scene'],
                            start_scale=c['start_scale'], start_phi=c['start_phi'], is_student_best=c is best,
                            traj_px=np.asarray(c['dpx'], float)))
        return dict(best=best_out, candidates=out, n_raw=len(cands), n_feasible=n_feas, n_picked=len(picked))
    finally:
        ppm, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis = saved
        set_scale(Z, ppm)


def _A(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [-s, -c]])


def pose_in_map(c, x, y, yaw):
    """Same transform as pose_tools.map_pose_px (odom_link x, y, yaw -> floor-plan px + heading)."""
    phi = math.radians(c['phi'])
    p = np.asarray(c['pos'], float) + c['ppm'] * (_A(phi) @ (np.array([x, y], float) - np.asarray(c['anchor'], float)))
    hd = phi + yaw + c['beta']
    return float(p[0]), float(p[1]), float((hd + math.pi) % (2 * math.pi) - math.pi)


def map_version(maps_root, floor):
    """Short content hash of the floor's localization map files (what the fit actually read)."""
    h = hashlib.sha256()
    d = Path(maps_root) / f'{floor}F' / 'maps'
    for p in sorted(d.glob('*.png')) + sorted(d.glob('*.yaml')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


def candidate_record(c, floor, mapver, odom_session, H, twoway=None, wall_pass=None, pass_id=None, now_wall=None):
    """LIVE2 E2 candidate record. x_m/y_m: floor-plan px / FITTED ppm, y up from the image bottom (scale unverified)."""
    px, py = c['pos']
    yaw_map = (math.radians(c['phi']) + c['beta'])          # heading of odom_link yaw 0 in the map; robot heading needs yaw
    return dict(floor=int(floor), px=round(px, 1), py=round(py, 1), x_m=round(px / c['ppm'], 2), y_m=round((H - py) / c['ppm'], 2),
                yaw_offset_deg=round(math.degrees(yaw_map) % 360, 1), phi_deg=round(c['phi'], 2), ppm=round(c['ppm'], 3),
                beta_deg=round(math.degrees(c['beta']) % 360, 2), beta_source=c.get('beta_source'), mirror=c['mirror'],
                merit=round(c['merit'], 4), twoway=None if twoway is None else round(float(twoway), 4), wall_pass=wall_pass,
                wall_cost_m=round(c['wall_cost_m'], 4), T2=round(c['T2'], 4), T3=round(c['T3'], 4), T4=round(c['T4'], 4),
                X_cross_m=round(c['X_cross_m'], 3), blocked=round(c['blocked'], 3), inlier_10cm=round(c['inlier_10cm'], 3),
                scene=c['scene'], is_student_best=bool(c['is_student_best']), anchor=[float(v) for v in c['anchor']],
                pos=[float(v) for v in c['pos']], phi=float(c['phi']), beta=float(c['beta']),
                fit_t=float(c['t']), window_s=round(c['window_s'], 1), map_version=mapver, odom_session=odom_session,
                pass_id=pass_id, created_wall=now_wall, status='UNVERIFIED_HYPOTHESIS', verified=False)


def frame_seconds_per_floor_estimate(path):
    """Mean measured E2 seconds per floor from earlier passes (e2_passes.jsonl), None if never measured."""
    try:
        import json
        v = [json.loads(l)['sec'] for l in open(path) if '"floor_done"' in l]
        return float(np.mean(v[-20:])) if v else None
    except (OSError, ValueError, KeyError):
        return None


def env_threads():
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(k, '1')
