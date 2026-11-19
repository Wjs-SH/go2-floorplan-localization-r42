"""E2 equivalence evidence (stored input, CPU only): student fit_window_v2 with the ORIGINAL CPU wall-fit vs
LIVE2 fit_window_v2_topk with live2.wallfit_cpu_fast. Same process, same Z, same t_now. Writes JSON evidence."""
import json, sys, time
cand, building, data, floor, ts, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
sys.path.insert(0, cand + '/external/sci_live/floor_id'); sys.path.insert(0, cand + '/live2')
import floor_id as F
cfg = F.load_cfg(); b = F.setup_env(cfg, building)
import numpy as np, rt_localizer as RT, run_floor_id as R, metric_fit_v2 as V, wallfit_cpu_fast as FW, e2_core as E
Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=floor, ppm=R.MID, data_dir=data, bag=None); V._maps(Z, floor)
SC = F.map_scales(b, R, int(b['n_maps']))[floor]; t_now = Z.L.T[0] + ts
RT._GPU = None
t0 = time.time(); a = V.fit_window_v2(Z, t_now, SC, floor=floor, W=60., dt=3.); ta = time.time() - t0
RT._GPU = FW
t0 = time.time(); r = E.fit_window_v2_topk(Z, t_now, SC, floor=floor, W=60., dt=3.); tb = time.time() - t0
bb = r['best']; diffs = {}
for k, v in a.items():
    w = bb.get(k)
    same = np.array_equal(np.asarray(v), np.asarray(w)) if not isinstance(v, (str, type(None))) else v == w
    if not same: diffs[k] = [str(v)[:80], str(w)[:80]]
res = dict(floor=floor, t_rel=ts, student_s=round(ta, 1), live2_topk_s=round(tb, 1), speedup=round(ta / tb, 2),
           best_identical=not diffs, diffs=diffs, n_candidates_returned=len(r['candidates']), n_raw=r['n_raw'],
           student_best=dict(pos=[float(v) for v in a['pos']], phi=a['phi'], ppm=a['ppm'], merit=a['cost_m']),
           topk=[dict(pos=[round(v, 1) for v in c['pos']], phi=round(c['phi'], 1), ppm=round(c['ppm'], 2), merit=round(c['merit'], 4),
                      is_best=c['is_student_best']) for c in r['candidates']], wallfit_stats=FW.STATS)
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res)[:1500])
