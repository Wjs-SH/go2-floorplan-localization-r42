"""Probe: one student fit_window_v2 on stored S6 input, CPU only. Prints time and RSS. Read-only on inputs."""
import os, sys, time, resource
sys.path.insert(0, sys.argv[1] + '/external/sci_live/floor_id')
import floor_id as F
cfg = F.load_cfg(); b = F.setup_env(cfg, sys.argv[2])
import numpy as np, rt_localizer as RT, run_floor_id as R, metric_fit_v2 as V
data, m, ts = sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
t0 = time.time(); Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=data, bag=None)
V._maps(Z, m); print(f'open {time.time()-t0:.1f}s rss {rss()} MB grid {Z.L.shp} map {Z.L.H}x{Z.L.W}', flush=True)
SC = F.map_scales(b, R, int(b['n_maps']))
t1 = time.time(); r = V.fit_window_v2(Z, Z.L.T[0] + ts, SC[m], floor=m, W=60., dt=3.)
print(f'fit {time.time()-t1:.1f}s rss {rss()} MB maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024} MB pos {r["pos"]} phi {r["phi"]:.1f} ppm {r["ppm"]:.2f} merit {r["cost_m"]:.3f} n_cand {r["n_candidates"]} beta {np.degrees(r["beta"]):.1f} mirror {r["mirror"]}', flush=True)
