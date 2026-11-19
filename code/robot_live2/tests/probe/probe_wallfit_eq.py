"""Bit-equality + speed of live2.wallfit_cpu_fast vs the student CPU expression on real S6 frames (sw 6F)."""
import sys, time
sys.path.insert(0, sys.argv[1] + '/external/sci_live/floor_id'); sys.path.insert(0, sys.argv[1] + '/live2')
import floor_id as F
cfg = F.load_cfg(); F.setup_env(cfg, 'sw')
import numpy as np, rt_localizer as RT, run_floor_id as R, wallfit_cpu_fast as FW
from wallfit_gpu import wallfit_cpu
Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=6, ppm=R.MID, data_dir=sys.argv[2], bag=None); L = Z.L
bad = 0; tc = tf = 0.0; n = 0
for t in np.linspace(10, 360, 12):
    pts, yaw = L._scan(L.T[0] + t)
    if len(pts) > 400: pts = pts[np.random.RandomState(int(L.T[0] + t) % 97).choice(len(pts), 400, replace=False)]
    for hd in (0.3, 2.1, 4.4):
        for ppm in (15.0 * 1.2, 24.0, 43.0 * 1.3):
            t0 = time.perf_counter(); a = wallfit_cpu(L, pts, hd, ppm); tc += time.perf_counter() - t0
            t0 = time.perf_counter(); b = FW.wallfit(L, pts, hd, ppm); tf += time.perf_counter() - t0
            n += 1; bad += int(not np.array_equal(a, b))
print("exact_table", FW.STATS); print(f"calls {n} not_bit_equal {bad} cpu_ms {1e3*tc/n:.1f} fast_ms {1e3*tf/n:.1f} speedup {tc/tf:.2f}x candidates {len(L.Pf)}")
