"""Pre-build the E2 map-preparation cache (RTLocalizer + metric_fit_v2._maps) per floor. Maps + code only; the data dir
is only needed by the constructor and is replaced by refresh() at run time (same as loc_worker's cache)."""
import sys, time, argparse
sys.path.insert(0, sys.argv[1] + "/candidate/live2")
import e2_worker as EW
ap = argparse.Namespace(building=sys.argv[2], store=sys.argv[3], cache_dir=sys.argv[1] + "/runtime/e2_cache", device="cpu",
                        wall_pass_top=4, out=None)
FR = EW.FloorRunner(ap)
for m in [int(x) for x in sys.argv[4].split(",")]:
    t = time.time(); Z, src, s = FR.open_floor(m); print(f"{sys.argv[2]} {m}F {src} {s:.1f}s rss {EW.rss_mb()} MB", flush=True); del Z
