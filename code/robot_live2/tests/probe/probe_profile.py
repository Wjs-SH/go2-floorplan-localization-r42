import cProfile, pstats, sys, os, time
sys.argv = sys.argv  # same args as probe_fit_once
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_fit_once.py')).read()
# profile only the fit: reuse probe body with one scale to keep it short
src = src.replace("SC[m], floor=m", "SC[m][2:4], floor=m")
pr = cProfile.Profile(); pr.enable(); exec(compile(src, 'probe', 'exec'), {'__name__': '__main__'}); pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(14)
