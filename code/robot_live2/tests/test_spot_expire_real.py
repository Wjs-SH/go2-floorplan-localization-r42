"""R34: spot-memory expiry through the REAL legacy spot_mean, the REAL Frontend hook names and observe_node's REAL _TravelMeter code; REAL field sims from scenes.jsonl (11-01 spot sequence), SYNTHETIC travel."""
import ast
import json
import sys
import threading
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAND = HERE.parent / "candidate"
SRC = (CAND / "src/sci_legacy_e5/live_event_runner.py").read_text()


def _method(src, name, cls=None):
    mod = ast.parse(src)
    for n in ast.walk(mod):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            code = compile(ast.Module([n], []), "m", "exec")
            ns = dict(np=np)
            exec(code, ns)
            return ns[name]
    raise KeyError(name)


spot_mean = _method(SRC, "spot_mean")
OBS = (CAND / "live1/observe_node.py").read_text()
_cls = next(n for n in ast.walk(ast.parse(OBS)) if isinstance(n, ast.ClassDef) and n.name == "_TravelMeter")
_ns = {}
exec(compile(ast.Module([_cls], []), "o", "exec"), _ns)
TM = _ns["_TravelMeter"]


def L(cur, m=0.7, nr=3):
    return types.SimpleNamespace(spot_recent=nr, spot_expire_m=m, cur_travel=cur)


def test_old_low_scores_from_earlier_travel_do_not_dilute_the_plate():
    sp = dict(sims=[0.02, 0.03, 0.14], trv=[1.0, 2.0, 5.0])
    assert abs(spot_mean(L(5.0), sp) - 0.14) < 1e-9
    assert abs(spot_mean(L(5.0, m=0.0), sp) - np.mean(sp["sims"])) < 1e-9        # off = the old rule


def test_unknown_travel_keeps_the_old_rule():
    sp = dict(sims=[0.02, 0.03, 0.14], trv=[None, None, None])
    assert abs(spot_mean(L(None), sp) - np.mean(sp["sims"])) < 1e-9


def test_standing_still_keeps_all_recent_observations():
    sp = dict(sims=[0.10, 0.12, 0.14], trv=[5.0, 5.0, 5.0])
    assert abs(spot_mean(L(5.0), sp) - 0.12) < 1e-9


def test_threshold_and_gates_are_untouched():
    pol = (CAND / "src/sci_visual_observation/policy.py").read_text()
    assert 'r["msim"] < thr' in pol and 'r["sim"] < c["sim_floor"]' in pol
    assert "SIM_MIN" in (CAND / "live2/visual_stop.py").read_text()


def test_travel_meter_uses_only_real_differences():
    src = types.SimpleNamespace(lock=threading.Lock(), odom=[])
    m = TM(src)
    assert m() is None                                                          # unknown until odometry exists
    src.odom = [(1.0, 0.0, 0.0, 0.0), (2.0, 3.0, 4.0, 0.0)]
    assert m() == 5.0
    src.odom = [(2.0, 3.0, 4.0, 0.0), (3.0, 3.0, 5.0, 0.0)]                      # a repeated sample is not counted twice
    assert m() == 6.0


def test_frontend_calls_the_hook_and_the_shared_mean():
    fe = (CAND / "src/sci_visual_observation/frontend.py").read_text()
    assert 'getattr(type(self), "travel_source", None)' in fe and "L.spot_mean(sp)" in fe
    assert "import frontend as FE" in OBS
