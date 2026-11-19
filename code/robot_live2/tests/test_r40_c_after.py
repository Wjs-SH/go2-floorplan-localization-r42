"""R40 scenario C: X20261103_003729 before / after. BEFORE = candidates the field E2 really had at that time. AFTER = the same + the unchanged exact fit of every floor on the window the new rule would use (data/exact40/c300_F*.json, offline). Real store, real E7 code, real classify. GT only scores. usage: test_r40_c_after.py OUT"""
import importlib.util, json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
A0 = sys.argv[1]
spec = importlib.util.spec_from_file_location("TF", HERE / "test_r40_field.py"); TF = importlib.util.module_from_spec(spec); spec.loader.exec_module(TF)
E3, EL = TF.E3, TF.EL
RUN_ID, TES, WINS = sys.argv[2], [float(x) for x in sys.argv[3].split(",")], sys.argv[4].split(",")   # usage: OUT RUN te1,te2 w1,w2
run = TF.Run(RUN_ID)
EX = []
for w in WINS:
    for f in range(1, 10):
        for r in json.loads((HERE.parent / f"data/exact40/{w}_F{f}.json").read_text()):
            EX += [dict(c, light=TF.lf(f), stage="exact", floor=f) for c in r.get("raw") or []]
def evaluate(te, extra):
    cs = run.cands(te)
    if extra:
        cs = cs + extra
    pf = {str(f): dict(light=TF.lf(f), status="DONE") for f in range(1, 10)}
    e2 = dict(per_floor=pf, coarse=dict(per_floor=pf), candidates_pass=[c for c in cs if c["stage"] == "exact"], candidates_coarse=[c for c in cs if c["stage"] == "coarse"], candidates_refresh=[c for c in cs if c["stage"] == "refresh"], floors_done=sorted({c["floor"] for c in cs if c["stage"] == "exact"}), floors_total=9, refresh=dict(n_passes=2))
    LC = {}
    def light(p):
        if p not in LC:
            LC[p] = E3.LightMap(p)
        return LC[p]
    return EL.evaluate(run.ST, e2, [], run.T0 + te, light)
out = dict(cases=[])
for name, extra in (("BEFORE", None), ("AFTER", EX)):
    hist = []
    for te in TES:
        r = evaluate(te, extra); hist.append(r); c = EL.classify(hist, 9)
        rows = []
        for x in r["floors"][:6]:
            rec = dict(pos=x["pos"], anchor=x["anchor"], phi=x["phi"], ppm=x["ppm"], beta=x["beta"], mirror=x.get("mirror"), light=None)
            px, py, h = E3.pose_in_map(rec, run.R[0][2], run.R[0][3], run.R[0][4])
            rows.append(dict(floor=x["floor"], stage=x["stage"], cost=round(x["cost"], 3), conv=x["converged"], xw=x.get("crossing"), scale=x.get("scale_rel"), start_err_m=round(math.hypot(px - run.GT["px"], py - run.GT["py"]) / 24.0, 1) if x["floor"] == run.GT["floor"] else None, head_err=round((math.degrees(h) - run.GT["yaw_deg"] + 180) % 360 - 180, 1) if x["floor"] == run.GT["floor"] else None, pos_rival=x.get("pos_rival")))
        d = dict(name=name, te=te, status=c["status"], why=c.get("why"), gap=c.get("next_gap_m"), close=c.get("close_floors"), floors=rows)
        out["cases"].append(d); print(json.dumps(d, ensure_ascii=False, default=str), flush=True)
Path(A0).write_text(json.dumps(out, ensure_ascii=False, default=str, indent=0))
