"""R40 scenario C: the REAL E3 tracker over the real store of X20261103_003729, virtual clock. BEFORE: only what the field E2 logged at the time it logged it (repeat-search candidates were not logged in R39, so they are absent). AFTER: + the unchanged exact fit on the window the new rule uses, at the time it would have finished (data/exact40, offline). GT only scores: start pose, by back-projection.
usage: test_r40_e3_replay.py OUT.json WINDOW AVAIL_S [run]"""
import json, math, sys, tempfile
from pathlib import Path
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "candidate/live2"), str(OPR / "candidate/external/sci_live/floor_id/localizer")]
import e3_tracker as E3
from test_e7fix_replay import CutStore, LROOT
RUN = "X20261103_003729"; X = OPR / "experiments" / RUN; T0 = json.loads((X / "experiment.json").read_text())["created_wall"]
GT = json.loads((OPR / "evaluation" / RUN / "gt.jsonl").read_text().splitlines()[0])
P = [json.loads(l) for l in open(X / "state/e2/e2_passes.jsonl")]
sess = json.loads((X / "archive/live2_store/session.json").read_text()).get("odom_session_id")
def lf(f): return str(next(LROOT.glob(f"kci_{f}F_*")))
def err(rec, R0):
    px, py, h = E3.pose_in_map(dict(rec, light=None), R0[2], R0[3], R0[4])
    return round(math.hypot(px - GT["px"], py - GT["py"]) / 24.0, 1), round((math.degrees(h) - GT["yaw_deg"] + 180) % 360 - 180, 1)
def sim(extra, t_end=388.0):
    td = Path(tempfile.mkdtemp()); (td / "e2").mkdir()
    tr = E3.Tracker(str(X / "archive/live2_store"), td / "e2", log_path=None)
    st0 = tr.store; st0.refresh(); cs = CutStore(st0); tr.store = cs
    now = [T0 + 60.0]; tr.clock = lambda: now[0]
    out = []; last = None
    while now[0] < T0 + t_end:
        now[0] += 2.0; cs.cut(now[0])
        pf = {str(f): dict(light=lf(f), status="DONE", W=4000, H=2000) for f in range(1, 10)}
        coarse, exact = [], []
        for p in P:
            if p["wall"] > now[0]:
                continue
            if p.get("kind") == "coarse_floor_done":
                coarse += [dict(c, stage="coarse", pass_id=1, floor=p["floor"]) for c in p.get("candidates") or []]
            if p.get("kind") == "floor_done":
                exact += [dict(c, stage="exact", pass_id=1, floor=p["floor"]) for c in p.get("candidates") or []]
        for av, c in extra:
            if T0 + av <= now[0]:
                exact.append(dict(c, stage="exact", pass_id=2))
        e2 = dict(odom_session=sess, per_floor=pf, coarse=dict(per_floor=pf), candidates_coarse=coarse, candidates_pass=exact, candidates_refresh=[])
        (td / "e2/e2_state.json").write_text(json.dumps(e2, default=float))
        for _ in range(4):
            tr.tick()
            if not tr.snapshot().get("kf_pending"):
                break
        al = sorted(tr.alive(), key=lambda h: h.rank_key())
        if not al or not cs.rows:
            continue
        b = al[0]
        if b.id != last or int(now[0] - T0) % 40 == 0:
            last = b.id
            b4 = [h for h in al if h.floor == GT["floor"]]
            out.append(dict(t=round(now[0] - T0), lead=b.id, floor=b.floor, score=round(b.score(), 3), n_kf=b.n_kf, ppm=round(b.rec["ppm"], 1), err=err(b.rec, cs.rows[0]) if b.floor == GT["floor"] else None, best4=None if not b4 else dict(id=b4[0].id, score=round(b4[0].score(), 3), err=err(b4[0].rec, cs.rows[0])), n_alive=len(al)))
    return out
if __name__ == "__main__":
    res = {}
    extra = []
    for w, av in zip(sys.argv[2::2], sys.argv[3::2]):
        for f in range(1, 10):
            for r in json.loads((HERE.parent / f"data/exact40/c{w}_F{f}.json").read_text()):
                extra += [(float(av), dict(c, floor=f)) for c in r.get("raw") or []]
    res["AFTER"] = sim(extra)
    Path(sys.argv[1]).write_text(json.dumps(res, default=str))
    for k, v in res.items():
        print("##", k)
        for x in v:
            print(json.dumps(x, default=str))
