"""R40: the three field runs of 11-03 through the real code, offline. GT only scores. usage: test_r40_field.py OUT
B: X20261103_001810: field photos, field OCR, field VLM on O002, the same VLM offline (data/vlm40.json) on the photos the field VLM never reached. Hypotheses: this run's E2 + an anchored fit from the reading, as the field code makes it.
A / C: real stores and E2 records through e7_long.evaluate + classify."""
import importlib.util, json, math, shutil, sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
L, E3 = RP.T.load(RP.STG); sys.path.insert(0, str(RP.STG))
import e6_semantic as ES, e8_pair as EP, semantic_anchor as SA, e7_policy as P7, e7_long as EL, plate_text as PT  # noqa
E7 = L.E7
SF = ES.SemanticFloors()
VLM = json.loads((HERE.parent / "data/vlm40.json").read_text())
def lf(f): return str(next(RP.LROOT.glob(f"kci_{f}F_*")))
RES = {f: 1.0 / EP.floor_res(f, m)[0] for f, m in SF.maps.items()}
class Run:
    def __init__(s, run):
        s.run, s.X = run, RP.OPR / "experiments" / run
        s.T0 = json.loads((s.X / "experiment.json").read_text())["created_wall"]
        s.GT = json.loads((RP.OPR / "evaluation" / run / "gt.jsonl").read_text().splitlines()[0])
        s.ST = E3.Store(str(s.X / "archive/live2_store")); s.ST.refresh(); s.R = list(s.ST.rows)
        s.P = [json.loads(l) for l in open(s.X / "state/e2/e2_passes.jsonl")]
    def err(s, rec):
        px, py, h = E3.pose_in_map(dict(rec, light=None), s.R[0][2], s.R[0][3], s.R[0][4])
        return round(math.hypot(px - s.GT["px"], py - s.GT["py"]) / 24.0, 2), round((math.degrees(h) - s.GT["yaw_deg"] + 180) % 360 - 180, 1)
    def cands(s, te):
        out = []
        for p in s.P:
            if p.get("kind") in ("floor_done", "coarse_floor_done", "refresh_floor_done") and p["wall"] <= s.T0 + te:
                st = dict(floor_done="exact", coarse_floor_done="coarse", refresh_floor_done="refresh")[p["kind"]]
                for c in p.get("candidates") or []:
                    if "pos" in c:
                        out.append(dict(c, light=lf(p["floor"]), stage=st, floor=p["floor"]))
        return out
def observations(run):
    A = {}
    for l in open(run.X / "state/e8_observations.jsonl"):
        d = json.loads(l)
        if d.get("event") in ("STORED", "READ", "VLM_READ"):
            A.setdefault(d["observation_id"], {}).update({k: v for k, v in d.items() if v is not None})
    out = {}
    for oid, d in A.items():
        v = d.get("vlm") or (VLM.get(oid) or {}).get("parsed")
        out[oid[-4:]] = dict(d, vlm=v, reading=dict(d.get("reading") or {}, vlm=dict(state="ASKED", crop=True)), read=True, experiment_id=run.run)
    return out
RESULT = dict(cases=[], ok=True)
def rec(name, ok=True, **kw):
    RESULT["cases"].append(dict(name=name, ok=ok, **kw)); RESULT["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(kw, ensure_ascii=False, default=str)[:520], flush=True)
def anchored(run, floor, label, ob, beta):
    m = SF.maps[floor]; sp = SF.spaces(floor, label)
    cor = [c["polygon_px"] for c in m.get("corridors") or [] if c.get("polygon_px")]
    fr = E7.frames(run.ST, float(ob["capture_wall"]), SA.WINDOW_S, structural=True)
    top, _ = SA.scan(E7.cost, fr, E3.LightMap(lf(floor)), ob["target_odom_link_xy"], SA.plate_wall_points(sp[0]["polygon_px"], cor, RES[floor]), RES[floor], beta, 1, floor)
    return [dict(t["rec"], floor=floor, light=lf(floor), merit=0.26, id=f"A{i + 1}_{label}") for i, t in enumerate(top[:2])]
def scenario_b(out):
    run = Run("X20261103_001810"); O = observations(run)
    table = []
    for k, o in sorted(O.items()):
        a, fu, dec, src = EP.reading_of(o)
        table.append(dict(obs=k, ocr=o.get("ocr_lines"), vlm=(o.get("vlm") or {}).get("room_number"), text=(o.get("vlm") or {}).get("text"), sharp=(o.get("image_quality") or {}).get("roi_laplacian_var"), labels=a, decisive=dec, src=src))
    print(json.dumps(table, ensure_ascii=False, indent=0)); RESULT["b_table"] = table
    rec("B O002 (407 + 로봇공학과, OCR empty) now counts", EP.reading_of(O["O002"])[2] and EP.reading_of(O["O002"])[0][0][0] == "407")
    rec("B blurred O008 / O010 do not count (VLM still invents a plate)", not EP.reading_of(O["O008"])[2] and not EP.reading_of(O["O010"])[2])
    rec("B bare number O005 (VLM 403, no department) does not count", not EP.reading_of(O["O005"])[2])
    te = float(O["O007"]["wall"]) - run.T0 + 15.0
    cs = run.cands(te); beta = math.radians(float(np.median([math.degrees(c["beta"]) % 360 for c in cs if c.get("beta") is not None])))
    A = anchored(run, 4, "405", O["O007"], beta) + anchored(run, 4, "407", O["O002"], beta)
    best = {}
    for c in sorted(cs, key=lambda c: c["merit"]):
        best.setdefault(c["floor"], c)
    hyps = [RP.T.Hyp(c["id"], dict(c), 1000 + i, light=c["light"]) for i, c in enumerate(A)] + [RP.T.Hyp(f"E2_{f}F", dict(c), 2000 + f, light=c["light"]) for f, c in best.items()]
    def full(group, name, want, floor_truth=4, te=te, hyps=hyps):
        cs = run.cands(te)
        rows = [r for r in run.R if r[1] <= run.T0 + te]
        ev = EP.evaluate_group(group, [dict(id=h.id, floor=h.floor, rec=h.rec) for h in hyps], SF, rows, None)
        r = dict(e8=ev["verdict"], why=str(ev.get("why"))[:140], cands=[c["id"] for c in ev.get("candidates") or []])
        if ev["verdict"] == "REGION_CANDIDATE_READY":
            sd = out / name; shutil.rmtree(sd, ignore_errors=True); (sd / "e2").mkdir(parents=True)
            (sd / "e2/e2_state.json").write_text(json.dumps(dict(odom_session=run.ST.session().get("odom_session_id"), candidates=cs, candidates_pass=[c for c in cs if c["stage"] == "exact"], per_floor={str(c["floor"]): dict(light=c["light"], status="DONE") for c in cs}, floors_done=sorted({c["floor"] for c in cs if c["stage"] == "exact"}), floors_total=9), default=float))
            st = RP.CutStore(run.ST); st.cut(run.T0 + te)
            tr = RP.Tracker(st, hyps, sd / "e2", run.ST.session().get("odom_session_id")); tr.E3 = E3; tr.experiment_id = run.run
            link = L.E6E7Link(sd, tr, sem_floors=SF, policy="SINGLE_DIAGNOSTIC"); link.max_odom_age_s = None; link.min_windows = 1   # the rule table of one window; the two-window rule has its own test
            link.pair = types.SimpleNamespace(last_eval=ev, group=group, on_e7=lambda o: None, snapshot=lambda: None)
            link.fastlio_degraded = lambda *a, **k: (False, None)
            ids = {c["id"] for c in ev["candidates"]}; cons = [h for h in hyps if h.id in ids]; riv = [h for h in hyps if h.id not in ids][:L.E7_RIVALS]
            link._run([dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:3] + riv], dict(observation_id=f"T_{name}", source="E8_PAIR", pair_observation_ids=[o["observation_id"] for o in group], consistent_ids=[h.id for h in cons]), cons[0].id if cons else None)
            d = json.loads((sd / "e7_state.json").read_text()); sh = json.loads((sd / "e7_semantic_shadow.json").read_text())
            win = next((h for h in hyps if h.id == d["semantic_hypothesis"]), None)
            cr = next((q for q in (d.get("e7") or {}).get("results") or [] if q.get("candidate_id") == d["semantic_hypothesis"]), {})
            r.update(te=round(te), semantic_converged=cr.get("convergence"), xm=sh.get("crossing_margin"), corr=cr.get("correction"), decision=d["global_decision"], path=d.get("path"), hyp=d["semantic_hypothesis"], floor=win.floor if win else None, start_err=run.err(win.rec) if win else None, gap=(d.get("e7") or {}).get("gap_m"), best=(d.get("e7") or {}).get("best"), shadow=sh.get("verdict"), blocked=sh.get("blocked_by"), rejected=sh.get("rejected_by"), evidence=sh.get("floor_evidence"))
            shutil.rmtree(sd, ignore_errors=True)
        rec(name, ok=want(r), **r)
        return r
    P7.semantic_policy = lambda *a, **k: dict(mode="ENABLED", why="same as the enable file")
    full([O["O002"], O["O007"]], "B_407_405_real_first_E7", lambda r: r.get("e8") == "REGION_CANDIDATE_READY" and r.get("floor") == 4 and r["start_err"][0] < 3.0 and "NO_FLOOR_EVIDENCE" not in (r.get("blocked") or []))
    full([O[k] for k in ("O002", "O005", "O007", "O008", "O010", "O012")], "B_all_six_photos_of_the_run", lambda r: r.get("e8") == "REGION_CANDIDATE_READY" and r.get("floor") == 4)
    series = []
    for t2 in (640, 700, 760, 820, 880, 950, 1020, 1060):        # E7 again on new path, as retry_e7 asks in the field; the first pass decides nothing
        r2 = full([O["O002"], O["O007"]], f"B_407_405_E7_retry_at_{t2}", lambda r: r.get("decision") != "GLOBAL_CONFIRMED" or (r.get("floor") == 4 and r["start_err"][0] < 3.0), te=float(t2))
        series.append(dict(te=t2, decision=r2.get("decision"), xm=r2.get("xm"), conv=r2.get("semantic_converged"), best=r2.get("best"), blocked=r2.get("blocked")))
    RESULT["b_retry"] = series
    print("RETRY", json.dumps(series, default=str))
    none = dict(O["O007"], vlm=None)
    full([O["O002"], none], "B_405_without_its_VLM (as in the field: memory refused)", lambda r: r.get("decision") != "GLOBAL_CONFIRMED")
    full([O["O007"], O["O012"]], "B_405_twice_same_plate", lambda r: r.get("decision") != "GLOBAL_CONFIRMED")
    def mis(o, d, room):
        return dict(o, ocr_lines=[[room, .98]], vlm=dict(o["vlm"], room_number=room))
    full([mis(O["O002"], 0, "507"), mis(O["O007"], 0, "505")], "B_wrong_507_505_with_real_plate_text", lambda r: r.get("decision") != "GLOBAL_CONFIRMED")
    # wrong answers that DO get a fitted candidate on the wrong floor, at a time the true pair confirms (+640 s), so only the rules stand between them and a FIX
    def wrong_hyps(fl, rooms):
        B = []
        for lab, ob in zip(rooms, (O["O007"], O["O002"])):
            B += anchored(run, fl, lab, ob, beta)
        return [RP.T.Hyp(c["id"], dict(c), 3000 + i, light=c["light"]) for i, c in enumerate(B)] + [RP.T.Hyp(f"E2_{f}F", dict(c), 2000 + f, light=c["light"]) for f, c in best.items() if f != fl] + [RP.T.Hyp(c["id"], dict(c), 1000 + i, light=c["light"]) for i, c in enumerate(A)]
    for fl in (5, 7, 8):
        ra, rb = f"{fl}05", f"{fl}07"
        if not (SF.spaces(fl, ra) and SF.spaces(fl, rb)):
            ra, rb = f"{fl}05", f"{fl}02"
        if not (SF.spaces(fl, ra) and SF.spaces(fl, rb)):
            print("skip", fl); continue
        H = wrong_hyps(fl, (ra, rb))
        o7, o2 = mis(O["O007"], 0, ra), mis(O["O002"], 0, rb)
        full([o2, o7], f"F_wrong_{rb}_{ra}_real_plate_text", lambda r: r.get("decision") != "GLOBAL_CONFIRMED", te=640.0, hyps=H)
        full([dict(o2, vlm=dict(o2["vlm"], text="")), dict(o7, vlm=dict(o7["vlm"], text=""))], f"F_wrong_{rb}_{ra}_numbers_only", lambda r: r.get("decision") != "GLOBAL_CONFIRMED", te=640.0, hyps=H)
    far = [h for h in hyps if h.id.startswith("A2_") or not h.id.startswith("A")]
    full([O["O002"], O["O007"]], "F_true_labels_wrong_place_only (A2: 40-56 m off)", lambda r: r.get("decision") != "GLOBAL_CONFIRMED", te=640.0, hyps=far)
    vo = dict(O["O002"], vlm=dict(O["O002"]["vlm"], room_number="507"))
    rec("B VLM-only 507 with 로봇공학과 is refused", not EP.reading_of(vo)[2])
def scenario_a_semantic():
    run = Run("X20261103_000434"); O = observations(run)
    t = {k: EP.reading_of(o) for k, o in O.items()}
    rows = [dict(obs=k, ocr=o.get("ocr_lines"), vlm=o.get("vlm"), labels=t[k][0], decisive=t[k][2]) for k, o in sorted(O.items())]
    print(json.dumps(rows, ensure_ascii=False)); RESULT["a_table"] = rows
    best = {}
    for c in sorted(run.cands(620.0), key=lambda c: c["merit"]):
        best.setdefault(c["floor"], c)
    hy = [dict(id=f"E2_{f}F", floor=f, rec=c) for f, c in best.items()]
    ev = EP.evaluate_group([O["O003"], O["O004"]], hy, SF, [r for r in run.R if r[1] <= run.T0 + 620.0], None)
    rec("A 402 photographed twice is one place, not a pair", ev["verdict"] != "REGION_CANDIDATE_READY" and (ev.get("n_valid_independent") or 0) < 2, verdict=ev["verdict"], why=str(ev.get("why"))[:160])


def long_case(name, runid, tes, want):
    run = Run(runid); hist = []; res = []
    for te in tes:
        cs = run.cands(te)
        pf = {str(f): dict(light=lf(f), status="DONE") for f in range(1, 10)}
        done = sorted({c["floor"] for c in cs if c["stage"] == "exact"})
        n_ref = len({p["wall"] // 1 for p in run.P if p.get("kind") == "refresh_start" and p["wall"] <= run.T0 + te})
        e2 = dict(per_floor=pf, coarse=dict(per_floor=pf), candidates_pass=[c for c in cs if c["stage"] == "exact"], candidates_coarse=[c for c in cs if c["stage"] == "coarse"], candidates_refresh=[c for c in cs if c["stage"] == "refresh"], floors_done=done, floors_total=9, refresh=dict(n_passes=n_ref))
        LC = {}
        def light(p):
            if p not in LC:
                LC[p] = E3.LightMap(p)
            return LC[p]
        r = EL.evaluate(run.ST, e2, [], run.T0 + te, light)
        r["floors"] = [dict(x, exact_done=x["exact_done"]) for x in r["floors"]]
        hist.append(r); c = EL.classify(hist, 9)
        lead = r["floors"][0]
        rr = dict(t=te, status=c["status"], why=c.get("why"), lead=lead["floor"], gap=c.get("next_gap_m"), close=c.get("close_floors"), scale=lead.get("scale_rel"), pos_rival=lead.get("pos_rival"), path=r["path_m"], stage=lead["stage"], top=[(x["floor"], x["stage"][0], round(x["cost"], 3)) for x in r["floors"][:4]])
        if lead["floor"] == run.GT["floor"]:
            px, py, h = E3.pose_in_map(dict(lead, light=None), run.R[0][2], run.R[0][3], run.R[0][4]) if lead.get("anchor") else (None, None, None)
            rr["start_err_m"] = None if px is None else round(math.hypot(px - run.GT["px"], py - run.GT["py"]) / 24.0, 2)
        res.append(rr); print("   ", json.dumps(rr, ensure_ascii=False, default=str)[:420], flush=True)
    rec(name, ok=want(res), last=res[-1])
    return res
def main(out):
    out = Path(out); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    if "--no-b" not in sys.argv:
        scenario_b(out)
        scenario_a_semantic()
    if "--no-long" not in sys.argv:
        A = long_case("A X20261103_000434 structure-rich: E7 comparison, no 9/9 wait", "X20261103_000434", [300, 420, 540, 600, 660, 720], lambda r: all(x["status"] != "WOULD_CONFIRM" or (x["lead"] == 4 and (x.get("start_err_m") or 99) < 3) for x in r))
        C = long_case("C X20261103_003729 ambiguous to features: no structure confirm on a wrong place", "X20261103_003729", [240, 300, 360, 385], lambda r: all(x["status"] != "WOULD_CONFIRM" for x in r))
        B2 = long_case("B2 X20261103_001810 corridor: structure stays unconfirmed", "X20261103_001810", [600, 800, 1000], lambda r: all(x["status"] != "WOULD_CONFIRM" for x in r))
    (out / "result.json").write_text(json.dumps(RESULT, ensure_ascii=False, default=str, indent=0))
    print("ALL_OK" if RESULT["ok"] else "SOME_FAILED", flush=True)
if __name__ == "__main__":
    main(sys.argv[1])
