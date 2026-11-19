#!/usr/bin/env python3
"""e7_long.py (R36): whole-path structure comparison of all floors. RECORD AND DISPLAY ONLY: no path to e7_state or a FIX.

Why it exists. E7's 30 s windows in a straight corridor say nothing (11-02 loc1: 6 windows, all 'runner-up within', best changing). Over the whole path so far the true floor leads steadily
(4 loc1 runs, LONG_STRUCTURE.md). What this module must not hide: the lead is real only against rivals that had a fair search. With a repeat search giving every floor a candidate from the same path,
4F, 5F, 7F, 8F came within 0.01-0.02 m at +150..+210 s (11-01 loc1 F4 0.044, F7 0.047, F8 0.051). So the status says which floors are not excluded, and 'WOULD_CONFIRM' needs exact rivals on all floors.

One evaluation runs in its own process at low priority, never on E3's core (cores() says which). Candidates per floor: the best E2 candidate (exact and coarse) and any live E3 hypothesis on that floor, by long-window cost;
each is registered with the unchanged E7 code. GT is never an input. Status:
  WAITING_MOTION / WAITING_E2   not enough to compare
  LEADER_UNRESOLVED             a leader, but some floors have only coarse candidates (their real cost may be much lower: F8 coarse 0.154 -> exact 0.032)
  AMBIGUOUS_FLOORS              exact rivals within MARGIN_M: the LiDAR does not separate them
  WOULD_CONFIRM                 all floors exact, leader first by >= MARGIN_M in N_RUN evaluations with >= NEW_PATH_M new path between them. SHADOW: what a structure rule would do, not a FIX
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP_S, MIN_PATH_M, NEW_PATH_M, MARGIN_M, N_RUN, MAX_FRAMES = 40.0, 15.0, 5.0, 0.05, 2, 90
TOP_K = 1
MIN_REFRESH = 2
POS_RIVAL_M, SCALE_TOL = 3.0, 0.15


def scale_rel(floor, ppm):
    """The candidate's px/m against the floor's: relative error or None. The prior is an estimate (semantic_scale_v1: stacked core; others the semantic map), so it is wide. X20261103_003729 led with 4F 33.1 px/m (prior 23.96), about 29 m off."""
    import auto_fsm as AF
    p = AF.scale_prior(floor)
    return None if not p or not ppm else round(ppm / p - 1.0, 3)


def atomic(p, obj):
    p = Path(p)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str))
    os.replace(tmp, p)


def recentered(c, fr, cr):
    """The corrected transform as a new start: the newest frame projects to the corrected pose. (The same record shape; used for the rivals' fairness, never as a result by itself.)"""
    import e3_tracker as E3
    tn, xn, yn, yawn, _ = fr[-1]
    px0, py0, _h = E3.pose_in_map(dict(c, light=None), xn, yn, yawn)
    return dict(c, pos=[px0 + cr["dx_m"] * c["ppm"], py0 - cr["dy_m"] * c["ppm"]], anchor=[xn, yn], phi=c["phi"] + cr["dphi_deg"], ppm=c["ppm"] * cr["scale"])


def evaluate(store, e2_state, hyps, t_end, light_of):
    """One whole-path comparison. -> dict(path_m, floors=[...best first]). store: e3_tracker.Store; hyps: [{id, floor, rec}] from E3; light_of: light dir -> LightMap."""
    import e6e7_link as LK
    E7 = LK.E7
    rows = [r for r in store.rows if r[1] <= t_end]
    if len(rows) < 5:
        return None
    fr = E7.frames(store, t_end, t_end - rows[0][1], max_frames=MAX_FRAMES, structural=True)
    path = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(rows, rows[1:]))
    cands = {}
    pf_x, pf_c = e2_state.get("per_floor") or {}, (e2_state.get("coarse") or {}).get("per_floor") or {}
    for stage, lst, pf in (("exact", e2_state.get("candidates_pass") or [], pf_x), ("coarse", e2_state.get("candidates_coarse") or [], pf_c), ("refresh", e2_state.get("candidates_refresh") or [], pf_c)):
        for c in lst:
            d = (pf.get(str(c["floor"])) or {}).get("light")
            if d:
                cands.setdefault(int(c["floor"]), []).append(dict(c, light=d, stage=stage))
    for h in hyps or []:
        cands.setdefault(int(h["floor"]), []).append(dict(h["rec"], light=h["light"], stage=h.get("stage") or "e3", hyp=h["id"]))
    out, per_order = [], {}
    done = {int(f) for f in e2_state.get("floors_done") or []}
    for f, cl in sorted(cands.items()):
        L = light_of(cl[0]["light"])
        sc = sorted((E7.cost(L, c, fr, (0, 0, 0, 1))[0], i) for i, c in enumerate(cl))
        per_order[f] = [i for _, i in sc]
        best = None
        reg = [i for _, i in sc[:TOP_K]]             # R40: + the floor's best exact fit: before registration the whole-path cost can prefer a far wrong place (X20261103_003729 4F: 28.9 m off chosen, 2.2 m never registered)
        ex = [i for i, c in enumerate(cl) if c.get("stage") == "exact"]
        if ex:
            j = min(ex, key=lambda i: float(cl[i].get("merit") if cl[i].get("merit") is not None else 9e9))
            if j not in reg:
                reg.append(j)
        for i in reg:
            r = E7.register(L, cl[i], fr)
            for _ in range(2):                      # R37/R40: a candidate stopped by the search bound continues from there, or a rival looks worse than it is (same as the fair replay)
                if not r.get("on_bound"):
                    break
                cl[i] = recentered(cl[i], fr, r["correction"])
                r = E7.register(L, cl[i], fr)
            e = dict(floor=f, stage=cl[i]["stage"], exact_done=f in done, cost=r["alignment_score"], inlier=r["inlier_ratio"], converged=r["convergence"], on_bound=r["on_bound"], crossing=(r.get("wall_crossing") or {}).get("violation"),
                     correction=r["correction"], hyp=cl[i].get("hyp"), pos=cl[i].get("pos"), phi=cl[i].get("phi"), ppm=cl[i].get("ppm"), anchor=cl[i].get("anchor"), beta=cl[i].get("beta"), mirror=cl[i].get("mirror"), _i=i, scale_rel=scale_rel(f, float(cl[i].get("ppm") or 0) * float((r["correction"] or {}).get("scale") or 1.0)))
            if best is None or e["cost"] < best["cost"]:
                best = e
        out.append(best)
    out.sort(key=lambda e: e["cost"])
    if out:                                         # R40: the floor can be right and the place wrong (X20261102_210840: cost 0.000, 94 m off). The same floor elsewhere must lose too
        import e3_tracker as E3
        lead = out[0]
        lc, tn = cands[lead["floor"]], fr[-1]
        px0, py0, h0 = E3.pose_in_map(dict(lc[lead["_i"]], light=None), tn[1], tn[2], tn[3])
        ppm0 = float(lc[lead["_i"]]["ppm"])
        lead["pos_rival"] = None
        for i in per_order[lead["floor"]]:
            if i == lead["_i"]:
                continue
            px, py, h = E3.pose_in_map(dict(lc[i], light=None), tn[1], tn[2], tn[3])
            if math.hypot(px - px0, py - py0) / ppm0 > POS_RIVAL_M or abs(math.degrees((h - h0 + math.pi) % (2 * math.pi) - math.pi)) > 30.0:
                rr = E7.register(light_of(lc[i]["light"]), lc[i], fr)
                lead["pos_rival"] = dict(cost=rr["alignment_score"], gap=round(rr["alignment_score"] - lead["cost"], 4), dist_m=round(math.hypot(px - px0, py - py0) / ppm0, 1), converged=rr["convergence"])
                break
    return dict(t_end=t_end, path_m=round(path, 1), n_frames=len(fr), floors=out, floors_total=e2_state.get("floors_total"), floors_exact=sorted(done), n_refresh=int((e2_state.get("refresh") or {}).get("n_passes") or 0))


def classify(hist, total):
    """hist = evaluations so far, oldest first. -> dict(status, leader, why, close_floors, ...). Pure."""
    if not hist:
        return dict(status="WAITING_E2", why="no evaluation yet")
    e = hist[-1]
    fl = e["floors"]
    if not fl:
        return dict(status="WAITING_E2", why="E2 has no candidates yet")
    lead = fl[0]
    close = [x["floor"] for x in fl[1:] if x["cost"] - lead["cost"] < MARGIN_M]
    run = 0
    for h in reversed(hist):
        if h["floors"] and h["floors"][0]["floor"] == lead["floor"]:
            run += 1
        else:
            break
    base = dict(leader_floor=lead["floor"], leader_cost=lead["cost"], leader_stage=lead["stage"], next_gap_m=None if len(fl) < 2 else round(fl[1]["cost"] - lead["cost"], 4), close_floors=close, leader_run=run,
                path_m=e["path_m"], floors_exact=len(e.get("floors_exact") or []), floors_total=total)
    fair = lambda x: x["exact_done"] or x["stage"] == "refresh"   # noqa: E731  R40: a repeat search on the newest window is a fair rival; waiting for every exact fit took 672 s in X20261103_000434 (operator ended 6 s before 9/9)
    missing = [x["floor"] for x in fl if not fair(x)]
    if close and not missing:
        return dict(base, status="AMBIGUOUS_FLOORS", why=f"{lead['floor']}F leads, but {close}F are within {MARGIN_M} m: the LiDAR does not separate them")
    if missing or len(fl) < int(total or 0):
        return dict(base, status="LEADER_UNRESOLVED", why=f"{lead['floor']}F leads; floors without a fair search yet: {missing}")
    if int(e.get("n_refresh") or 0) < MIN_REFRESH:         # R37: E2 searches one early window. 11-02: F7 long-window 0.145 from its exact/coarse candidates, 0.024 from a repeat search; 4F 0.028
        return dict(base, status="LEADER_UNRESOLVED", why=f"{lead['floor']}F leads; repeat searches of the other floors {e.get('n_refresh') or 0} of {MIN_REFRESH}: their real cost may be much lower")
    sr = lead.get("scale_rel")
    if sr is not None and abs(sr) > SCALE_TOL:
        return dict(base, status="LEADER_UNRESOLVED", why=f"{lead['floor']}F leads with a map scale {sr:+.0%} from that floor's: the place cannot be right")
    pr = lead.get("pos_rival")
    if pr and pr.get("gap") is not None and pr["gap"] < MARGIN_M:
        return dict(base, status="LEADER_UNRESOLVED", pos_rival=pr, why=f"{lead['floor']}F leads, but the same floor {pr['dist_m']} m away fits within {pr['gap']} m: the floor may be right, the place not")
    good = [h for h in hist if h["floors"] and h["floors"][0]["floor"] == lead["floor"] and len(h["floors"]) > 1 and h["floors"][1]["cost"] - h["floors"][0]["cost"] >= MARGIN_M and all(fair(x) for x in h["floors"])]
    if len(good) >= N_RUN and good[-1]["path_m"] - good[0]["path_m"] >= NEW_PATH_M and hist[-1] is good[-1]:
        return dict(base, status="WOULD_CONFIRM", why=f"{lead['floor']}F first by >= {MARGIN_M} m against exact rivals in {len(good)} evaluations. SHADOW: not a FIX")
    return dict(base, status="LEADER_UNRESOLVED", why="more path needed")


class LongStructure:
    """Runs inside e34: starts one worker at a time, reads its result, writes state/e7_long.json(l)."""

    def __init__(self, state_dir, tracker, clock=time.time, py=None):
        self.state, self.tr, self.clock = Path(state_dir), tracker, clock
        self.py = py or sys.executable
        self.p, self.t_last, self.hist, self.path_last = None, 0.0, [], 0.0
        self.d = dict(kind="LIVE2_E7_LONG", status="WAITING_MOTION", authority="NONE: record and display only, cannot produce a FIX", why="waiting for motion")

    def tick(self, now=None):
        now = self.clock() if now is None else now
        if os.environ.get("LIVE2_E7_LONG", "1") != "1":
            return None
        if self.p is not None:
            if self.p.poll() is None:
                return self.d
            self._collect(now)
        rows = getattr(self.tr.store, "rows", None)
        due = getattr(self, "_now_wanted", False)
        if not rows or (now - self.t_last < STEP_S and not due):
            return self.d
        path = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(rows, rows[1:]))
        if not due and (path < MIN_PATH_M or (self.hist and path - self.path_last < 3.0)):
            self.d.update(status="WAITING_MOTION", why=f"path {path:.0f} m (needs {MIN_PATH_M:.0f} m and 3 m of new path)") if not self.hist else None
            return self.d
        e2p = Path(self.tr.e2_dir) / "e2_state.json"
        try:
            e2 = json.loads(e2p.read_text())
        except (OSError, ValueError):
            return self.d
        if e2.get("odom_session") != getattr(self.tr, "session", None) or not (e2.get("candidates_coarse") or e2.get("candidates_pass")):
            return self.d
        hyps = [dict(id=h.id, floor=h.floor, rec=dict(h.rec, light=None), light=h.light_dir, stage=h.stage) for h in sorted(self.tr.alive(), key=lambda h: h.rank_key())[:9]]
        job = self.state / "e7_long_job.json"
        res = self.state / "e7_long_result.json"
        atomic(job, dict(store=str(self.tr.store.root), e2_state=str(e2p), hyps=hyps, t_end=rows[-1][1], out=str(res), started_wall=now))
        self.t_last, self.path_last, self._now_wanted = now, path, False
        self.p = subprocess.Popen(["taskset", "-c", self.cores(), "nice", "-n", "5", self.py, "-s", "-B", str(Path(__file__).resolve()), str(job)], stdout=subprocess.DEVNULL, stderr=open(self.state / "e7_long.err", "ab"), env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        return self.d

    def request_now(self):
        """final_verify: one evaluation as soon as the worker is free, at rest (the newest LiDAR, no new-path condition)."""
        self._now_wanted = True

    def latest_after(self, wall):
        """The newest evaluation that STARTED after `wall`, or None."""
        for h in reversed(self.hist):
            if float(h.get("started_wall") or 0) >= wall:
                return h
        return None

    def cores(self):
        """R39: the E2 cores (0,1 shared with SigLIP, which runs at normal priority and wins them); nice 10, never E3's core. X20261102_210254: 76 s for one comparison of 5 floors on free cores."""
        return "0,1,6,7"

    def _collect(self, now):
        self.p = None
        try:
            r = json.loads((self.state / "e7_long_result.json").read_text())
        except (OSError, ValueError):
            self.d.update(error="no result from the last evaluation")
            atomic(self.state / "e7_long.json", dict(self.d, written_wall=now))
            return
        if not r.get("floors"):
            return
        if self.hist and r["t_end"] == self.hist[-1]["t_end"]:
            return
        self.hist.append(r)
        c = classify(self.hist, r.get("floors_total"))
        self.d = dict(kind="LIVE2_E7_LONG", authority="NONE: record and display only, cannot produce a FIX", n_eval=len(self.hist), sec=r.get("sec"), **c, floors=[{k: x.get(k) for k in ("floor", "stage", "exact_done", "cost", "inlier", "converged", "crossing", "scale_rel", "pos_rival")} for x in r["floors"]])
        atomic(self.state / "e7_long.json", dict(self.d, written_wall=now))
        with open(self.state / "e7_long.jsonl", "a") as f:
            f.write(json.dumps(dict(self.d, wall=now, t_end=r["t_end"]), default=str) + "\n")


def worker(job_path):
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "external" / "sci_live" / "floor_id" / "localizer"))
    import e3_tracker as E3
    t0 = time.time()
    j = json.loads(Path(job_path).read_text())
    st = E3.Store(j["store"]); st.refresh()
    e2 = json.loads(Path(j["e2_state"]).read_text())
    lc = {}

    def light(p):
        if p not in lc:
            lc[p] = E3.LightMap(p)
        return lc[p]
    r = evaluate(st, e2, j["hyps"], j["t_end"], light) or {}
    atomic(j["out"], dict(r, sec=round(time.time() - t0, 1), started_wall=j.get("started_wall")))


if __name__ == "__main__":
    worker(sys.argv[1])
