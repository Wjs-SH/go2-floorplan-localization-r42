"""STAGING ONLY (2026-10-30): E7 PATH A (STRUCTURE_ONLY) through the REAL e6e7_link.structure_tick + e7_rematch.evaluate.
No ROS, no robot, no GPU. Maps = the operational E2 light-map cache (read-only, mmap). LiDAR = SYNTHETIC scans ray-cast on the
4F wall map along a straight walk (then at rest), expressed in odom_link with the E2 candidate transform (exact inverse of
e7_rematch.project). Candidate transforms = real E2 pass candidates of the body-IMU offline run (floor 4 one re-anchored on the
synthetic walk; rivals unchanged). This is a CODE-PATH check (positive control + wrong-FIX prevention), not a field accuracy.
  operational e7_rematch  -> AssertionError inside E7 -> NOT_CONFIRMED always              (the defect)
  staging   true leader   -> 2 E7 windows, best == leader, gap >= 0.05 -> CONFIRMED
  staging   wrong leader  -> E7 best != leader                       -> NOT_CONFIRMED
  staging   twin rival    -> same geometry under another floor id    -> ambiguous / indistinguishable -> NOT_CONFIRMED
  staging   E4 AMBIGUOUS  -> PATH A never starts
Run: python3 -s -B test_e7_structure_fix.py OUT_DIR"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
OP = HOME / "sci_ws/repairs/livefield2_20261018_151606/candidate/live2"
STG = Path(__file__).resolve().parents[1] / "candidate/live2"
LIGHT = HOME / "sci_ws/repairs/livefield2_20261018_151606/runtime/e2_cache/light"
E2S = HOME / "sci_ws_alt_imu_20261029_204740/runs/offline_body_live2_20261030_060840/e2_offline/state/e2/e2_state.json"
DT, WALK_S, TOTAL_S, SPEED = 1.5, 30.0, 150.0, 0.33


def load(code_dir):
    for m in [k for k in sys.modules if k in ("e6e7_link", "e7_rematch", "e3_tracker", "e6_semantic", "e8_pair", "e6_pipeline")]:
        del sys.modules[m]
    sys.path.insert(0, str(code_dir))
    spec = importlib.util.spec_from_file_location("e6e7_link", code_dir / "e6e7_link.py")
    L = importlib.util.module_from_spec(spec); sys.modules["e6e7_link"] = L; spec.loader.exec_module(L)
    import e3_tracker as E3
    sys.path.remove(str(code_dir))
    return L, E3


def _A(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [-s, -c]])


def light_dir(floor):
    return str(next(LIGHT.glob(f"kci_{floor}F_*")))


def build_walk(E3, rec):
    """straight free run in the 4F 'ok' mask from the candidate pose -> odom_link rows + body scans."""
    Lm = E3.LightMap(light_dir(rec["floor"]))
    ok, wall, ppm = np.asarray(Lm.ok), np.asarray(Lm.wall_real) > 0, rec["ppm"]
    p0 = np.array(rec["pos"], float)
    best = None
    for a in np.radians(np.arange(0, 360, 5)):
        u = np.array([math.cos(a), math.sin(a)])
        n = 0
        for k in range(1, int(15 * ppm)):
            x, y = np.rint(p0 + k * u).astype(int)
            if not (0 <= x < Lm.W and 0 <= y < Lm.H and ok[y, x]):
                break
            n = k
        if best is None or n > best[0]:
            best = (n, u)
    run_px = min(best[0] - 0.8 * ppm, WALK_S * SPEED * ppm)
    assert run_px >= 6 * ppm, f"free run only {best[0] / ppm:.1f} m"
    phi, A = math.radians(rec["phi"]), _A(math.radians(rec["phi"]))
    angs = np.radians(np.arange(0, 360, 1.0))
    rays = np.stack([np.cos(angs), np.sin(angs)], 1)
    steps = np.arange(4, int(10 * ppm))
    rows, pts = [], {}
    t0 = 1.0e9
    for i, t in enumerate(np.arange(0, TOTAL_S + 1e-6, DT)):
        c = p0 + best[1] * min(t * SPEED * ppm, run_px)
        xy = np.asarray(rec["anchor"], float) + A @ ((c - p0) / ppm)            # inverse of c = pos + ppm*A(phi)(xy - anchor)
        yaw = 0.3 * math.sin(t / 20.0)                                          # small heading changes, robot frame irrelevant
        P = c[None, None, :] + rays[:, None, :] * steps[None, :, None]
        X, Y = np.rint(P[..., 0]).astype(int), np.rint(P[..., 1]).astype(int)
        inb = (X >= 0) & (X < Lm.W) & (Y >= 0) & (Y < Lm.H)
        hit = np.zeros_like(inb); hit[inb] = wall[Y[inb], X[inb]]
        k = np.where(hit.any(1), hit.argmax(1), -1)
        m = P[np.arange(len(rays))[k >= 0], k[k >= 0]]
        hd = phi + yaw + rec["beta"]
        v = ((m - c) / ppm) @ _A(hd).T                                           # A is an involution: A^-1 = A
        sid = f"S{i:05d}"
        pts[sid] = v * np.array([1.0, rec["mirror"]])
        rows.append((sid, t0 + t, float(xy[0]), float(xy[1]), yaw))
    return rows, pts, dict(run_m=round(run_px / ppm, 2), dir=best[1].tolist())


class Store:
    """growing synthetic LiveStore: a row becomes visible when the replay clock passes its stamp."""
    def __init__(self, rows, pts, clock):
        self.all, self.p, self.clock, self.rows = rows, pts, clock, []

    def session(self):
        return dict(odom_session_id="SYNTH_E7")

    def refresh(self):
        tn = self.clock()
        self.rows = [r for r in self.all if r[1] <= tn]

    def points(self, sid):
        return self.p.get(sid)


class Hyp:
    def __init__(self, hid, rec, k, light=None):
        self.id, self.rec, self.floor, self.light_dir, self.k = hid, rec, rec["floor"], light or light_dir(rec["floor"]), k

    def rank_key(self):
        return (0, 1, self.k)


class Tracker:
    def __init__(self, store, hyps, E3):
        self.store, self.H, self.E3, self._l = store, hyps, E3, {}

    def alive(self):
        return list(self.H)

    def light(self, d):
        if d not in self._l:
            self._l[d] = self.E3.LightMap(d)
        return self._l[d]


def scenario(code_dir, name, leader_of, extra, e4_state, out):
    L, E3 = load(code_dir)
    t_start = time.time()
    clock = lambda: 1.0e9 + 75.0 + (time.time() - t_start)                      # the robot has walked/stood 75 s at t_start
    e2 = json.loads(E2S.read_text())
    cands = sorted(e2["candidates_pass"], key=lambda c: -c["merit"])
    true4 = dict(next(c for c in cands if c["floor"] == 4))
    rows, pts, walk = build_walk(E3, true4)
    rivals = [dict(c) for c in cands if c["floor"] != 4]
    rivals = [next(c for c in rivals if c["floor"] == f) for f in (5, 7, 8)]
    hyps = [Hyp("H_TRUE_F4", true4, 0.30)] + [Hyp(f"H_R{i}_F{r['floor']}", r, 0.40 + 0.01 * i) for i, r in enumerate(rivals)]
    hyps += extra(true4)
    tr = Tracker(Store(rows, pts, clock), hyps, E3)
    st = out / name; st.mkdir(parents=True)
    link = L.E6E7Link(st, tr, policy="SINGLE_DIAGNOSTIC")
    leader = leader_of(hyps)
    e4 = dict(state=e4_state, metrics=dict(best_id=leader.id, best_floor=leader.floor, gap=0.2))
    sa, t0 = None, time.time()
    while time.time() - t0 < 200:
        tr.store.refresh()                                                     # what the e34 loop's E3 tick does every cycle
        sa = link.structure_tick(e4, time.time(), stop=dict(stopped=True, why="SYNTH_AT_REST"))
        if sa is None or sa.get("status") in ("CONFIRMED", "NOT_CONFIRMED"):
            break
        time.sleep(0.5)
    if (st / "structure_path.json").exists():                                  # after a FIX structure_tick returns None: read the record
        sa = json.loads((st / "structure_path.json").read_text())
    e7 = {}
    if (st / "e7_state.json").exists():
        e7 = json.loads((st / "e7_state.json").read_text())
    res = dict(scenario=name, code=str(code_dir), leader=leader.id, e4_state=e4_state, walk=walk,
               status=None if sa is None else sa.get("status"), why=None if sa is None else sa.get("why"),
               passes=[] if sa is None else [{k: p.get(k) for k in ("ok", "best", "final_decision", "gap_m", "n_frames", "e7_s", "independence", "why")}
                                             for p in sa.get("passes", [])],
               global_decision=e7.get("global_decision"), wall_s=round(time.time() - t_start, 1))
    if link.job:
        link.job.join(60)
    return res


def main(out):
    out = Path(out); out.mkdir(parents=True)
    none = lambda t: []
    twin = lambda t: [Hyp("H_TWIN_F7", dict(t, floor=7), 0.31, light=light_dir(4))]   # identical geometry under another floor id
    R = []
    R.append(scenario(OP, "A_operational_true_leader", lambda h: h[0], none, "CONFIDENT_GEOMETRY", out))
    R.append(scenario(STG, "B_staging_true_leader", lambda h: h[0], none, "CONFIDENT_GEOMETRY", out))
    R.append(scenario(STG, "C_staging_wrong_leader", lambda h: h[1], none, "CONFIDENT_GEOMETRY", out))
    R.append(scenario(STG, "D_staging_twin_rival", lambda h: h[0], twin, "CONFIDENT_GEOMETRY", out))
    R.append(scenario(STG, "E_staging_e4_ambiguous", lambda h: h[0], none, "AMBIGUOUS", out))
    exp = dict(A_operational_true_leader=lambda r: r["status"] == "NOT_CONFIRMED" and "AssertionError" in (r["why"] or ""),
               B_staging_true_leader=lambda r: r["status"] == "CONFIRMED" and r["global_decision"] == "GLOBAL_CONFIRMED"
               and len(r["passes"]) == 2 and all(p["ok"] and p["best"] == "H_TRUE_F4" for p in r["passes"]),
               C_staging_wrong_leader=lambda r: r["status"] == "NOT_CONFIRMED" and r["global_decision"] != "GLOBAL_CONFIRMED",
               D_staging_twin_rival=lambda r: r["status"] == "NOT_CONFIRMED" and r["global_decision"] != "GLOBAL_CONFIRMED",
               E_staging_e4_ambiguous=lambda r: r["status"] is None and r["global_decision"] is None)
    for r in R:
        r["expected"] = bool(exp[r["scenario"]](r))
    summ = dict(results=R, all_ok=all(r["expected"] for r in R))
    (out / "e7_structure_result.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    return summ["all_ok"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
