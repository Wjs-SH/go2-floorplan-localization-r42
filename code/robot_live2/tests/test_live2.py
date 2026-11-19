"""LIVE2 unit tests (no ROS, no GPU, no device, no network beyond 127.0.0.1 for the web test).

Run: tools/run_live2_tests.sh   (student venv python, read-only; tmp under the LIVE2 runtime)
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

W = Path(__file__).resolve().parents[1]
CAND = W / "candidate"
sys.path.insert(0, str(CAND / "live2"))
sys.path.insert(0, str(CAND / "live1"))
sys.path.insert(0, str(CAND / "external" / "sci_live" / "realtime"))
sys.path.insert(0, str(CAND / "external" / "sci_live" / "floor_id" / "localizer"))


# ------------------------------------------------------------------ wallfit_cpu_fast: bit identity
class _L:
    def __init__(self, seed=0, H=300, Wd=420, step=11):
        from scipy.ndimage import distance_transform_edt
        rs = np.random.RandomState(seed)
        wall = rs.rand(H, Wd) > 0.993
        wall[:, :3] = True
        self.DFpx = distance_transform_edt(~wall)
        self.H, self.W = H, Wd
        gx, gy = np.meshgrid(np.arange(5, Wd, step), np.arange(5, H, step))
        self.Pf = np.c_[gx.ravel(), gy.ravel()].astype(float)


def _orig(L, pts, hd, ppm):
    c, s = np.cos(hd), np.sin(hd)
    ex = L.Pf[:, 0][:, None] + ((pts[:, 0] * c - pts[:, 1] * s) * ppm)[None]
    ey = L.Pf[:, 1][:, None] + (-(pts[:, 0] * s + pts[:, 1] * c) * ppm)[None]
    ci = np.clip(np.round(ex).astype(int), 0, L.W - 1)
    ri = np.clip(np.round(ey).astype(int), 0, L.H - 1)
    dd = np.clip(L.DFpx[ri, ci], 0, 10)
    return -(np.sort(dd, 1)[:, :int(ex.shape[1] * 0.7)].mean(1) / 5) ** 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wallfit_fast_bit_identical(seed):
    import wallfit_cpu_fast as FW
    L = _L(seed)
    rs = np.random.RandomState(10 + seed)
    for n in (37, 400, 401):
        pts = rs.randn(n, 2) * 4
        for hd in (0.0, 1.3, -2.9):
            for ppm in (15.0, 24.0, 55.5):
                assert np.array_equal(FW.wallfit(L, pts, hd, ppm), _orig(L, pts, hd, ppm))
    assert FW.STATS["exact_table"] is True and FW.STATS["fallback_calls"] == 0


def test_wallfit_fast_falls_back_when_table_not_exact():
    import wallfit_cpu_fast as FW
    L = _L(3)
    L.DFpx = L.DFpx + 1e-7                    # no longer sqrt(integer): the exact table must refuse, not approximate
    pts = np.random.RandomState(0).randn(50, 2) * 3
    before = FW.STATS["fallback_calls"]
    assert np.array_equal(FW.wallfit(L, pts, 0.4, 20.0), _orig(L, pts, 0.4, 20.0))
    assert FW.STATS["fallback_calls"] == before + 1


# ------------------------------------------------------------------ e2_feed: frame chain + session
def _rot_z(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def test_feed_recovers_base_frame_points_exactly(tmp_path):
    import e2_feed as EF
    from live_store import LiveStore
    R_ext = np.array([-0.747, -0.652, 0.129, -0.661, 0.706, -0.256, 0.076, -0.276, -0.958]).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_ext); R_ext = U @ Vt                      # proper rotation like the real yaml (det +1)
    T_ext = np.array([0.171, 0.0, -0.0908])
    feed = EF.E2Feed(tmp_path / "store", (R_ext, T_ext), sample_dt=0.0, merge=1, live_store_cls=LiveStore)
    # a wall 3 m in front of base_link (x forward), 1 m high band, a second wall 12 m to the left (far)
    ys, zs = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-0.1, 1.2, 14))
    wall = np.c_[np.full(ys.size, 3.0), ys.ravel(), zs.ravel()]
    xs, zs2 = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(0.3, 1.5, 10))
    far = np.c_[xs.ravel(), np.full(xs.size, 12.0), zs2.ravel()]
    pb = np.vstack([wall, far])
    Rbl, tbl = EF.STATIC_BASE_LIDAR[:3, :3], EF.STATIC_BASE_LIDAR[:3, 3]
    p_lidar = (pb - tbl) @ Rbl                                            # inverse of base <- lidar
    p_imu = p_lidar @ R_ext.T + T_ext                                     # FAST-LIO body = R_ext p_lidar + T_ext
    R_w = _rot_z(0.7) @ np.array([[1, 0, 0], [0, math.cos(0.05), -math.sin(0.05)], [0, math.sin(0.05), math.cos(0.05)]])
    t_w = np.array([5.0, -2.0, 0.3])
    p_w = p_imu @ R_w.T + t_w
    st = 1_800_000_000_000_000_000
    feed.on_odom(st, t_w, _quat(R_w))
    assert feed.on_cloud(st, p_w.astype(np.float32))
    sid = feed.sample()
    assert sid is not None
    d = np.load(tmp_path / "store" / "samples" / sid / "data_short.npz")
    cells = d["a1_3_1_all_candidate_xy"]
    assert len(cells) > 10 and np.all(np.abs(cells[:, 0] - 3.0) < 0.09) and np.all(np.abs(cells[:, 1]) < 1.09)
    fz = np.load(tmp_path / "store" / "far_points.npz")[sid]
    assert len(fz) >= 10 and np.allclose(fz[:, 1], 12.0, atol=1e-3)       # far wall kept at its true base position
    row = (tmp_path / "store" / "odom_link.csv").read_text().splitlines()[1].split(",")
    yaw = math.atan2(R_w[1, 0], R_w[0, 0])
    assert abs(float(row[2]) + 5.0) < 1e-6 and abs(float(row[3]) + 2.0) < 1e-6 and abs(float(row[4]) + yaw) < 1e-6   # (-x, y, -yaw)


def test_feed_needs_exact_stamp_pose_and_resets_on_session_change(tmp_path):
    import e2_feed as EF
    from live_store import LiveStore
    feed = EF.E2Feed(tmp_path / "s", (np.eye(3), np.zeros(3)), sample_dt=0.0, merge=1, live_store_cls=LiveStore)
    feed.on_odom(1_000_000_000, (0, 0, 0), (0, 0, 0, 1))
    pts = np.c_[np.full(200, 3.0), np.linspace(-1, 1, 200), np.linspace(0, 1, 200)].astype(np.float32)
    feed.on_cloud(1_100_000_000, pts)                          # no odometry with this stamp: never interpolated
    assert feed.sample() is None and feed.stats["no_pose"] == 1
    s1 = feed.session["odom_session_id"]
    feed.on_odom(2_000_000_000, (0, 0, 0), (0, 0, 0, 1), session_changed=True)
    assert feed.session["odom_session_id"] != s1 and feed.stats["reset"] == 2


# ------------------------------------------------------------------ E3 tracker
def _light(tmp, H=200, Wd=300):
    from scipy.ndimage import distance_transform_edt
    wall = np.zeros((H, Wd), bool)
    wall[50, 20:280] = True; wall[150, 20:280] = True                     # a corridor along x between y=50 and y=150
    d = tmp / "light"; d.mkdir()
    q = lambda a: np.minimum(np.rint(a * 4), 65535).astype(np.uint16)   # noqa: E731
    np.save(d / "F_q4.npy", q(distance_transform_edt(~wall)))
    np.save(d / "outd_q4.npy", np.zeros((H, Wd), np.uint16))
    np.save(d / "wall_tr.npy", wall)
    np.save(d / "ok.npy", ~wall)
    np.save(d / "wall_real.npy", wall)
    (d / "meta.json").write_text(json.dumps(dict(H=H, W=Wd)))
    return str(d)


def _store(tmp, n, step=1.0, sid0=0, session="S1"):
    from live_store import LiveStore
    root = tmp / "store"
    st = LiveStore(root, reset=not root.exists())
    (root / "session.json").write_text(json.dumps(dict(odom_session_id=session)))
    for i in range(sid0, sid0 + n):
        # robot moves along +x in odom_link frame; walls at +-5 m (left/right)
        xs = np.linspace(-4, 4, 80)
        cells = np.vstack([np.c_[xs, np.full(80, 5.0)], np.c_[xs, np.full(80, -5.0)]]).astype(np.float32)
        st.add(100.0 + i * step, i * 0.6, 0.0, 0.0, cells, np.zeros(len(cells), np.int8), np.zeros((0, 2), np.float32))
    return root


def _e2(tmp, light, cands, session="S1", status="DONE", floors=(1,), done=(1,)):
    d = tmp / "e2"; d.mkdir(exist_ok=True)
    pf = {str(f): dict(light=light, H=200, W=300, status="DONE") for f in floors}
    (d / "e2_state.json").write_text(json.dumps(dict(status=status, pass_id=1, odom_session=session, candidates_pass=cands,
                                                      per_floor=pf, floors_total=len(floors), floors_done=list(done),
                                                      floors_failed={}, floors_pending=[f for f in floors if f not in done])))
    return d


def _cand(floor=1, px=50.0, py=100.0, phi=0.0, merit=0.3, fit_t=100.0, pass_id=1):
    # odom_link (x, y) -> map px: pos + ppm * A(phi) (xy - anchor); with phi=0, beta=0: +x odom -> +x px, +y odom -> -y px
    return dict(floor=floor, px=px, py=py, phi_deg=phi, phi=phi, pos=[px, py], anchor=[0.0, 0.0], beta=0.0, mirror=1, ppm=10.0,
                merit=merit, twoway=-1.0, fit_t=fit_t, pass_id=pass_id, map_version="T", H=200)


def test_e3_evidence_counted_once_and_kept_while_stationary(tmp_path):
    import e3_tracker as E3
    light = _light(tmp_path)
    root = _store(tmp_path, 12)
    e2 = _e2(tmp_path, light, [_cand(), _cand(py=60.0, merit=0.31)])      # right place vs. a pose 4 m off (inside a wall band)
    tr = E3.Tracker(root, e2)
    for _ in range(5):
        tr.tick()
    h0 = tr.H[0]
    n_kf = h0.n_kf
    assert n_kf >= 8 and len(h0.used) == n_kf                             # each keyframe sample counted once
    s_before = h0.score()
    for _ in range(5):                                                    # no new sample (robot stationary)
        tr.tick()
    assert h0.n_kf == n_kf and h0.score() == s_before and h0.status == "ACTIVE"
    good, bad = sorted(tr.H, key=lambda h: h.score())
    assert good.rec["py"] == 100.0 and bad.score() > good.score()


def test_e3_single_weak_observation_does_not_suspend(tmp_path):
    import e3_tracker as E3
    light = _light(tmp_path)
    root = _store(tmp_path, 6)
    e2 = _e2(tmp_path, light, [_cand(), _cand(py=99.0, merit=0.30)])
    tr = E3.Tracker(root, e2)
    for _ in range(3):
        tr.tick()
    h = [x for x in tr.H if x.rec["py"] == 99.0][0]
    best = min(x.score() for x in tr.H)
    h.ev.append(dict(frame_merit=9.0, twoway=None)); h.n_kf += 1          # one terrible frame
    tr.lifecycle()
    assert h.status == "ACTIVE"                                           # needs SUSP_N consecutive worse keyframes


def test_e3_suspended_hypothesis_recovers(tmp_path):
    import e3_tracker as E3
    light = _light(tmp_path)
    root = _store(tmp_path, 4)
    e2 = _e2(tmp_path, light, [_cand(), _cand(py=99.0)])
    tr = E3.Tracker(root, e2)
    tr.tick()
    h = tr.H[1]
    h.status = "SUSPENDED"
    _store(tmp_path, 8, sid0=4)
    for _ in range(4):
        tr.tick()
    assert h.status == "ACTIVE" and h.status_why == "RECOVERED"
    assert any(e["kind"] == "reactivated" for e in tr.events)


def test_e3_session_change_invalidates_everything(tmp_path):
    import e3_tracker as E3
    light = _light(tmp_path)
    root = _store(tmp_path, 5)
    e2 = _e2(tmp_path, light, [_cand()])
    tr = E3.Tracker(root, e2)
    tr.tick()
    assert tr.alive()
    (root / "session.json").write_text(json.dumps(dict(odom_session_id="S2")))
    tr.tick()
    assert not tr.alive() and tr.dropped[-1]["why"] == "ODOM_SESSION_CHANGED" and tr.request["why"] == "ODOM_SESSION_CHANGED"


def test_e3_merges_duplicate_from_research(tmp_path):
    import e3_tracker as E3
    light = _light(tmp_path)
    root = _store(tmp_path, 6)
    e2d = _e2(tmp_path, light, [_cand()])
    tr = E3.Tracker(root, e2d)
    tr.tick()
    # pass 2 finds the same place again (propagated pose identical within 1.5 m): merged, not duplicated
    c2 = _cand(px=50.0 + 10 * 0.6 * 3, fit_t=103.0, pass_id=2)
    c2["anchor"] = [3 * 0.6, 0.0]
    s = json.loads((e2d / "e2_state.json").read_text()); s["candidates_pass"] = [c2]; s["pass_id"] = 2
    (e2d / "e2_state.json").write_text(json.dumps(s))
    tr.tick()
    assert len(tr.alive()) == 1 and tr.alive()[0].passes == [1, 2]


# ------------------------------------------------------------------ E4 states
def _h(i, floor, score, n_kf=10, recent=0.2, pose=(100, 100), hd=0.0, viol=0, fit_t=0.0):
    return dict(id=f"H{i}", floor=floor, status="ACTIVE", score=score, n_kf=n_kf, recent=recent, pose_px=list(pose),
                heading_deg=hd, ppm=10.0, violations=viol, fit_t=fit_t, twoway_mean=-1.0)


def _rows(path_m):
    return [("s%d" % i, float(i), i * path_m / 20.0, 0.0, 0.0) for i in range(21)]


def test_e4_states():
    import e4_ambiguity as E4
    now = 1000.0
    lio = dict(odom=dict(last_rx_age_s=0.1))
    feed = dict(last=dict(wall=now - 0.5))
    e2_full = dict(status="DONE", pass_id=1, floors_total=3, floors_done=[1, 2, 3], floors_failed={})
    e3 = lambda hs, st=20: dict(session="S", hypotheses=hs, events=[], leader_streak_kf=st)  # noqa: E731
    assert E4.classify(None, dict(session=None, hypotheses=[], events=[]), None, None, [], now)["state"] == "INITIALIZING"
    assert E4.classify(dict(status="WAITING_DATA"), e3([]), feed, lio, [], now)["state"] == "INITIALIZING"
    part = dict(status="RUNNING", pass_id=1, floors_total=3, floors_done=[1], floors_failed={}, floors_pending=[2, 3])
    assert E4.classify(part, e3([_h(1, 1, 0.2)]), feed, lio, _rows(10), now)["state"] == "SEARCH_INCOMPLETE"
    amb = E4.classify(e2_full, e3([_h(1, 1, 0.20), _h(2, 3, 0.22)]), feed, lio, _rows(10), now)
    assert amb["state"] == "AMBIGUOUS" and amb["metrics"]["separation"] == "OTHER_FLOOR"
    # same place (duplicate pose) is not a competitor
    dom = E4.classify(e2_full, e3([_h(1, 1, 0.20), _h(2, 1, 0.205, pose=(101, 100)), _h(3, 2, 0.30)]), feed, lio, _rows(3), now)
    assert dom["state"] == "DOMINANT_UNVERIFIED" and dom["metrics"]["competitor_id"] == "H3"
    conf = E4.classify(e2_full, e3([_h(1, 1, 0.20), _h(2, 2, 0.40)]), feed, lio, _rows(12), now)
    assert conf["state"] == "CONFIDENT_GEOMETRY" and any("GEOMETRY ONLY" in r for r in conf["reasons"])
    stale = E4.classify(e2_full, e3([_h(1, 1, 0.20), _h(2, 2, 0.40)]), feed, dict(odom=dict(last_rx_age_s=5.0)), _rows(12), now)
    assert stale["state"] == "RECOVERY_REQUIRED"
    sess = dict(session="S", hypotheses=[], events=[dict(kind="session_changed", wall=now - 1)])
    assert E4.classify(e2_full, sess, feed, lio, [], now)["state"] == "RECOVERY_REQUIRED"


def test_e4_semantic_targets_from_real_map_layout(tmp_path):
    import e4_ambiguity as E4
    root = tmp_path / "sem"
    for f, rn in ((1, "101"), (2, "201")):
        d = root / f"{f}F"; d.mkdir(parents=True)
        (d / f"{f}F_semantic_map.json").write_text(json.dumps(dict(map_meta=dict(image_size=dict(width=300, height=200)),
                                                                   spaces=[dict(id=f"r{rn}", room_number=rn, type="room", name=rn,
                                                                                polygon_px=[[130, 40], [170, 40], [170, 70], [130, 70]],
                                                                                center_px=[150, 55])])))
    sem = E4.SemanticMaps(root, "kci", loc_sizes={1: (300, 200), 2: (300, 200)})
    hyps = [_h(1, 1, 0.2, pose=(150, 100), hd=90.0), _h(2, 2, 0.21, pose=(150, 100), hd=90.0)]
    walls = {1: np.zeros((200, 300), bool), 2: np.zeros((200, 300), bool)}
    need = E4.semantic_need(dict(state="AMBIGUOUS"), hyps, sem, walls)
    assert need["needed"] and need["priority"] == "HIGH" and set(need["wanted_labels"]) == {"101", "201"}
    t = need["targets"][0]
    assert t["observability"] == "OBSERVABLE_NOW" and t["discriminates_floors"] and abs(t["bearing_deg"]) < 1.0 and abs(t["dist_m"] - 3.0) < 0.01


# ------------------------------------------------------------------ E4 -> E5 link
def test_link_identical_when_budget_not_binding_and_promotes_when_binding(tmp_path):
    import types
    import e5_e4_link as EL
    calls = []

    def n_tiles(b, tile=100):
        return int(math.ceil((b[2] - b[0]) / tile) * math.ceil((b[3] - b[1]) / tile))

    def plan_budget(regs, max_crops, prev_deferred=(), width=None, tile=100, ctx_per_region=1):
        calls.append(1)
        cost = [n_tiles(b) + 1 for b in regs]
        if sum(cost) <= max_crops:
            return list(range(len(regs))), [], dict(binding=False)
        send, used = [], 0
        for i in range(len(regs)):
            if used + cost[i] <= max_crops:
                send.append(i); used += cost[i]
        return send, [i for i in range(len(regs)) if i not in send], dict(binding=True)
    B = types.SimpleNamespace(plan_budget=plan_budget, n_tiles=n_tiles)
    feed = tmp_path / "e4_feed.json"
    hub = tmp_path / "hub"; hub.mkdir()
    (hub / "index.json").write_text(json.dumps(dict(caminfo=dict(K=[600.0, 0, 320.0, 0, 600.0, 240.0, 0, 0, 1]))))
    feed.write_text(json.dumps(dict(stamp=time.time(), level="AMBIGUOUS", priority="HIGH",
                                    targets=[dict(label="401", floor=4, bearing_deg=-20.0, dist_m=3.0, observability="OBSERVABLE_NOW")])))
    EL.install(B, feed, hub)
    regs = [[0, 0, 90, 90], [100, 0, 190, 90], [520, 100, 610, 190]]        # third region: u=565 -> bearing atan((320-565)/600) = -22 deg
    assert B.plan_budget(regs, 100)[0] == [0, 1, 2]                          # not binding: unchanged
    send, deferred, info = B.plan_budget(regs, 4)                            # binding: only 2 regions fit
    assert 2 in send and info["e4_promoted"] == [2] and EL.LINK.last["promoted_total"] == 1
    feed.write_text(json.dumps(dict(stamp=time.time() - 10, level="AMBIGUOUS", priority="HIGH", targets=[])))
    send, deferred, info = B.plan_budget(regs, 4)                            # stale E4: original order
    assert send == [0, 1] and EL.LINK.last["last_scene"]["why"] == "E4_UNAVAILABLE_OR_STALE"


# ------------------------------------------------------------------ ctl refusals (no approval in this workspace)
def test_ctl_refuses_live_modules_without_approval():
    import subprocess
    ctl = W / "scripts" / "live2_ctl.sh"
    for m in ("fastlio", "lio2", "observe", "hub", "e2", "e34"):
        r = subprocess.run([str(ctl), "start", m, "--run-s", "30"], capture_output=True, text=True, timeout=120, env=dict(os.environ, HOME="/home/unitree"))
        assert "REFUSED" in r.stdout, (m, r.stdout[-400:])
    r = subprocess.run([str(ctl), "start", "e2", "--device", "gpu", "--test-only", "--run-s", "30"], capture_output=True, text=True, timeout=60)
    assert "REFUSED" in r.stdout


def test_e3_tracking_pulls_offset_hypothesis_back_and_never_jumps(tmp_path):
    import e3_tracker as E3
    light = _light(tmp_path)
    root = _store(tmp_path, 14)
    e2 = _e2(tmp_path, light, [_cand(py=101.5)])                          # 0.15 m off the true corridor centre (py=100)
    tr = E3.Tracker(root, e2)
    for _ in range(6):
        tr.tick()
    h = tr.H[0]
    assert h.corr["n"] >= 1 and abs(h.pose[1] - 100.0) < 0.6               # crept back to the centre line
    assert h.corr["xy_m"] <= 0.2 * math.sqrt(2) * h.corr["n"] + 1e-9         # every step bounded by the grid


def test_e4_no_confidence_right_after_a_leader_change():
    import e4_ambiguity as E4
    now = 1000.0
    e2 = dict(status="DONE", pass_id=1, floors_total=3, floors_done=[1, 2, 3], floors_failed={})
    e3 = dict(session="S", hypotheses=[_h(1, 1, 0.20), _h(2, 2, 0.40)], events=[], leader_streak_kf=3)
    r = E4.classify(e2, e3, dict(last=dict(wall=now)), dict(odom=dict(last_rx_age_s=0.1)), _rows(12), now)
    assert r["state"] == "DOMINANT_UNVERIFIED" and any("leader stable" in x for x in r["reasons"])


def test_e4_research_pass_keeps_judging_existing_hypotheses():
    import e4_ambiguity as E4
    now = 1000.0
    e2 = dict(status="RUNNING", pass_id=2, pass_why="RESEARCH", floors_total=3, floors_done=[], floors_failed={}, floors_pending=[1, 2, 3])
    e3 = dict(session="S", hypotheses=[_h(1, 1, 0.20), _h(2, 3, 0.21)], events=[], leader_streak_kf=20)
    r = E4.classify(e2, e3, dict(last=dict(wall=now)), dict(odom=dict(last_rx_age_s=0.1)), _rows(12), now)
    assert r["state"] == "AMBIGUOUS" and "research_running" in r["metrics"]


def test_e2_waits_for_motion_and_operator_pause(tmp_path):
    """Field 2026-10-18: a stationary window failed calibration on every floor -> E2 now waits for >= 2 m of window motion;
    the operator pause holds new passes and new floor fits."""
    import types
    import e2_worker as E2W
    store, out = tmp_path / "store", tmp_path / "state" / "e2"
    store.mkdir(); out.mkdir(parents=True)

    def rows(step):
        with open(store / "odom_link.csv", "w") as f:
            f.write("sample_id,t_sec,fastlio_x,fastlio_y,fastlio_yaw_rad\n")
            for i in range(61):
                f.write(f"S{i},{100 + 1.5 * i},{step * i},0,0\n")
    co = types.SimpleNamespace(store=str(store), W=60.0, out=out, pass_id=0, state={},
                               a=types.SimpleNamespace(min_span=70.0, min_window_path_m=2.0, min_pass_gap_s=120.0))
    co.span = lambda: (100.0, 190.0)
    co.paused = lambda: E2W.Coordinator.paused(co)
    co.window_path_m = lambda t1: E2W.Coordinator.window_path_m(co, t1)
    rows(0.0005)                                                     # 4 cm in the window: robot standing
    why, note = E2W.Coordinator.want_pass(co, 0.0)
    assert why is None and note.startswith("WAITING_MOTION")
    rows(0.2)                                                        # 8 m in the window
    assert E2W.Coordinator.want_pass(co, 0.0)[0] == "INITIAL"
    (out.parent / "control").mkdir()
    (out.parent / "control" / "pause.json").write_text(json.dumps(dict(paused=True)))
    assert E2W.Coordinator.want_pass(co, 0.0) == (None, "PAUSED_BY_OPERATOR")


def test_e2_wait_status_accepts_every_note_type():
    """2026-10-19 field: want_pass() returns the store span as a float while the 60 s window fills; the idle status
    code called .startswith on it and E2 exited rc=1 two seconds after start."""
    import e2_worker as E2W
    assert E2W.wait_status(12.3, 0) == "WAITING_DATA"
    assert E2W.wait_status(None, 0) == "WAITING_DATA" and E2W.wait_status(None, 2) == "IDLE"
    assert E2W.wait_status("PAUSED_BY_OPERATOR", 1) == "PAUSED"
    assert E2W.wait_status("WAITING_MOTION path 0.4 m < 2.0 m", 0) == "WAITING_MOTION"
    assert E2W.wait_status("REQUEST_RATE_LIMITED", 3) == "IDLE"


def test_e2_pass_status_interrupted_is_not_failed():
    """2026-10-19 21:35 field: experiment end stopped pass 1 after 68.5 s (0 done, 0 failed) -> was labelled FAILED."""
    import e2_worker as E2W
    assert E2W.pass_status([], {}, True) == "INTERRUPTED"
    assert E2W.pass_status([1, 2], {}, True) == "PARTIAL_INTERRUPTED"
    assert E2W.pass_status([1], {}, False) == "DONE"
    assert E2W.pass_status([], {"3": "CalibrationUnavailable"}, False) == "FAILED"
    assert E2W.pass_status([], {}, False) == "NO_RESULT"
