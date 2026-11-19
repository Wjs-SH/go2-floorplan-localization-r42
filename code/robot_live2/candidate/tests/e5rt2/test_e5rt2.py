"""E5RT2 integration tests (ROBOT_CPU_FAKE: real candidate callers with fakes; no device/ROS/model/network/GPU).

Each test names the acceptance row (04_ACCEPTANCE_MATRIX.csv A14..A34) it evidences. They drive the REAL entry points:
GatedNode.loop (async scene path), handle_scene/after_scene, p2_tick, track_frame, start_event -> aim -> device worker
-> capture -> enqueue_reader_job -> ReaderCore -> done -> attach_readings -> store/status.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
HERE = Path(__file__).resolve().parent
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_common", WS / "src" / "sci_pantilt",
          WS / "src" / "sci_text_reading", WS / "external" / "sci_event" / "e5_onboard", HERE.parent / "r1"):
    sys.path.insert(0, str(p))

import r1_contracts as RC  # noqa: E402
import e5_node as N  # noqa: E402
import test_r1_e5 as TE  # noqa: E402  (before e5_onboard: R1 tests capture the SCI Node class at import)
import e5_onboard as OB  # noqa: E402
import stop_policy as SP  # noqa: E402
import r1_jobs as J  # noqa: E402
import r1_fakes_e5 as F  # noqa: E402
import test_r1_reader as TR  # noqa: E402
from event_store import EventStore  # noqa: E402
from timing import P2Policy, load_p2, p2_display  # noqa: E402
from policy import StagePolicy, load_cfg  # noqa: E402
from features import Geometry  # noqa: E402
from e5rt import gates as G  # noqa: E402
from e5rt import observation as OBS  # noqa: E402
from e5rt import snapshot as SN  # noqa: E402
from e5rt.budget import BudgetedSiglip, plan_budget, n_tiles, DEFERRED  # noqa: E402
from e5rt.rt_frontend import BoxAssociator, RtSiglipCache  # noqa: E402
from e5rt.scheduler import VerifyScheduler  # noqa: E402
from e5rt.tracker import LIVE  # noqa: E402

CS, RS, RR = RC.CaptureStatus, RC.ReleaseStatus, RC.ReadReason


# ------------------------------------------------------------------------------------------------ helpers
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "WS", tmp_path / "ws")
    monkeypatch.setattr(N, "SHM", tmp_path / "shm")
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    monkeypatch.setenv("SCI_AIM_TIMEOUT_S", "0.6")
    monkeypatch.setenv("SCI_AIM_CANCEL_GRACE_S", "0.4")
    monkeypatch.delenv("SCI_E5RT2_SCENE_MODE", raising=False)
    monkeypatch.delenv("SCI_ARM_WITHOUT_UI", raising=False)
    monkeypatch.delenv("SCI_E5RT_HOME_WITHOUT_ARM", raising=False)
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    return tmp_path


def write_gate(env, monkeypatch, state="walking", grant=True, estop=None, operator_abort=False, ttl=600.0, **extra):
    gp = env / "kci_gate.json"
    g = dict(id="test-grant", expires_t=time.time() + ttl) if grant else None
    d = G.gate_payload(state, time.time(), grant=g, operator_abort=operator_abort, estop=estop)
    d.update(extra)
    gp.write_text(json.dumps(d))
    monkeypatch.setattr(OB, "GATE", gp)
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    return gp


def gnode(env, console, run="e5rt2", execute=True, **kw):
    clk = TE.Clock()
    planner = OB.Planner(N.Geometry(), console="http://fake-console", execute=execute, http_get=console.http_get)
    deps = dict(frontend=kw.pop("frontend", F.FakeFrontend()), monitor=F.FakeMon(), planner=planner, trial_pointer=None,
                clock=kw.pop("clock", clk), threaded=kw.pop("threaded", False),
                body_speed=lambda now: SP.BodySample(0.0, now, "sportmodestate.velocity_hypot_xy"))
    g = OB.GatedNode(F.args(env, run_name=run, pantilt_execute=execute, **kw), deps=deps)
    return g, clk


def tex(seed=1, h=480, w=640):
    r = np.random.RandomState(seed)
    g = cv2.GaussianBlur((r.rand(h, w) * 255).astype(np.uint8), (5, 5), 0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class LiveSource(F.FakeSource):
    """Real-time camera/odometry stand-in: frames + odom appended by a thread at `hz` (wall-clock stamps)."""

    def __init__(self, hz=15.0, n=200, img=None):
        super().__init__()
        self.hz, self.n, self.img = hz, n, img if img is not None else tex()
        self.odom_session = 0
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        t0 = time.time()
        for i in range(self.n):
            if not self.alive:
                return
            st = t0 + i / self.hz
            with self.lock:
                self.queue.append((st, i, self.img))
                self.odom_hist = (self.odom_hist + [(st, 0.01 * i, 0.0, 0.0)])[-200:]
                self.last_odom_wall = time.time()
            time.sleep(max(0.0, t0 + (i + 1) / self.hz - time.time()))

    def poll(self):
        with self.lock:
            out, self.queue = self.queue, []
        return out


class SlowFrontend(F.FakeFrontend):
    """Scene takes `delay` s on the worker; records the window and the buffer object it was given."""

    def __init__(self, delay=2.0):
        super().__init__(scene_delay_s=delay)
        self.windows, self.bufs = [], []

    def scene(self, t, dimg, pose, allow_stop=True, buf=None):
        t0 = time.monotonic()
        n0 = len(buf.items) if buf is not None else None
        r = super().scene(t, dimg, pose, allow_stop)
        self.windows.append((t0, time.monotonic()))
        self.bufs.append((type(buf).__name__, n0, len(buf.items) if buf is not None else None, len(self.buf.items)))
        if buf is not None:
            r["base"] = buf.newest()
        return r


def max_gap(ts, a, b):
    pts = [a] + sorted(x for x in ts if a <= x <= b) + [b]
    return max(y - x for x, y in zip(pts, pts[1:]))


def run_loop(node, src, seconds):
    th = threading.Thread(target=node.loop, args=(src,), daemon=True)
    th.start()
    time.sleep(seconds)
    src.alive = False
    th.join(timeout=15)
    node.stop_event_thread()
    assert not th.is_alive()


# ======================================================================================= A14 real caller timing
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_A14_blocking_2s_scene_p2_tick_and_tracking_continue_only_in_async(env, monkeypatch, mode):
    """REAL GatedNode.loop with a 2 s scene. async: P2 ticks + owner tracking continue (gap <= 0.35 s) during the scene.
    sync (= E5-RT1 behaviour): both stop for the whole scene (gap >= 1.8 s) -> the defect the async path removes."""
    monkeypatch.setenv("SCI_E5RT2_SCENE_MODE", mode)
    monkeypatch.setattr(OB, "GATE_ON", False)                   # observe always; no grant -> no STOP/physical action
    fe = SlowFrontend(2.0)
    node, _ = gnode(env, F.FakeConsole(), frontend=fe, threaded=True)
    node.clock = time.monotonic
    assert node.scene_mode == mode
    src = LiveSource(hz=15.0, n=120)
    run_loop(node, src, 6.5)
    assert fe.windows, "no scene was computed"
    a, b = fe.windows[0]
    assert b - a >= 1.9
    ticks = [m for m, _ in node.tick_log]
    trk = list(node.track_stats["update_mono"])
    g_tick, g_trk = max_gap(ticks, a + 0.05, b - 0.05), max_gap(trk, a + 0.05, b - 0.05)
    if mode == "async":
        assert g_tick <= 0.35 and g_trk <= 0.35, (g_tick, g_trk)
        assert len([x for x in ticks if a <= x <= b]) >= 10
        kind, n0, n1, live_n = fe.bufs[0]
        assert kind == "FrozenFrames" and n0 == n1               # A15: the snapshot did not change during the scene
        sc = [json.loads(x) for x in (node.run / "scenes.jsonl").read_text().splitlines() if x.strip()]
        assert any(r.get("admission") and r["admission"]["evidence_ok"] for r in sc if not r.get("diagnostic_only"))
    else:
        assert g_tick >= 1.8 and g_trk >= 1.8, (g_tick, g_trk)


# ======================================================================================= A15 snapshot
def test_A15_snapshot_binds_frame_depth_odom_identity_and_is_immutable():
    class Buf:
        def __init__(self):
            self.items, self.keep = [], 4.0

        def nearest_tb(self, t):
            return min(self.items, key=lambda f: abs(f["tb"] - t))

        def nearest_th(self, t):
            return min(self.items, key=lambda f: abs(f["th"] - t))

        def newest(self):
            return self.items[-1] if self.items else None
    b = Buf()
    for i in range(5):
        b.items.append(dict(id=i, tb=i * 0.1, th=10 + i * 0.1, bgr=np.zeros((480, 640, 3), np.uint8)))
    fz = SN.FrozenFrames(b)
    snap = SN.make_snapshot("run", "boot", 3, 7, 0.4, 10.4, fz, 10.39, (1.0, 2.0, 0.1), 10.41)
    b.items.append(dict(id=5, tb=0.5, th=10.5, bgr=np.zeros((480, 640, 3), np.uint8)))
    b.items.pop(0)
    assert [f["id"] for f in fz.items] == [0, 1, 2, 3, 4] and fz.nearest_tb(0.45)["id"] == 4
    d = snap.to_dict()
    assert d["newest_frame_id"] == 4 and d["odom_session"] == 7 and d["generation"] == 3 and d["image_size"] == [640, 480]
    assert d["depth_stamp"] == 10.39 and d["pose_stamp"] == 10.41 and d["boot_id"] == "boot"
    with pytest.raises(Exception):
        snap.generation = 4                                         # frozen dataclass
    with pytest.raises(RuntimeError):
        fz.add(9, 0, 0, None)


# ======================================================================================= A16 / A17 late, generation
def test_A16_admission_late_generation_session_resolution_pose():
    snap = SN.ObservationSnapshot(1, "r", "B", 2, 5, 1.0, 1.0, 1, 1.0, 1.0, (0.0, 0.0, 0.0), 1.0, (640, 480), 100.0, 0.0)
    cur = dict(boot_id="B", generation=2, odom_session=5, image_size=(640, 480))
    a = SN.result_admission(snap, cur, 100.5)
    assert a["evidence_ok"] and a["stop_ok"]
    a = SN.result_admission(snap, cur, 103.0)
    assert a["evidence_ok"] and not a["stop_ok"] and "LATE_RESULT_NO_STOP" in a["reasons"]
    for k, v, why in (("generation", 3, "STALE_GENERATION"), ("odom_session", 6, "STALE_ODOM_SESSION"),
                      ("boot_id", "X", "STALE_BOOT"), ("image_size", (1280, 720), "RESOLUTION_CHANGED")):
        a = SN.result_admission(snap, dict(cur, **{k: v}), 100.1)
        assert not a["evidence_ok"] and not a["stop_ok"] and why in a["reasons"]
    bad = SN.ObservationSnapshot(1, "r", "B", 2, 5, 1.0, 1.0, 1, 1.0, 1.0, (float("nan"), 0.0, 0.0), 1.0, None, 100.0, 0.0)
    assert "NONFINITE_POSE" in SN.result_admission(bad, cur, 100.1)["reasons"]


def test_A16_A17_odom_reset_during_scene_result_not_applied_tracks_invalidated(env, monkeypatch):
    monkeypatch.setenv("SCI_E5RT2_SCENE_MODE", "async")
    monkeypatch.setattr(OB, "GATE_ON", False)
    fe = SlowFrontend(1.5)
    node, _ = gnode(env, F.FakeConsole(), frontend=fe, threaded=True)
    node.clock = time.monotonic
    node.p2.tracks[3] = dict(spot=3, state="VERIFIED", n_verify=2, xy=np.array([2.0, 1.0]), n_xy=None, up=0.0, h_m=None,
                             last_t=0.0, box=[0, 0, 1, 1], tile=[0, 0, 1, 1], z_last=2.0, sim=0.2, t_verified=0.0)
    src = LiveSource(hz=15.0, n=100)

    def reset_odom():
        while not fe.windows and not getattr(fe, "_entered", False):
            if node.scene_worker is not None and node.scene_worker.in_flight():
                break
            time.sleep(0.01)
        src.odom_session = 1                                        # odometry reset while the scene is computing
    threading.Thread(target=reset_odom, daemon=True).start()
    run_loop(node, src, 4.0)
    sc = [json.loads(x) for x in (node.run / "scenes.jsonl").read_text().splitlines() if x.strip()]
    diag = [r for r in sc if r.get("diagnostic_only")]
    assert diag and any("STALE_GENERATION" in r["admission"]["reasons"] or "STALE_ODOM_SESSION" in r["admission"]["reasons"] for r in diag)
    assert node.p2.tracks[3]["state"] == "STALE_ODOM_SESSION"
    st = json.loads((N.SHM / "status.json").read_text())
    assert st["generation"]["invalidated_tracks"] == 1 and "odom_session" in st["generation"]["why"]
    faults = (node.run / "faults.jsonl").read_text()
    assert "SCENE_GENERATION_CHANGED" in faults and "SCENE_RESULT_NOT_APPLIED" in faults


def test_A16_logical_deadline_late_result_is_diagnostic_only(env, monkeypatch):
    monkeypatch.setenv("SCI_E5RT2_SCENE_MODE", "async")
    monkeypatch.setattr(OB, "GATE_ON", False)
    fe = SlowFrontend(1.6)
    node, _ = gnode(env, F.FakeConsole(), frontend=fe, threaded=True, scene_deadline_s=0.5)
    node.clock = time.monotonic
    run_loop(node, LiveSource(hz=15.0, n=90), 4.0)
    faults = (node.run / "faults.jsonl").read_text()
    assert "SCENE_LOGICAL_DEADLINE" in faults and "worker NOT assumed" in faults
    sc = [json.loads(x) for x in (node.run / "scenes.jsonl").read_text().splitlines() if x.strip()]
    assert all(r.get("diagnostic_only") for r in sc) and any("LATE_AFTER_LOGICAL_DEADLINE" in r["admission"]["reasons"] for r in sc)
    assert node.status["scene_worker"]["late_completed"] >= 1


def test_A16_scene_worker_one_in_flight_latest_pending_superseded_counted():
    gate = threading.Event()
    done = []

    def fn(job):
        gate.wait(2.0)
        done.append(job["id"])
        return job["id"]
    w = SN.SceneWorker(fn, deadline_s=5.0)
    w.submit(dict(id=1))
    t0 = time.time()
    while w.in_flight() is None and time.time() - t0 < 2:
        time.sleep(0.01)
    for i in (2, 3, 4):
        w.submit(dict(id=i))                                       # 2 and 3 are superseded by 4 (never computed)
    assert w.in_flight()["id"] == 1 and w.slot.depth() == 2
    gate.set()
    t0 = time.time()
    while len(done) < 2 and time.time() - t0 < 3:
        time.sleep(0.01)
    w.close()
    assert done == [1, 4] and w.counters["superseded"] == 2 and [r["id"] for r in w.drain()] == [1, 4]


def test_A17_generation_change_invalidates_cache_revision_and_marks_tracks(env):
    node, _ = gnode(env, F.FakeConsole())

    class Client:
        model, prompts, want_board = "m", ["p"], True

        def score(self, img, regs):
            n = len(regs)
            self.last_adv, self.last_board, self.last_ceil = [0] * n, [-1] * n, [-1] * n
            self.last_tiles, self.last_emb, self.last_ctx_plate, self.last_neg, self.last_ctx = [list(r) for r in regs], [b"x"] * n, [0] * n, {}, {}
            return np.full(n, .12), 0
    cache = RtSiglipCache(Client(), F.FakeBuf(), scheduler=VerifyScheduler({"model": "m"}), clock=lambda: 1.0)
    cache.sched.store(5, {"aspect": 1.0}, {"sim": .2}, now=1.0)
    node.frontend = F.FakeFrontend()
    node.frontend.vlm = BudgetedSiglip(cache, max_crops=32)
    node.rt_tracker.tracks[5] = dict(id=5, status=LIVE, frame_id=1, box=[0, 0, 10, 10])
    node.scene_generation += 1
    node.on_generation_change("odom_session 0->1")
    assert len(cache.sched.cache) == 0 and cache.sched.revision.get("odom_generation") == 1
    assert node.rt_tracker.tracks[5]["meta"]["odom_generation_changed"] == "odom_session 0->1"
    assert node.status["generation"]["scene_generation"] == 1


# ======================================================================================= A18-A20 tracking
def _scene_res(fid, regions):
    return dict(regions=regions, base=dict(id=fid))


def _region(box, spot=None, sim=0.15, ss="FRESH"):
    return dict(box=list(box), tile=list(box), sim=sim, spot=spot, score_status=ss)


def _fill_buf(node, n, img_fn=lambda i: tex(1)):
    node.frontend = F.FakeFrontend()
    for i in range(1, n + 1):
        im = img_fn(i)
        node.frontend.buf.items.append(dict(id=i, tb=i * 0.1, th=100 + i * 0.1, bgr=im, gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))


def test_A18_A19_candidate_seeds_geometry_only_and_catch_up_bounded_to_tracked_frame(env):
    node, _ = gnode(env, F.FakeConsole())
    shift = lambda i: np.roll(tex(1), 2 * i, axis=1)  # noqa: E731
    _fill_buf(node, 10, shift)
    for it in node.frontend.buf.items[:7]:                         # the owner has tracked up to frame 7
        node.track_frame(it["id"], it["th"], it["bgr"])
    assert node.rt_tracker.prev[1] == 7
    regs = [_region([200, 150, 260, 210], spot=4), _region([400, 200, 450, 250], spot=None)]
    dec = {"A3": [dict(display="CANDIDATE"), dict(display="CANDIDATE")]}
    p2recs = [dict(state="TRACK"), dict(state="VERIFIED")]
    n_verify_before = {k: v.get("n_verify") for k, v in node.p2.tracks.items()}
    node.after_scene(_scene_res(3, regs), dec, p2recs, None, dict(frame_id=3), None)
    tr = node.rt_tracker.tracks
    assert node.track_stats["caught_up_frames"] == 4               # frames 4..7 only (source < f <= newest tracked)
    assert tr[4]["status"] == LIVE and tr[4]["frame_id"] == 7      # unverified candidate keeps geometry
    assert abs(tr[4]["box"][0] - (200 + 2 * 4)) < 1.5               # moved with the image (frame 3 -> 7 = +8 px)
    assert node.sem_by_track[4]["p2_state"] == "TRACK" and node.sem_by_track[4]["source_frame_id"] == 3
    assert {k: v.get("n_verify") for k, v in node.p2.tracks.items()} == n_verify_before   # no semantic evidence added
    neg = [k for k in tr if k < 0]
    assert neg and tr[neg[0]]["frame_id"] == 7
    for it in node.frontend.buf.items[7:]:
        node.track_frame(it["id"], it["th"], it["bgr"])
    assert tr[4]["frame_id"] == 10                                 # continues forward, never back to an older frame


def test_A18_source_frame_evicted_is_not_substituted(env):
    node, _ = gnode(env, F.FakeConsole())
    _fill_buf(node, 5)
    node.after_scene(_scene_res(99, [_region([10, 10, 60, 60], spot=1)]), {"A3": [dict(display="VERIFYING")]},
                     [dict(state="VERIFIED")], None, dict(frame_id=99), None)
    assert node.track_stats["seed_rejected"] == 1 and 1 not in node.rt_tracker.tracks
    assert node.status["e5rt2_seed_last"]["why"] == "SOURCE_FRAME_EVICTED_NOT_SUBSTITUTED"


def test_A20_occlusion_loses_track_and_old_semantics_are_not_shown_or_handed_off(env, monkeypatch):
    node, _ = gnode(env, F.FakeConsole())
    _fill_buf(node, 3)
    node.after_scene(_scene_res(3, [_region([200, 150, 280, 230], spot=9, sim=0.2)]), {"A3": [dict(display="VERIFYING")]},
                     [dict(state="VERIFIED")], None, dict(frame_id=3), None)
    node.track_frame(3, 100.3, tex(1))
    occ = tex(1).copy()
    occ[140:240, 190:290] = 128                                     # occluder (textureless) over the sign
    node.track_frame(4, 100.4, occ)
    assert node.rt_tracker.tracks[9]["status"] != LIVE
    boxes, _ = OB.LR.live_overlay(4, node.rt_tracker.live_on(4), node.sem_by_track, time.monotonic())
    assert boxes == []                                              # nothing drawn as current
    stop = dict(F.a3_stop(box=(200, 150, 280, 230)), spot=9)
    oc = OBS.build_observation("r", "g", "k", stop, dict(node.rt_tracker.tracks[9]), dict(frame_id=4, header_stamp=100.4), None)
    assert oc.roi_box_latest is None and oc.latest_frame_id is None and not oc.geometry_confirmed
    # re-entry of (possibly another) target: a NEW seed with its own source frame; the old semantic age is not reused
    t_old = node.sem_by_track[9]["t_mono"]
    node.frontend.buf.items.append(dict(id=5, tb=0.5, th=100.5, bgr=tex(1), gray=cv2.cvtColor(tex(1), cv2.COLOR_BGR2GRAY)))
    node.track_frame(5, 100.5, tex(1))
    node.after_scene(_scene_res(5, [_region([200, 150, 280, 230], spot=9, sim=0.11)]), {"A3": [dict(display="VERIFYING")]},
                     [dict(state="VERIFIED")], None, dict(frame_id=5), None)
    assert node.sem_by_track[9]["source_frame_id"] == 5 and node.sem_by_track[9]["t_mono"] > t_old
    assert node.rt_tracker.tracks[9]["seed_frame"] == 5


def test_A20_box_associator_lk_none_time_backwards_future_base():
    a = BoxAssociator()
    g = cv2.cvtColor(tex(2), cv2.COLOR_BGR2GRAY)
    buf = type("B", (), {})()
    buf.items = [dict(id=1, th=1.0, gray=g), dict(id=2, th=1.2, gray=g.copy())]
    ids0 = a.assign(buf, buf.items[0], [[100, 100, 150, 150]])
    old = cv2.calcOpticalFlowPyrLK
    try:
        cv2.calcOpticalFlowPyrLK = lambda *x, **k: (None, None, None)
        ids1 = a.assign(buf, buf.items[1], [[100, 100, 150, 150]])
    finally:
        cv2.calcOpticalFlowPyrLK = old
    assert ids1 != ids0 and a.stats["lk_fail"] >= 1                 # explicit failure -> new id (no exception)
    a.assign(buf, dict(id=0, th=0.5, gray=g), [[100, 100, 150, 150]])
    assert a.stats["time_backwards"] == 1
    a.assign(buf, dict(id=9, th=5.0, gray=g), [[100, 100, 150, 150]])
    assert a.stats["future_base"] == 1


# ======================================================================================= A21 / A22 budget + typed defer
def test_A21_budget_counts_tiles_plus_context_and_worker_receives_planned_crops():
    regs = [[0, 0, 300, 200], [300, 0, 640, 200], [0, 250, 90, 330], [500, 300, 560, 380], [100, 300, 400, 470]]
    assert [n_tiles(b) for b in regs] == [6, 8, 1, 1, 6]
    send, deferred, info = plan_budget(regs, 12, width=640)
    assert info["binding"] and info["crops_planned_upper"] <= 12 and set(send) | set(deferred) == set(range(5))
    assert 2 in send and 3 in send                                  # cheap small candidates of both halves first

    class Worker:
        def __init__(self):
            self.received = []

        def score(self, img, rr):
            self.received.append([list(r) for r in rr])
            n = len(rr)
            self.last_n_tiles = sum(n_tiles(b) for b in rr)
            self.last_adv, self.last_board, self.last_ceil = [0.0] * n, [-1.0] * n, [-1.0] * n
            self.last_tiles, self.last_emb, self.last_ctx_plate = [list(r) for r in rr], [b"e"] * n, [0.1] * n
            self.last_neg, self.last_ctx = {"poster": [0.05] * n}, {"poster": [0.01] * n}
            return np.full(n, 0.12), 0.01
    L = type("L", (), {})()
    w = Worker()
    b = BudgetedSiglip(w, legacy_events=L, max_crops=12)
    sim, _ = b.score(np.zeros((480, 640, 3), np.uint8), regs)
    got = w.received[-1]
    assert sum(n_tiles(x) + 1 for x in got) <= 12 and b.counters["tiles_mismatch"] == 0
    st = b.last_score_status
    assert [st[i] for i in deferred] == [DEFERRED] * len(deferred) and all(sim[i] == -1.0 for i in deferred)
    assert all(b.last_tiles[i] is None and b.last_neg["poster"][i] is None for i in deferred)
    assert L.no_evidence_keys == {tuple(float(v) for v in regs[i]) for i in deferred}
    # next scene: the waited regions go first (fair: no starvation of a deferred region)
    b.score(np.zeros((480, 640, 3), np.uint8), regs)
    assert any(r in w.received[-1] for r in [list(map(float, regs[i])) for i in deferred])
    # non-binding budget = byte-identical request (decision-preserving)
    w2 = Worker()
    b2 = BudgetedSiglip(w2, legacy_events=L, max_crops=32)
    b2.score(np.zeros((480, 640, 3), np.uint8), regs)
    assert w2.received[-1] == [list(map(float, r)) for r in regs] and b2.last_score_status == ["FRESH"] * 5 and L.no_evidence_keys == set()


def test_A21_budget_over_cache_wrapper_never_exceeds_and_cache_status_kept():
    class Client:
        model, prompts, want_board = "m", ["p"], True

        def __init__(self):
            self.calls = []

        def score(self, img, rr):
            self.calls.append(len(rr))
            n = len(rr)
            self.last_n_tiles = sum(n_tiles(b) for b in rr)
            self.last_adv, self.last_board, self.last_ceil = [0] * n, [-1] * n, [-1] * n
            self.last_tiles, self.last_emb, self.last_ctx_plate, self.last_neg, self.last_ctx = [list(r) for r in rr], [b"x"] * n, [0] * n, {}, {}
            return np.full(n, .12), 0
    c = Client()
    cache = RtSiglipCache(c, type("B", (), {"items": []})(), scheduler=VerifyScheduler({"model": "m"}, max_fresh=3), clock=lambda: 5.0)
    cache.assoc.assign = lambda buf, base, boxes: list(range(len(boxes)))
    L = type("L", (), {})()
    cache.L = L
    b = BudgetedSiglip(cache, legacy_events=L, max_crops=6)
    regs = [[10 + 60 * i, 10, 60 + 60 * i, 60] for i in range(6)]
    b.score(np.zeros((100, 400, 3), np.uint8), regs)
    assert max(c.calls) <= 3                                         # budget (6 crops = 3 regions) and scheduler (3)
    assert b.last_score_status.count("FRESH") <= 3 and b.last_score_status.count(DEFERRED) >= 3


def test_A22_deferred_is_typed_end_to_end_in_p2_and_stage_policies():
    cfg = load_cfg(str(WS / "config" / "e5_policy.json"))
    geom = Geometry()
    p2 = P2Policy(cfg, load_p2(str(WS / "config" / "e5_policy_p2.yaml")), geom)
    r = dict(box=[100, 100, 140, 140], tile=[100, 100, 140, 140], sim=-1.0, spot=3, xyz=[2, 0, 1.5], z=2.0, msim=0.2, nobs=3,
             nscene=3, board=-1, adv=0, hgt=1.5, fired=False, like=False, neg={}, ctx={}, ctx_plate=None, score_status=DEFERRED)
    f = dict(glare=dict(sat=0.0, spec=0.0, contrast=0.0, sharp=0.0), plane=dict(ok=False), edge=50, speed=0.0, z_tile=2.0, up=None, pantilt=dict(ok=True), text_px=30,
             obliquity_pred=10.0, aim_px=[120, 120])
    recs, stop, _ = p2.step(1.0, (0, 0, 0), 0.0, [r], [f], 0.1, None)
    assert recs[0]["state"] == "DEFERRED" and stop is None and 3 not in p2.tracks
    assert p2_display(recs[0])["display"] == "DEFERRED_BUDGET"
    assert len(p2.gates.hist[3]) == 0                                # no persistence vote from a missing score
    sp = StagePolicy("A3", cfg, geom, a1=True, a2=True, a3=True)
    out, st = sp.step(1.0, [r], [f], 0.1, None)
    assert out[0]["state"] == "DEFER" and out[0]["reasons"] == [DEFERRED] and st is None and len(sp.hist[3]) == 0


# ======================================================================================= A28 / A29 hand-off
def _confirmed_setup(node, z_mm=2000, mixed=False):
    d = np.full((480, 640), z_mm, np.uint16)
    if mixed:
        d[220:260, 300:340] = 4200                                   # sign box: 80 % far background
        d[220:228, 300:340] = z_mm
    node.scene_depth[5] = dict(dimg=d, depth_stamp=100.0, pose=[0, 0, 0], pose_stamp=100.0)
    with node._seed_lock:
        node.rt_tracker.tracks[7] = dict(id=7, status=LIVE, frame_id=9, header_t=101.0, box=[302.0, 221.0, 342.0, 261.0], seed_frame=5,
                                         quality=1.0, why=None)
        node._trk_latest = (dict(frame_id=9, header_stamp=101.0), np.zeros((480, 640, 3), np.uint8))
        node.sem_by_track[7] = dict(display="VERIFYING", sim=0.2, t_mono=time.monotonic(), source_frame_id=5)


def _stop():
    return dict(F.a3_stop(), plane=dict(ok=True, z_med=2.0), z_tile=2.0)


def test_A28_same_observation_through_planner_capture_reader_and_event(env, monkeypatch):
    write_gate(env, monkeypatch, estop=False)
    con = F.FakeConsole()
    node, clk = gnode(env, con)
    _confirmed_setup(node)
    node.start_event(_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None,
                     latest_frame=dict(frame_id=9, header_stamp=101.0))
    TE.drive(node, F.FakeSource(), clk, 6.0, until=lambda n: n.fsm is not None and not n.fsm.active and not n.device.busy)
    rec = node.store.get(node.fsm.key)
    oid = rec["observation_id"]
    ob = rec["observation"]
    assert oid and ob["geometry_confirmed"] and ob["source_frame_id"] == 5 and ob["latest_frame_id"] == 9
    assert ob["roi_box_latest"] == [302.0, 221.0, 342.0, 261.0] and ob["depth"]["ok"] and ob["depth"]["uncertainty"] == "LOW"
    assert ob["boot_id"] == RC.BOOT_ID and ob["run_id"] == node.run_id and ob["event_key"] == node.fsm.key
    assert rec["aim_request"]["observation_id"] == oid and rec["capture_status"] == CS.CAPTURE_QUALITY_OK
    aims = [q for p, q in con.calls if p == "/aim"]
    assert aims and aims[0]["req"] == rec["request_id"]              # the console request is THIS event's request
    job = json.loads(next((node.run / "jobs" / "pending").glob("*.json")).read_text())
    assert job["observation_id"] == oid and job["capture_id"] == rec["capture_id"] and job["request_id"] == rec["request_id"]
    core = J.ReaderCore(node.run, dict(TR.CFG), ocr_fn=TR.FakeOcr([("418", 0.95)]), engine_info=TR.ENGINE, vlm_fn=None,
                        admit_fn=lambda n: (False, "fake"), clock=TR.Clock(), wall=lambda: 1.7e9, boot_id="B")
    for _ in range(5):
        core.step()
    node.event_step(F.FakeSource(), None)
    rec = node.store.get(node.fsm.key)
    assert rec["reading"]["observation_id"] == oid and rec["reading"]["terminal"] and rec["reading"]["capture_id"] == rec["capture_id"]
    st = json.loads((N.SHM / "status.json").read_text())
    ev = [e for e in st["events"] if e["event_key"] == node.fsm.key][0]
    assert ev["reading"] is not None and ev["capture_id"] == rec["capture_id"]
    fsm = (node.run / "fsm.jsonl").read_text()
    assert '"OBSERVATION"' in fsm and oid in fsm
    # a reading that belongs to another hand-off is never attached
    res = dict(rec["reading"], observation_id="obs-other", result_revision=99)
    assert node.store.attach_result(res) == "OBSERVATION_MISMATCH"


def test_A28_unconfirmed_geometry_refuses_aim_without_any_device_command(env, monkeypatch):
    write_gate(env, monkeypatch, estop=False)
    con = F.FakeConsole()
    node, clk = gnode(env, con)
    node.start_event(_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None, latest_frame=dict(frame_id=9))
    TE.drive(node, F.FakeSource(), clk, 6.0, until=lambda n: n.fsm is not None and not n.fsm.active and not n.device.busy)
    rec = node.store.get(node.fsm.key)
    assert rec["capture_status"] == CS.NO_CAPTURE and "SOURCE_DEPTH_UNAVAILABLE" in json.dumps(rec["observation"]["depth"])
    assert not [p for p, _ in con.calls if p in ("/aim", "/home", "/focus")]
    fsm = [json.loads(x) for x in (node.run / "fsm.jsonl").read_text().splitlines()]
    assert ("NO_CAPTURE", "geometry_unconfirmed_no_aim") in [(r["state"], r["reason"]) for r in fsm]
    assert rec["release"] == RS.RELEASE_READY_OBSERVATION_FAILED


def test_A29_mixed_depth_median_mad_not_trusted():
    d = np.full((480, 640), 4200, np.uint16)
    d[220:228, 300:340] = 2000                                     # 20 % sign, 80 % background inside the ROI
    p = OBS.depth_profile(d, [300, 220, 340, 260], ref_z=2.0)
    assert p["ok"] and abs(p["median_m"] - 4.2) < 0.01 and p["mad_m"] == 0.0     # median/MAD look perfectly stable ...
    assert p["multimodal"] and p["ref_consistent"] is False and p["uncertainty"] == "HIGH"   # ... but are flagged
    ok, why = OBS.geometry_confirmation(p, 3, True)
    assert not ok and "MULTIMODAL_DEPTH" in why and any(w.startswith("REFERENCE_INCONSISTENT") for w in why)   # SEM1: renamed (same-source check)
    u = np.full((480, 640), 2000, np.uint16)
    q = OBS.depth_profile(u, [300, 220, 340, 260], ref_z=2.05)
    assert q["uncertainty"] == "LOW" and OBS.geometry_confirmation(q, 2, True)[0]
    assert not OBS.geometry_confirmation(q, 1, True)[0] and not OBS.geometry_confirmation(q, 2, False)[0]
    n = OBS.depth_profile(u, [300, 220, 340, 260], ref_z=None)
    assert n["uncertainty"] == "MEDIUM" and not OBS.geometry_confirmation(n, 3, True)[0] and n["sign_mask"] is None


# ======================================================================================= A30 / A31 reader + release
def test_A30_reading_done_before_home_release_not_lost(tmp_path):
    st = EventStore(tmp_path / "run", "run")
    r = st.new_event(1, dict(t=0, stage="P2", spot=1, box=[0, 0, 1, 1], z=1, sim=.2, nobs=2))
    k = r["event_key"]
    st.transition(k, CS.CAPTURE_QUALITY_OK, "quality_ok", capture_status=CS.CAPTURE_QUALITY_OK)
    st.set_fields(k, capture_id="c1")
    st.transition(k, "HOME", "return_to_safe_posture")
    assert st.attach_result(dict(event_key=k, capture_id="c1", terminal=True, reason_code=None, result_revision=1)) == "ATTACHED"
    assert st.get(k)["state"] == "READ_DONE"
    assert st.transition(k, RS.RELEASE_READY_CAPTURE_OK, "device_state_confirmed", release=RS.RELEASE_READY_CAPTURE_OK, terminal=True)
    rec = st.get(k)
    assert rec["release"] == RS.RELEASE_READY_CAPTURE_OK and rec["terminal"] and rec["state"] == "READ_DONE"
    assert rec["release_state"] == RS.RELEASE_READY_CAPTURE_OK and rec["reading"]["terminal"]
    hist = (tmp_path / "run" / "events_r1" / "history.jsonl").read_text()
    assert "RELEASE_RECORDED_STATE_KEPT" in hist


def test_A30_home_finishes_before_reading_both_kept(tmp_path):
    st = EventStore(tmp_path / "run", "run")
    k = st.new_event(1, dict(t=0, stage="P2", spot=1, box=[0, 0, 1, 1], z=1, sim=.2, nobs=2))["event_key"]
    st.set_fields(k, capture_id="c1")
    st.transition(k, RS.RELEASE_READY_CAPTURE_OK, "device_state_confirmed", release=RS.RELEASE_READY_CAPTURE_OK, terminal=True)
    assert st.attach_result(dict(event_key=k, capture_id="c1", terminal=True, reason_code=None, result_revision=1)) == "ATTACHED"
    rec = st.get(k)
    assert rec["release"] == RS.RELEASE_READY_CAPTURE_OK and rec["state"] == "READ_DONE" and rec["reading"]["terminal"]
    # a late non-terminal result after terminal does not overwrite
    assert st.attach_result(dict(event_key=k, capture_id="c1", terminal=False, result_revision=2)) == "LATE_AFTER_TERMINAL"


def test_A31_ocr_exception_bounded_and_does_not_block_other_jobs(tmp_path):
    run = tmp_path / "kci"
    (run / "jobs" / "pending").mkdir(parents=True)

    class Ocr:                                                      # job 1 (1280x720 capture) always crashes the engine
        def __call__(self, img):
            if img.shape[0] == 720:
                raise RuntimeError("paddle crashed")
            return TR.FakeOcr([("418", 0.95)])(img)
    clk = TR.Clock()
    TR.make_job(run, 1, seed=1)
    c = TR.core(run, ocr=Ocr(), clock=clk, cfg=dict(TR.CFG, max_ocr_errors=2, ocr_retry_s=1.0))
    c.step()                                                        # job1 raises once (bounded back-off)
    p = run / "captures" / "ev002_cap1.jpg"
    TR.make_image(p, seed=2, size=(480, 640))
    j2 = dict(schema_version="r1.1", run_id=run.name, event_id=2, event_key=f"{run.name}#ev002", capture_id="cap1", request_id="r2",
              image=str(p), image_sha256=J.file_sha256(p), capture_status=CS.CAPTURE_QUALITY_OK, pantilt_ok=True,
              trial=str(run / "trials" / "ev002"), created=1.0)
    (run / "jobs" / "pending" / "ev002.json").write_text(json.dumps(j2))
    c.step()                                                        # job2 is processed although job1 is older
    assert TR.done_current(run, 2)["terminal"]
    clk.t += 2.0
    for _ in range(3):
        c.step()
        clk.t += 2.0
    d1 = TR.done_current(run, 1)
    assert d1["terminal"] and d1["reason_code"] == RR.OCR_ENGINE_ERROR and c.counters["ocr_errors"] == 2


def test_A31_slow_vlm_async_logical_timeout_other_jobs_continue_late_result_history(tmp_path):
    run = tmp_path / "kci"
    (run / "jobs" / "pending").mkdir(parents=True)
    release = threading.Event()

    def vlm(img, timeout_s):
        release.wait(3.0)                                           # slow VLM (still running after the logical deadline)
        return dict(raw="{}", parsed=None, sec=3.0, image_px=list(img.shape[:2]))
    cfg = dict(TR.CFG, vlm_async=True, inference_timeout_s=0.3, resource_wait_s=30)
    c = J.ReaderCore(run, cfg, ocr_fn=TR.FakeOcr([("41", 0.5)]), engine_info=TR.ENGINE, vlm_fn=vlm,
                     admit_fn=lambda n: (True, "ok"), clock=time.monotonic, wall=time.time, boot_id="B")
    TR.make_job(run, 1, seed=1)
    c.step()                                                        # OCR -> waiting
    c.step()                                                        # VLM submitted asynchronously, step returns
    st1 = c.states[next(iter(c.states))]
    assert st1["status"] == "VLM_RUNNING"
    TR.make_job(run, 2, seed=2)
    t0 = time.monotonic()
    c.step()                                                        # job 2 ingested + OCR'd while the VLM runs
    assert time.monotonic() - t0 < 0.5 and len(c.states) == 2
    time.sleep(0.35)
    c.step()
    assert TR.done_current(run, 1)["reason_code"] == RR.VLM_TIMEOUT and c.counters["vlm_logical_timeouts"] == 1
    release.set()
    time.sleep(0.1)
    c.step()
    assert c.counters["vlm_late_results"] == 1 and TR.done_current(run, 1)["reason_code"] == RR.VLM_TIMEOUT   # unchanged
    hist = list((run / "jobs" / "history").glob("*ev001*"))[0].read_text()
    assert "late_result" in hist


# ======================================================================================= A32 safety at all entrypoints
def test_A32_no_grant_estop_abort_shadow_expired_block_every_entrypoint(env, monkeypatch):
    con = F.FakeConsole()
    node, clk = gnode(env, con)
    cases = [dict(grant=False), dict(estop=True), dict(operator_abort=True), dict(ttl=-1.0)]
    for c in cases:
        write_gate(env, monkeypatch, **c)
        assert not node.can_start_physical(), c
    write_gate(env, monkeypatch)
    monkeypatch.setattr(OB, "OBSERVE_MODE", "shadow")
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    g = OB.gate_full()
    assert g["observe_enabled"] and not g["request_enabled"] and not g["actuator_armed"] and node.event_busy()
    monkeypatch.setattr(OB, "OBSERVE_MODE", "walking")
    # entrypoint 1: proposal -> event start refused without grant
    write_gate(env, monkeypatch, grant=False)
    node.proposals.put(dict(id=7, stop=_stop(), stamp=100.0, pose=[0, 0, 0], bgr=np.zeros((480, 640, 3), np.uint8), latest_frame=None))
    node.event_step(F.FakeSource(), None)
    assert node.fsm is None and "operator arm/heartbeat missing" in (node.run / "faults.jsonl").read_text()
    # entrypoint 2: grant lost before aim (abort latched mid-event)
    write_gate(env, monkeypatch, estop=False)
    _confirmed_setup(node)
    node.start_event(_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None, latest_frame=dict(frame_id=9))
    write_gate(env, monkeypatch, operator_abort=True)
    TE.drive(node, F.FakeSource(), clk, 3.0, until=lambda n: not n.fsm.active)
    fsm = [json.loads(x) for x in (node.run / "fsm.jsonl").read_text().splitlines()]
    assert ("NO_CAPTURE", "operator_arm_lost_before_aim") in [(r["state"], r["reason"]) for r in fsm]
    assert not [p for p, _ in con.calls if p in ("/aim", "/home", "/focus")]


def test_A32_home_blocked_with_unknown_estop_but_allowed_with_explicit_safe_inputs(env, monkeypatch):
    for estop, expect_home in ((None, False), (False, True)):
        con = F.FakeConsole()
        write_gate(env, monkeypatch, estop=estop)
        node, clk = gnode(env, con, run=f"home_{estop}")
        _confirmed_setup(node)
        node.start_event(_stop(), 100.0, (0.0, 0.0, 0.0), np.zeros((480, 640, 3), np.uint8), None, latest_frame=dict(frame_id=9))
        TE.drive(node, F.FakeSource(), clk, 6.0, until=lambda n: n.fsm is not None and not n.fsm.active and not n.device.busy)
        rec = node.store.get(node.fsm.key)
        homes = [p for p, _ in con.calls if p == "/home"]
        assert bool(homes) == expect_home, (estop, homes)
        assert rec["release"] == RS.RELEASE_READY_CAPTURE_OK
        hd = rec["home_decision"]
        assert (hd["why"] == "OPERATOR_SAFETY_STATE_UNKNOWN") == (estop is None) and hd["ok"] == expect_home


def test_A32_viewer_writer_grant_lifecycle(tmp_path, monkeypatch):
    sys.path.insert(0, str(WS / "external" / "sci_live" / "realtime" / "tools"))
    monkeypatch.setenv("SCI_SHM", str(tmp_path / "shm"))
    import live_viewer as LV
    s = LV.State.__new__(LV.State)
    s.exp, s.e5_grant, s.e5_abort = dict(state="waiting"), None, False
    assert LV.State.e5_arm(s, {})["why"] == "trial_not_walking"
    s.exp = dict(state="walking")
    s.write_gate()
    d = json.loads((tmp_path / "shm" / "kci_gate.json").read_text())
    assert d["armed"] is False and d["schema"] == "e5rt-gate-2" and "estop" not in d
    assert G.gate_state(d, time.time(), now=time.time())["actuator_armed"] is False
    r = LV.State.e5_arm(s, dict(ttl_s=60))
    d = json.loads((tmp_path / "shm" / "kci_gate.json").read_text())
    g = G.gate_state(d, time.time(), now=time.time())
    assert r["ok"] and d["armed"] and g["actuator_armed"] and g["estop_state"] is None
    assert not G.gate_state(d, time.time(), now=d["arm_expires_t"] + 1)["actuator_armed"]    # expired grant re-read
    LV.State.e5_disarm(s, abort=True)
    d = json.loads((tmp_path / "shm" / "kci_gate.json").read_text())
    assert d["armed"] is False and d["operator_abort"] is True
    s.exp = dict(state="computing")
    LV.State.e5_arm(s, {})
    s.write_gate()
    assert s.e5_grant is None                                       # leaving walking revokes


# ======================================================================================= A33 / A34 UI
def test_A33_live_image_only_boxes_verified_on_that_frame(env, monkeypatch):
    monkeypatch.setattr(N, "SHM", env / "shm")
    node, _ = gnode(env, F.FakeConsole())
    im = np.zeros((480, 640, 3), np.uint8)
    with node._seed_lock:
        node.rt_tracker.tracks[3] = dict(id=3, status=LIVE, frame_id=11, box=[100, 100, 200, 200], quality=1.0)
        node.sem_by_track[3] = dict(display="VERIFYING", sim=0.2, t_mono=time.monotonic())
    on11 = node.render_live((1.0, 11, im))
    on12 = node.render_live((1.1, 12, im))
    assert on11[100:201, 100:201].sum() > 0 and on12[100:201, 100:201].sum() == 0
    with node._seed_lock:
        node.sem_by_track[3]["display"] = "CANDIDATE"                 # geometry-only candidate: never drawn as a sign
    assert node.render_live((1.0, 11, im))[100:201, 100:201].sum() == 0


def test_A34_status_separates_wall_and_monotonic_and_carries_new_fields(env, monkeypatch):
    write_gate(env, monkeypatch)
    node, _ = gnode(env, F.FakeConsole())
    node.publish_status(F.FakeSource(), None, force=True)
    st = json.loads((N.SHM / "status.json").read_text())
    assert isinstance(st["now"], float) and st["now"] > 1e9 and isinstance(st["status_mono_ns"], int)
    assert "e5rt2_p2_tick" in st and st["scene_mode"] in ("sync", "async") and "e5rt_tracker" in st
    assert st["e5rt_gate"]["actuator_armed"] is True and st["home_last"] is None
    js = (WS / "external" / "sci_live" / "realtime" / "tools" / "live_viewer.py").read_text()
    assert "Date.now()-" not in js.replace(" ", "") or "mono" not in js[js.index("Date.now()"):js.index("Date.now()") + 80]


def test_A22_frontend_export_with_budget_deferred_regions_does_not_crash():
    """Found by the onboard replay (D_cand_120_150, budget 32 bound once): export crashed on emb=None of a deferred
    region (b64encode(None)). Real Frontend.export with a BudgetedSiglip-shaped client: deferred = typed, no crash."""
    import types
    import frontend as FE
    fe = FE.Frontend.__new__(FE.Frontend)
    regs = [[10.0, 10.0, 60.0, 60.0], [100.0, 10.0, 150.0, 60.0]]
    fe.last_pre_regs = [list(r) for r in regs]
    fe.vlm = types.SimpleNamespace(last_neg={"poster": [0.05, None]}, last_ctx={"poster": [0.01, None]}, last_ctx_plate=[0.1, None],
                                   last_emb=[b"\x00\x01", None], last_tiles=[[10, 10, 60, 60], None],
                                   last_score_status=["FRESH", DEFERRED])
    fe.L = types.SimpleNamespace(spots=[], spot_recent=0, board_of={}, adv_of={}, fired=set(), floor_up=None)
    orig = FE.M.region_world
    FE.M.region_world = lambda b, d, vp, pose: (None, None)
    try:
        out = fe.export([[0.12, regs[0]], [-1.0, regs[1]]], None, None, (0, 0, 0), 0.1, [])
    finally:
        FE.M.region_world = orig
    assert out[0]["score_status"] == "FRESH" and out[0]["emb"]
    assert out[1]["score_status"] == DEFERRED and out[1]["emb"] is None and out[1]["like"] is False and out[1]["tile"] == regs[1]


# ======================================================================================= A24 threshold mode
def test_A24_threshold_mode_legacy_default_uncalibrated_blocks_stop_verified_artifact_fixed(env, tmp_path, monkeypatch):
    from e5rt import calibration as CAL
    res = dict(thr=float("inf"), regions=[dict(msim=0.12, like=False, score_status="FRESH"), dict(msim=0.12, like=False, score_status=DEFERRED)])
    leg = CAL.ThresholdMode(mode="legacy_p90")
    assert leg.apply(dict(res))["thr_used"] == float("inf")                     # cold start measured as-is (inf)
    un = CAL.ThresholdMode(mode="calibrated", artifact_path=None)
    a = un.apply(dict(res))
    assert un.state == CAL.UNCALIBRATED and not a["stop_allowed"] and "NO_CALIBRATION_ARTIFACT" in a["why"]
    art = dict(method="m", split="DEV", score="raw cosine", thresholds=dict(global_=None), data=dict(x=1), created="t")
    art["thresholds"] = {"global": 0.105}
    for hv, expect in ((False, CAL.UNCALIBRATED), (True, CAL.ACTIVE)):
        d = dict(art, human_verified=hv)
        d["calibration_id"] = CAL.artifact_id(d)
        p = tmp_path / f"cal_{hv}.json"
        p.write_text(json.dumps(d))
        tm = CAL.ThresholdMode(mode="calibrated", artifact_path=str(p))
        assert tm.state == expect, tm.why
    r2 = json.loads(json.dumps(res))
    out = tm.apply(r2)
    assert out["thr_used"] == 0.105 and r2["thr"] == 0.105 and r2["regions"][0]["like"] is True and r2["regions"][1]["like"] is False
    d = json.loads(p.read_text()); d["thresholds"]["global"] = 0.05; p.write_text(json.dumps(d))       # tampered content
    assert CAL.ThresholdMode(mode="calibrated", artifact_path=str(p)).state == CAL.UNCALIBRATED
    monkeypatch.setenv("SCI_E5RT2_THRESHOLD_MODE", "calibrated")
    node, _ = gnode(env, F.FakeConsole(), run="cal_uncal")
    assert node.threshold_mode.state == CAL.UNCALIBRATED and node.event_busy()                  # no STOP from tentative semantics
