"""E5-RT candidate tests (SIMULATED / STORED_INPUT_CPU). No torch, ROS, devices or network (rt_guard enforces it)."""
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
sys.path.insert(0, str(WS / "src" / "sci_visual_observation"))
sys.path.insert(0, str(WS / "external" / "sci_event" / "e5_onboard"))

from e5rt import contracts as C  # noqa: E402
from e5rt import evidence as EV  # noqa: E402
from e5rt import gates as G  # noqa: E402
from e5rt import handoff as H  # noqa: E402
from e5rt import home_policy as HP  # noqa: E402
from e5rt import live_render as LR  # noqa: E402
from e5rt import scheduler as S  # noqa: E402
from e5rt import status_writer as SW  # noqa: E402
from e5rt.rt_frontend import RtSiglipCache  # noqa: E402
from e5rt.scout import Scout, map_box  # noqa: E402
from e5rt.tracker import LIVE, LOST, LKTracker  # noqa: E402


def textured(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    im = (rng.random((h, w)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(im, (5, 5), 1.0)


def shift(im, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ V03 concurrent status writes
def test_V03_revisioned_writer_blocks_stale_snapshot(tmp_path):
    w = SW.RevisionedWriter()
    p = tmp_path / "status.json"
    r_old, r_new = w.take(), w.take()                      # old snapshot built first, written last
    assert w.write_json(p, dict(v="new"), r_new) is True
    assert w.write_json(p, dict(v="old"), r_old) is False   # stale overwrite refused
    assert json.loads(p.read_text())["v"] == "new" and w.stale_dropped == 1


def test_V03_unique_tmp_concurrent_writers_no_error(tmp_path):
    p = tmp_path / "s.json"
    errs = []

    def run(k):
        try:
            for i in range(300):
                SW.atomic_write_unique(p, json.dumps(dict(k=k, i=i)).encode())
        except Exception as e:                               # noqa: BLE001
            errs.append(e)
    th = [threading.Thread(target=run, args=(k,)) for k in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs and json.loads(p.read_text())["i"] == 299
    assert not list(tmp_path.glob(".*.tmp"))                 # no leftovers


def test_V03_status_snapshots_monotonic_under_two_threads(tmp_path):
    """publish order == build order when building+writing is serialised (GatedNode._status_lock pattern)."""
    w, lock, p = SW.RevisionedWriter(), threading.Lock(), tmp_path / "status.json"
    seen = []

    def pub(k):
        for i in range(200):
            with lock:
                rev = w.take()
                w.write_json(p, dict(k=k, i=i), rev)
            seen.append(json.loads(p.read_text())["_rev"])
    th = [threading.Thread(target=pub, args=(k,)) for k in range(2)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert w.stale_dropped == 0 and json.loads(p.read_text())["_rev"] == 400


# ------------------------------------------------------------------ V06/V07/V10 odom samples
def test_V07_odom_rejects_nan_future_backwards():
    import e5_onboard as OB
    now = 1000.0
    assert OB.odom_sample_ok(999.0, 0, 0, 0, None, now) == (True, None)
    assert OB.odom_sample_ok(float("nan"), 0, 0, 0, None, now)[1] == "non_finite"
    assert OB.odom_sample_ok(999.0, float("inf"), 0, 0, None, now)[1] == "non_finite"
    assert OB.odom_sample_ok(1010.0, 0, 0, 0, None, now)[1] == "future_stamp"
    assert OB.odom_sample_ok(990.0, 0, 0, 0, 999.0, now)[1] == "time_backwards"     # reset -> new session
    assert OB.odom_sample_ok(999.0, 0, 0, 0, 999.0, now)[1] == "duplicate_or_reordered"


def test_V10_odom_reset_starts_new_session_and_drops_history():
    import e5_onboard as OB
    from collections import deque

    class Hdr:
        def __init__(self, t):
            self.stamp = t

    class Msg:
        def __init__(self, t, x):
            self.header = Hdr(t)
            self.pose = type("P", (), {})()
            self.pose.pose = type("Q", (), {})()
            self.pose.pose.position = type("V", (), dict(x=x, y=0.0, z=0.0))()
            self.pose.pose.orientation = type("O", (), dict(x=0.0, y=0.0, z=0.0, w=1.0))()
    s = OB.BaseOdomRosSource.__new__(OB.BaseOdomRosSource)
    s.lock, s.odom, s.n_odom = threading.Lock(), deque(maxlen=500), 0
    s._st = lambda h: h.stamp
    now = time.time()
    for k in range(5):
        s._on_odom(Msg(now - 5 + k * 0.1, k * 0.1))
    assert len(s.odom_hist) == 5 and s.odom_session == 0
    s._on_odom(Msg(now - 20, 0.0))                           # stamp jumps back 15 s: FAST-LIO restart
    assert s.odom_session == 1 and len(s.odom_hist) == 0 and s.odom_rejected["time_backwards"] == 1
    s._on_odom(Msg(now - 19.9, 0.0))
    assert len(s.odom_hist) == 1


# ------------------------------------------------------------------ V09 depth
def test_V09_invalid_depth_not_filled():
    d = np.zeros((480, 640), np.uint16)
    st = C.roi_depth_stats(d, [100, 100, 130, 150])
    assert not st["ok"] and st["median_m"] is None and st["reason"] == "INSUFFICIENT_VALID_DEPTH"
    d[100:150, 100:130] = 4200
    d[100:110, 100:130] = 9000                                # partial background behind a small plate
    st = C.roi_depth_stats(d, [100, 100, 130, 150])
    assert st["ok"] and abs(st["median_m"] - 4.2) < 1e-6 and st["valid_ratio"] == 1.0 and st["mad_m"] == 0.0


# ------------------------------------------------------------------ V13/V14/V15 scout + mapping
def test_V14_exact_inverse_mapping_non_640():
    assert map_box([10, 20, 30, 40], (240, 424), (480, 848)) == [20.0, 40.0, 60.0, 80.0]
    assert map_box([1, 1, 2, 2], (360, 640), (720, 1280)) == [2.0, 2.0, 4.0, 4.0]


def test_V13_scout_downscale_and_rescue_and_skip():
    calls = []

    def prop(img):
        calls.append(img.shape)
        h, w = img.shape[:2]
        return [[0.1 * w, 0.1 * h, 0.2 * w, 0.3 * h, 7.0], [0.5 * w, 0.5 * h, 0.6 * w, 0.7 * h, 3.0]]
    sc = Scout(prop, scale=0.5, rescue_every=3)
    img = np.zeros((480, 640, 3), np.uint8)
    b1, i1 = sc.step(img)
    assert calls[-1] == (240, 320, 3) and i1["source"].startswith("scout") and abs(b1[0][0] - 64.0) < 1e-6 and b1[0][4] == 7.0
    b2, i2 = sc.step(img, tracked_boxes=[[64, 48, 128, 144]])
    assert i2["skipped_tracked"] == 1 and len(b2) == 1
    b3, i3 = sc.step(img, tracked_boxes=[[64, 48, 128, 144]])
    assert calls[-1] == (480, 640, 3) and i3["source"] == "full_res_rescue" and len(b3) == 2   # rescue never skips


# ------------------------------------------------------------------ V16/V17 tracker
def test_tracker_follows_translation_full_res_coords():
    g0 = textured()
    tr = LKTracker(track_scale=0.5)
    tr.seed(1, [200, 150, 240, 210], g0, 1, 0.0)
    g1 = shift(g0, 6, -4)
    tr.update(g1, 2, 0.13)
    t = tr.tracks[1]
    assert t["status"] == LIVE and t["frame_id"] == 2
    assert abs(t["box"][0] - 206) < 1.5 and abs(t["box"][1] - 146) < 1.5
    assert tr.live_on(2) and not tr.live_on(1)


def test_V17_textureless_roi_is_uncertain_then_lost():
    g0 = np.full((480, 640), 128, np.uint8)
    tr = LKTracker()
    t = tr.seed(1, [200, 150, 240, 210], g0, 1, 0.0)
    assert t["status"] == "UNCERTAIN_TEXTURELESS"
    tr.update(g0.copy(), 2, 0.1)
    assert tr.tracks[1]["status"] == LOST and not tr.live_on(2)


def test_V16_occlusion_or_jump_loses_track_and_gap_loses_track():
    g0 = textured(seed=1)
    tr = LKTracker()
    tr.seed(1, [300, 200, 340, 260], g0, 1, 0.0)
    tr.update(textured(seed=99), 2, 0.1)                     # appearance replaced (occlusion / cut)
    assert tr.tracks[1]["status"] == LOST
    tr2 = LKTracker(max_gap_s=0.6)
    tr2.seed(1, [300, 200, 340, 260], g0, 1, 0.0)
    tr2.update(shift(g0, 1, 0), 2, 1.5)                       # 1.5 s gap: not propagated
    assert tr2.tracks[1]["status"] == LOST and "frame_gap" in tr2.tracks[1]["why"]


def test_tracker_left_view_lost():
    g0 = textured(seed=3)
    tr = LKTracker(track_scale=1.0)
    tr.seed(1, [620, 200, 639, 240], g0, 1, 0.0)
    tr.update(shift(g0, 40, 0), 2, 0.1)
    assert tr.tracks[1]["status"] == LOST


# ------------------------------------------------------------------ V18/V19/V20 scheduler
def test_V18_cache_hit_reduces_forwards_and_is_not_evidence():
    sch = S.VerifyScheduler(dict(model="m", prompts="p"), max_age_s=2.0, clock=lambda: 0.0)
    sig = S.crop_signature([0, 0, 40, 60], sharp=50, mean_gray=100)
    f, c, d = sch.plan([dict(key=0, track_id=7, sig=sig)], now=0.0)
    assert f == [0] and d[0] == S.NEW
    sch.store(7, sig, dict(sim=0.11), now=0.0)
    f, c, d = sch.plan([dict(key=0, track_id=7, sig=sig)], now=1.0)
    assert f == [] and c == [0] and d[0] == S.CACHED_FRESH
    f, c, d = sch.plan([dict(key=0, track_id=7, sig=S.crop_signature([0, 0, 60, 90], 50, 100))], now=1.2)
    assert d[0] == S.IMPROVED_SCALE
    f, c, d = sch.plan([dict(key=0, track_id=7, sig=sig)], now=2.5)
    assert d[0] == S.AGED
    assert sch.set_revision(model="m2") and sch.plan([dict(key=0, track_id=7, sig=sig)], now=2.6)[2][0] == S.NEW
    q = EV.QualityEvidence()
    q.update(dict(t=0.0, frame_id=1, sim=0.12, thr=0.10, conf=0.05))
    n0 = q.n_distinct
    q.update(dict(t=1.0, frame_id=2, sim=0.12, thr=0.10, conf=0.05, cached=True))
    assert q.n_distinct == n0                                  # V37: cache hit adds no evidence


def test_V19_no_starvation_under_budget():
    sch = S.VerifyScheduler(dict(model="m"), max_fresh=2, max_age_s=0.5)
    served = {k: 0 for k in range(6)}
    for step in range(12):
        now = float(step)
        items = [dict(key=k, track_id=k, sig=S.crop_signature([0, 0, 40, 40])) for k in range(6)]
        f, c, d = sch.plan(items, now=now)
        for k in f:
            served[k] += 1
            sch.store(k, items[k]["sig"], dict(sim=0.1), now=now)
    assert min(served.values()) >= 3 and sch.counters["deferred_budget"] > 0 and sch.drops


def test_V20_latest_slot_bounded_with_drop_reasons():
    sl = S.LatestSlot()
    for i in range(10):
        sl.put(dict(id=i))
    assert sl.depth() == 1 and len(sl.dropped) == 9 and all(d["why"] == "superseded" for d in sl.dropped)
    it = sl.take(timeout=0.01)
    assert it["id"] == 9 and sl.depth() == 1
    sl.done()
    assert sl.depth() == 0


# ------------------------------------------------------------------ RtSiglipCache (B3) with a fake worker
class FakeClient:
    model, prompts, want_board = "fake", ["p"], True

    def __init__(self):
        self.calls, self.n_crops = 0, 0

    def score(self, bgr, regs):
        self.calls += 1
        self.n_crops += len(regs)
        n = len(regs)
        self.last_adv, self.last_board, self.last_ceil = np.zeros(n), np.full(n, -0.1), np.full(n, -0.5)
        self.last_tiles = [list(r) for r in regs]
        self.last_emb = [b"x"] * n
        self.last_neg = {"poster": [0.01] * n}
        self.last_ctx, self.last_ctx_plate = {"poster": [0.0] * n}, [0.0] * n
        self.last_infer = 0.01
        return np.array([0.1 + 0.001 * r[0] for r in regs]), 0.01


class Buf:
    def __init__(self):
        self.items = []

    def add(self, fid, t, bgr):
        self.items.append(dict(id=fid, tb=t, th=t, bgr=bgr, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))


class FakeL:
    no_evidence_keys = set()


def test_cache_wrapper_sends_only_fresh_and_marks_no_evidence():
    fc, buf, L = FakeClient(), Buf(), FakeL()
    clk = [0.0]
    w = RtSiglipCache(fc, buf, legacy_events=L, mode="cache", clock=lambda: clk[0])
    g = textured(seed=5)
    bgr0 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    regs = [[100.0, 100.0, 140.0, 160.0], [300.0, 200.0, 340.0, 260.0]]
    buf.add(1, 0.0, bgr0)
    s0, _ = w.score(bgr0, regs)
    assert fc.n_crops == 2 and w.last_cached == [False, False] and L.no_evidence_keys == set()
    for k, t in enumerate((0.2, 0.4, 0.6, 0.8, 1.0)):          # robot (almost) still: same image
        buf.add(2 + k, t, bgr0.copy())
    bgr1 = buf.items[-1]["bgr"]
    clk[0] = 1.0
    s1, _ = w.score(bgr1, regs)
    assert fc.n_crops == 2 and w.last_cached == [True, True]  # no new forward
    assert np.allclose(s1, s0) and len(L.no_evidence_keys) == 2
    clk[0] = 2.5                                               # aged -> fresh again
    buf.add(10, 2.5, bgr1.copy())
    w.score(buf.items[-1]["bgr"], regs)
    assert fc.n_crops == 4 and w.last_cached == [False, False]


def test_cache_wrapper_off_is_passthrough():
    fc, buf = FakeClient(), Buf()
    w = RtSiglipCache(fc, buf, legacy_events=FakeL(), mode="off")
    bgr = np.zeros((480, 640, 3), np.uint8)
    buf.add(1, 0.0, bgr)
    for _ in range(3):
        w.score(bgr, [[0.0, 0.0, 50.0, 50.0]])
    assert fc.n_crops == 3


def test_legacy_decide_skips_no_evidence_keys():
    """E5RT2 replacement (was a source-text assertion of the E5-RT1 rule 'or not spot.sims', which let a cached score
    become the FIRST observation of a spot and always entered self.seen = RP11). Behavioural AFTER check on the real
    decide() prefix: a no-evidence (cached/deferred) region keeps its spot association but adds nothing to spot sims,
    self.seen (p90 warm-up) or the spot mean; a fresh region still does."""
    import ast, copy, types
    f = WS / "src" / "sci_legacy_e5" / "live_event_runner.py"
    root = ast.parse(f.read_text())
    cl = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == "LiveEvents")
    fn = copy.deepcopy(next(n for n in cl.body if isinstance(n, ast.FunctionDef) and n.name == "decide"))
    ix = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "base" for t in n.targets))
    fn.body = fn.body[:ix + 1] + [ast.Return(value=ast.Name(id="base", ctx=ast.Load()))]
    env = dict(np=np, M=types.SimpleNamespace(region_world=lambda *a: ([1, 1, 1], 1)), P={"radius": .7, "quant": .9, "warm": 50},
               match=lambda *a: 0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(f), "exec"), env)
    box = [1., 1., 2., 2.]
    o = types.SimpleNamespace(spots=[dict(sims=[], ts=[])], seen=[.1] * 49, no_evidence_keys={tuple(box)})
    thr = env["decide"](o, None, 1., [(.2, box)], None, None, None, False)
    assert o.spots[0]["sims"] == [] and len(o.seen) == 49 and not np.isfinite(thr)      # cached FIRST obs: no evidence
    o.no_evidence_keys = set()
    env["decide"](o, None, 2., [(.2, box)], None, None, None, False)
    assert o.spots[0]["sims"] == [.2] and len(o.seen) == 50                             # fresh obs still counts


# ------------------------------------------------------------------ V21 live overlay
def test_V21_live_overlay_draws_only_tracked_boxes_of_this_frame():
    tracks = [dict(id=3, frame_id=12, box=[10, 10, 50, 60], quality=0.9, status=LIVE),
              dict(id=4, frame_id=11, box=[100, 100, 150, 160], quality=0.9, status=LIVE)]
    sem = {3: dict(display="VERIFIED", sim=0.11, t_mono=99.0), 4: dict(display="VERIFIED", sim=0.12, t_mono=99.0)}
    boxes, note = LR.live_overlay(12, tracks, sem, 100.0, mode="tracked")
    assert len(boxes) == 1 and boxes[0][0] == (10, 10, 50, 60) and "sem 1.0s" in boxes[0][1]
    legacy = dict(frame_id=5, boxes=[([1, 1, 9, 9], "sign", 2)], t_drawn_mono=99.0)
    lb, ln = LR.live_overlay(12, [], {}, 100.0, legacy_meta=legacy, mode="legacy")
    assert lb and "LEGACY" in ln                              # student overlay reproducible, clearly labelled
    nb, _ = LR.live_overlay(12, [], {}, 100.0, legacy_meta=legacy, mode="tracked")
    assert nb == []                                            # no old box on a newer frame in candidate mode


# ------------------------------------------------------------------ evidence policies (V32/V33/V36/V37)
def run_policy(cls, obs, **kw):
    p = cls(**kw)
    states = [p.update(o) for o in obs]
    return p, states


def seq(sims, thr=0.104, conf=None, q=None, dt=1.0):
    return [dict(t=i * dt, frame_id=i, sim=s, thr=thr, conf=(conf[i] if conf else s - 0.03), quality=(q[i] if q else 1.0), scene_t=i * dt)
            for i, s in enumerate(sims)]


def test_V32_true_plaque_with_one_blurred_frame():
    sims = [0.110, 0.112, 0.080, 0.113, 0.114, 0.112]          # frame 2 blurred while walking
    q = [1.0, 1.0, 0.15, 1.0, 1.0, 1.0]
    m3, s_m3 = run_policy(EV.Mean3Legacy, seq(sims, q=q))
    qe, s_qe = run_policy(EV.QualityEvidence, seq(sims, q=q))
    # legacy mean3 LOSES confirmation for the next 3 scenes after one blurred frame (window of the mean)
    assert s_m3[1] == EV.CONFIRMED and s_m3[2:5] == [EV.UNKNOWN] * 3
    assert qe.t_confirm == 1.0 and all(s == EV.CONFIRMED for s in s_qe[1:])   # held: blurred shortfall down-weighted


def test_V33_single_glare_spike_not_confirmed_by_quality_policy():
    sims = [0.07, 0.072, 0.135, 0.071, 0.07]                    # one reflection frame
    conf = [0.08, 0.08, 0.16, 0.08, 0.08]                       # glare confuser even stronger on that frame
    mx, _ = run_policy(EV.MaxOnly, seq(sims, conf=conf))
    qe, _ = run_policy(EV.QualityEvidence, seq(sims, conf=conf))
    assert mx.t_confirm == 2.0                                  # max-only limitation reproduced
    assert qe.t_confirm is None and qe.state != EV.CONFIRMED    # a single spike (with a stronger confuser) never confirms


def test_far_low_scores_do_not_reject_a_sign_that_gets_closer():
    sims = [0.06, 0.065, 0.07, 0.08, 0.112, 0.115, 0.118]        # approaching: small/far first, clear later
    qe, st = run_policy(EV.QualityEvidence, seq(sims, conf=[s - 0.02 for s in sims]))
    assert EV.REJECTED not in st and qe.t_confirm == 5.0


def test_V36_ambiguous_low_res_stays_unknown():
    sims = [0.100, 0.103, 0.101, 0.102]
    qe, st = run_policy(EV.QualityEvidence, seq(sims, q=[0.3] * 4))
    assert qe.t_confirm is None and st[-1] in (EV.UNKNOWN, EV.TENTATIVE)


def test_V37_repeated_frame_counts_once():
    qe = EV.QualityEvidence()
    o = dict(t=0.0, frame_id=5, sim=0.13, thr=0.10, conf=0.05, appearance="a")
    for k in range(4):
        qe.update(dict(o, t=k * 0.1))
    assert qe.n_distinct == 1 and qe.t_confirm is None          # one frame repeated is one observation


def test_warmup_threshold_is_pending_not_negative():
    qe = EV.QualityEvidence()
    assert qe.update(dict(t=0, frame_id=1, sim=0.2, thr=float("inf"))) == EV.UNKNOWN and "warmup" in qe.reason
    m3 = EV.Mean3Legacy()
    m3.update(dict(t=0, frame_id=1, sim=0.2, thr=float("inf")))
    assert "warmup" in m3.reason


def test_V23_score_is_raw_cosine_no_sigmoid():
    src = (WS / "src" / "sci_visual_observation" / "e5rt" / "evidence.py").read_text()
    assert "RAW COSINE" in src and "sigmoid(" not in src


# ------------------------------------------------------------------ V40 evaluation guard
def test_V40_zero_stop_policy_has_no_precision():
    sys.path.insert(0, str(Path(os.environ["RT_ROOT"]) / "benchmark" / "evidence"))
    import eval_metrics as M
    r = M.spot_metrics(tp=0, fp=0, fn=7)
    assert r["precision"] is None and r["precision_null_reason"] == "NO_POSITIVE_DECISIONS" and r["recall"] == 0.0


# ------------------------------------------------------------------ V41/V42 gates + E4
def test_V41_global_computing_keeps_search_but_blocks_requests():
    now = 100.0
    grant = dict(id="g1", expires_t=now + 60)
    for st in ("walking", "waiting", "computing"):
        g = G.gate_state(G.gate_payload(st, now), now, now=now + 1)
        assert g["observe_enabled"] and g["trial_logging_active"]
        # E5RT2 replacement: walking is NOT an actuator grant (RP01/RP02). E5-RT1 asserted armed == walking.
        assert g["request_enabled"] == (st == "walking") and g["actuator_armed"] is False
        g = G.gate_state(G.gate_payload(st, now, grant=grant, operator_abort=False), now, now=now + 1)
        assert g["actuator_armed"] == (st == "walking")                          # explicit live grant: still possible
    g = G.gate_state(G.gate_payload("computing", now), now, now=now + 1)
    assert g["global_match_state"] == "computing" and any(r.startswith("TRIAL_GATE_BLOCKED") for r in g["reasons"])
    g = G.gate_state(G.gate_payload("done", now), now, now=now + 1)
    assert not g["observe_enabled"]                            # trial closed: search off (not computing)
    g = G.gate_state(G.gate_payload("computing", now), now, now=now + 30)
    assert not g["observe_enabled"] and not g["operator_link_alive"]   # heartbeat stale: nothing latched


def test_V41_old_viewer_schema_keeps_old_meaning():
    now = 100.0
    g = G.gate_state(dict(active=False, why="층 계산 중", t=now), now, now=now + 1)
    assert g["legacy_schema"] and not g["observe_enabled"] and "LEGACY_GATE_SCHEMA" in g["reasons"]


def test_V41_shadow_mode_never_requests_or_arms():
    now = 100.0
    g = G.gate_state(G.gate_payload("walking", now), now, now=now + 1, mode="shadow", arm_without_ui=True)
    assert g["observe_enabled"] and not g["request_enabled"] and not g["actuator_armed"]


def test_V42_e4_standalone_is_explicit():
    assert "UNAVAILABLE" in C.E4_STATES and C.E4_STANDALONE_MODE == "E5_STANDALONE_E4_UNAVAILABLE"


# ------------------------------------------------------------------ V45/V49 HOME policy
SAFE = dict(estop=False, operator_abort=False)            # E5RT2: safety inputs must be stated explicitly


@pytest.mark.parametrize("kw,ok,why", [
    (dict(device_used=False, device_idle_confirmed=True), False, "NO_DEVICE_ACTION_TO_UNDO"),
    (dict(device_used=True, device_idle_confirmed=False, **SAFE), False, "DEVICE_STATE_UNKNOWN_NO_AUTO_HOME"),
    (dict(device_used=True, device_idle_confirmed=True, estop=True), False, "ESTOP_OR_OPERATOR_ABORT_NO_AUTO_MOTION"),
    (dict(device_used=True, device_idle_confirmed=True, owner_busy_other=True, **SAFE), False, "DEVICE_OWNED_BY_OTHER_JOB"),
    (dict(device_used=True, device_idle_confirmed=True, home_after=False), False, "HOME_AFTER_DISABLED"),
    # E5RT2 replacement of the E5-RT1 row (armed=False -> True 'cleanup_of_own_action_without_arm', RP03):
    (dict(device_used=True, device_idle_confirmed=True, armed=False), False, "OPERATOR_SAFETY_STATE_UNKNOWN"),
    (dict(device_used=True, device_idle_confirmed=True, armed=False, cleanup_without_arm=True, **SAFE), False,
     "EVENT_SCOPED_CLEANUP_GRANT_REQUIRED"),
    (dict(device_used=True, device_idle_confirmed=True, armed=False, **SAFE), False, "NOT_ARMED"),
    (dict(device_used=True, device_idle_confirmed=True, armed=True, **SAFE), True, "HOME_CLEANUP_ALLOWED:armed"),
    (dict(device_used=True, device_idle_confirmed=True, armed=True, estop=None, operator_abort=False), False,
     "OPERATOR_SAFETY_STATE_UNKNOWN"),
])
def test_V49_home_policy(kw, ok, why):
    assert HP.home_decision(**kw) == (ok, why)


def test_V49_candidate_has_no_direct_http_home_thread():
    src = (WS / "external" / "sci_event" / "e5_onboard" / "e5_onboard.py").read_text()
    body = src[src.index("class GatedNode"):]
    assert "_console_home" not in body and "urlopen" not in body


# ------------------------------------------------------------------ V50 hand-off provenance
def test_V50_handoff_keeps_same_frame_provenance_and_rejects_mismatch():
    # E5RT2 replacement: the track must state the frame its box is on (RP06); E5-RT1 built it without frame_id
    tr = dict(id=3, status=LIVE, box=[12, 10, 52, 70], frame_id=46, header_t=1.8)
    sem = dict(box_source=[10, 10, 50, 70], sim=0.12, conf=0.07, t_mono=5.0, frame_id=40, state="CONFIRMED", n_distinct=3)
    oc = H.build_candidate("run", "g1", tr, sem, dict(frame_id=40, header_stamp=1.0), dict(frame_id=46, header_stamp=1.8),
                           depth_stats=dict(ok=False, reason="INSUFFICIENT_VALID_DEPTH"), image_size=[640, 480])
    d = oc.to_dict()
    assert d["source_frame_id"] == 40 and d["latest_frame_id"] == 46 and abs(d["margin"] - 0.05) < 1e-9
    assert d["stop_feasibility"] == C.GEOMETRY_UNAVAILABLE and d["depth"]["median_m"] is None if "median_m" in d["depth"] else True
    with pytest.raises(ValueError):
        H.build_candidate("run", "g1", tr, dict(sem, frame_id=41), dict(frame_id=40), dict(frame_id=46))
    lost = H.build_candidate("run", "g1", dict(tr, status=LOST), sem, dict(frame_id=40), dict(frame_id=46))
    assert lost.roi_box_latest is None and lost.latest_frame_id is None   # no stale ROI handed to the aimer


# ------------------------------------------------------------------ V02 import path
def test_V02_candidate_modules_load_from_this_tree():
    import e5_onboard as OB
    import e5rt
    assert Path(OB.__file__).resolve().is_relative_to(WS.resolve()) if hasattr(Path, "is_relative_to") else str(WS) in OB.__file__
    assert str(WS) in e5rt.__file__ and str(WS) in OB.N.__file__


def test_V48_quality_default_is_r1_and_override_explicit():
    import e5_onboard as OB
    assert OB.QUALITY_SHARP_MIN == "" and OB.N.QUALITY_SHARP_MIN == 40.0
    prof = (WS / "config" / "e5rt_profiles" / "kci_20261017_field.env").read_text()
    assert "SCI_QUALITY_SHARP_MIN=20" in prof and "NOT comparable" in prof
