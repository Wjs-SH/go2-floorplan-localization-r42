"""E5RT2 BEFORE reproductions through REAL callers (expected to FAIL on RT2_ROOT/baseline = E5-RT1 candidate, PASS on
RT2_ROOT/candidate). Only APIs present in both trees are used; each test asserts the CORRECT behaviour.
Evidence: ROBOT_CPU_FAKE (fakes for console/source/frontend; no device, ROS, model, GPU or network)."""
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_common", WS / "src" / "sci_pantilt",
          WS / "src" / "sci_text_reading", WS / "external" / "sci_event" / "e5_onboard", WS / "tests" / "r1"):
    sys.path.insert(0, str(p))

import r1_contracts as RC  # noqa: E402
import e5_node as N  # noqa: E402
import test_r1_e5 as _TE_FIRST  # noqa: E402,F401  (before e5_onboard: R1 tests capture the SCI Node class)
import e5_onboard as OB  # noqa: E402
import stop_policy as SP  # noqa: E402
import r1_jobs as J  # noqa: E402
import r1_fakes_e5 as F  # noqa: E402
import test_r1_reader as TR  # noqa: E402
from event_store import EventStore  # noqa: E402

CS, RS = RC.CaptureStatus, RC.ReleaseStatus


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "WS", tmp_path / "ws")
    monkeypatch.setattr(N, "SHM", tmp_path / "shm")
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    for k in ("SCI_ARM_WITHOUT_UI", "SCI_E5RT_HOME_WITHOUT_ARM", "SCI_E5RT2_SCENE_MODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    return tmp_path


def test_before_viewer_walking_payload_arms_wrapper(env):
    """RP01/RP02 on the real wrapper: the E5-RT1 viewer payload (walking, no grant) armed physical actions."""
    p = env / "kci_gate.json"
    p.write_text(json.dumps(dict(active=True, why="걷는 중", t=time.time(), trial_state="walking", global_match_state="idle",
                                 schema="e5rt-gate-1")))
    g = OB.gate_state(time.time(), path=p, mode="walking", gate_on=True)
    assert g["observe"] and not g["arm"], "walking without an explicit grant arms pan/tilt/HOME"


def test_before_wrapper_default_home_without_arm_or_safety_inputs(env):
    """RP03 on the real wrapper: default env allowed HOME with no arm and an E-stop state nobody provides."""
    g = OB.GatedNode.__new__(OB.GatedNode)
    g.device = types.SimpleNamespace(busy=False, busy_key=None)
    g.store = types.SimpleNamespace(set_fields=lambda *a, **k: True)
    f = types.SimpleNamespace(key="k#ev001", device_used=True, device_idle_confirmed=True)
    assert OB.GatedNode.home_allowed(g, f) is False, "HOME (motion) allowed without arm / with unknown E-stop"


class _SlowFE(F.FakeFrontend):
    def __init__(self):
        super().__init__(scene_delay_s=2.0)
        self.windows = []

    def scene(self, t, dimg, pose, allow_stop=True, **kw):
        a = time.monotonic()
        r = super().scene(t, dimg, pose, allow_stop)
        self.windows.append((a, time.monotonic()))
        return r


def test_before_scene_blocks_p2_timing_ticks(env, monkeypatch):
    """A14: in the real GatedNode.loop a 2 s scene stops every P2 timing tick (E5-RT1: tracking was display-only)."""
    monkeypatch.setattr(OB, "GATE_ON", False)
    fe = _SlowFE()
    planner = OB.Planner(N.Geometry(), console="http://fake", execute=False, http_get=F.FakeConsole().http_get)
    node = OB.GatedNode(F.args(env, run_name="before_tick"), deps=dict(frontend=fe, monitor=F.FakeMon(), planner=planner,
                                                                       trial_pointer=None, clock=time.monotonic, threaded=True,
                                                                       body_speed=lambda now: SP.BodySample(0.0, now, "sportmodestate.velocity_hypot_xy")))
    ticks = []
    orig = node.p2.tick
    node.p2.tick = lambda *a, **k: (ticks.append(time.monotonic()), orig(*a, **k))[1]
    src = F.FakeSource()
    img = np.zeros((480, 640, 3), np.uint8)

    def feed():
        t0 = time.time()
        for i in range(90):
            with src.lock:
                src.queue.append((t0 + i / 15.0, i, img))
                src.odom_hist = (src.odom_hist + [(t0 + i / 15.0, 0.01 * i, 0.0, 0.0)])[-200:]
            time.sleep(1 / 15.0)
    threading.Thread(target=feed, daemon=True).start()

    def poll():
        with src.lock:
            out, src.queue = src.queue, []
        return out
    src.poll = poll
    th = threading.Thread(target=node.loop, args=(src,), daemon=True)
    th.start()
    time.sleep(6.0)
    src.alive = False
    th.join(timeout=15)
    node.stop_event_thread()
    assert fe.windows
    a, b = fe.windows[0]
    pts = [a + 0.05] + sorted(x for x in ticks if a <= x <= b) + [b - 0.05]
    assert max(y - x for x, y in zip(pts, pts[1:])) <= 0.35, "P2 timing ticks stopped for the whole scene"


def test_before_reading_before_home_drops_release(tmp_path):
    """A30: a reading that finishes while HOME runs moved the state to READ_DONE; the later RELEASE was rejected as a
    backward transition and the release status was lost."""
    st = EventStore(tmp_path / "run", "run")
    k = st.new_event(1, dict(t=0, stage="P2", spot=1, box=[0, 0, 1, 1], z=1, sim=.2, nobs=2))["event_key"]
    st.set_fields(k, capture_id="c1")
    st.transition(k, "HOME", "return_to_safe_posture")
    st.attach_result(dict(event_key=k, capture_id="c1", terminal=True, reason_code=None, result_revision=1))
    st.transition(k, RS.RELEASE_READY_CAPTURE_OK, "device_state_confirmed", release=RS.RELEASE_READY_CAPTURE_OK, terminal=True)
    assert st.get(k)["release"] == RS.RELEASE_READY_CAPTURE_OK, "RELEASE lost after an early reading"


def test_before_ocr_exception_blocks_reader(tmp_path):
    """A31: an OCR engine exception escaped step(); the failing (oldest) job was retried forever and blocked others."""
    run = tmp_path / "kci"
    (run / "jobs" / "pending").mkdir(parents=True)

    def ocr(img):
        if img.shape[0] == 720:
            raise RuntimeError("engine crashed")
        return TR.FakeOcr([("418", 0.95)])(img)
    TR.make_job(run, 1, seed=1)
    p = run / "captures" / "ev002.jpg"
    TR.make_image(p, seed=2, size=(480, 640))
    (run / "jobs" / "pending" / "ev002.json").write_text(json.dumps(dict(
        schema_version="r1.1", run_id=run.name, event_id=2, event_key=f"{run.name}#ev002", capture_id="cap1", request_id="r2",
        image=str(p), image_sha256=J.file_sha256(p), capture_status=CS.CAPTURE_QUALITY_OK, pantilt_ok=True,
        trial=str(run / "trials" / "ev002"), created=1.0)))
    c = TR.core(run, ocr=ocr, cfg=dict(TR.CFG))
    errors = 0
    for _ in range(4):
        try:
            c.step()
        except Exception:                                          # noqa: BLE001
            errors += 1
    assert errors == 0 and TR.done_current(run, 2)["terminal"], "OCR exception escaped / blocked the other job"


def test_before_deferred_score_is_a_normal_negative_in_p2(env):
    """A22: a region without a fresh SigLIP score (sim -1, DEFERRED_BUDGET) was judged as a normal low score (SEARCH)."""
    from timing import P2Policy, load_p2
    from policy import load_cfg
    p2 = P2Policy(load_cfg(str(WS / "config" / "e5_policy.json")), load_p2(str(WS / "config" / "e5_policy_p2.yaml")), N.Geometry())
    r = dict(box=[100, 100, 140, 140], tile=[100, 100, 140, 140], sim=-1.0, spot=3, xyz=[2, 0, 1.5], z=2.0, msim=0.2, nobs=3,
             nscene=3, board=-1, adv=0, hgt=1.5, fired=False, like=False, neg={}, ctx={}, ctx_plate=None, score_status="DEFERRED_BUDGET")
    f = dict(glare=dict(sat=0.0, spec=0.0, contrast=0.0, sharp=0.0), plane=dict(ok=False), edge=50, speed=0.0, z_tile=2.0, up=None, pantilt=dict(ok=True), text_px=30,
             obliquity_pred=10.0, aim_px=[120, 120])
    recs, _, _ = p2.step(1.0, (0, 0, 0), 0.0, [r], [f], 0.1, None)
    assert recs[0]["state"] not in ("SEARCH", "REJECT", "TRACK"), f"deferred judged as {recs[0]['state']}"


def test_before_reader_job_has_no_observation_handoff(env, monkeypatch):
    """A28: the planner/reader never received the ObservationCandidate (module only)."""
    import test_r1_e5 as TE
    con = F.FakeConsole()
    gp = env / "gate.json"
    gp.write_text(json.dumps(dict(active=True, armed=True, trial_state="walking", t=time.time())))
    monkeypatch.setattr(OB, "GATE", gp)
    monkeypatch.setenv("SCI_ARM_WITHOUT_UI", "1")
    planner = OB.Planner(N.Geometry(), console="http://fake", execute=True, http_get=con.http_get)
    clk = TE.Clock()
    node = OB.GatedNode(F.args(env, run_name="before_obs"), deps=dict(frontend=F.FakeFrontend(), monitor=F.FakeMon(), planner=planner,
                                                                      trial_pointer=None, clock=clk, threaded=False,
                                                                      body_speed=lambda now: SP.BodySample(0.0, now, "sportmodestate.velocity_hypot_xy")))
    if hasattr(OB, "E5RT2_REQUIRE_CONFIRMED_GEOMETRY"):
        monkeypatch.setattr(OB, "E5RT2_REQUIRE_CONFIRMED_GEOMETRY", False)
    TE.run_event(node, clk)
    jobs = list((node.run / "jobs" / "pending").glob("*.json"))
    assert jobs, "no reader job"
    assert json.loads(jobs[0].read_text()).get("observation_id") is not None, "reader job carries no observation hand-off"


def test_before_first_verification_p2_stop_has_no_source_frame(env, monkeypatch):
    """N16: a P2 STOP fired in the same scene as the first verification carried box_frame=None (setdefault kept None),
    so the event/hand-off lost its source frame lineage. Real Node.handle_scene (GatedNode) with a fake frontend; the
    gate file states walking + an explicit live grant (accepted by both trees)."""
    import cv2
    import test_r1_e5 as TE
    gp = env / "gate_lineage.json"
    gp.write_text(json.dumps(dict(active=True, trial_state="walking", t=time.time(), armed=True, arm_grant_id="g",
                                  arm_expires_t=time.time() + 600, operator_abort=False, schema="e5rt-gate-2")))
    monkeypatch.setattr(OB, "GATE", gp)
    monkeypatch.setattr(OB, "_gate_cache", [0.0, None])
    node, _ = TE.make_node(env, F.FakeConsole(), run="before_lineage", stage="P2")
    r = np.random.RandomState(3)
    img = cv2.cvtColor(cv2.GaussianBlur((r.rand(480, 640) * 255).astype(np.uint8), (5, 5), 0), cv2.COLOR_GRAY2BGR)
    src = F.FakeSource()
    src.odom_hist = [(100.0 + 0.1 * i, 0.0, 0.0, 0.0) for i in range(40)]
    node.note_frame(src, 100.4, 4)
    reg = dict(box=[380.0, 200.0, 430.0, 240.0], tile=[380.0, 200.0, 430.0, 240.0], sim=0.15, msim=0.15, like=True, spot=11, nobs=3,
               nscene=3, xyz=[2.5, -0.35, 0.1], z=2.5, hgt=1.3, board=-1.0, adv=0.03, neg={}, ctx={}, ctx_plate=None, fired=False)
    res = dict(t=1.0, base=dict(id=4, bgr=img, tb=0.4), regs=[], vp=None, thr=0.10, legacy_ev=None, timing=dict(scene_ms=1.0), regions=[reg])
    node.handle_scene(src, res, 1.0, 100.4, np.full((480, 640), 2500, np.uint16), (0.0, 0.0, 0.0), img, 1, 0, origin=100.0)
    p = node.proposals.take()
    assert p is not None, "no P2 STOP proposal"
    assert (p["stop"].get("box_frame") or {}).get("frame_id") == 4, "STOP source frame lineage lost"
