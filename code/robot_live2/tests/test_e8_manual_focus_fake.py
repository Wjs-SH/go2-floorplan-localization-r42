"""E8 manual-focus E6 path (2026-10-26 directive): DC focus motor excluded, operator focus ring + [완료] on the 8770 page.
OFFLINE: fake aim console (no serial, no camera, no network), rendered IMX477 frames, the real e6_pipeline / e6_manual /
e8_pair / obs_flow / visual_stop code. Nothing here is hardware or field evidence."""
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

if "serial" not in sys.modules:                 # the test interpreter has no pyserial; no port is ever opened in these tests
    sys.modules["serial"] = types.ModuleType("serial")

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "candidate" / "live2"))
sys.path.insert(0, str(W / "candidate" / "live1"))
import e6_manual as M  # noqa: E402
import e6_pipeline as P  # noqa: E402
import e8_pair as EP  # noqa: E402
import obs_flow as OF  # noqa: E402
import visual_stop as VS  # noqa: E402
from test_e8_pair_fake import Hyp, OCR, T401, floors, manager, stored, wait  # noqa: E402

ENV = dict(pan=(85.0, 106.0), tilt=(147.0, 168.0), source="TEST_ENVELOPE")
HOME = (95.352, 157.189)
OBS = dict(observation_id="O_m1", source="AUTO_E5", target_pose_in_odom=dict(odom_link_xy=[1.0, 0.5]),
           odom_pose_at_stop=dict(t=100.0, x=0.0, y=0.0, yaw=0.0), stopped_wall=1000.0,
           redetection=dict(status="REDETECTED", pixel=[330, 230], depth_m=1.4, pantilt_command=dict(inside_limits=True)))


def render(sharp=True, at=(960, 540)):
    import cv2
    img = np.full((1080, 1920, 3), 120, np.uint8)
    x, y = at
    cv2.rectangle(img, (x - 400, y - 250), (x + 400, y + 250), (235, 235, 235), -1)
    cv2.putText(img, "418", (x - 300, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 10, (25, 25, 25), 28)
    return img if sharp else cv2.GaussianBlur(img, (0, 0), 25)


class FakeConsole:
    """Student aim console + LIVE2 manual routes, faked. Records every job kind; never has a focus motor."""

    def __init__(self, frame, sharp=80.0, pose=HOME, corrupt=0, device="IDLE"):
        import cv2
        self.jpg = cv2.imencode(".jpg", frame)[1].tobytes()
        self.calls, self.sharp, self.pose, self.corrupt, self.seq, self.device = [], sharp, pose, corrupt, 100, device
        self.cam = []

    def run(self, kind, params, *a, **k):
        self.calls.append((kind, dict(params)))
        if kind == "aim":
            assert params["focus"] == 0, "manual path must never ask for autofocus"
            return dict(status="COMPLETED", device_state=self.device, reason_code="captured", capture=dict(capture_id="c_aim"),
                        record=dict(plan=dict(pan=100.0, tilt=160.0)))
        if kind == "live2/manual_capture":
            sha = hashlib.sha256(self.jpg).hexdigest()
            cap = dict(capture_id=f"cap_m{len(self.calls)}", sha256=sha, width=1920, height=1080,
                       metrics=dict(sharp=dict(roi=self.sharp, centre=self.sharp / 2, full=self.sharp / 4), roi=[320, 180, 960, 540],
                                    roi_source="SIGN_COMPONENT", auto_roi_source="SIGN_COMPONENT", roi_warning=[]))
            return dict(status="COMPLETED", device_state="IDLE", record=dict(result=dict(detail=dict(capture=cap))))
        if kind == "live2/pose":
            return dict(status="COMPLETED", device_state="IDLE", record=dict(result=dict(detail=dict(pan=self.pose[0], tilt=self.pose[1]))))
        return dict(status="COMPLETED", device_state="IDLE")          # home

    def fetch_capture(self, cid):
        if cid == "c_aim":
            return self.jpg
        if self.corrupt > 0:                                        # disk / transfer corruption -> sha mismatch
            self.corrupt -= 1
            return self.jpg[:-10] + b"0123456789"
        return self.jpg

    def camera(self, q):
        self.cam.append(dict(q))
        return dict(on_demand=True, open=True, first_frame=True, last_open_s=1.05, opens=len(self.cam))

    def shake(self):
        self.shakes = getattr(self, "shakes", 0) + 1
        return dict(stale=False, shift_px=self.shake_px[min(self.shakes - 1, len(self.shake_px) - 1)] if hasattr(self, "shake_px") else 1.0,
                    response=0.9)

    def metrics(self, roi=None):
        self.seq += 3
        return dict(sharp=dict(roi=self.sharp, centre=self.sharp / 2, full=self.sharp / 4), frame_age_s=0.08, seq=self.seq,
                    roi=list(roi) if roi else [320, 180, 960, 540], roi_source="OPERATOR_SELECTED" if roi else "SIGN_COMPONENT")

    def kinds(self):
        return [k for k, _ in self.calls]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t


def build(tmp, console, script=(), stopped=True, gate_ok=True, moved=None, clock=None, review="accept"):
    """script: list of command dicts (or callables) appended one per sleep() of the wait loop.
    review (2026-11-01): 'accept' = the operator confirms every reviewed photo at once (the pre-review semantics of these tests);
    None = the script itself must answer CAPTURE_REVIEW."""
    ch = M.Channel(tmp, clock=(clock or Clock()).time if clock else time.time)
    queue = list(script)
    clk = clock or types.SimpleNamespace(time=time.time)

    seen_rev = set()

    def sleep(s):
        if clock:
            clock.t += s
        rv = ch.d.get("review") if ch.d.get("phase") == "CAPTURE_REVIEW" else None
        if review == "accept" and rv and rv.get("sha256") + str(rv.get("n")) not in seen_rev:
            seen_rev.add(rv.get("sha256") + str(rv.get("n")))
            M.append_command(tmp, dict(observation_id="O_m1", experiment_id="X1", request_id=f"acc{len(seen_rev)}", hands_clear=True,
                                       action="accept_capture"))
            return
        if queue:
            c = queue.pop(0)
            c = c() if callable(c) else c
            if c:
                M.append_command(tmp, dict(dict(observation_id="O_m1", experiment_id="X1", request_id=f"r{len(queue)}", hands_clear=True), **c))
    gates = []

    def gate(kind, **kw):
        gates.append(kind)
        return (gate_ok, None if gate_ok else "approval B APPROVAL_EXPIRED")
    st = stopped if callable(stopped) else (lambda: dict(stopped=stopped, why="STOPPED" if stopped else "MOVING speed 0.3 m/s"))
    p = P.E6Pipeline(console, gate, st, tmp / "caps", envelope=ENV, focus_mode="MANUAL_FIXED_NO_MOTOR", defer_reading=True,
                     manual=ch, experiment_id="X1", sleep=sleep, clock=clk, moved_check=moved, log=lambda *_: None)
    p.SETTLE = dict(p.SETTLE, body_min_s=0.0, aim_min_s=0.0)          # settle rule itself: test_field_20261027_fixes
    return p, ch, gates


def first(cmd):
    """Command already waiting when the wait loop starts."""
    return [None, cmd]


def phases(ch):
    return [h["phase"] for h in ch.d["history"]]


# ------------------------------------------------------------------ 2 / 3 / 4 / 19: stop gate, range, manual entry, no DC motor
def test_02_no_aim_and_no_manual_flow_before_stop_confirmed(tmp_path):
    c = FakeConsole(render())
    p, ch, gates = build(tmp_path, c, stopped=False)
    rec = p.run(OBS, 1)
    assert rec["status"] == "ROBOT_NOT_STOPPED" and c.calls == [] and gates == [] and not ch.d
    assert not (tmp_path / "e6_manual.json").exists()


def test_03_outside_range_including_the_lash_point_is_refused_before_any_command(tmp_path, monkeypatch):
    import e6_core as G
    c = FakeConsole(render())
    monkeypatch.setattr(G, "calibrated_aim", lambda u, v, z: dict(pan_cmd=88.0, tilt_cmd=160.0, inside_limits=True))
    p, ch, gates = build(tmp_path, c)
    rec = p.run(OBS, 1)                                  # target 88 is inside, its 5 deg lash point 83 is not
    assert rec["status"] == "OUTSIDE_VERIFIED_ENVELOPE" and "lash_pre" in rec["why"] and c.calls == [] and gates == []
    assert ch.d["phase"] == "AIM_FAILED"


def test_04_19_manual_focus_entered_after_aim_and_no_focus_motor_command_ever(tmp_path):
    c = FakeConsole(render())
    p, ch, gates = build(tmp_path, c, script=first(dict(action="capture")))
    rec = p.run(OBS, 1)
    assert phases(ch)[:6] == ["AIMING", "WAIT_BODY_SETTLE", "AIMING", "WAIT_AIM_SETTLE", "AIM_READY", "WAIT_MANUAL_FOCUS"]
    assert c.kinds() == ["aim", "live2/manual_capture", "home", "live2/pose"]          # no 'focus' job at all
    assert all(k != "focus" for k in c.kinds()) and "focus" not in gates
    tr = {t["state"]: t.get("outcome") for t in rec["e6_trace"]}
    assert tr["AUTOFOCUS_IF_NEEDED"] == "NOT_SENT_MANUAL_FIXED_FOCUS" and tr["WAIT_MANUAL_FOCUS"] == "OPERATOR_CONFIRMED"
    assert rec["focus_mode"] == "MANUAL_FIXED_NO_MOTOR" and ch.d["dc_motor"].startswith("NOT_USED")


def test_19_console_refuses_focus_motor_in_manual_mode(monkeypatch):
    import e6_aim_console_live2 as L
    monkeypatch.setenv("LIVE2_FOCUS_MODE", "MANUAL_FIXED_NO_MOTOR")
    ok, why = L.approval_gate("focus")
    assert not ok and "MANUAL_FIXED_NO_MOTOR" in why


# ------------------------------------------------------------------ 8 / 5 / 6 / 7: capture -> HOME -> resume, duplicates, failures
def test_08_capture_saved_verified_then_home_confirmed_then_resume_ready(tmp_path):
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture", roi="300,150,980,560", best_seen=91.0, verdict="sharp")))
    rec = p.run(OBS, 1)
    assert rec["status"] == "CAPTURE_STORED" and rec["home_confirmed"] is True
    order = phases(ch)
    assert order.index("CAPTURING") < order.index("CAPTURE_SAVED") < order.index("RETURNING_HOME") < order.index("HOME_CONFIRMED")
    f = Path(rec["capture_file"])
    assert f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == rec["image_sha256"] == hashlib.sha256(c.jpg).hexdigest()
    q = rec["image_quality"]
    assert q["ok"] and q["basis"] == "OPERATOR_CONFIRMED_MANUAL_FOCUS" and q["sharp_line_met"] and not q["operator_override_low_sharpness"]
    assert {"roi_laplacian_var", "centre_laplacian_var", "full_laplacian_var"} <= set(q)            # three scales kept apart
    assert rec["manual"]["best_seen_roi"] == 91.0 and rec["manual"]["operator_verdict"] == "sharp"
    assert "physical angle NOT measured" in rec["home"]["checks"]["basis"]
    # the record feeds e8_pair; the operator gets RESUME_READY only after it was handled
    pm, _, _ = manager(tmp_path / "pair", [Hyp("H4", 4)], floors(4), OCR)
    (tmp_path / "pair").mkdir(exist_ok=True)
    snap = pm.on_precision(rec)
    o6 = [dict(observation_id="O_m1", state="STOPPED_TARGET_READY", created_wall=990, stopped_wall=1000)]
    man = json.loads((tmp_path / "e6_manual.json").read_text())
    now = man["phase_since"] + 1
    assert OF.flow(o6, {}, {}, None, True, True, False, now, pair=snap, manual=man)["state"] == "E6_OBSERVING"   # record not yet there
    f2 = OF.flow(o6, {"O_m1": rec}, {}, None, True, True, False, now, pair=snap, manual=man)
    assert f2["state"] == "RESUME_READY"


def test_05_duplicate_completion_requests_give_one_capture(tmp_path):
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=[None])
    M.append_command(tmp_path, dict(observation_id="O_m1", experiment_id="X1", request_id="a", hands_clear=True, action="capture"))
    M.append_command(tmp_path, dict(observation_id="O_m1", experiment_id="X1", request_id="b", hands_clear=True, action="capture"))
    M.append_command(tmp_path, dict(observation_id="O_m1", experiment_id="X1", request_id="c", hands_clear=True, action="cancel"))
    rec = p.run(OBS, 1)
    assert c.kinds().count("live2/manual_capture") == 1
    rep = {r["request_id"]: r for r in ch.d["replies"]}
    # 2026-11-01: a capture now opens CAPTURE_REVIEW; a second capture is refused (REVIEW_PENDING), a cancel during the review
    # is a valid decision (the photo is archived, never read, the observation is not counted)
    assert rep["a"]["accepted"] and not rep["b"]["accepted"] and "REVIEW_PENDING" in rep["b"]["why"]
    assert rep["c"]["accepted"] and rec["status"] == "USER_CANCELLED" and rec["manual"]["attempts"][0]["decision"].startswith("NOT_ACCEPTED")


def test_06_save_failure_keeps_waiting_without_home_then_retry_succeeds(tmp_path):
    c = FakeConsole(render(), corrupt=1)
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture")) + [dict(action="capture")])
    rec = p.run(OBS, 1)
    k = c.kinds()
    assert k.count("live2/manual_capture") == 2 and k.index("home") > [i for i, x in enumerate(k) if x == "live2/manual_capture"][1]
    assert "CAPTURE_FAILED" in phases(ch) and rec["status"] == "CAPTURE_STORED"
    fail = [t for t in rec["e6_trace"] if t.get("outcome") == "CAPTURE_FAILED"]
    assert fail and "SHA256_MISMATCH" in fail[0]["why"]
    assert phases(ch).index("CAPTURE_FAILED") < phases(ch).index("RETURNING_HOME")


def test_06b_save_failure_then_no_decision_never_homes_or_resumes_until_cancel(tmp_path):
    c = FakeConsole(render(), corrupt=5)
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture")) + [None, None, dict(action="cancel")])
    rec = p.run(OBS, 1)
    assert rec["status"] == "USER_CANCELLED" and c.kinds()[-2:] == ["home", "live2/pose"]
    pm, _, _ = manager(tmp_path / "pair", [Hyp("H4", 4)], floors(4), OCR)
    (tmp_path / "pair").mkdir(exist_ok=True)
    snap = pm.on_precision(rec)
    assert pm.group == []                                                 # a cancelled observation is never counted


def test_07_home_not_confirmed_blocks_resume_everywhere(tmp_path):
    c = FakeConsole(render(), pose=(100.0, 160.0))                         # Nano target != HOME after the HOME job
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture")))
    rec = p.run(OBS, 1)
    assert rec["home_confirmed"] is False and ch.d["phase"] == "HOME_NOT_CONFIRMED" and p.blocked
    man = json.loads((tmp_path / "e6_manual.json").read_text())
    o6 = [dict(observation_id="O_m1", state="STOPPED_TARGET_READY", created_wall=990, stopped_wall=1000)]
    pm, _, _ = manager(tmp_path / "pair", [Hyp("H4", 4)], floors(4), OCR)
    (tmp_path / "pair").mkdir(exist_ok=True)
    snap = pm.on_precision(rec)
    assert snap["state"] == "HOLD_ERROR" and not snap["resume_ok"]
    now = man["phase_since"] + 1
    assert OF.flow(o6, {"O_m1": rec}, {}, None, True, True, False, now, pair=snap, manual=man)["state"] == "HOLD_ERROR"
    o6n = [dict(o6[0], stopped_wall=now - 5)]                              # without the manual snapshot: the record alone holds
    assert OF.flow(o6n, {"O_m1": rec}, {}, None, True, True, False, now, pair=None, manual=None)["state"] == "HOLD_ERROR"
    assert p.run(dict(OBS, observation_id="O_m2"), 2)["status"] == "BLOCKED"                     # no further device command


# ------------------------------------------------------------------ operator decisions / robot motion / sharpness
def test_low_sharpness_is_a_warning_not_a_refusal_in_manual_mode(tmp_path):
    """2026-10-28 field: manual focus must never be blocked by the AF sharpness line; the numbers become quality flags."""
    c = FakeConsole(render(False), sharp=6.0)
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture", verdict="sharp")))
    rec = p.run(OBS, 1)
    assert ch.d["replies"][0]["accepted"] and "LOW_SHARPNESS" in ch.d["replies"][0]["why"]
    q = rec["image_quality"]
    assert rec["status"] == "CAPTURE_STORED" and q["ok"] and not q["sharp_line_met"] and "LOW_SHARPNESS" in q["flags"]
    assert "MANUAL_ACCEPTED" in q["flags"] and rec["home_confirmed"]


def test_capture_anyway_is_recorded_as_operator_override(tmp_path):
    c = FakeConsole(render(False), sharp=6.0)
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture_anyway", verdict="insufficient")))
    rec = p.run(OBS, 1)
    assert rec["image_quality"]["operator_override_low_sharpness"] and rec["manual"]["operator_verdict"] == "insufficient"


def test_capture_refused_while_moving_and_moved_robot_ends_observation_with_home(tmp_path):
    clk = Clock()
    seq = iter([True] * 5 + [False] * 200)                                 # stopped through both settles, then moving
    stop = lambda: dict(stopped=next(seq), why="MOVING speed 0.4 m/s")   # noqa: E731
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture")) + [None] * 40, stopped=stop, clock=clk)
    rec = p.run(OBS, 1)
    assert not ch.d["replies"][0]["accepted"] and "ROBOT_NOT_STOPPED" in ch.d["replies"][0]["why"]
    assert rec["status"] == "ROBOT_MOVED_DURING_FOCUS" and "live2/manual_capture" not in c.kinds() and "home" in c.kinds()


def test_abort_never_homes_and_holds_e6(tmp_path):
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(dict(action="abort")))
    rec = p.run(OBS, 1)
    assert rec["status"] == "USER_ABORTED" and c.kinds() == ["aim"] and p.blocked and ch.d["phase"] == "USER_ABORTED"
    man = json.loads((tmp_path / "e6_manual.json").read_text())
    o6 = [dict(observation_id="O_m1", state="STOPPED_TARGET_READY", created_wall=990, stopped_wall=1000)]
    assert OF.flow(o6, {}, {}, None, True, True, False, man["phase_since"] + 1, manual=man)["state"] == "HOLD_ERROR"


def test_wrong_session_or_observation_commands_are_refused(tmp_path):
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(dict(action="capture", experiment_id="X_OLD")) +
                     [dict(action="capture", observation_id="O_other"), dict(action="cancel")])
    rec = p.run(OBS, 1)
    rep = ch.d["replies"]
    assert [r["accepted"] for r in rep] == [False, False, True] and all("SESSION_MISMATCH" in r["why"] for r in rep[:2])
    assert rec["status"] == "USER_CANCELLED" and "live2/manual_capture" not in c.kinds()


def test_manual_wait_budget_is_a_notice_never_an_automatic_end_or_home(tmp_path):
    clk = Clock()
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=[None] * 2500 + [dict(action="cancel")], clock=clk)
    rec = p.run(OBS, 1)
    assert ch.d.get("over_budget") and rec["status"] == "USER_CANCELLED" and not p.blocked
    assert c.kinds().count("home") == 1                                   # HOME only after the operator's cancel (hands clear)


def test_blurry_aim_frame_without_sign_still_lets_the_operator_decide(tmp_path):
    c = FakeConsole(np.full((1080, 1920, 3), 90, np.uint8))               # featureless / out of focus: no sign component
    p, ch, _ = build(tmp_path, c, script=first(dict(action="cancel")))
    rec = p.run(OBS, 1)
    assert "WAIT_MANUAL_FOCUS" in phases(ch) and ch.d["target_seen_at_aim"] is False
    assert rec["status"] == "USER_CANCELLED"                              # never reported as an aim success


# ------------------------------------------------------------------ 1 / 9 / 10 / 17: SigLIP dedup, reading, UI responsiveness
def test_01_siglip_stop_requests_are_not_repeated_while_the_robot_stays_still(tmp_path):
    v = VS.VisualStop(tmp_path)
    reg = lambda spot: dict(regions=[dict(spot=spot, sim=0.2, z=2.0, box=[1, 1, 9, 9], dec=dict(P2=dict(state="VERIFIED")))], t=1.0)  # noqa: E731
    line = lambda x: dict(odom_raw=dict(x=x, y=0.0), frame_id=1, depth_dt_s=0.02)        # noqa: E731
    out, k = [], 0
    for i in range(0, 600):                                               # 5 minutes still; a new spot id every 8 s
        t = 100.0 + 0.5 * i
        r = reg(f"s{k}") if i % 16 == 0 else dict(regions=[], t=1.0)
        k += i % 16 == 0
        o = v.feed(r, t, line(0.0), now=t)
        if i % 16 == 0:
            out.append(o)
    assert out[0] is not None and all(o is None for o in out[1:]) and v.n_suppressed >= 30
    got = None
    for i in range(0, 20):                                                # the robot drives on: a new plate may stop it again
        t = 420.0 + 0.5 * i
        got = got or v.feed(reg(f"m{i}"), t, line(0.4 * i), now=t)
    assert got is not None


def test_09_10_17_first_reading_runs_in_background_and_keeps_the_observation_id(tmp_path):
    pm, tr, e7 = manager(tmp_path, [Hyp("H4", 4)], floors(4), OCR)
    slow = pm.reader

    def reader(img):
        time.sleep(1.5)
        return slow(img)
    pm.reader = reader
    t0 = time.time()
    snap = pm.on_precision(stored(tmp_path, "O1", 10, T401))
    assert time.time() - t0 < 0.5 and snap["resume_ok"]                   # the tablet / HOME never wait for OCR
    wait(pm)
    ev = pm.last_eval
    po = ev["per_observation"][0]
    assert ev["verdict"] == "PROVISIONAL" and po["observation_id"] == "O1" and po["capture_id"] == "cap_O1"
    assert po["provisional"] and po["provisional"]["floor"] == 4 and not e7
    rows = [json.loads(x) for x in (tmp_path / "e8_observations.jsonl").read_text().splitlines()]
    rd = [r for r in rows if r["event"] == "READ"]
    assert rd[0]["observation_id"] == "O1" and rd[0]["reading"]["capture_id"] == "cap_O1"


# ------------------------------------------------------------------ console side (8770): action validation, live routes
def test_18_console_action_checks_current_observation_session_phase_and_writes_one_command(tmp_path):
    root = tmp_path / "ws"
    (root / "runtime" / "state").mkdir(parents=True)
    (root / "LIVE2_FAKE_WORKSPACE.json").write_text("{}")
    (root / "candidate").symlink_to(W / "candidate")
    st = root / "runtime" / "state"
    snap = dict(observation_id="O_m1", experiment_id="X1", phase="WAIT_MANUAL_FOCUS", wall=time.time())
    (st / "e6_manual.json").write_text(json.dumps(snap))
    code = r'''
import json, sys, types
sys.path.insert(0, sys.argv[1] + "/candidate/live2"); sys.path.insert(0, sys.argv[1] + "/candidate/live1")
import console as K
ev = []
fake = types.SimpleNamespace(exp=types.SimpleNamespace(d=dict(exp_id="X1"), event=lambda k, **kw: ev.append(k)))
f = lambda **b: K.Controller.a_e6_manual(fake, "rid00000001", b, "sid")
out = [f(action="capture", observation_id="O_m1", roi="300,150,980,560"), f(action="capture", observation_id="O_x"),
       f(action="explode", observation_id="O_m1"), f(action="capture", observation_id="O_m1", roi="1,2")]
json.dump(dict(out=out, ev=ev), sys.stdout)
'''
    env = dict(PATH="/usr/bin:/bin", LIVE1_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1", HOME=str(tmp_path))
    r = subprocess.run(["/usr/bin/python3", "-I", "-c", code, str(root)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads(r.stdout)
    o = res["out"]
    assert o[0]["ok"] and not o[1]["ok"] and "NOT_THE_CURRENT" in o[1]["why"] and o[2]["why"] == "BAD_ACTION" and o[3]["why"] == "BAD_ROI"
    lines = (st / "control" / "e6_manual_cmd.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["experiment_id"] == "X1" and res["ev"] == ["e6_manual_cmd"]
    snap.update(phase="CAPTURING")                                        # not accepted while the capture runs
    (st / "e6_manual.json").write_text(json.dumps(snap))
    r = subprocess.run(["/usr/bin/python3", "-I", "-c", code, str(root)], capture_output=True, text=True, env=env, timeout=60)
    assert "PHASE_CAPTURING" in json.loads(r.stdout)["out"][0]["why"]


def test_console_capture_job_writes_full_frame_with_sha_and_returns_refusals_as_results(tmp_path):
    import e6_aim_console_live2 as L
    frame = render()

    class Stats:
        def __init__(self, fresh):
            self.fresh = fresh

        def get(self):
            return 5, 0.0

        def fresh_frame(self, after, timeout):
            return (6, frame, 123) if self.fresh else None
    owner = types.SimpleNamespace(capture_dir=str(tmp_path))
    r = L.manual_capture_job(types.SimpleNamespace(stats=Stats(True)), owner, "rq:1", None)
    cap = r["capture"]
    data = Path(cap["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == cap["sha256"] and cap["width"] == 1920 and cap["height"] == 1080
    assert set(cap["metrics"]["sharp"]) == {"roi", "centre", "full"}
    assert L.manual_capture_job(types.SimpleNamespace(stats=Stats(False)), owner, "rq:2", None)["capture"] is None


def test_metrics_scales_are_separate_and_background_roi_is_warned():
    img = render()
    m = M.measure(img, None)
    assert m["roi_source"] == "SIGN_COMPONENT" and set(m["sharp"]) == {"roi", "centre", "full"} and not m["roi_warning"]
    bg = M.measure(img, (0, 0, 120, 120))                                  # plain grey corner
    assert bg["roi_source"] == "OPERATOR_SELECTED" and bg["roi_warning"] and any("배경" in w for w in bg["roi_warning"])


def test_home_rule_with_focus_fault_lock_in_manual_console(tmp_path, monkeypatch):
    import e6_aim_console_ctl as K
    monkeypatch.setattr(K, "PIDF", tmp_path / "pid")
    (tmp_path / "pid").write_text("123 /x manual_no_motor\n")
    assert K.manual_console()
    (tmp_path / "pid").write_text("123 /x student_stepwise\n")
    assert not K.manual_console()


def test_console_routes_through_the_real_student_owner_give_the_record_shape_the_pipeline_reads(tmp_path, monkeypatch):
    """install_manual_routes on a fake console Handler + the REAL r1_device_owner.DeviceOwner (no serial: fake Nano)."""
    import threading
    sys.path.insert(0, str(W / "candidate" / "external" / "sci_event" / "code"))
    import r1_device_owner as R1O
    import e6_aim_console_live2 as L
    monkeypatch.setenv("LIVE2_FOCUS_MODE", "MANUAL_FIXED_NO_MOTOR")
    frame = render()
    nano_cmds = []

    class Stats:
        lock = threading.Lock()
        seq, frame, rx_mono_ns = 7, None, time.monotonic_ns()

        def get(self):
            return self.seq, 0.0

        def fresh_frame(self, after, timeout):
            return (after + 1, frame, time.monotonic_ns())
    st = Stats()
    st.frame = frame
    a = types.SimpleNamespace(stats=st, nano=types.SimpleNamespace(cmd=lambda c: nano_cmds.append(c) or dict(pan=95.4, tilt=157.2)))
    hw = types.SimpleNamespace(stop_actuators=lambda: nano_cmds.append("F0"))
    owner = R1O.DeviceOwner(hw, str(tmp_path / "caps"), log=lambda *_: None)

    class Handler:
        def __init__(self, path):
            self.path, self.out = path, None

        def do_GET(self):
            self.out = (404, b"")

        def reply(self, body, ctype="text/plain", code=200):
            self.out = (code, body)
    AC = types.SimpleNamespace(Handler=Handler, AIMER={"a": a}, OWNER={"o": owner})
    L.install_manual_routes(AC)

    def get(path):
        h = Handler(path)
        h.do_GET()
        return h.out
    code, body = get("/live2/metrics?roi=300,150,980,560")
    m = json.loads(body)
    assert code == 200 and m["roi_source"] == "OPERATOR_SELECTED" and set(m["sharp"]) == {"roi", "centre", "full"} and m["seq"] == 7
    acc = json.loads(get("/live2/manual_capture?req=live2:ev001:a3:x&owner=e5:live2")[1])
    assert acc["status"] == "ACCEPTED"
    t0 = time.time()
    while time.time() - t0 < 5:
        rec = owner.get("live2:ev001:a3:x")
        if rec["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.02)
    cap = rec["result"]["detail"]["capture"]                                # exactly what E6Pipeline._manual_capture reads
    assert rec["status"] == "COMPLETED" and rec["device_state"] == "IDLE" and owner.capture_path(cap["capture_id"])
    assert hashlib.sha256(Path(owner.capture_path(cap["capture_id"])).read_bytes()).hexdigest() == cap["sha256"]
    acc = json.loads(get("/live2/pose?req=live2:ev001:a8:y&owner=e5:live2")[1])
    time.sleep(0.3)
    assert owner.get("live2:ev001:a8:y")["result"]["detail"]["pan"] == 95.4 and nano_cmds == ["?"]   # one status query, no motion
    monkeypatch.setenv("LIVE2_FOCUS_MODE", "AUTO")
    assert get("/live2/manual_capture?req=r2&owner=o")[0] == 403                # route exists only in the manual mode
    assert get("/other")[0] == 404                                              # everything else: the student handler
