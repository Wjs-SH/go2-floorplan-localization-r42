"""LIVE2 E6 pipeline (state machine) — offline with a FAKE console (no device, no serial, no network).
Frames: the real AIM #2 IMX477 capture (blurry) and rendered sign frames; OCR is injected except in one test."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "candidate" / "live2"))
sys.path.insert(0, str(W / "candidate" / "live1"))
import e6_pipeline as P  # noqa: E402

AIM2 = W / "runtime" / "e6_b2b3" / "e6_captures" / "single_1792482932.jpg"
ENV = dict(pan=(90.0, 112.0), tilt=(150.0, 166.0), source="TEST_ENVELOPE")
OBS = dict(observation_id="O_test", source="AUTO_E5", target_pose_in_odom=dict(odom_link_xy=[1.0, 0.5]),
           redetection=dict(status="REDETECTED", pixel=[175, 217], depth_m=1.36, pantilt_command=dict(inside_limits=True)))


def render(sharp=True, at=(700, 330), text="418"):
    import cv2
    img = np.full((1080, 1920, 3), 120, np.uint8)
    x, y = at
    cv2.rectangle(img, (x - 400, y - 250), (x + 400, y + 250), (235, 235, 235), -1)
    cv2.putText(img, text, (x - 300, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 10, (25, 25, 25), 28)
    return img if sharp else cv2.GaussianBlur(img, (0, 0), 25)


class FakeConsole:
    def __init__(self, aim_img, focus_img=None, focus_status="AF_CONVERGED", aim_device="IDLE"):
        import cv2
        self.jpg = {"c_aim": cv2.imencode(".jpg", aim_img)[1].tobytes()}
        if focus_img is not None:
            self.jpg["c_focus"] = cv2.imencode(".jpg", focus_img)[1].tobytes()
        self.focus_status, self.aim_device, self.calls = focus_status, aim_device, []

    def run(self, kind, params, *a, **k):
        self.calls.append((kind, dict(params)))
        if kind == "aim":
            assert params["focus"] == 0 and params["strict"] == 1
            return dict(status="COMPLETED", device_state=self.aim_device, reason_code="captured", capture=dict(capture_id="c_aim"),
                        record=dict(plan=dict(pan=109.36, tilt=160.57)))
        if kind == "focus":
            return dict(status="COMPLETED", device_state="IDLE", focus=dict(status=self.focus_status, sec=7.3),
                        capture=dict(capture_id="c_focus") if "c_focus" in self.jpg else None)
        return dict(status="COMPLETED", device_state="IDLE")

    def fetch_capture(self, cid):
        return self.jpg.get(cid)


def make(console, tmp_path, gate_ok=True, stopped=True, ocr=None, vlm=None):
    gates = []

    def gate(kind, **kw):
        gates.append(kind)
        return (gate_ok, None if gate_ok else "approval B APPROVAL_EXPIRED")
    p = P.E6Pipeline(console, gate, lambda: dict(stopped=stopped, fresh=True), tmp_path / "caps", envelope=ENV,
                     ocr=ocr or (lambda img: ([["418", 0.997]], dict(ocr_s=0.1))), vlm=vlm, log=lambda *_: None)
    return p, gates


def states(rec):
    return [(t["state"], t.get("outcome")) for t in rec["e6_trace"]]


def test_sharp_offcentre_sign_full_path_home_then_ocr(tmp_path):
    c = FakeConsole(render(True, at=(560, 300)))                       # sign up-left like AIM #2
    p, gates = make(c, tmp_path)
    rec = p.run(OBS, 1)
    s = dict(states(rec))
    assert rec["status"] == "CAPTURED" and rec["ocr_lines"] == [["418", 0.997]]
    assert s["IMX477_TARGET_CHECK"] == "OFF_CENTER" and s["FINE_AIM_IF_NEEDED"] == "PROPOSED_NOT_APPLIED"
    assert rec["fine_aim"]["applied"] is False
    assert s["FOCUS_CHECK"] == "ALREADY_SHARP" and s["AUTOFOCUS_IF_NEEDED"] == "SKIPPED_ALREADY_SHARP"
    assert [k for k, _ in c.calls] == ["aim", "home"] and gates == ["aim", "home"]
    order = [t["state"] for t in rec["e6_trace"]]
    assert order == list(P.STATES)                                       # every state, in order
    assert order.index("HOME") < order.index("OCR")
    assert s["SEMANTIC_MAP_MATCH"] == "DEFERRED_TO_E34_LINK"
    assert Path(rec["capture_file"]).exists() and rec["resolution"] == [1920, 1080]
    assert all(v is not None for v in rec["e6_timings_ms"].values())


@pytest.mark.skipif(not AIM2.exists(), reason="AIM #2 capture not present")
def test_real_blurry_capture_focus_fault_blocks_home_and_ocr(tmp_path):
    import cv2
    blurry = cv2.imread(str(AIM2))
    c = FakeConsole(blurry, focus_img=blurry, focus_status="SUSPECTED_MECHANICAL_FAULT")
    p, gates = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "FOCUS_FAULT_BLOCKED" and rec["image_quality"]["ok"] is False
    assert [k for k, _ in c.calls] == ["aim", "focus"]                # no HOME after a focus fault
    assert dict(states(rec))["HOME"] == "NOT_SENT_BLOCKED" and rec["ocr_lines"] == []
    assert p.blocked and p.run(dict(OBS, observation_id="O2"), 2)["status"] == "BLOCKED" and len(c.calls) == 2


def test_blurry_then_af_success_uses_focus_capture(tmp_path):
    c = FakeConsole(render(False), focus_img=render(True))
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert [k for k, _ in c.calls] == ["aim", "focus", "home"]
    assert rec["status"] == "CAPTURED" and rec["capture_id"] == "c_focus" and rec["image_quality"]["ok"]


def test_blurry_af_no_improvement_is_not_success(tmp_path):
    c = FakeConsole(render(False), focus_img=render(False), focus_status="AF_NO_IMPROVEMENT")
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "CAPTURED_BLURRY_REOBSERVE" and rec["ocr_lines"] == []
    assert [k for k, _ in c.calls] == ["aim", "focus", "home"]          # normal end -> HOME, no OCR


def test_not_stopped_and_no_approval_send_nothing(tmp_path):
    c = FakeConsole(render(True))
    p, _ = make(c, tmp_path, stopped=False)
    assert p.run(OBS, 1)["status"] == "ROBOT_NOT_STOPPED" and c.calls == []
    p2, gates = make(c, tmp_path, gate_ok=False)
    rec = p2.run(OBS, 1)
    assert rec["status"] == "REFUSED_NO_VALID_APPROVAL" and c.calls == [] and gates == ["aim"]


def test_outside_envelope_refused_before_any_command(tmp_path):
    c = FakeConsole(render(True))
    p, _ = make(c, tmp_path)
    far = dict(OBS, redetection=dict(OBS["redetection"], pixel=[40, 217]))   # pan > 112
    assert p.run(far, 1)["status"] == "OUTSIDE_VERIFIED_ENVELOPE" and c.calls == []


def test_target_not_in_imx477_goes_home(tmp_path):
    c = FakeConsole(np.full((1080, 1920, 3), 110, np.uint8))
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "TARGET_NOT_IN_IMX477" and [k for k, _ in c.calls] == ["aim", "home"]


def test_device_error_on_aim_no_home(tmp_path):
    c = FakeConsole(render(True), aim_device="UNKNOWN")
    p, _ = make(c, tmp_path)
    assert p.run(OBS, 1)["status"] == "DEVICE_ERROR_BLOCKED" and [k for k, _ in c.calls] == ["aim"]


def test_vlm_only_when_ocr_insufficient_and_vlm_only_number_is_no_label(tmp_path):
    import e6_semantic as S
    asked = []
    vlm = lambda img: asked.append(1) or dict(parsed=dict(sign_type="room_number_plate", room_number="201", text=""), sec=1.0)  # noqa: E731
    c = FakeConsole(render(True))
    p, _ = make(c, tmp_path, ocr=lambda img: ([], dict(ocr_s=0.1)), vlm=vlm)
    rec = p.run(OBS, 1)
    assert asked and rec["vlm"]["room_number"] == "201" and dict(states(rec))["VLM_IF_NEEDED"] == "ASKED"
    alts, info = S.label_alternatives([], rec["vlm"])
    assert alts == [] and info["fusion"]["status"] == "VLM_ONLY_UNCONFIRMED"    # never a location label
    c2 = FakeConsole(render(True))
    asked.clear()
    p2, _ = make(c2, tmp_path, vlm=vlm)                                 # confident OCR 418 -> VLM not needed
    assert dict(states(p2.run(OBS, 1)))["VLM_IF_NEEDED"] == "NOT_NEEDED" and not asked


def test_hypothesis_update_rows_carry_the_contract_fields():
    import e6e7_link as L
    r = dict(observation_id="O1", source="AUTO_E5", status="CAPTURED", capture_timestamp=123.0, capture_file="x.jpg",
             image_quality=dict(ok=True, roi_laplacian_var=42.0, min_sharp=30.0), ocr_lines=[["418", 0.997]], vlm=None)
    res = dict(alternatives=[("418", 0.798)], floor_likelihood={4: 0.862, 7: 0.017}, fusion=dict(status="OCR_ONLY"),
               per_hypothesis={"H4": dict(verdict="SEMANTIC_CONSISTENT", observed_room_id="418", map_room_id="room_418",
                                          map_room_name="418 계측실협심", semantic_score=0.798, dist_m=0.0),
                               "H7": dict(verdict="FLOOR_MISMATCH")})
    rows = L.hypothesis_updates(r, res, True, "ok", [dict(id="H4", floor=4), dict(id="H7", floor=7)])
    need = {"observation_id", "hypothesis_id", "observed_room", "map_room", "ocr_confidence", "semantic_likelihood",
            "observation_timestamp", "supporting_evidence", "update_reason"}
    assert all(need <= set(x) for x in rows)
    h4, h7 = rows
    assert h4["map_room"]["id"] == "room_418" and h4["ocr_confidence"] == 0.997 and h4["semantic_likelihood"]["floor"] == 0.862
    assert h4["update_reason"].startswith("DECISIVE") and h4["applied_to_ranking"] and h4["gt_used"] is False
    assert h7["verdict"] == "FLOOR_MISMATCH"
    blurry = L.hypothesis_updates(dict(r, status="CAPTURED_BLURRY_REOBSERVE"), dict(alternatives=[], per_hypothesis={}), False,
                                  "status CAPTURED_BLURRY_REOBSERVE", [dict(id="H4", floor=4)])
    assert blurry[0]["update_reason"].startswith("NO_EVIDENCE") and not blurry[0]["applied_to_ranking"]


def test_want_vlm_student_rule():
    assert P.want_vlm([["418", 0.997]])[0] is False
    assert P.want_vlm([["418", 0.70]])[0] is True
    assert P.want_vlm([])[0] is True


def _manual(console, tmp_path):
    return P.E6Pipeline(console, lambda kind, **kw: (True, None), lambda: dict(stopped=True, fresh=True), tmp_path / "caps",
                        envelope=ENV, ocr=lambda img: ([["418", 0.997]], dict(ocr_s=0.1)), log=lambda *_: None,
                        focus_mode="MANUAL_FIXED_NO_MOTOR")


def test_manual_fixed_focus_never_sends_focus_and_blurry_is_not_success(tmp_path):
    """2026-10-21 plan: AF drive suspected faulty -> optional NO_MOTOR mode. No /focus, blurry -> re-observe, no OCR label."""
    c = FakeConsole(render(False), focus_img=render(True))           # a focus call WOULD return sharp: it must not be made
    rec = _manual(c, tmp_path).run(OBS, 1)
    assert [k for k, _ in c.calls] == ["aim", "home"] and rec["focus_mode"] == "MANUAL_FIXED_NO_MOTOR"
    assert dict(states(rec))["AUTOFOCUS_IF_NEEDED"] == "NOT_SENT_MANUAL_FIXED_FOCUS"
    assert rec["status"] == "CAPTURED_BLURRY_REOBSERVE" and rec["ocr_lines"] == [] and rec.get("focus") is None


def test_manual_fixed_focus_sharp_capture_reads_without_motor(tmp_path):
    c = FakeConsole(render(True))
    rec = _manual(c, tmp_path).run(OBS, 1)
    assert [k for k, _ in c.calls] == ["aim", "home"] and rec["status"] == "CAPTURED" and rec["ocr_lines"] == [["418", 0.997]]
    assert dict(states(rec))["AUTOFOCUS_IF_NEEDED"] == "SKIPPED_ALREADY_SHARP"
    with pytest.raises(ValueError):
        P.E6Pipeline(c, None, None, tmp_path, focus_mode="PWM_MAX")
    src = (W / "candidate" / "live2" / "e6_precision_bridge.py").read_text()
    assert 'focus=0 if a.focus_mode == "manual_fixed" else 1' in src            # --single manual: /aim focus=0 (no AF)


def test_vlm_never_shares_the_gpu_with_active_siglip(tmp_path):
    """Student rule (SigLIP paused before VLM) is now enforced: ACTIVE fresh SigLIP request -> VLM not admitted."""
    now = time.time()
    (tmp_path / "siglip_request.json").write_text(json.dumps(dict(mode="ACTIVE", stamp=now - 1)))
    assert P.siglip_idle(tmp_path, now)[0] is False
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "pause.json").write_text(json.dumps(dict(paused=True)))
    assert P.siglip_idle(tmp_path, now)[0] is True                       # operator pause stops SigLIP
    (tmp_path / "control" / "pause.json").write_text(json.dumps(dict(paused=False)))
    (tmp_path / "siglip_request.json").write_text(json.dumps(dict(mode="ACTIVE", stamp=now - 60)))
    assert P.siglip_idle(tmp_path, now)[0] is True                       # stale request: SigLIP not driven
    assert P.siglip_idle(tmp_path / "none", now)[0] is True


def test_bridge_never_aims_from_a_stale_or_moved_stop():
    import math
    import e6_precision_bridge as B
    now = 5000.0
    o = dict(stopped_wall=now - 5, odom_pose_at_stop=dict(x=10.0, y=-3.0, yaw=0.5))
    assert B.stop_still_valid(o, (now, 10.05, -3.02, 0.51), now)[0]
    assert not B.stop_still_valid(o, (now, 10.6, -3.0, 0.5), now)[0]                     # moved 0.6 m
    assert not B.stop_still_valid(o, (now, 10.0, -3.0, 0.5 + math.radians(9)), now)[0]  # turned 9 deg
    assert not B.stop_still_valid(dict(o, stopped_wall=now - 90), (now, 10.0, -3.0, 0.5), now)[0]
    assert not B.stop_still_valid(o, None, now)[0]


@pytest.mark.parametrize("st", ["SUSPECTED_DRIVE_FAULT", "AF_BLOCKED_AFTER_FAULT"])
def test_student_af_drive_fault_blocks_home_and_ocr(tmp_path, st):
    c = FakeConsole(render(False), focus_img=render(False), focus_status=st)
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "FOCUS_FAULT_BLOCKED" and [k for k, _ in c.calls] == ["aim", "focus"] and rec["ocr_lines"] == []


@pytest.mark.parametrize("st", ["AF_TARGET_LOST", "AF_LOW_EXPOSURE", "AF_FRAME_STALE", "AF_BUDGET_INFEASIBLE"])
def test_student_af_non_fault_failures_end_normally_without_ocr_success(tmp_path, st):
    c = FakeConsole(render(False), focus_img=render(False), focus_status=st)
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "CAPTURED_BLURRY_REOBSERVE" and [k for k, _ in c.calls] == ["aim", "focus", "home"]


@pytest.mark.parametrize("st", ["AF_DRIVE_LIMIT"])           # AF_NO_RESPONSE: no HOME at all (test_e6_focus_safety_fake)
def test_drive_cap_without_confirmed_drive_goes_home_then_blocks_the_experiment(tmp_path, st):
    """2026-10-22: cap reached -> this observation ends (HOME sent, no OCR on a blurry frame) and NO further aim / AF in this
    experiment (no automatic retry)."""
    c = FakeConsole(render(False), focus_img=render(False), focus_status=st)
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "FOCUS_FAULT_BLOCKED" and [k for k, _ in c.calls] == ["aim", "focus", "home"] and rec["ocr_lines"] == []
    assert p.blocked and st in p.blocked
    rec2 = p.run(dict(OBS, observation_id="O_next"), 2)
    assert rec2["status"] == "BLOCKED" and [k for k, _ in c.calls] == ["aim", "focus", "home"]   # nothing new sent


def test_drive_cap_with_a_sharp_capture_keeps_the_reading_but_blocks_later_observations(tmp_path):
    c = FakeConsole(render(False), focus_img=render(True), focus_status="AF_DRIVE_LIMIT")
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "CAPTURED" and rec["ocr_lines"] and p.blocked
    assert p.run(dict(OBS, observation_id="O_next"), 2)["status"] == "BLOCKED"


def test_focus_comm_error_blocks_without_home(tmp_path):
    c = FakeConsole(render(False), focus_img=render(False), focus_status="FOCUS_COMM_ERROR")
    p, _ = make(c, tmp_path)
    rec = p.run(OBS, 1)
    assert rec["status"] == "FOCUS_FAULT_BLOCKED" and [k for k, _ in c.calls] == ["aim", "focus"]


def test_focus_mode_from_approval_b(tmp_path, monkeypatch):
    import e6_precision_bridge as B
    monkeypatch.setattr(B, "FAULT_FILE", tmp_path / "e6_focus_fault.json")
    prof = json.loads((W / "candidate" / "live2" / "e6_af_profile_kci.json").read_text())
    ok_lim = dict(focus_mode="AUTOFOCUS_V2", focus_mechanical_check="PASSED", af_profile_id=prof["profile_id"],
                  af_profile_verification_run=True)
    assert B.focus_mode_from_approval(dict(limits=ok_lim)) == ("AUTO", None)
    assert B.focus_mode_from_approval(dict(limits=dict(focus_mode="MANUAL_FIXED_NO_MOTOR"))) == ("MANUAL_FIXED_NO_MOTOR", None)
    assert B.focus_mode_from_approval(dict(limits=dict(ok_lim, focus_mechanical_check=None)))[0] is None
    assert B.focus_mode_from_approval(dict(limits={}))[0] is None
    (tmp_path / "e6_focus_fault.json").write_text("{}")                     # a recorded fault needs focus_fault_cleared
    assert B.focus_mode_from_approval(dict(limits=ok_lim))[0] is None
    assert B.focus_mode_from_approval(dict(limits=dict(ok_lim, focus_fault_cleared=True))) == ("AUTO", None)
    out = B.bridge_state(tmp_path, state="BLOCKED", why="FOCUS_AF_NO_RESPONSE")
    assert json.loads((tmp_path / "e6_bridge.json").read_text())["state"] == "BLOCKED" and out["kind"] == "LIVE2_E6_BRIDGE"


def test_link_reports_why_an_observation_has_no_fix(tmp_path):
    """Every processed observation gets <state>/e7_observation.json: NOT_RUN with the reason (here: blurry capture)."""
    import types
    import e6e7_link as L
    st = tmp_path / "state"
    st.mkdir()
    rec = dict(kind="LIVE2_E6_PRECISION", observation_id="O_blur", source="AUTO_E5", status="CAPTURED_BLURRY_REOBSERVE",
               image_quality=dict(ok=False, roi_laplacian_var=6.0), ocr_lines=[], target_odom_link_xy=[1.0, 0.5])
    (st / "e6_precision_results.jsonl").write_text(json.dumps(rec) + "\n")
    hs = [types.SimpleNamespace(id="H4", floor=4, rec={}, light_dir="/L4", sem_verdict=None, status="ACTIVE", rank_key=lambda: 0)]
    tr = types.SimpleNamespace(alive=lambda: hs, apply_semantic=lambda *a: 0, light=lambda d: d,
                               store=types.SimpleNamespace(refresh=lambda: None, rows=[], session=lambda: {}))
    link = L.E6E7Link(st, tr, sem_floors=object(), policy="SINGLE_DIAGNOSTIC")
    link.tick()
    ob = json.loads((st / "e7_observation.json").read_text())
    assert ob["observation_id"] == "O_blur" and ob["outcome"] == "E7_NOT_RUN" and "NO_EVIDENCE" in ob["why"]
    assert not (st / "e7_state.json").exists()                              # the last E7 run (a FIX) is never overwritten


def test_bridge_auto_mode_skips_old_stops_runs_new_ones_and_stays_blocked(tmp_path, monkeypatch):
    """E8 automatic mode (ctl module e6): focus mode from approval B; a stop that existed before the bridge started is never
    aimed at; a new STOPPED_TARGET_READY runs the pipeline once; after a block the bridge stays alive, sends nothing more and
    reports BLOCKED for the tablet. OFFLINE: approval, console client and pipeline are fakes."""
    import types
    import e6_precision_bridge as B
    import e6_pipeline as PP
    import live1_common as C
    st = tmp_path / "state"
    st.mkdir()
    monkeypatch.setattr(C, "approval_status", lambda kind, mod=None: dict(valid=True, why=None, file="B_test.json", expires_epoch=9e9,
                                                                         limits=dict(focus_mode="MANUAL_FIXED_NO_MOTOR")))
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    monkeypatch.setitem(sys.modules, "aim_client", types.SimpleNamespace(AimClient=lambda url: types.SimpleNamespace(probe=lambda: dict(ok=True))))
    old, new = dict(observation_id="O_old", state="STOPPED_TARGET_READY"), dict(observation_id="O_new", state="STOPPED_TARGET_READY")
    calls = iter([[old]] + [[old, new]] * 100)
    monkeypatch.setattr(B, "ready", lambda state: next(calls))
    monkeypatch.setattr(B, "stop_still_valid", lambda o, odom, now: (True, None))
    ran, modes = [], []

    class FakePipe:
        def __init__(self, *a, focus_mode=None, **k):
            modes.append(focus_mode)
            self.blocked = None

        def run(self, o, idx):
            ran.append(o["observation_id"])
            self.blocked = "FOCUS_AF_NO_RESPONSE — test"
            return dict(kind="LIVE2_E6_PRECISION", observation_id=o["observation_id"], status="FOCUS_FAULT_BLOCKED")
    monkeypatch.setattr(PP, "E6Pipeline", FakePipe)
    monkeypatch.setattr(sys, "argv", ["e6_precision_bridge.py", "--state", str(st), "--hub", str(tmp_path / "hub"), "--execute",
                                      "--focus-mode", "from_approval", "--max-s", "3"])
    assert B.main() == 0
    assert modes == ["MANUAL_FIXED_NO_MOTOR"] and ran == ["O_new"]
    res = [json.loads(x) for x in (st / "e6_precision_results.jsonl").read_text().splitlines()]
    assert [r["observation_id"] for r in res] == ["O_new"]
    b = json.loads((st / "e6_bridge.json").read_text())
    assert b["state"] == "BLOCKED_ENDED" and "AF_NO_RESPONSE" in b["why"] and b["focus_mode"] == "MANUAL_FIXED_NO_MOTOR"


def test_bridge_auto_mode_refuses_without_a_usable_focus_mode(tmp_path, monkeypatch):
    import types
    import e6_precision_bridge as B
    import live1_common as C
    st = tmp_path / "state"
    st.mkdir()
    monkeypatch.setattr(C, "approval_status", lambda kind, mod=None: dict(valid=True, why=None, limits=dict(focus_mode="AUTOFOCUS_V2")))
    monkeypatch.setenv("SCI_PANTILT_APPROVED", "1")
    monkeypatch.setitem(sys.modules, "aim_client", types.SimpleNamespace(AimClient=lambda url: None))
    monkeypatch.setattr(sys, "argv", ["e6_precision_bridge.py", "--state", str(st), "--execute", "--focus-mode", "from_approval"])
    with pytest.raises(SystemExit) as e:
        B.main()
    assert "REFUSED" in str(e.value) and json.loads((st / "e6_bridge.json").read_text())["state"] == "REFUSED"
