"""STAGING ONLY (2026-10-30): E6 run mode AUTO | MANUAL_SETTLE through the REAL console + ctl + supervisor in a FAKE workspace
(test_console_fake infrastructure; no ROS, no device, no robot). No approval B record exists in the fake workspace, so the E6 bridge
is OFF exactly as in the 2026-10-30 field runs: the tablet must still show the confirmed stop and refuse the pan-tilt button."""
import json
import time

from test_console_fake import WS, Client, ws, FULL, LIM  # noqa: F401  (fixture re-used)
from test_input_mode_fake import _console, _start_gt, _end


def _settle(c, act, oid):
    c.n += 1
    return c.req("POST", "/api/ctl/e6_settle", dict(action=act, observation_id=oid, request_id=f"settle{c.n}{int(time.time() * 1000) % 100000}"),
                 {"X-CSRF-Token": c.csrf or ""})


def _run(ws, c, **kw):
    _start_gt(c)
    code, j = c.act("start", building="kci", siglip=False, **kw)
    assert code == 202, j
    return c.wait_job(timeout=120)


def test_a_default_auto_recorded(ws):
    c = _console(ws)
    s = _run(ws, c)
    assert s["experiment"]["e6_mode"] == "AUTO"
    assert s["e6_settle"]["mode"] == "AUTO"
    tl = [t for t in s["timeline"] if t["kind"] == "experiment_start"]
    assert tl and tl[-1].get("e6_mode") == "AUTO"
    _end(ws, c)


def test_b_bad_e6_mode_refused(ws):
    _start_gt(ws.client)
    code, j = ws.client.act("start", building="kci", siglip=False, e6_mode="FULL_AUTO_NOW")
    assert code == 400 and "BAD_E6_MODE" in j["why"], (code, j)


def test_c_manual_settle_without_b_shows_stop_and_refuses(ws):
    c = ws.client
    s = _run(ws, c, e6_mode="MANUAL_SETTLE", input="body")
    assert s["experiment"]["e6_mode"] == "MANUAL_SETTLE" and s["experiment"]["input"] == "body"
    e6 = s["experiment"]
    st = c.state()
    assert st["modules"]["e6"]["exp"]["state"] == "OFF"                     # no approval B in the fake workspace
    # a confirmed stop as e6_tracker writes it (e34 owns e6_state.json); the console derives the panel from it
    oid = "live1_observe_TEST_O001"
    now = time.time()
    (ws.root / "runtime" / "state").mkdir(parents=True, exist_ok=True)
    (ws.root / "runtime" / "state" / "e6_state.json").write_text(json.dumps(dict(kind="LIVE2_E6_STATE", recent=[dict(
        experiment_id=e6["exp_id"], observation_id=oid, state="STOPPED_TARGET_READY", created_timestamp=now - 8, stopped_wall=now - 2,
        detection_timestamp=now - 10, stop_check=dict(stopped=True, speed_mps=0.03, yaw_rate_dps=0.2))])))
    v = c.state()["e6_settle"]
    assert v["mode"] == "MANUAL_SETTLE" and v["pending_stop"]["observation_id"] == oid and v["approval_b"]["valid"] is False
    assert v["message"].startswith("로봇 정지 완료")
    code, j = _settle(c, "approve", oid)
    assert code == 409 and "APPROVAL_B_REQUIRED" in j["why"], (code, j)
    code, j = _settle(c, "cancel", oid)
    assert code == 409 and "E6_NOT_RUNNING" in j["why"], (code, j)
    # page refresh = a new GET: the same event is shown again from the server state
    v2 = c.state()["e6_settle"]
    assert v2["pending_stop"]["observation_id"] == oid
    _end(ws, c)


def test_d_settle_action_in_auto_experiment_refused(ws):
    c = ws.client
    _run(ws, c)
    code, j = _settle(c, "approve", "x")
    assert code == 409 and "NOT_MANUAL_SETTLE_MODE" in j["why"], (code, j)
    code, j = _settle(c, "launch", "x")
    assert code == 409 and "BAD_ACTION" in j["why"]
    _end(ws, c)


def test_e_ctl_e6_mode_only_for_e6(ws):
    rc, out = ws.ctl("start", "lio2", "--e6-mode", "MANUAL_SETTLE", "--run-s", "60")
    assert rc == 3 and "only 'e6' takes --e6-mode" in out, out
