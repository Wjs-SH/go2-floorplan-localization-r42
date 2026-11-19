"""STAGING ONLY (2026-10-30): FAST-LIO input mode l1 | body through the REAL console + ctl + supervisor in a FAKE workspace
(same infrastructure as test_console_fake.py; no ROS, no FAST-LIO, no sensors, no robot).
Checks: default start = l1 (experiment, ctl args, owner record); 'body' is recorded end to end; the body gate checks the alt-workspace
files; a running FAST-LIO of the other input is never silently reused; bad values are refused; /api/ctl/state shows the input."""
import json
import time

from test_console_fake import WS, Client, ws, FULL, LIM  # noqa: F401  (fixture re-used)


def _console(ws):
    rc, out = ws.ctl("console-token")
    ws.token = (ws.root / "runtime" / "ctl" / "operator_token").read_text().strip()
    ws.approve(modules=FULL + ["console"], limits=dict(LIM, console={"max_run_s": 3600}))
    rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "900")
    assert rc == 0, out
    c = Client(ws.port)
    for _ in range(100):                     # current console: device mode -> GET /api/session issues the device session + csrf
        try:
            code, info = c.req("GET", "/api/session")
            if code == 200 and info.get("csrf"):
                c.csrf = info["csrf"]
                break
            if code == 200 and info.get("mode") != "device" and c.login(ws.token)[0] == 200:
                break
        except OSError:
            pass
        time.sleep(0.2)
    assert c.csrf, info
    ws.client = c
    return c


def _start_gt(c):
    """evaluation-only start mark required by the console before [start] (fake workspace; never an estimator input)"""
    code, j = c.act("gt_save", building="kci", floor=4, kind="start", px=100.0, py=100.0, yaw_deg=0.0)
    assert code == 200, (code, j)


def _run(ws, c, **kw):
    _start_gt(c)
    code, j = c.act("start", building="kci", siglip=False, **kw)
    assert code == 202, j
    s = c.wait_job(timeout=120)
    return s


def _end(ws, c):
    code, j = c.act("end")
    assert code in (200, 202), j
    t0 = time.time()
    while time.time() - t0 < 120:
        s = c.state()
        if s["experiment"].get("state") in ("ENDED", None) and not (s.get("job") or {}).get("done") is False:
            if not any(ws.alive(m) for m in ("fastlio", "lio2", "hub", "rec", "observe", "e34", "e2")):
                return s
        time.sleep(0.5)
    raise AssertionError("experiment did not end")


def test_a_default_is_l1(ws):
    c = _console(ws)
    s = _run(ws, c)
    assert s["experiment"]["state"] == "RUNNING", s["job"]
    assert s["experiment"]["input"] == "l1"
    assert ws.owner("fastlio").get("input") == "l1"
    assert s["fastlio_input"]["input"] == "l1"
    order = [t["module"] for t in s["job"]["steps"]]
    assert [m for m in order if m != "e6"] == ["fastlio", "lio2", "hub", "rec", "observe", "e34", "e2"], order   # e6 = OFF without approval B
    _end(ws, c)


def test_b_body_recorded_end_to_end(ws):
    c = ws.client
    s = _run(ws, c, input="body")
    assert s["experiment"]["state"] == "RUNNING", s["job"]
    assert s["experiment"]["input"] == "body"
    rec = ws.owner("fastlio")
    assert rec.get("input") == "body" and rec.get("body_calib", "").endswith("CANDIDATE.yaml") and len(rec.get("body_calib_sha256", "")) == 64
    # FAKE workspaces use ctl.fake_gates (no file/graph gates); the real body gates are checked in test_f below
    st = c.state()
    assert st["fastlio_input"]["input"] == "body" and st["experiment"]["input"] == "body"
    _end(ws, c)


def test_c_bad_input_refused(ws):
    _start_gt(ws.client)
    code, j = ws.client.act("start", building="kci", siglip=False, input="lidar_only")
    assert code == 400 and "BAD_INPUT" in j["why"], (code, j)


def test_d_ctl_refuses_input_on_other_modules(ws):
    rc, out = ws.ctl("start", "lio2", "--input", "body", "--run-s", "60")
    assert rc == 3 and "only 'fastlio' takes --input" in out, out


def test_e_running_l1_fastlio_not_reused_for_body(ws):
    c = ws.client
    ws.ctl("stop", "fastlio", timeout=120)                                    # clear the ended experiment's owner record
    rc, out = ws.ctl("start", "fastlio", "--run-s", "300")                 # an L1 FAST-LIO left running outside an experiment
    assert rc == 0, out
    assert ws.owner("fastlio").get("input") == "l1"
    _start_gt(c)
    code, j = c.act("start", building="kci", siglip=False, input="body")
    s = c.wait_job(timeout=120)
    st = s["job"]["steps"][0]
    assert st["module"] == "fastlio" and not st["ok"] and "not reused" in str(st.get("why")), s["job"]
    _end(ws, c)
    ws.ctl("stop", "fastlio", timeout=120)


def test_f_real_body_gates_static():
    """real (non-FAKE) gate code path for --input body, evaluated with the ROS graph probe stubbed out (no ROS needed)."""
    import importlib.util, os, sys
    from pathlib import Path
    cand = Path(__file__).resolve().parents[1] / "candidate"
    spec = importlib.util.spec_from_file_location("ctl_stage", cand / "live1" / "ctl.py")
    import pwd
    os.environ.pop("LIVE1_ROOT", None)
    os.environ["HOME"] = pwd.getpwuid(os.getuid()).pw_dir                  # the runner's HOME is a test dir; ctl paths use the real home
    sys.path.insert(0, str(cand / "live1"))
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    assert not ctl.C.FAKE
    for f in (ctl.BODY_LAUNCHER, ctl.BODY_ADAPTER_BIN, ctl.BODY_BRIDGE_BIN, ctl.BODY_CALIB):
        assert f.exists(), f
    ok_graph = dict(ok=True, lidar_points_pubs=1, lidar_points_subs=0, imu_pubs=1, odom_pubs=0, odom_subs=0, cloud_registered_pubs=0,
                    cloud_registered_subs=0, color_pubs=1, color_subs=0, depth_pubs=1, info_pubs=1, nodes=[])
    ctl.graph = lambda: dict(ok_graph)
    ctl.approval_gates = lambda g, *a, **k: None
    g = ctl.gates("fastlio", run_s=60, input_mode="body")
    assert g["fastlio_body_input"]["ok"] and g["fastlio_topology"]["ok"] and g["fastlio_topology"]["input"] == "body", g
    ctl.graph = lambda: dict(ok_graph, odom_pubs=1)                       # an L1 FAST-LIO already publishing /Odometry
    assert not ctl.gates("fastlio", run_s=60, input_mode="body")["fastlio_topology"]["ok"]
    ctl.graph = lambda: dict(ok_graph, imu_pubs=0)                        # L1 rule unchanged: needs the /utlidar/imu publisher
    assert not ctl.gates("fastlio", run_s=60, input_mode="l1")["fastlio_topology"]["ok"]
    assert ctl.gates("fastlio", run_s=60, input_mode="body")["fastlio_topology"]["ok"]   # body does not need /utlidar/imu
