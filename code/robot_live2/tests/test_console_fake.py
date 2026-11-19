"""LIVE2 operator console — FAKE workspace verification (no ROS, no FAST-LIO, no GPU, no sensor, no robot).

A tmp workspace with only LIVE2_FAKE_WORKSPACE.json makes ctl launch live2/fake_module.py under the REAL live1_supervisor;
approval records, run limits, PID+starttime ownership, the console HTTP API, auth, CSRF and de-duplication are the real code.
Tests run in file order and share one workspace (one experiment lifecycle after another).
"""
import http.cookiejar
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

W = Path(__file__).resolve().parents[1]
CAND = W / "candidate"
CTL = CAND / "live1" / "ctl.py"
FULL = ["fastlio", "lio2", "hub", "observe", "web", "e2", "e34", "frame_probe"]
LIM = {m: {"max_run_s": 1800} for m in FULL}
LIM["lio2"]["accum_cloud_registered"] = True
LIM["web"] = {"max_run_s": 3600}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class WS:
    def __init__(self, root):
        self.root = root
        self.port = free_port()
        import pwd                                   # real home: maps are only READ from ~/sci_ws (runner HOME is a test dir)
        self.env = dict(PATH="/usr/bin:/bin", HOME=pwd.getpwuid(os.getuid()).pw_dir, LIVE1_ROOT=str(root), LANG="C.UTF-8",
                        PYTHONDONTWRITEBYTECODE="1")

    def ctl(self, *args, timeout=120):
        r = subprocess.run(["/usr/bin/python3", "-I", "-B", str(CTL), *args], capture_output=True, text=True, env=self.env, timeout=timeout)
        return r.returncode, r.stdout + r.stderr

    def approve(self, modules=FULL, limits=LIM, minutes=30):
        t = self.root / "approval_text.txt"
        t.write_text("FAKE workspace test approval text (pytest) — not a real user approval")
        rc, out = self.ctl("record-approval", "A", "--text-file", str(t), "--minutes", str(minutes), "--modules", ",".join(modules),
                           "--limits", json.dumps(limits), "--recorded-by", "pytest FAKE")
        assert rc == 0, out
        time.sleep(1.05)                     # record file names have 1 s resolution; the newest record wins
        return sorted((self.root / "runtime" / "approvals").glob("A_*.json"))[-1]

    def edit_approval(self, **kv):
        p = sorted((self.root / "runtime" / "approvals").glob("A_*.json"))[-1]
        d = json.loads(p.read_text())
        d.update(kv)
        p.write_text(json.dumps(d))

    def owner(self, m):
        p = self.root / "runtime" / "owned" / f"{m}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def alive(self, m):
        r = self.owner(m)
        if not r:
            return False
        try:
            s = open(f"/proc/{r['pid']}/stat").read()
            return int(s[s.rindex(")") + 2:].split()[19]) == r["start"]
        except OSError:
            return False


class Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.csrf = None
        self.n = 0

    def req(self, method, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        h = {"Content-Type": "application/json"} if data is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with self.op.open(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def login(self, tok):
        c, j = self.req("POST", "/api/login", {"token": tok})
        self.csrf = j.get("csrf")
        return c, j

    def act(self, action, rid=None, **body):
        self.n += 1
        body["request_id"] = rid or f"req{os.getpid()}x{self.n:04d}{int(time.time() * 1000) % 100000}"
        return self.req("POST", "/api/ctl/" + action, body, {"X-CSRF-Token": self.csrf or ""})

    def state(self):
        return self.req("GET", "/api/ctl/state")[1]

    def wait_job(self, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.state()
            if s.get("job") and s["job"].get("done"):
                return s
            time.sleep(0.3)
        raise AssertionError(f"job not done: {self.state().get('job')}")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("live2_fake_ws")
    (root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="pytest")))
    (root / "candidate").symlink_to(CAND)
    for d in ("runtime/approvals", "runtime/state", "logs", "reports"):
        (root / d).mkdir(parents=True, exist_ok=True)
    w = WS(root)
    yield w
    w.ctl("stop", "all", timeout=200)
    left = [p for p in os.listdir("/proc") if p.isdigit() and str(root) in open(f"/proc/{p}/cmdline", "rb").read().decode(errors="replace")]
    assert not left, f"processes left in the fake workspace: {left}"


def test_00_fake_mode_never_in_real_workspace():
    r = subprocess.run(["/usr/bin/python3", "-I", "-c", "import sys; sys.path.insert(0, %r); import live1_common as C; print(C.FAKE, C.MARKER.name)"
                        % str(CAND / "live1")], capture_output=True, text=True, env=dict(PATH="/usr/bin", LIVE1_ROOT=str(W)))
    assert r.stdout.split() == ["False", "LIVE_FIELD2_WORKSPACE.json"]
    rc = subprocess.run(["/usr/bin/python3", "-I", str(CAND / "live2" / "fake_module.py"), "--module", "hub", "--root", str(W)],
                        capture_output=True, text=True, timeout=10)
    assert rc.returncode != 0 and "REFUSED" in rc.stderr


def test_01_no_approval_refuses_console_and_modules(ws):
    rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "600")
    assert rc == 3 and "NO_APPROVAL_RECORD" in out
    rc, out = ws.ctl("start", "fastlio", "--run-s", "60")
    assert rc == 3 and "approval_A" in out and not ws.alive("fastlio")


def test_02_console_up_web_only_approval_refuses_experiment(ws):
    ws.approve(modules=["web"], limits={"web": {"max_run_s": 3600}})
    rc, out = ws.ctl("console-token")
    assert rc == 0
    ws.token = (ws.root / "runtime" / "ctl" / "operator_token").read_text().strip()
    assert oct((ws.root / "runtime" / "ctl" / "operator_token").stat().st_mode)[-3:] == "600"
    rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "900")
    assert rc == 0, out
    c = Client(ws.port)
    for _ in range(40):
        try:
            if c.req("GET", "/api/session")[0] == 200:
                break
        except OSError:
            time.sleep(0.25)
    assert c.login(ws.token)[0] == 200
    code, j = c.act("start", building="kci", siglip=True)
    assert code == 403 and "MODULES_NOT_APPROVED" in j["why"]
    ws.client = c


def test_03_auth_csrf_origin_json_requestid(ws):
    anon = Client(ws.port)
    assert anon.act("start", building="kci")[0] == 401                              # not logged in
    assert anon.login("wrong")[0] == 401
    c = ws.client
    code, j = c.req("POST", "/api/ctl/pause", {"request_id": "abcdefgh123"}, {"X-CSRF-Token": "bad"})
    assert code == 403 and j["why"] == "CSRF_TOKEN_MISMATCH"
    code, j = c.req("POST", "/api/ctl/pause", {"request_id": "abcdefgh124"}, {"X-CSRF-Token": c.csrf, "Origin": "http://evil.example"})
    assert code == 403 and j["why"] == "CROSS_ORIGIN_REFUSED"
    code, j = c.req("POST", "/api/ctl/pause", raw=b"request_id=x", headers={"Content-Type": "application/x-www-form-urlencoded",
                                                                           "X-CSRF-Token": c.csrf})
    assert code == 415
    code, j = c.req("POST", "/api/ctl/pause", {}, {"X-CSRF-Token": c.csrf})
    assert code == 400 and j["why"] == "REQUEST_ID_REQUIRED"
    code, j = c.act("shell", cmd="id")
    assert code == 404 and j["why"] == "ACTION_NOT_ALLOWED"
    code, j = c.req("GET", "/api/state", headers={"Host": "evil.example:8770"})
    assert code == 421
    assert c.req("GET", "/api/session", headers={"Host": f"localhost:{ws.port}"})[0] == 200
    audit = [json.loads(x) for x in open(ws.root / "runtime" / "ctl" / "audit.jsonl")]
    assert any(a.get("action") == "host_refused" and a.get("host") == "evil.example:8770" for a in audit)
    assert any(a.get("why") == "CSRF_TOKEN_MISMATCH" for a in audit) and all("token" not in (a.get("params") or {}) for a in audit)


def test_04_start_runs_every_module_in_order_with_ownership(ws):
    ws.approve()
    c = ws.client
    code, j = c.act("start", rid="startreq0001", building="kci", siglip=True)
    assert code == 202, j
    s = c.wait_job()
    assert s["experiment"]["state"] == "RUNNING", s["job"]
    order = [t["module"] for t in s["job"]["steps"]]
    assert order == ["fastlio", "lio2", "hub", "rec", "observe", "e34", "e2"]
    for m in order:
        assert ws.alive(m), m
        sup = json.loads((ws.root / "runtime" / "owned" / f"{m}.sup_state.json").read_text())
        assert sup["status"] == "RUNNING"
    code, j = c.act("start", rid="startreq0001", building="kci")                  # same request id: not executed again
    assert code == 200 and j["duplicate"]
    code, j = c.act("start", building="kci")                                        # new id: one experiment only
    assert code == 409 and "EXPERIMENT_ALREADY_ACTIVE" in j["why"]
    st = c.req("GET", "/api/state")[1]
    assert st["meta"]["fake"] and st["meta"]["experiment_state"] == "RUNNING" and not st["meta"]["live"]
    ws.exp_id = s["experiment"]["exp_id"]


def test_05_other_session_cannot_control(ws):
    other = Client(ws.port)
    other.login(ws.token)
    code, j = other.act("pause")
    assert code == 423
    code, j = other.act("takeover", token="wrong")
    assert code == 401


def test_06_pause_holds_analysis_and_resume_restarts_siglip(ws):
    c = ws.client
    obs_pid = ws.owner("observe")["pid"]
    assert c.act("pause")[0] == 202
    s = c.wait_job()
    assert s["experiment"]["state"] == "PAUSED"
    pause = json.loads((ws.root / "runtime" / "state" / "control" / "pause.json").read_text())
    assert pause["paused"] is True
    assert not ws.alive("observe") and ws.alive("fastlio") and ws.alive("lio2") and ws.alive("hub")
    time.sleep(0.6)
    assert json.loads((ws.root / "runtime" / "state" / "e2" / "e2_state.json").read_text())["status"] == "PAUSED"
    assert c.act("e2_research")[0] == 409                                           # paused: resume first
    assert c.act("resume")[0] == 202
    s = c.wait_job()
    assert s["experiment"]["state"] == "RUNNING" and ws.alive("observe") and ws.owner("observe")["pid"] != obs_pid
    assert json.loads((ws.root / "runtime" / "state" / "control" / "pause.json").read_text())["paused"] is False


def test_07_siglip_toggle_and_e2_research(ws):
    c = ws.client
    assert c.act("siglip_off")[0] == 202 and c.wait_job()["job"]["result"]["ok"] and not ws.alive("observe")
    assert c.act("siglip_on")[0] == 202 and c.wait_job()["job"]["result"]["ok"] and ws.alive("observe")
    code, j = c.act("e2_research")
    assert code == 200 and json.loads((ws.root / "runtime" / "state" / "e2" / "e2_request.json").read_text())["why"] == "OPERATOR_REQUEST"


def test_08_gt_and_detection_labels_go_to_the_evaluation_store_only(ws):
    c = ws.client
    code, j = c.act("gt_save", kind="start", building="kci", floor=1, px=1500, py=600, yaw_deg=90.0, note="pytest")
    assert code == 200 and j["gt"]["used_by_inference"] is False and j["gt"]["map_m"]
    assert c.act("gt_save", kind="start", building="kci", floor=1, px=99999, py=1)[0] == 400
    assert c.act("gt_save", kind="truth", building="kci", floor=1, px=10, py=10)[0] == 400
    code, j = c.act("det_snapshot")
    assert code == 200, j
    snap = j["snapshot"]
    assert snap["review"] == "UNREVIEWED" and snap["run_id"] == "FAKE_observe"
    assert c.act("det_label", snap_id=snap["snap_id"], label="TP", region_index=0)[0] == 200
    assert c.act("det_label", snap_id=snap["snap_id"], label="TP", region_index=7)[0] == 400       # not a region of this frame
    assert c.act("det_label", snap_id=snap["snap_id"], label="MISSED")[0] == 400                   # needs a position
    assert c.act("det_label", snap_id=snap["snap_id"], label="MISSED", point=[400, 200], sign_text="301")[0] == 200
    code, j = c.act("det_label", snap_id=snap["snap_id"], label="AMBIGUOUS")
    assert code == 200 and j["label"]["ground_truth_confirmed"] is False
    ev = ws.root / "evaluation" / ws.exp_id
    assert (ev / "gt.jsonl").exists() and len(open(ev / "detections.jsonl").readlines()) == 3
    assert (ev / "snapshots" / f"{snap['snap_id']}.jpg").exists()
    code_files = [*(CAND / "live2").glob("*.py"), *(CAND / "live1").glob("*.py"), *(CAND / "external" / "sci_event" / "e5_onboard").glob("*.py")]
    for f in code_files:                                                            # no E2..E7 / E5 code touches the evaluation store
        if f.name != "console.py":
            t = f.read_text(errors="replace")
            assert not any(k in t for k in ('"evaluation"', "'evaluation'", "evaluation/", "gt.jsonl", "detections.jsonl")), f


def test_09_end_stops_owned_modules_and_archives(ws):
    c = ws.client
    assert c.act("end")[0] == 202
    s = c.wait_job(timeout=200)
    assert s["experiment"]["state"] == "ENDED", s["job"]
    for m in ("fastlio", "lio2", "hub", "rec", "observe", "e34", "e2"):
        assert not ws.alive(m), m
    rd = Path(s["experiment"]["record_dir"])
    assert (rd / "state" / "lio_state.json").exists() and (rd / "timeline.jsonl").exists() and (rd / "experiment.json").exists()
    st = c.req("GET", "/api/state")[1]
    assert st["lio"]["status"] == "DATA_UNAVAILABLE"                               # old data is not shown after the end
    assert st["meta"]["experiment_state"] == "ENDED"
    assert ws.alive("console")                                                      # ending the experiment keeps the console


def test_10_start_failure_is_partial_and_recoverable(ws):
    c = ws.client
    (ws.root / "fake_inject.json").write_text(json.dumps({"gate_fail": {"hub": ["camera_topology"]}}))
    assert c.act("start", building="kci", siglip=False)[0] == 202
    s = c.wait_job()
    X = s["experiment"]
    assert X["state"] == "PARTIAL" and X["failed"] == ["hub"]
    assert s["modules"]["hub"]["exp"]["state"] == "FAILED" and "camera_topology" in " ".join(s["modules"]["hub"]["exp"]["why"])
    assert ws.alive("fastlio") and ws.alive("e2")                                   # non-critical failure: the rest runs
    (ws.root / "fake_inject.json").write_text("{}")
    assert c.act("start_missing")[0] == 202
    s = c.wait_job()
    assert s["experiment"]["state"] == "RUNNING" and ws.alive("hub")
    assert c.act("end")[0] == 202 and c.wait_job(timeout=200)["experiment"]["state"] == "ENDED"


def test_10b_module_dying_after_start_is_shown(ws):
    c = ws.client
    (ws.root / "fake_inject.json").write_text(json.dumps({"exit": {"hub": 3}}))   # passes health, then exits rc 3
    assert c.act("start", building="kci", siglip=False)[0] == 202
    c.wait_job()
    t0 = time.time()
    while time.time() - t0 < 15:
        s = c.state()
        if (s["modules"]["hub"]["exp"] or {}).get("state") == "ENDED_UNEXPECTEDLY":
            break
        time.sleep(0.5)
    assert s["modules"]["hub"]["exp"]["state"] == "ENDED_UNEXPECTEDLY" and s["experiment"]["state"] == "PARTIAL"
    assert "hub" in s["experiment"]["failed"] and s["modules"]["hub"]["exp"]["why"][0]
    (ws.root / "fake_inject.json").write_text("{}")
    assert c.act("end")[0] == 202 and c.wait_job(timeout=200)["experiment"]["state"] == "ENDED"


def test_11_critical_failure_skips_dependants(ws):
    c = ws.client
    (ws.root / "fake_inject.json").write_text(json.dumps({"gate_fail": {"fastlio": ["fastlio_topology"]}}))
    assert c.act("start", building="kci", siglip=False)[0] == 202
    s = c.wait_job()
    mods = s["modules"]
    assert s["experiment"]["state"] == "PARTIAL" and mods["fastlio"]["exp"]["state"] == "FAILED"
    assert "fastlio_topology" in " ".join(mods["fastlio"]["exp"]["why"])
    assert all(mods[m]["exp"]["state"] == "SKIPPED" for m in ("lio2", "hub", "rec", "e34", "e2"))
    assert not any(ws.alive(m) for m in ("fastlio", "lio2", "hub", "e2"))
    (ws.root / "fake_inject.json").write_text("{}")
    assert c.act("end")[0] == 202 and c.wait_job(timeout=200)["experiment"]["state"] == "ENDED"


def test_12_expired_and_other_boot_approval_refused(ws):
    c = ws.client
    ws.edit_approval(expires_epoch=time.time() - 5)
    code, j = c.act("start", building="kci")
    assert code == 403 and "APPROVAL_EXPIRED" in j["why"]
    rc, out = ws.ctl("start", "lio2", "--accum", "--run-s", "60")
    assert rc == 3 and "APPROVAL_EXPIRED" in out
    ws.edit_approval(expires_epoch=time.time() + 600, boot_id="00000000-other-boot")
    code, j = c.act("start", building="kci")
    assert code == 403 and "APPROVAL_FOR_OTHER_BOOT" in j["why"]


def test_13_run_limit_and_unapproved_module(ws):
    lim = json.loads(json.dumps(LIM))
    lim["fastlio"]["max_run_s"] = 60
    ws.approve(modules=["fastlio", "lio2", "hub", "web", "e2", "e34"], limits=lim)  # observe not approved, fastlio 60 s
    c = ws.client
    code, j = c.act("start", building="kci", siglip=True)
    assert code == 403 and "observe" in j["why"]
    rc, out = ws.ctl("start", "fastlio", "--run-s", "100")                          # beyond the per-module approved limit
    assert rc == 3 and "run_limit" in out and not ws.alive("fastlio")


def test_14_approval_expiry_mid_experiment_auto_ends(ws):
    ws.approve()
    c = ws.client
    assert c.act("start", building="kci", siglip=False)[0] == 202
    assert c.wait_job()["experiment"]["state"] == "RUNNING"
    ws.edit_approval(expires_epoch=time.time() - 1)
    t0 = time.time()
    while time.time() - t0 < 120:
        s = c.state()
        if s["experiment"]["state"].startswith("ENDED"):
            break
        time.sleep(0.5)
    assert s["experiment"]["state"] == "ENDED" and "AUTO_END APPROVAL_EXPIRED" in s["experiment"]["end_reason"]
    assert not any(ws.alive(m) for m in ("fastlio", "lio2", "hub", "e2", "e34"))


def test_15_ownership_pid_reuse_is_never_signalled(ws):
    victim = subprocess.Popen(["/bin/sleep", "60"])
    try:
        rec = dict(module="e2", pid=victim.pid, start=1, pgid=victim.pid, started_wall="fake", boot_id="x")   # wrong starttime
        (ws.root / "runtime" / "owned" / "e2.json").write_text(json.dumps(rec))
        rc, out = ws.ctl("stop", "e2")
        assert "NOT signalled" in out
        time.sleep(0.3)
        assert victim.poll() is None                                                # the unrelated process is untouched
    finally:
        victim.send_signal(signal.SIGTERM)
        victim.wait(5)


def test_16_console_exit_does_not_stop_modules(ws):
    ws.approve()
    rc, out = ws.ctl("start", "hub", "--run-s", "120")
    assert rc == 0 and ws.alive("hub")
    rc, out = ws.ctl("stop", "console")
    assert rc == 0 and not ws.alive("console")
    time.sleep(1.0)
    assert ws.alive("hub")                                                          # modules are not children of the console
    assert ws.ctl("stop", "hub")[0] == 0 and not ws.alive("hub")
