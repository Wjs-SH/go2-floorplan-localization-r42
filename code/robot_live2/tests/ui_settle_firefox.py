"""STAGING UI test (2026-10-30): the REAL 8770 console + web_v2 page in REAL Firefox (headless, Marionette), FAKE workspace (fake modules,
fake approval records that exist only inside the temp workspace; no ROS, no device, no robot). The approval wait is the REAL
e6_settle.SettleGate running in this process on the fake runtime, so the page's button -> console -> command file -> gate -> page path
is exercised end to end. Screenshots + a JSON result go to OUT. Usage: python3 -B ui_settle_firefox.py OUT"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_console_fake as F  # noqa: E402

root = OUT / "fake_ws"
if root.exists():
    shutil.rmtree(root)
root.mkdir()
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_settle_firefox")))
(root / "candidate").symlink_to(F.CAND)
for d in ("runtime/approvals", "runtime/state", "logs", "reports"):
    (root / d).mkdir(parents=True, exist_ok=True)
ws = F.WS(root)
os.environ["LIVE1_ROOT"] = str(root)                       # this process reads the SAME fake runtime as the console
sys.path[:0] = [str(F.CAND / "live1"), str(F.CAND / "live2")]
import live1_common as C  # noqa: E402
import e6_settle as ST  # noqa: E402
assert C.FAKE and str(C.RUNTIME).startswith(str(root)), C.RUNTIME
RES = dict(steps=[], ok=True)


def step(name, ok, **kw):
    RES["steps"].append(dict(name=name, ok=bool(ok), **kw))
    RES["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, ensure_ascii=False, default=str)[:300], flush=True)


# ---------------------------------------------------------------- fake workspace: approvals A + B (test-only records), console
ws.approve(modules=F.FULL + ["console"], limits=dict(F.LIM, console={"max_run_s": 3600}))
txt = root / "b.txt"; txt.write_text("FAKE workspace UI test approval B (pantilt) — exists only in this temp dir")
rc, out = ws.ctl("record-approval", "B", "--text-file", str(txt), "--minutes", "30", "--modules", "pantilt",
                 "--limits", json.dumps(dict(focus_mode="MANUAL_FIXED_NO_MOTOR")), "--recorded-by", "ui test FAKE")
step("fake approval B recorded in temp workspace", rc == 0, out=out[-200:])
rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "900")
step("console started (fake ws)", rc == 0, out=out[-200:])
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", ws.port), 0.5).close(); break
    except OSError:
        time.sleep(0.2)


# ---------------------------------------------------------------- Marionette (raw protocol, no selenium)
class Mn:
    def __init__(self, port=2828):
        for _ in range(150):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), 1.0); break
            except OSError:
                time.sleep(0.2)
        self.s.settimeout(60); self.n = 0; self._read()
        self.cmd("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"acceptInsecureCerts": True}}})

    def _read(self):
        buf = b""
        while b":" not in buf:
            buf += self.s.recv(1)
        ln, rest = buf.split(b":", 1); ln = int(ln)
        while len(rest) < ln:
            rest += self.s.recv(ln - len(rest))
        return json.loads(rest)

    def cmd(self, name, params=None):
        self.n += 1
        m = json.dumps([0, self.n, name, params or {}]).encode()
        self.s.sendall(str(len(m)).encode() + b":" + m)
        while True:
            r = self._read()
            if r[0] == 1 and r[1] == self.n:
                if r[2]:
                    raise RuntimeError(f"{name}: {r[2]}")
                return r[3]

    def js(self, script, *args):
        return self.cmd("WebDriver:ExecuteScript", {"script": script, "args": list(args)}).get("value")

    def state(self):
        return self.cmd("WebDriver:ExecuteAsyncScript", {"script": "const cb=arguments[arguments.length-1]; fetch('/api/ctl/state').then(r=>r.json()).then(cb).catch(e=>cb(null));", "args": []}).get("value")

    def click(self, css):
        el = self.cmd("WebDriver:FindElement", {"using": "css selector", "value": css})["value"]
        eid = list(el.values())[0]
        self.cmd("WebDriver:ElementClick", {"id": eid})

    def shot(self, name):
        png = base64.b64decode(self.cmd("WebDriver:TakeScreenshot", {"full": True})["value"])
        (OUT / f"{name}.png").write_bytes(png)
        return str(OUT / f"{name}.png")


prof = OUT / "ff_profile"; prof.mkdir(exist_ok=True)
(prof / "user.js").write_text('user_pref("marionette.port", 2829);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n')
ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(prof), "about:blank"],
                      stdout=open(OUT / "firefox.out", "w"), stderr=subprocess.STDOUT, start_new_session=True,
                      env=dict(os.environ, MOZ_HEADLESS="1"))
gate_thread = None
try:
    mn = Mn(2829)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1100, "height": 1500})
    base = f"http://127.0.0.1:{ws.port}/"
    mn.cmd("WebDriver:Navigate", {"url": base})
    time.sleep(3.0)
    step("page loaded, E6 mode select present (default MANUAL_SETTLE)", mn.js("return document.getElementById('e6mode') && document.getElementById('e6mode').value") == "MANUAL_SETTLE")
    # 2026-10-30 duplicate-id regression: the existing E6 manual-focus card (#e6m) must still be the element renderE6M shows
    vis = mn.js("""try{ const now=Date.now()/1000;
        renderE6M({live2:{e6_manual:{phase:'WAIT_MANUAL_FOCUS',age_s:1,wall:now,phase_since:now,observation_id:'UI_DUP_ID_CHECK'}}},{experiment:{}});
      }catch(e){}
      const c=document.getElementById('e6m'), s=document.getElementById('e6mode');
      const r=[c&&c.tagName, c&&!c.classList.contains('hide'), s&&s.tagName, s&&!s.classList.contains('hide')];
      try{ renderE6M({live2:{}},{experiment:{}}); }catch(e){}
      return r;""")
    step("E6 manual-focus card #e6m is the DIV renderE6M shows; E6-mode select stays visible", vis == ["DIV", True, "SELECT", True], got=vis)
    mn.js("window.confirm=()=>true; window.alert=()=>{};")
    r = mn.js("""return (async()=>{ const s=await fetch('/api/session').then(r=>r.json()); return s; })()""")
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:1013.3,py:705.6,yaw_deg:0}); return 1;")
    time.sleep(1.5)
    mn.js("document.getElementById('inp').value='body'; return 1;")
    mn.shot("01_start_card")
    mn.js("return 1;")
    mn.click("#b_start")
    t0 = time.time(); X = {}
    while time.time() - t0 < 120:
        X = (mn.state() or {}).get("experiment") or {}
        if X.get("state") in ("RUNNING", "PARTIAL"):
            break
        time.sleep(1.0)
    step("experiment started from the page with e6_mode=MANUAL_SETTLE, input=body", X.get("e6_mode") == "MANUAL_SETTLE" and X.get("input") == "body", exp=X)
    exp = X.get("exp_id")
    e6alive = (((mn.state() or {}).get("modules") or {}).get("e6") or {}).get("alive")
    step("E6 module running (fake module, fake approval B valid)", bool(e6alive))

    # body-IMU health for this run (fake run dir), as the bridge would write it
    body = root / "runtime" / "fastlio" / "body_ui"; body.mkdir(parents=True, exist_ok=True)
    rec = json.loads((root / "runtime" / "owned" / "fastlio.json").read_text()); rec.update(input="body", body_run_dir=str(body))
    (root / "runtime" / "owned" / "fastlio.json").write_text(json.dumps(rec))
    imu = dict(gyro_rms_dps=3.1, gyro_peak_dps=6.0, acc_norm_std=0.11)
    stop_flag = threading.Event()

    def imu_writer():
        while not stop_flag.is_set():
            with open(body / "bridge_health.jsonl", "a") as f:
                f.write(json.dumps(dict(wall_ns=int(time.time() * 1e9), health="READY", publish_hz=499, **imu)) + "\n")
            time.sleep(0.5)
    threading.Thread(target=imu_writer, daemon=True).start()

    # e34's e6_tracker record of a confirmed stop (what the console + bridge read)
    now = time.time(); oid = "live1_observe_UI_O001"
    o = dict(experiment_id=exp, observation_id=oid, source="AUTO_E5", state="STOPPED_TARGET_READY", detection_timestamp=now - 9,
             created_timestamp=now - 7, stopped_wall=now - 1, stop_check=dict(stopped=True, speed_mps=0.03, yaw_rate_dps=0.2),
             odom_pose_at_stop=dict(x=0, y=0, yaw=0))
    (C.STATE / "e6_state.json").write_text(json.dumps(dict(kind="LIVE2_E6_STATE", recent=[o])))
    world = dict(stop=dict(stopped=True, speed_mps=0.02, yaw_rate_dps=0.3, why="STOPPED"))
    gate = ST.SettleGate(C.STATE, C.RUNTIME)
    result = {}

    def bridge():
        result["r"] = gate.wait(o, exp, lambda: dict(world["stop"]), lambda: dict(shift_px=2.0),
                                lambda: C.approval_status("B", "pantilt"), lambda _o: (False, None))
    gate_thread = threading.Thread(target=bridge, daemon=True); gate_thread.start()
    time.sleep(3.0)
    txt_ = mn.js("return document.getElementById('settle_panel').innerText")
    btn = mn.js("const b=[...document.querySelectorAll('#settle_panel button')][0]; return b?{t:b.innerText,dis:b.disabled}:null")
    step("waiting panel: big message + approve button enabled (stopped, stable, B valid)",
         "로봇 정지 완료" in (txt_ or "") and btn and btn["t"].startswith("① 진동 안정 확인 · 팬틸트 조준 시작") and not btn["dis"], button=btn)   # 2026-10-31 label
    mn.shot("02_waiting_ready")

    imu.update(gyro_rms_dps=14.0, gyro_peak_dps=45.0, acc_norm_std=1.2)       # body still shaking
    time.sleep(1.5)
    mn.shot("03_shaking_warning")
    mn.click("#settle_panel button")
    time.sleep(5.0)                                                         # client-side 4 s lockout + gate reply
    snap = ST.read_snapshot(C.STATE) or {}
    rep = (snap.get("replies") or [None])[-1] or {}
    step("approve while shaking -> refused by the gate, still waiting, no approval time", snap.get("state") == "WAITING_FOR_SETTLE_APPROVAL"
         and rep.get("accepted") is False and ST.MSG_UNSTABLE in (rep.get("why") or "") and not snap["times"].get("t_manual_approval"),
         reply=rep)
    mn.shot("04_refused_unstable")

    imu.update(gyro_rms_dps=2.9, gyro_peak_dps=5.0, acc_norm_std=0.09)
    time.sleep(1.5)
    mn.js("const b=document.querySelector('#settle_panel button'); b.click(); b.click(); setTimeout(()=>{const c=document.querySelector('#settle_panel button'); if(c) c.click();},150); return 1;")
    gate_thread.join(20)
    out_, why_, rec_ = result.get("r") or (None, None, None)
    ncmd = sum(1 for x in open(C.STATE / "control" / "e6_settle_cmd.jsonl") if '"approve"' in x)
    step("stable approve (triple tap) -> exactly one E6_APPROVED", out_ == "E6_APPROVED" and len([1 for e in open(C.STATE / "e6_settle_events.jsonl") if '"E6_APPROVED"' in e]) == 1,
         outcome=out_, approve_commands_in_file=ncmd)
    time.sleep(2.0)
    mn.shot("05_approved")
    mn.cmd("WebDriver:Navigate", {"url": base}); time.sleep(3.0)
    txt2 = mn.js("return document.getElementById('settle_panel').innerText") or ""
    step("page reload restores the event + its state from the server", oid in txt2 and "승인됨" in txt2, panel=txt2[:200])
    mn.shot("06_after_reload")

    # approval B gone (expired / other boot): new stop -> red notice, button disabled, server refuses
    for p in (root / "runtime" / "approvals").glob("B_*.json"):
        p.rename(p.with_suffix(".json.removed_by_test"))
    now = time.time(); oid2 = "live1_observe_UI_O002"
    o2 = dict(o, observation_id=oid2, created_timestamp=now - 6, stopped_wall=now - 1, detection_timestamp=now - 8)
    (C.STATE / "e6_state.json").write_text(json.dumps(dict(kind="LIVE2_E6_STATE", recent=[o, o2])))
    gate2 = ST.SettleGate(C.STATE, C.RUNTIME, wait_max_s=25)
    result2 = {}
    th2 = threading.Thread(target=lambda: result2.setdefault("r", gate2.wait(o2, exp, lambda: dict(world["stop"]), lambda: dict(shift_px=2.0),
                                                                             lambda: C.approval_status("B", "pantilt"), lambda _o: (False, None))), daemon=True)
    th2.start(); time.sleep(3.0)
    btn2 = mn.js("const b=document.querySelector('#settle_panel button'); return b?{t:b.innerText,dis:b.disabled}:null")
    txt3 = mn.js("return document.getElementById('settle_panel').innerText") or ""
    code = mn.js(f"""return (async()=>{{ const [c,j]=await jpost('/api/ctl/e6_settle',{{request_id:'ui'+Date.now()+Math.floor(Math.random()*1e6),action:'approve',observation_id:'{oid2}'}}); return [c,j]; }})()""")
    step("no approval B (= no per-experiment pan-tilt consent): notice shown, button disabled, server refuses APPROVAL_B_REQUIRED",
         "팬틸트 사용 동의 없음" in txt3 and btn2 and btn2["dis"] and code[0] == 409 and "APPROVAL_B_REQUIRED" in code[1]["why"], button=btn2, server=code)
    mn.shot("07_no_approval_b")
    mn.js(f"""return (async()=>{{ await jpost('/api/ctl/e6_settle',{{request_id:'ui'+Date.now()+Math.floor(Math.random()*1e6),action:'cancel',observation_id:'{oid2}'}}); return 1; }})()""")
    th2.join(30)
    step("cancel without B -> USER_CANCELLED (no E6)", (result2.get("r") or [None])[0] == "USER_CANCELLED", outcome=(result2.get("r") or [None])[0])
    stop_flag.set()

    mn.js("act('end'); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120:
        X2 = (mn.state() or {}).get("experiment") or {}
        if X2.get("state") == "ENDED":
            break
        time.sleep(1.0)
    arch = root / "experiments" / exp / "state"
    ok_arch = (arch / "e6_settle_events.jsonl").exists() if arch.exists() else any((root / "experiments" / exp).rglob("e6_settle_events.jsonl"))
    step("experiment ended; settle records archived with the experiment", X2.get("state") == "ENDED" and ok_arch)
    tm = root / "evaluation" / exp / "time_metrics.json"
    step("time_metrics.json written at the end (no FIX -> no TTF)", tm.exists() and json.loads(tm.read_text()).get("TTF_raw_s") is None,
         metrics=json.loads(tm.read_text()) if tm.exists() else None)
    mn.shot("08_ended")
except Exception as e:                                   # noqa: BLE001
    step("harness exception", False, error=f"{type(e).__name__}: {e}")
finally:
    try:
        os.killpg(ff.pid, 15)
    except OSError:
        pass
    ws.ctl("stop", "all", timeout=200)
    (OUT / "ui_result.json").write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=str))
    print("ALL_OK" if RES["ok"] else "SOME_FAILED", flush=True)
