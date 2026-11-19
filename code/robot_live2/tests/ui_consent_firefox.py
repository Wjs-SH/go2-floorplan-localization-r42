"""STAGING UI test (2026-10-31): per-experiment pan-tilt consent on the start screen (fake workspace, real console/ctl, real Firefox).
Base: '위치추정 진행' card + staged view in REAL Firefox (headless, Marionette) on the REAL console in a FAKE
workspace. E2/E3/E4 state files are COPIES of a real offline replay of the loc1 LiveStore (bench run dir argv[2]) placed where e2/e34
would write them. Checks: exact x/9 progress, per-floor chips, Top-1 shown as 미확정, Top-5, E3 counts, E4 reason, E7 line, staged
view; FIX only with an e7 GLOBAL_CONFIRMED record. Usage: python3 -B ui_progress_firefox.py OUT BENCH_RUN_DIR"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(sys.argv[2]).resolve() / "state"
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_console_fake as F  # noqa: E402

root = OUT / "fake_ws"
if root.exists():
    shutil.rmtree(root)
root.mkdir()
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_consent_firefox")))
(root / "candidate").symlink_to(F.CAND)
for d in ("runtime/approvals", "runtime/state", "logs", "reports"):
    (root / d).mkdir(parents=True, exist_ok=True)
ws = F.WS(root)
RES = dict(steps=[], ok=True)


def step(name, ok, **kw):
    RES["steps"].append(dict(name=name, ok=bool(ok), **kw)); RES["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, ensure_ascii=False, default=str)[:300], flush=True)


class Mn:
    def __init__(self, port):
        for _ in range(150):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), 1.0); break
            except OSError:
                time.sleep(0.2)
        self.s.settimeout(60); self.n = 0; self._read()
        self.cmd("WebDriver:NewSession", {"capabilities": {}})

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

    def js(self, script):
        return self.cmd("WebDriver:ExecuteScript", {"script": script, "args": []}).get("value")

    def state(self, path="/api/ctl/state"):
        return self.cmd("WebDriver:ExecuteAsyncScript", {"script": f"const cb=arguments[arguments.length-1]; fetch('{path}').then(r=>r.json()).then(cb).catch(e=>cb(null));", "args": []}).get("value")

    def shot(self, name):
        (OUT / f"{name}.png").write_bytes(base64.b64decode(self.cmd("WebDriver:TakeScreenshot", {"full": True})["value"]))


def place(n_exact=None, conf=False, fix=False):
    """copy the replay state; optionally truncate the E2 progress to n_exact floors (as it looked mid-search)."""
    st = root / "runtime" / "state"
    (st / "e2").mkdir(parents=True, exist_ok=True)
    e2 = json.loads((SRC / "e2" / "e2_state.json").read_text())
    if n_exact is not None:
        e2["floors_done"] = e2["floors_done"][:n_exact]
        e2["floors_pending"] = [f for f in e2["floors"] if f not in e2["floors_done"]]
        e2["status"] = "RUNNING"
    now = time.time()
    e2["updated_wall"] = now
    (st / "e2" / "e2_state.json").write_text(json.dumps(e2))
    e3 = json.loads((SRC / "e3_state.json").read_text()); e3["written_wall"] = now
    (st / "e3_state.json").write_text(json.dumps(e3))
    hist = [json.loads(x) for x in open(SRC / "e4_history.jsonl")]
    pick = next((h for h in hist if h["state"] == "CONFIDENT_GEOMETRY"), hist[-1]) if conf else \
        next((h for h in hist if h["state"] == "SEARCH_INCOMPLETE" and (h.get("metrics") or {}).get("floors_done") == n_exact), hist[len(hist) // 2])
    e4 = json.loads((SRC / "e4_state.json").read_text()); e4.update({k: pick[k] for k in ("state", "reasons", "metrics") if k in pick}); e4["written_wall"] = now
    e4.pop("display_ko", None); e4["milestones"] = {}                      # replay wall times predate this fake experiment
    (st / "e4_state.json").write_text(json.dumps(e4))
    if fix:
        (st / "e7_state.json").write_text(json.dumps(dict(kind="LIVE2_E7_STATE", global_decision="GLOBAL_CONFIRMED", path="SYNTHETIC_TEST",
                                                         why="SYNTHETIC_TEST record (UI only)", written_wall=now, semantic_hypothesis=e3["leader"])))
    elif (st / "e7_state.json").exists():
        (st / "e7_state.json").unlink()


ws.approve(modules=F.FULL + ["console"], limits=dict(F.LIM, console={"max_run_s": 3600}))
rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "900")
step("console started (fake ws)", rc == 0)
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", ws.port), 0.5).close(); break
    except OSError:
        time.sleep(0.2)


def bstatus():
    r = subprocess.run(["/usr/bin/python3", "-I", "-B", "-c", "import sys,json; sys.path.insert(0, %r); import live1_common as C; "
                        "print(json.dumps(C.approval_status('B','pantilt')))" % str(F.CAND / "live1")],
                       capture_output=True, text=True, env=dict(PATH="/usr/bin", LIVE1_ROOT=str(root)))
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:                                    # noqa: BLE001
        return dict(valid=None, why=r.stderr[-300:])


prof = OUT / "ff_profile"; prof.mkdir(exist_ok=True)
(prof / "user.js").write_text('user_pref("marionette.port", 2831);\n')
ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(prof), "about:blank"],
                      stdout=open(OUT / "firefox.out", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, MOZ_HEADLESS="1"))
try:
    mn = Mn(2831)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1100, "height": 1500})
    mn.cmd("WebDriver:Navigate", {"url": f"http://127.0.0.1:{ws.port}/"}); time.sleep(3)
    mn.js("window.__errs=[]; window.addEventListener('error',e=>window.__errs.push(String(e.message)+' @'+e.lineno)); window.confirm=()=>true; window.alert=(m)=>{window.__alert=String(m)}; return 1;")
    step("before start: approval B for pan-tilt not valid", bstatus().get("valid") is False, b=bstatus())
    txt = mn.js("return document.getElementById('ptc_box').innerText") or ""
    step("start screen shows the consent box", "이번 실험에서 팬틸트 카메라 사용" in txt, box=txt[:200])
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:1011.4,py:705.8,yaw_deg:0.5}); return 1;"); time.sleep(1.5)
    # (1) incomplete consent: the page refuses, and the API refuses a forged incomplete body
    mn.js("document.getElementById('ptc_use').checked=true; ptcSync(); document.getElementById('inp').value='body'; document.getElementById('e6mode').value='MANUAL_SETTLE'; startExp(); return 1;")
    time.sleep(2)
    st = ((mn.state() or {}).get("experiment") or {}).get("state")
    step("incomplete consent: page refuses to start", st not in ("RUNNING", "PARTIAL", "STARTING") and "모두 체크" in (mn.js("return window.__alert||''") or ""), state=st)
    mn.js("act('start',{building:'kci',siglip:true,input:'body',e6_mode:'MANUAL_SETTLE',pantilt_consent:{use:true,acks:{operator_supervises:true}}}); return 1;")
    time.sleep(3)
    st = ((mn.state() or {}).get("experiment") or {}).get("state")
    aud = (root / "runtime/ctl/audit.jsonl").read_text() if (root / "runtime/ctl/audit.jsonl").exists() else ""
    step("forged incomplete consent: API refuses (PANTILT_CONSENT_INCOMPLETE)", st not in ("RUNNING", "PARTIAL", "STARTING") and "PANTILT_CONSENT_INCOMPLETE" in aud, state=st)
    # (2) full consent
    mn.js("document.querySelectorAll('.ptc_ack').forEach(e=>e.checked=true); startExp(); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120 and ((mn.state() or {}).get("experiment") or {}).get("state") not in ("RUNNING", "PARTIAL"):
        time.sleep(1)
    X = (mn.state() or {}).get("experiment") or {}
    step("experiment running with consent", X.get("state") in ("RUNNING", "PARTIAL"), state=X.get("state"))
    recs = sorted((root / "runtime/approvals").glob("B_*.json"))
    rec = json.loads(recs[-1].read_text()) if recs else {}
    lim = rec.get("limits") or {}
    step("consent record: kind B, this boot, bound to this exp_id, manual focus, operator text", rec.get("kind") == "B" and lim.get("exp_id") == X.get("exp_id")
         and lim.get("focus_mode") == "MANUAL_FIXED_NO_MOTOR" and lim.get("source") == "TABLET_EXPERIMENT_CONSENT" and "운영자가 로봇을 직접 감독" in rec.get("user_text", ""),
         file=str(recs[-1]) if recs else None)
    b = bstatus()
    step("approval_status('B','pantilt') valid during the experiment", b.get("valid") is True, b=b)
    mods = ((mn.state() or {}).get("modules") or {})
    xj = json.loads((root / "experiments" / X["exp_id"] / "experiment.json").read_text()) if (root / "experiments" / X["exp_id"] / "experiment.json").exists() else {}
    tl = json.dumps(xj.get("timeline") or []) + json.dumps(json.loads((root / "runtime/ctl/experiment.json").read_text()).get("timeline") or [])
    step("e6 bridge started because of the consent (module_started E6 bridge, no e6_off)", "E6 bridge RUNNING" in tl and "e6_off" not in tl)
    mn.shot("c1_running_with_consent")
    mn.js("act('end'); return 1;"); time.sleep(10)
    b2 = bstatus()
    step("after [실험 종료]: consent no longer valid", b2.get("valid") is False, b=b2)
    recs2 = sorted((root / "runtime/approvals").glob("B_*.json"))
    step("closing record written", len(recs2) == len(recs) + 1 and recs2[-1].name.endswith("_END.json"), last=recs2[-1].name)
    # (3) a new experiment WITHOUT consent: B invalid, e6 OFF
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:1011.4,py:705.8,yaw_deg:0.5}); return 1;"); time.sleep(1.5)
    mn.cmd("WebDriver:Navigate", {"url": f"http://127.0.0.1:{ws.port}/"}); time.sleep(3)
    mn.js("window.confirm=()=>true; window.alert=(m)=>{window.__alert=String(m)}; document.getElementById('ptc_use').checked=false; ptcSync(); startExp(); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120 and ((mn.state() or {}).get("experiment") or {}).get("state") not in ("RUNNING", "PARTIAL"):
        time.sleep(1)
    X3 = (mn.state() or {}).get("experiment") or {}
    b3 = bstatus()
    step("next experiment without consent: B invalid (old consent bound to the ended exp), e6 OFF",
         b3.get("valid") is False and ((X3.get("modules") or {}).get("e6") or {}).get("state") in ("OFF", None), b=b3, e6=(X3.get("modules") or {}).get("e6"))
    mn.js("act('end'); return 1;"); time.sleep(8)
    errs = mn.js("return window.__errs") or []
    step("no JS errors", not errs, errs=errs)
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
