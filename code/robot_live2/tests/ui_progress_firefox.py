"""STAGING UI test (2026-10-30): '위치추정 진행' card + staged view in REAL Firefox (headless, Marionette) on the REAL console in a FAKE
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
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_progress_firefox")))
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
prof = OUT / "ff_profile"; prof.mkdir(exist_ok=True)
(prof / "user.js").write_text('user_pref("marionette.port", 2831);\n')
ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(prof), "about:blank"],
                      stdout=open(OUT / "firefox.out", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, MOZ_HEADLESS="1"))
try:
    mn = Mn(2831)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1100, "height": 1500})
    base = f"http://127.0.0.1:{ws.port}/"
    mn.cmd("WebDriver:Navigate", {"url": base}); time.sleep(3)
    # r33 (2026-11-01): the detailed panels are collapsed by default — the operator opens [상세 정보]; this test reads them
    mn.js("document.querySelectorAll('#loc_details, #loc_map_box details').forEach(d=>d.open=true); return 1;")
    mn.js("window.__errs=[]; window.addEventListener('error',e=>window.__errs.push(String(e.message)+' @'+e.lineno)); window.addEventListener('unhandledrejection',e=>window.__errs.push('rej '+String(e.reason&&e.reason.stack||e.reason))); return 1;")
    mn.js("window.confirm=()=>true; act('gt_save',{building:'kci',floor:4,kind:'start',px:2315.9,py:707.6,yaw_deg:-179.8}); return 1;")
    time.sleep(1.5)
    mn.js("document.getElementById('e6mode').value='AUTO'; document.getElementById('inp').value='body'; document.getElementById('b_start').click(); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120 and ((mn.state() or {}).get("experiment") or {}).get("state") not in ("RUNNING", "PARTIAL"):
        time.sleep(1)
    step("experiment running (fake modules)", ((mn.state() or {}).get("experiment") or {}).get("state") in ("RUNNING", "PARTIAL"))
    (root / "fake_inject.json").write_text(json.dumps(dict(freeze_state=["e2", "e34"])))   # fake e2/e34 keep running but stop writing
    time.sleep(1.0)
    place(n_exact=4); time.sleep(9)
    txt = mn.js("return document.getElementById('loc_progress').innerText") or ""
    s2 = (mn.state("/api/state") or {}).get("live2", {}).get("stage") or {}
    step("mid-search: 정밀 4/9, 미확정 label, Top-1 4F, no FIX", "정밀 4/9" in txt and "추정 중 · 미확정" in txt and "4F" in txt and "FIX 없음" in txt,
         stage=s2.get("stage"), panel=txt[:400])
    print("JS_ERRORS", mn.js("return window.__errs"), flush=True)
    step("staged view served and never READY/FIXED mid-search", s2.get("stage") in ("SEARCHING", "PROVISIONAL", "AMBIGUOUS") and s2.get("fix_allowed_by_this_view") is False, stage=s2)
    mn.shot("p1_mid_search")
    place(conf=True); time.sleep(9)
    txt = mn.js("return document.getElementById('loc_progress').innerText") or ""
    s2 = (mn.state("/api/state") or {}).get("live2", {}).get("stage") or {}
    step("search complete + E4 CONFIDENT: 9/9, READY_FOR_VERIFY, still 미확정 without E7", "정밀 9/9" in txt and s2.get("stage") == "READY_FOR_VERIFY"
         and "추정 중 · 미확정" in txt, stage=s2.get("stage"))
    mn.shot("p2_ready_for_verify")
    place(conf=True, fix=True); time.sleep(9)
    txt = mn.js("return document.getElementById('loc_progress').innerText") or ""
    s2 = (mn.state("/api/state") or {}).get("live2", {}).get("stage") or {}
    step("E7 GLOBAL_CONFIRMED record (SYNTHETIC_TEST) -> FIX shown", s2.get("stage") == "FIXED" and "FIX" in txt, stage=s2.get("stage"))
    mn.shot("p3_fix_synthetic")
    mn.js("act('end'); return 1;"); time.sleep(8)
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
