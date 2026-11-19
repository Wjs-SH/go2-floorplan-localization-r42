"""STAGING UI test (2026-11-01, r32): REAL 8770 console + web_v2 page in REAL Firefox (headless, Marionette), FAKE workspace.
End to end, with the REAL e6_manual channel and the REAL E6Pipeline manual-focus loop running in this process (fake aim console
client: no serial, no camera, no robot). The live view and the captures are the REAL saved field photos of X20261031_231940
(O005 defocused, O002 '402'); a tiny fake aim console on 127.0.0.1:8088 serves them to the console's /img/e6live.jpg and
/api/e6/metrics routes (metrics = the real e6_manual.measure on that photo). Checked:
  F3   semantic card on the REAL loc2 e8_pair_state.json: VLM state text per reading + OCR time; single-reading recovery row
  A    sign ROI dragged on the live view with real pointer input -> metrics computed on that ROI (OPERATOR_SELECTED)
  B/C  zoom canvas drawn from the ROI; sharpness / contrast / size / brightness / change shown
  D    [②] without an operator ROI -> warning first
  3    capture -> CAPTURE_REVIEW: the saved photo (sha) + zoom shown, ② hidden; [다시 촬영] -> same observation, photo kept as
       _try1, never read; 2nd capture -> [사진 확인 · 판독 진행] -> HOME -> one stored record; OCR of the accepted photo with the
       operator ROI reads '402'
Usage: python3 -B ui_r32_firefox.py OUT"""
import base64
import hashlib
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.parse
from pathlib import Path

OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_console_fake as F  # noqa: E402

FIELD = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261031_231940/state"
CAP = FIELD / "e6_captures"
root = OUT / "fake_ws"
if root.exists():
    shutil.rmtree(root)
root.mkdir()
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_r32_firefox")))
(root / "candidate").symlink_to(F.CAND)
for d in ("runtime/approvals", "runtime/state", "logs", "reports"):
    (root / d).mkdir(parents=True, exist_ok=True)
ws = F.WS(root)
os.environ["LIVE1_ROOT"] = str(root)
sys.path[:0] = [str(F.CAND / "live1"), str(F.CAND / "live2")]
if "serial" not in sys.modules:
    sys.modules["serial"] = types.ModuleType("serial")
import cv2  # noqa: E402
import live1_common as C  # noqa: E402
import e6_manual as M  # noqa: E402
import e6_pipeline as P  # noqa: E402
import e6_ocr  # noqa: E402
import e8_pair as EP  # noqa: E402
assert C.FAKE and str(C.RUNTIME).startswith(str(root)), C.RUNTIME
RES = dict(steps=[], ok=True)


def step(name, ok, **kw):
    RES["steps"].append(dict(name=name, ok=bool(ok), **kw))
    RES["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, ensure_ascii=False, default=str)[:300], flush=True)


# ---------------------------------------------------------------- real field photos
def jpg_bytes(p):
    return Path(p).read_bytes()


IMG = {k: CAP / f"live1_observe_20261031_232020_{k}.jpg" for k in ("O002", "O005")}
VIEW = dict(frame=cv2.imread(str(IMG["O005"])), seq=100)


# ---------------------------------------------------------------- fake aim console (loopback 8088) serving the field photos
class AimH(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        VIEW["seq"] += 1
        fr = VIEW["frame"]
        if u.path == "/live2/preview.jpg":
            w = int(q.get("w", "960"))
            im = cv2.resize(fr, (w, int(w * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
            b = cv2.imencode(".jpg", im)[1].tobytes()
            self.send_response(200)
            for k, v in (("Content-Type", "image/jpeg"), ("Content-Length", str(len(b))), ("X-Seq", str(VIEW["seq"])), ("X-Frame-Age-S", "0.08")):
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(b)
            return
        if u.path == "/live2/metrics":
            roi = M.parse_roi(q.get("roi")) if q.get("roi") else None
            m = M.measure(fr, roi, auto=True)
            m.update(frame_age_s=0.08, seq=VIEW["seq"])
            b = json.dumps(m).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
            return
        self.send_response(404); self.end_headers()


aim_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8088), AimH)
threading.Thread(target=aim_srv.serve_forever, daemon=True).start()


class FakeAim:
    """The pipeline's aim-console client, faked (records every job; the 'device' never exists)."""

    def __init__(self):
        self.calls, self.cap = [], jpg_bytes(IMG["O005"])

    def run(self, kind, params, *a, **k):
        self.calls.append(kind)
        if kind == "aim":
            assert params["focus"] == 0
            return dict(status="COMPLETED", device_state="IDLE", reason_code="captured", capture=dict(capture_id="c_aim"),
                        record=dict(plan=dict(pan=100.0, tilt=160.0)))
        if kind == "live2/manual_capture":
            roi = M.parse_roi(params.get("roi")) if params.get("roi") else None
            m = M.measure(cv2.imdecode(__import__("numpy").frombuffer(self.cap, "uint8"), cv2.IMREAD_COLOR), roi)
            cap = dict(capture_id=f"cap_ui{len(self.calls)}", sha256=hashlib.sha256(self.cap).hexdigest(), width=1920, height=1080, metrics=m)
            return dict(status="COMPLETED", device_state="IDLE", record=dict(result=dict(detail=dict(capture=cap))))
        if kind == "live2/pose":
            return dict(status="COMPLETED", device_state="IDLE", record=dict(result=dict(detail=dict(pan=95.352, tilt=157.189))))
        return dict(status="COMPLETED", device_state="IDLE")

    def fetch_capture(self, cid):
        return self.cap if cid != "c_aim" else jpg_bytes(IMG["O005"])

    def camera(self, q):
        return dict(on_demand=True, open=True, first_frame=True)

    def shake(self):
        return dict(stale=False, shift_px=1.0, response=0.9)

    def metrics(self, roi=None):
        VIEW["seq"] += 3
        m = M.measure(VIEW["frame"], roi)
        m.update(frame_age_s=0.08, seq=VIEW["seq"])
        return m


# ---------------------------------------------------------------- console
ws.approve(modules=F.FULL + ["console"], limits=dict(F.LIM, console={"max_run_s": 3600}))
txt = root / "b.txt"; txt.write_text("FAKE workspace UI test approval B (pantilt) — exists only in this temp dir")
rc, out = ws.ctl("record-approval", "B", "--text-file", str(txt), "--minutes", "30", "--modules", "pantilt",
                 "--limits", json.dumps(dict(focus_mode="MANUAL_FIXED_NO_MOTOR")), "--recorded-by", "ui test FAKE")
step("fake approval B recorded in temp workspace", rc == 0)
rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "900")
step("console started (fake ws)", rc == 0, out=out[-160:])
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", ws.port), 0.5).close(); break
    except OSError:
        time.sleep(0.2)


class Mn:
    def __init__(self, port):
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
        self.cmd("WebDriver:ElementClick", {"id": list(el.values())[0]})

    def click_text(self, sel, text):
        return self.js("const b=[...document.querySelectorAll(arguments[0])].find(x=>x.innerText.includes(arguments[1])&&!x.disabled&&x.offsetParent); if(!b) return false; b.click(); return true;", sel, text)

    def drag(self, x0, y0, x1, y1):
        self.cmd("WebDriver:PerformActions", {"actions": [{"type": "pointer", "id": "mouse1", "parameters": {"pointerType": "mouse"}, "actions": [
            {"type": "pointerMove", "x": int(x0), "y": int(y0), "origin": "viewport", "duration": 0}, {"type": "pointerDown", "button": 0},
            {"type": "pointerMove", "x": int((x0 + x1) / 2), "y": int((y0 + y1) / 2), "origin": "viewport", "duration": 100},
            {"type": "pointerMove", "x": int(x1), "y": int(y1), "origin": "viewport", "duration": 100}, {"type": "pointerUp", "button": 0}]}]})
        self.cmd("WebDriver:ReleaseActions")

    def shot(self, name):
        (OUT / f"{name}.png").write_bytes(base64.b64decode(self.cmd("WebDriver:TakeScreenshot", {"full": True})["value"]))


def wait_phase(ch, want, t=30):
    t0 = time.time()
    while time.time() - t0 < t:
        if (ch.d or {}).get("phase") == want:
            return True
        time.sleep(0.2)
    return False


NONBLANK = "const c=document.getElementById(arguments[0]); if(!c) return -1; const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data; let n=0; for(let i=0;i<d.length;i+=4){ if(d[i]>30||d[i+1]>30||d[i+2]>30) n++; } return n;"

prof = OUT / "ff_profile"; prof.mkdir(exist_ok=True)
(prof / "user.js").write_text('user_pref("marionette.port", 2831);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n')
ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(prof), "about:blank"],
                      stdout=open(OUT / "firefox.out", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, MOZ_HEADLESS="1"))
pipe_thread = None
try:
    mn = Mn(2831)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1280, "height": 1600})
    base = f"http://127.0.0.1:{ws.port}/"
    mn.cmd("WebDriver:Navigate", {"url": base}); time.sleep(3.0)
    # r33 (2026-11-01): the detailed panels are collapsed by default — the operator opens [상세 정보]; this test reads them
    mn.js("document.querySelectorAll('#loc_details, #loc_map_box details').forEach(d=>d.open=true); return 1;")
    mn.js("window.confirm=()=>true; window.alert=()=>{}; return 1;")
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:1009.5,py:709.2,yaw_deg:0}); return 1;"); time.sleep(1.5)
    mn.js("document.getElementById('inp').value='body'; return 1;")
    mn.js("return 1;")
    mn.click("#b_start")
    t0 = time.time(); X = {}
    while time.time() - t0 < 120:
        X = (mn.state() or {}).get("experiment") or {}
        if X.get("state") in ("RUNNING", "PARTIAL"):
            break
        time.sleep(1.0)
    exp = X.get("exp_id")
    step("experiment started from the page (MANUAL_SETTLE, body input)", X.get("e6_mode") == "MANUAL_SETTLE" and bool(exp), exp=exp)
    (C.OWNED / "e6_aim_console.pid").write_text(f"{os.getpid()}\n")         # the console's aim_console_alive(): this process

    # ------------------------------------------------ F3 on the REAL loc2 E8 state
    shutil.copy(FIELD / "e8_pair_state.json", C.STATE / "e8_pair_state.json")
    reg = json.loads((Path.home() / "sci_ws/reports/field_20261031_analysis/data/r32_single_region_real_loc2.json").read_text())[0]["region"]
    rec1 = dict(kind="LIVE2_SEMANTIC_RECOVERY_SINGLE", observation_id=reg["observation_id"], status="REGION_REQUESTED", region=reg,
                why="regional E2 search requested from one decisive reading", wall=time.time())
    (C.STATE / "semantic_recovery_single.json").write_text(json.dumps(rec1))
    time.sleep(4.0)
    sem = mn.js("const e=document.getElementById('sem_card'); return e?e.innerText:''") or ""
    step("F3: 1st reading '402' shows 'VLM 실행 안 함 (OCR 이 호실 번호를 읽음 — VLM 결과 아님)' and OCR 67 s",
         "402" in sem and "OCR 이 호실 번호를 읽음" in sem and "VLM 결과 아님" in sem and "(OCR 67 s)" in sem, text=sem[:400])
    step("F3: 2nd / 3rd readings show 'VLM 서버 꺼짐' (not '실행 안 함') and OCR 119 s", sem.count("VLM 서버 꺼짐") == 2 and "(OCR 119 s)" in sem)
    body = mn.js("return document.body.innerText") or ""
    step("F3: single-reading recovery row shown as candidate generation only", "판독 1개 기반 지역 E2" in body and "요청됨" in body
         and "FIX 아님" in body, snippet=body[body.find("판독 1개 기반"):body.find("판독 1개 기반") + 160])
    mn.shot("01_semcard_real_loc2")

    # ------------------------------------------------ real manual-focus loop (pipeline thread)
    oid = "live1_observe_UI_R32_O001"
    ch = M.Channel(C.STATE)
    aim = FakeAim()
    env = dict(pan=(85.0, 106.0), tilt=(147.0, 168.0), source="TEST_ENVELOPE")
    pipe = P.E6Pipeline(aim, lambda kind, **kw: (True, None), lambda: dict(stopped=True, why="STOPPED"), C.STATE / "e6_captures",
                        envelope=env, focus_mode="MANUAL_FIXED_NO_MOTOR", defer_reading=True, manual=ch, experiment_id=exp,
                        log=lambda *_: None)
    pipe.SETTLE = dict(pipe.SETTLE, body_min_s=0.0, aim_min_s=0.0)
    obs = dict(observation_id=oid, source="AUTO_E5", target_pose_in_odom=dict(odom_link_xy=[1.0, 0.5]),
               odom_pose_at_stop=dict(t=100.0, x=0.0, y=0.0, yaw=0.0), stopped_wall=time.time(),
               redetection=dict(status="REDETECTED", pixel=[330, 230], depth_m=1.4, pantilt_command=dict(inside_limits=True)))
    result = {}
    pipe_thread = threading.Thread(target=lambda: result.setdefault("rec", pipe.run(obs, 1)), daemon=True); pipe_thread.start()
    step("pipeline reached WAIT_MANUAL_FOCUS", wait_phase(ch, "WAIT_MANUAL_FOCUS"))
    time.sleep(4.0)
    mn.js("document.getElementById('e6m').scrollIntoView(); return 1;"); time.sleep(1.5)
    live = mn.js("const i=document.getElementById('e6m_img'); return [i.naturalWidth, !document.getElementById('e6m').classList.contains('hide')]")
    step("E6 card visible with the live view (field photo O005 through /img/e6live.jpg)", live and live[0] >= 900 and live[1], got=live)
    mn.shot("02_focus_no_roi")
    # D: ② without operator ROI -> warning first
    mn.js("document.getElementById('e6m_go').click(); return 1;"); time.sleep(0.8)
    ask = mn.js("return document.getElementById('e6m_ask').innerText") or ""
    step("D: [②] without a selected sign ROI -> warning (no command sent)", "표지판 영역을 직접 선택하지 않았습니다" in ask
         and not any(x.get("action") for x in ch.d.get("replies") or []), ask=ask[:120])
    mn.shot("03_warn_no_roi")
    mn.click_text("#e6m_ask button", "표지판 영역 선택하기"); time.sleep(0.5)
    # A: drag the sign ROI on the canvas (plate of O002/O005 region, 1280x720 coords ~ (420,505)-(545,615))
    rect = mn.js("const r=document.getElementById('e6m_cv').getBoundingClientRect(); return [r.left,r.top,r.width,r.height]")
    sc = min(rect[2] / 1280, rect[3] / 720); ox = rect[0] + (rect[2] - 1280 * sc) / 2; oy = rect[1] + (rect[3] - 720 * sc) / 2
    roi_t = (420, 505, 545, 615)
    mn.drag(ox + roi_t[0] * sc, oy + roi_t[1] * sc, ox + roi_t[2] * sc, oy + roi_t[3] * sc)
    time.sleep(2.5)
    st = mn.js("return window.eval(\"[E6M.roi, (E6M.m||{}).roi_source]\").concat([document.getElementById('e6m_zcap').innerText, document.getElementById('e6m_q').innerText])")
    ok_roi = bool(st and st[0] and all(abs(a - b) <= 6 for a, b in zip(st[0], roi_t)))
    step("A: ROI dragged with real pointer input -> E6M.roi set; metrics computed on it (OPERATOR_SELECTED)", ok_roi and st[1] == "OPERATOR_SELECTED",
         roi=st and st[0], src=st and st[1])
    nz = mn.js(NONBLANK, "e6m_zoom")
    step("B: zoom canvas drawn from the selected ROI ('운용자 선택 영역')", nz and nz > 5000 and "운용자 선택 영역" in (st[2] or ""), nonblank_px=nz, cap=st and st[2])
    q = st[3] or ""
    step("C: sharpness / change / contrast·brightness / sign size shown on that ROI", all(k in q for k in ("선명도", "직전 대비", "명암 대비", "표지판 크기", "px")),
         q=q[:200])
    mn.shot("04_roi_zoom_metrics")
    # capture 1 (defocused O005 photo)
    for _ in range(3):
        mn.js("document.getElementById('e6m_go').click(); return 1;"); time.sleep(0.8)
        if mn.click_text("#e6m_ask button", "그대로 촬영") or mn.click_text("#e6m_ask button", "현재 상태로 촬영"):
            time.sleep(0.6)
        if mn.click_text("#e6m_ask button", "예 — 손 뗐음"):
            break
    step("capture 1 -> CAPTURE_REVIEW (head not moved: no HOME yet)", wait_phase(ch, "CAPTURE_REVIEW") and "home" not in aim.calls)
    time.sleep(4.0)
    rv = mn.js("""const e=document.getElementById('e6m_rev'), i=document.getElementById('e6m_rev_img');
                 return [!e.classList.contains('hide'), i?i.naturalWidth:0, document.getElementById('e6m_go').classList.contains('hide'), e.innerText]""")
    nz2 = mn.js(NONBLANK, "e6m_rev_zoom")
    step("3: review panel shows the SAVED photo (1920 px, /img/e8) + its zoom; ② hidden; buttons shown",
         rv and rv[0] and rv[1] == 1920 and rv[2] and "사진 확인 · 판독 진행" in rv[3] and "다시 촬영" in rv[3] and nz2 > 3000, got=rv and rv[:3], zoom_px=nz2)
    mn.shot("05_review_1_defocused")
    sha1 = (ch.d.get("review") or {}).get("sha256")
    mn.js("window.eval('E6M.busy=0'); return 1;")
    mn.click_text("#e6m_rev button", "다시 촬영")
    step("3: [다시 촬영] -> back to WAIT_MANUAL_FOCUS in the SAME observation", wait_phase(ch, "WAIT_MANUAL_FOCUS") and ch.d.get("observation_id") == oid)
    try1 = C.STATE / "e6_captures" / f"{oid}_try1.jpg"
    step("3: rejected photo kept as _try1 with the same sha256", try1.exists() and hashlib.sha256(try1.read_bytes()).hexdigest() == sha1)
    # operator refocuses: the live view and the next capture become the sharp O002 photo
    VIEW["frame"] = cv2.imread(str(IMG["O002"])); aim.cap = jpg_bytes(IMG["O002"])
    time.sleep(3.0)
    mn.js("window.eval('E6M.busy=0'); return 1;")
    for _ in range(3):
        mn.js("document.getElementById('e6m_go').click(); return 1;"); time.sleep(0.8)
        if mn.click_text("#e6m_ask button", "그대로 촬영") or mn.click_text("#e6m_ask button", "현재 상태로 촬영"):
            time.sleep(0.6)
        if mn.click_text("#e6m_ask button", "예 — 손 뗐음"):
            break
    ok2 = wait_phase(ch, "CAPTURE_REVIEW") and (ch.d.get("review") or {}).get("n") == 2
    time.sleep(4.0)
    mn.shot("06_review_2_sharp")
    step("capture 2 -> CAPTURE_REVIEW n=2", ok2)
    mn.js("window.eval('E6M.busy=0'); return 1;")
    mn.click_text("#e6m_rev button", "사진 확인 · 판독 진행"); time.sleep(0.6)
    mn.click_text("#e6m_ask button", "예 — 손 뗐음, 판독 진행")
    pipe_thread.join(40)
    rec = result.get("rec") or {}
    a = (rec.get("manual") or {}).get("attempts") or []
    step("3: accept -> HOME -> ONE stored record; attempts [RETAKE, ACCEPTED]; record file = accepted photo",
         rec.get("status") == "CAPTURE_STORED" and [x["decision"] for x in a] == ["RETAKE", "ACCEPTED"] and aim.calls.count("home") == 1
         and rec.get("image_sha256") == hashlib.sha256(jpg_bytes(IMG["O002"])).hexdigest(), status=rec.get("status"), attempts=[x["decision"] for x in a])
    q2 = rec.get("image_quality") or {}
    step("operator ROI recorded in the image quality (source of the OCR ROI)", q2.get("roi_source") == "OPERATOR_SELECTED", roi=q2.get("roi"))
    img = cv2.imread(rec["capture_file"])
    lines, meta = e6_ocr.sign_read(img, roi=EP.operator_roi(dict(image_quality=q2)))
    step("OCR of the accepted photo with the operator ROI (one model load) reads '402'", any(t == "402" for t, _ in lines) and meta["roi_source"] == "OPERATOR_SELECTED",
         lines=lines, load_s=meta.get("load_s"), roi_s=meta.get("roi_s"), full_s=meta.get("full_s"), threads=meta.get("threads"))
    time.sleep(2.0)
    mn.shot("07_after_accept")
    mn.js("act('end'); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120:
        if ((mn.state() or {}).get("experiment") or {}).get("state") == "ENDED":
            break
        time.sleep(1.0)
except Exception as e:                                   # noqa: BLE001
    step("harness exception", False, error=f"{type(e).__name__}: {e}")
finally:
    try:
        os.killpg(ff.pid, 15)
    except OSError:
        pass
    aim_srv.shutdown()
    ws.ctl("stop", "all", timeout=200)
    (OUT / "ui_result.json").write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=str))
    print("ALL_OK" if RES["ok"] else "SOME_FAILED", flush=True)
