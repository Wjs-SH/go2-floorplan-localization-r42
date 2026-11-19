"""STAGING UI test (2026-11-01, r33): REAL 8770 console + web_v2 page in REAL Firefox (headless), FAKE workspace (fake modules
frozen so the test owns the state files). State files written into the fake runtime, in order:
  loc1  the REAL final state files of X20261031_231209 (e2 4/9, E3 H073_F4 4F leader, E4 SEARCH_INCOMPLETE, loc_overlay 82.5 % on
        wall) — first as recorded (old frame time -> must be shown as NOT live), then with the frame time set to now (SYNTHETIC
        timing only)
  E7    the E7 records produced OFFLINE by the real e6e7_link on the real loc1 record (test_r33_fix_view data/): REJECTED,
        AMBIGUOUS, MORE_OBSERVATION_REQUIRED, GLOBAL_CONFIRMED (+ its fix_overlay) — their written time is set to now so that they
        belong to this fake experiment (a SYNTHETIC FIX for the UI; never a field result); a structure_path RUNNING record of the real
        loc2 run and a SHADOW_WOULD_CONFIRM record (synthetic)
  clock settle / focus events written live (synthetic), page reload, experiment end -> time_metrics.json
Checked: no result popup before a FIX (incl. the old time-budget popup), the final card ONLY with the console's final_fix.valid,
final pose = the E7-corrected pose of the snapshot, the map shows E7's points, the two clocks (pause / +1 s / +3 s).
Usage: python3 -B ui_r33_firefox.py OUT FIXVIEW_DATA_DIR"""
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
FV = Path(sys.argv[2]).resolve()
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_console_fake as F  # noqa: E402

L1 = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261031_231209/state"
L2 = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261031_231940/state"
root = OUT / "fake_ws"
if root.exists():
    shutil.rmtree(root)
root.mkdir()
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_r33_firefox")))
(root / "candidate").symlink_to(F.CAND)
for d in ("runtime/approvals", "runtime/state", "logs", "reports"):
    (root / d).mkdir(parents=True, exist_ok=True)
(root / "fake_inject.json").write_text(json.dumps(dict(freeze_state=["e2", "e34"])))
ws = F.WS(root)
os.environ["LIVE1_ROOT"] = str(root)
sys.path[:0] = [str(F.CAND / "live1"), str(F.CAND / "live2")]
import live1_common as C  # noqa: E402
assert C.FAKE and str(C.RUNTIME).startswith(str(root)), C.RUNTIME
RES = dict(steps=[], ok=True)


def step(name, ok, **kw):
    RES["steps"].append(dict(name=name, ok=bool(ok), **kw)); RES["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, ensure_ascii=False, default=str)[:300], flush=True)


def put(name, obj):
    p = C.STATE / name; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("." + p.name + ".tmp"); tmp.write_text(json.dumps(obj, default=float)); os.replace(tmp, p)


def rm(name):
    p = C.STATE / name
    if p.exists():
        p.unlink()


EXP = {}


def jl(name, rec):
    rec = dict(rec, mono=time.monotonic(), experiment_id=EXP.get("id"))      # the real gate / bridge record format
    with open(C.STATE / name, "a") as f:
        f.write(json.dumps(rec) + "\n")


class Mn:
    def __init__(self, port):
        for _ in range(150):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), 1.0); break
            except OSError:
                time.sleep(0.2)
        self.s.settimeout(60); self.n = 0; self._read()
        self.cmd("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {}}})

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

    def js(self, s):
        return self.cmd("WebDriver:ExecuteScript", {"script": s, "args": []}).get("value")

    def state(self):
        return self.cmd("WebDriver:ExecuteAsyncScript", {"script": "const cb=arguments[arguments.length-1]; fetch('/api/ctl/state').then(r=>r.json()).then(cb).catch(e=>cb(null));", "args": []}).get("value")

    def shot(self, name):
        (OUT / f"{name}.png").write_bytes(base64.b64decode(self.cmd("WebDriver:TakeScreenshot", {"full": True})["value"]))


def view(mn):
    return mn.js("""const g=i=>document.getElementById(i); const vis=i=>g(i)&&!g(i).classList.contains('hide');
      return {now:vis('now_card')?g('now_card').innerText:'', status:g('now_status')?g('now_status').innerText:'', final:vis('final_card'),
              finalText:vis('final_card')?g('final_card').innerText:'', resdlg:vis('resdlg'), fixdlg:vis('fixdlg'), map:vis('loc_map_box'),
              mapCanvas:vis('loc_map'), empty:vis('loc_map_empty'), legend:g('loc_map_legend')?g('loc_map_legend').innerHTML:'',
              details:g('loc_details')?g('loc_details').open:null, title:g('loc_map_title')?g('loc_map_title').innerText:''};""")


GREEN = "const c=document.getElementById('loc_map'); const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data; let g=0,r=0; for(let i=0;i<d.length;i+=4){ if(d[i+1]>110&&d[i]<80&&d[i+2]<120) g++; if(d[i]>170&&d[i+1]<80&&d[i+2]<80) r++; } return [g,r,c.width,c.height];"

prof = OUT / "ff"; prof.mkdir(exist_ok=True)
(prof / "user.js").write_text('user_pref("marionette.port", 2834);\n')
ws.approve(modules=F.FULL + ["console"], limits=dict(F.LIM, console={"max_run_s": 3600}))
rc, out = ws.ctl("start", "console", "--port", str(ws.port), "--host", "127.0.0.1", "--run-s", "900")
step("console started (fake ws)", rc == 0)
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", ws.port), 0.5).close(); break
    except OSError:
        time.sleep(0.2)
ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(prof), "about:blank"],
                      stdout=open(OUT / "ff.out", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, MOZ_HEADLESS="1"))
try:
    mn = Mn(2834)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1100, "height": 1600})
    base = f"http://127.0.0.1:{ws.port}/"
    mn.cmd("WebDriver:Navigate", {"url": base}); time.sleep(3)
    mn.js("window.confirm=()=>true; window.alert=()=>{}; return 1;")
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:2319.7,py:709.0,yaw_deg:179.3}); return 1;"); time.sleep(1.5)
    mn.js("document.getElementById('inp').value='body'; return 1;")
    mn.cmd("WebDriver:ExecuteScript", {"script": "document.getElementById('b_start').click(); return 1;", "args": []})
    t_start = time.time(); X = {}
    while time.time() - t_start < 120:
        X = (mn.state() or {}).get("experiment") or {}
        if X.get("state") in ("RUNNING", "PARTIAL"):
            break
        time.sleep(1)
    t0 = X.get("created_wall")
    EXP["id"] = X.get("exp_id")
    step("experiment running", bool(t0), exp=X.get("exp_id"))
    for n in ("e3_state.json", "e4_state.json", "loc_overlay.json", "e7_state.json", "structure_path.json", "fix_overlay.json"):
        rm(n)
    time.sleep(3)
    v = view(mn)
    step("S0 no candidate: '주변 구조를 분석하여 현재 위치를 탐색하고 있습니다.' · map placeholder · no result", "주변 구조를 분석하여 현재 위치를 탐색하고 있습니다." in v["status"]
         and v["map"] and v["empty"] and not v["final"] and not v["resdlg"] and not v["fixdlg"], status=v["status"])
    step("details are collapsed by default", v["details"] is False)
    # ---- loc1 real files (as recorded: old frame time)
    for n in ("e3_state.json", "e4_state.json", "loc_overlay.json"):
        put(n, json.loads((L1 / n).read_text()))
    put("e2/e2_state.json", json.loads((L1 / "e2/e2_state.json").read_text()))
    O = json.loads((L1 / "loc_overlay.json").read_text())
    time.sleep(4)
    v = view(mn)
    x_m = round(O["pose_px"][0] / O["ppm"], 1)
    step("S1 real loc1: '현재 가장 유력한 위치를 찾았습니다.' · 4F · x from the overlay pose · 잠정 · not live (old frame) · no result popup",
         "현재 가장 유력한 위치를 찾았습니다." in v["status"] and "4F" in v["now"] and f"{x_m}" in v["now"] and "잠정 위치추정" in v["now"]
         and "실시간 값 아님" in v["now"] and not v["final"] and not v["resdlg"] and not v["fixdlg"], x_m=x_m, now=v["now"][:300])
    O2 = dict(O, t_end=time.time(), written_wall=time.time())                      # SYNTHETIC timing: as if the frame were new
    put("loc_overlay.json", O2); time.sleep(4)
    v = view(mn); px = mn.js(GREEN)
    step("S1b fresh frame: live (no delay note) · map drawn with the LiDAR points (green on-wall / red)", "실시간 값 아님" not in v["now"]
         and v["mapCanvas"] and px[0] > 200 and "잠정" in v["title"], green_red=px, title=v["title"])
    mn.js("window.eval('renderResult(S,CS,true); resOpen(true)'); return 1;"); time.sleep(0.5)
    step("the old time-budget / best-candidate popup can no longer open", not view(mn)["resdlg"])
    mn.js("document.getElementById('loc_details').open=true; return 1;"); time.sleep(1.5)
    det = mn.js("return document.getElementById('loc_progress').innerText") or ""
    step("details: E2 4/9 · E4 판정 · E7 실행 안 됨 · FIX 정책 kept", "4/9" in det and "E4 판정" in det and "실행 안 됨" in det and "FIX 정책" in det)
    mn.js("document.getElementById('loc_details').open=false; return 1;")
    mn.shot("01_loc1_provisional")
    # ---- E4 ambiguous (real loc2 E4 record)
    e4 = json.loads((L2 / "e4_state.json").read_text())
    put("e4_state.json", dict(e4, state="AMBIGUOUS", written_wall=time.time())); time.sleep(3)
    step("S2 E4 AMBIGUOUS -> '여러 위치 후보가 유사하게 일치합니다…'", "여러 위치 후보가 유사하게 일치합니다" in view(mn)["status"])
    put("e4_state.json", json.loads((L1 / "e4_state.json").read_text()))
    # ---- E7 verifying: structure path RUNNING (real loc2 record)
    sp = json.loads((L2 / "structure_path.json").read_text())
    put("structure_path.json", dict(sp, status="RUNNING", phase="VERIFY_WHILE_MOVING")); time.sleep(3)
    v = view(mn)
    step("S3 E7 actually running -> '최종 위치 검증 중' · no result", "최종 위치 검증 중" in v["status"] and not v["final"])
    rm("structure_path.json")
    # ---- E7 failures (offline real E7 records), each written NOW
    for name, label in (("MISREAD", "REJECTED"), ("STRAIGHT", "AMBIGUOUS"), ("NO_E2", "MORE_OBSERVATION_REQUIRED")):
        e = json.loads((FV / f"e7_state_{name}.json").read_text()); e["written_wall"] = time.time()
        put("e7_state.json", e); time.sleep(3)
        v = view(mn)
        step(f"S4 E7 {label} -> '현재 후보를 최종 확정하지 못했습니다…' · no result card · no popup", "최종 확정하지 못했습니다" in v["status"]
             and not v["final"] and not v["fixdlg"] and (mn.state() or {}).get("final_fix", {}).get("valid") is False)
    rm("e7_state.json")
    # ---- SHADOW would-confirm
    put("structure_path.json", dict(sp, status="SHADOW_WOULD_CONFIRM", phase="SHADOW_WOULD_CONFIRM", ended_wall=time.time())); time.sleep(3)
    v = view(mn)
    step("S5 SHADOW_WOULD_CONFIRM -> '현재 유력 위치 표시 중 · 최종 확정 기능 비활성화' · no result", "최종 확정 기능 비활성화" in v["status"]
         and not v["final"] and not v["fixdlg"])
    mn.shot("02_shadow")
    rm("structure_path.json")
    # ---- a GLOBAL_CONFIRMED record from BEFORE this experiment -> ignored
    e = json.loads((FV / "e7_state_SEM_OK.json").read_text()); e["written_wall"] = t0 - 100
    put("e7_state.json", e); time.sleep(3)
    step("S6 GLOBAL_CONFIRMED older than the experiment -> no result", not view(mn)["final"]
         and "older" in ((mn.state() or {}).get("final_fix") or {}).get("why", ""))
    # ---- manual clocks (synthetic live events)
    cs0 = mn.state()["op_clock"]
    oid = "UI_R33_O1"
    jl("e6_settle_events.jsonl", dict(kind="STATE", state="WAITING_FOR_SETTLE_APPROVAL", observation_id=oid, wall=time.time()))
    time.sleep(2.5); a = mn.state()["op_clock"]; time.sleep(3); b = mn.state()["op_clock"]
    v = view(mn)
    step("C1 settle wait: normalized clock paused while the real clock runs; page shows '일시정지 — 진동 안정화 중'",
         b["raw_s"] - a["raw_s"] >= 2.5 and abs(b["normalized_s"] - a["normalized_s"]) <= 0.15 and b["paused"]["kind"] == "SETTLE"
         and "진동 안정화 중" in v["now"], a=a["normalized_s"], b=b["normalized_s"], raw=[a["raw_s"], b["raw_s"]])
    jl("e6_settle_events.jsonl", dict(kind="COMMAND", action="approve", accepted=False, observation_id=oid, wall=time.time()))
    time.sleep(1.5); c = mn.state()["op_clock"]
    step("C2 refused approval keeps the pause (not counted)", c["paused"] and c["N_settle"] == 0)
    jl("e6_settle_events.jsonl", dict(kind="COMMAND", action="approve", accepted=True, observation_id=oid, wall=time.time()))
    jl("e6_manual_events.jsonl", dict(kind="PHASE", phase="WAIT_MANUAL_FOCUS", observation_id=oid, wall=time.time() + 2.0))
    time.sleep(1.0); d1 = mn.state()["op_clock"]
    step("C3 approved -> +1 s, clock runs again", d1["N_settle"] == 1 and d1["paused"] is None)
    time.sleep(5); e1 = mn.state()["op_clock"]; v = view(mn)
    step("C4 manual focus: paused '수동 초점 중'", e1["paused"] and e1["paused"]["kind"] == "FOCUS" and "수동 초점 중" in v["now"])
    mn.cmd("WebDriver:Navigate", {"url": base}); time.sleep(4)
    step("C5 page reload: same server clock, still paused", "수동 초점 중" in view(mn)["now"])
    jl("e6_manual_events.jsonl", dict(kind="COMMAND", action="capture", accepted=True, observation_id=oid, wall=time.time()))
    time.sleep(1.5); g1 = mn.state()["op_clock"]
    exp_norm = g1["raw_s"] - g1["removed_s"] + 1 + 3
    step("C6 [②] -> +3 s; normalized = raw - manual + 1 + 3", g1["N_focus"] == 1 and g1["paused"] is None and abs(g1["normalized_s"] - exp_norm) <= 0.2,
         clock=g1)
    jl("e6_manual_events.jsonl", dict(kind="PHASE", phase="WAIT_MANUAL_FOCUS", observation_id=oid, wall=time.time()))   # [다시 촬영]
    time.sleep(2); jl("e6_manual_events.jsonl", dict(kind="PHASE", phase="USER_CANCELLED", observation_id=oid, wall=time.time()))
    time.sleep(1.5); h1 = mn.state()["op_clock"]
    step("C7 retake wait then cancel -> not normalized (still 1 focus)", h1["N_focus"] == 1 and h1["paused"] is None)
    # ---- FIX: GLOBAL_CONFIRMED (offline real E7 + gates) written NOW with its snapshot
    tw = time.time()
    e = json.loads((FV / "e7_state_SEM_OK.json").read_text()); e["written_wall"] = tw
    fo = json.loads((FV / "fix_overlay_SEM_OK.json").read_text()); fo["e7_written_wall"] = tw
    put("fix_overlay.json", fo); put("e7_state.json", e); time.sleep(4)
    v = view(mn); st = mn.state(); FFv = st.get("final_fix") or {}
    step("S7 backend GLOBAL_CONFIRMED (this experiment) -> console final_fix.valid, snapshot accepted", FFv.get("valid") is True and FFv.get("fix_overlay_ok") is True,
         final_fix={k: FFv.get(k) for k in ("valid", "path", "hypothesis", "TTF_raw_s", "TTF_normalized_s", "fix_overlay_ok")})
    step("S7 final card: title · E7-corrected pose of the snapshot · E7 correction · both TTFs + note", v["final"] and "최종 초기 전역 위치추정 완료" in v["finalText"]
         and f"{fo['pose_m'][0]:.2f}" in v["finalText"] and f"{fo['pose_m'][1]:.2f}" in v["finalText"] and "Δx 0.42 m" in v["finalText"]
         and "실제 최초 위치 확정시간" in v["finalText"] and "수동조작 기준 보정 최초 확정시간" in v["finalText"] and "1초·수동 초점 3초를 가정한 분석값" in v["finalText"]
         and "GT가 없어" in v["finalText"] and not v["fixdlg"], text=v["finalText"][:500])
    time.sleep(1.5); px = mn.js(GREEN); v = view(mn)
    step("S7 same map switches to the final pose + E7's LiDAR points", "최종 위치 (E7 보정)" in v["title"] and "E7 검증 LiDAR 구조점" in v["legend"] and px[0] > 200,
         green_red=px, title=v["title"])
    step("S7 top card says FIX", "FIX" in v["now"] and "최종 초기 전역 위치추정 완료" in v["status"])
    mn.shot("03_final_fix_synthetic")
    fo2 = dict(fo, e7_written_wall=tw - 50)                                     # snapshot of ANOTHER E7 run
    put("fix_overlay.json", fo2); time.sleep(3); v = view(mn)
    step("S8 snapshot not of this FIX -> card shows no E7 point/pose claim (reason shown)", v["final"] and "E7 점군 스냅샷" in v["finalText"]
         and "E7 검증 LiDAR 구조점" not in v["legend"])
    # ---- R40: through the real console: the popup, and the FIX that arrives while the head is in use
    v = mn.js("const c=document.getElementById('final_card'); return {pos:c.style.position,t:c.innerText,cv:!!document.getElementById('fin_map')};")
    step("R40 FIX with E6 idle: full-screen popup with the map, first candidate row, both TTF", v["pos"] == "fixed" and v["cv"] and "최초 위치 후보" in v["t"] and "실제 최초 위치 확정시간" in v["t"], pos=v["pos"])
    put("e6_manual.json", dict(kind="LIVE2_E6_MANUAL", observation_id="O_hw", experiment_id=X["exp_id"], phase="WAIT_MANUAL_FOCUS", wall=time.time()))
    tw2 = time.time(); e2_ = dict(e, written_wall=tw2); put("fix_overlay.json", dict(fo, e7_written_wall=tw2)); put("e7_state.json", e2_); time.sleep(4)
    PR = ((mn.state() or {}).get("final_fix") or {}).get("presentation") or {}
    v = mn.js("const c=document.getElementById('final_card'); return {pos:c.style.position,t:c.innerText,pend:!!document.getElementById('final_pending')};")
    step("R40 FIX during manual focus: recorded, no popup, one line waits", PR.get("state") == "FINAL_PRESENTATION_PENDING" and PR.get("t_fix") == tw2 and v["pend"] and v["pos"] != "fixed" and "최종 초기" not in v["t"], pr=PR)
    put("e6_manual.json", dict(kind="LIVE2_E6_MANUAL", observation_id="O_hw", experiment_id=X["exp_id"], phase="HOME_CONFIRMED", wall=time.time())); time.sleep(4)
    PR = ((mn.state() or {}).get("final_fix") or {}).get("presentation") or {}
    v = mn.js("const c=document.getElementById('final_card'); return {pos:c.style.position,t:c.innerText};")
    step("R40 after HOME: popup shows; the FIX time did not move, the display time is kept apart", PR.get("state") == "FINAL_RESULT_VISIBLE" and PR.get("t_fix") == tw2 and PR.get("t_visible") > tw2 + 1 and v["pos"] == "fixed" and "최종 초기 전역 위치추정 완료" in v["t"], pr=PR)
    mn.shot("04_final_popup_r40")
    # ---- end -> time_metrics with raw + normalized values
    mn.js("act('end'); return 1;")
    t1 = time.time()
    while time.time() - t1 < 120:
        if ((mn.state() or {}).get("experiment") or {}).get("state") == "ENDED":
            break
        time.sleep(1)
    tm = json.loads((root / "evaluation" / X["exp_id"] / "time_metrics.json").read_text())
    step("END time_metrics: raw TTF and normalized TTF stored side by side (SYNTHETIC FIX) + end clocks + note",
         tm.get("TTF_raw_s") is not None and tm.get("TTF_normalized_s") is not None and tm["TTF_normalized_s"] < tm["TTF_raw_s"]
         and tm.get("T_end_raw_s") and "분석값" in tm.get("normalized_note", ""),
         TTF_raw=tm.get("TTF_raw_s"), TTF_norm=tm.get("TTF_normalized_s"), N=(tm.get("operator_clock") or {}).get("at_fix", {}).get("N_focus"))
except Exception as ex:                                  # noqa: BLE001
    step("harness exception", False, error=f"{type(ex).__name__}: {ex}")
finally:
    try:
        os.killpg(ff.pid, 15)
    except OSError:
        pass
    ws.ctl("stop", "all", timeout=200)
    (OUT / "ui_result.json").write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=str))
    print("ALL_OK" if RES["ok"] else "SOME_FAILED", flush=True)
