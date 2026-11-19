"""STAGING UI test (2026-10-31 r31: LiDAR-on-map card, 의미정보 측정 결과 card, VERIFYING stage — display placements are SYNTHETIC files,
the overlay points are a REAL overlay of loc1 X20261030_223346 relabelled to the replay leader for display only). Base: '위치추정 진행' card + staged view in REAL Firefox (headless, Marionette) on the REAL console in a FAKE
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
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_r31_firefox")))
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
    # ---- 2026-10-31 r31 display checks (no FIX record: the previous step's synthetic e7 GLOBAL_CONFIRMED is removed by place())
    place(conf=True); time.sleep(3)
    st = root / "runtime" / "state"
    e3 = json.loads((st / "e3_state.json").read_text())
    lead = sorted([h for h in e3["hypotheses"] if h.get("status") in ("ACTIVE", "SUSPENDED")], key=lambda h: h.get("score", 9))[0]
    ov = json.loads((Path.home() / "sci_ws_alt_imu_20261029_204740/runs/r31/overlay1/overlay_H080_F4.json").read_text())
    dx, dy = lead["pose_px"][0] - ov["pose_px"][0], lead["pose_px"][1] - ov["pose_px"][1]
    ov.update(leader=lead["id"], floor=lead["floor"], pose_px=lead["pose_px"], written_wall=time.time(),
              pts=[[p[0] + dx, p[1] + dy] for p in ov["pts"]])
    (st / "loc_overlay.json").write_text(json.dumps(ov)); time.sleep(9)
    mn.js("document.querySelectorAll('button').forEach(b=>{ if(b.innerText.trim()==='닫기'&&b.offsetParent) b.click(); }); return 1;"); time.sleep(4)
    PIX = ("const c=document.getElementById('loc_map'), d=c.getContext('2d').getImageData(0,0,c.width,c.height).data; let g=0,r=0,b=0;"
           "for(let i=0;i<d.length;i+=4){ const R=d[i],G=d[i+1],B=d[i+2]; if(Math.abs(R-47)<25&&Math.abs(G-133)<25&&Math.abs(B-90)<25) g++;"
           " else if(Math.abs(R-197)<25&&Math.abs(G-48)<25&&Math.abs(B-48)<25) r++; else if(Math.abs(R-43)<25&&Math.abs(G-108)<25&&Math.abs(B-176)<25) b++; }"
           " return {green:g,red:r,blue:b,w:c.width,h:c.height};")
    vis = mn.js("return !document.getElementById('loc_map_box').classList.contains('hide')")
    leg = mn.js("return document.getElementById('loc_map_legend').innerText") or ""
    px = mn.js(PIX)
    step("LiDAR-on-map card visible with on-wall share and 'provisional — not verified'", vis and "벽 0.3 m 이내" in leg and "잠정 위치 — 아직 검증되지 않음" in leg, legend=leg)
    step("LiDAR points really drawn on the map canvas (green on-wall pixels, no blue before E7)", px and px["green"] > 200 and px["blue"] < 50, pixels=px)
    mn.cmd("WebDriver:ExecuteScript", {"script": "document.getElementById('loc_map_box').scrollIntoView(); return 1;", "args": []}); time.sleep(1)
    mn.shot("p4_lidar_map_provisional")
    ov["corrected"] = dict(correction=dict(dx_m=0.2, dy_m=-0.1, dphi_deg=1.0, scale=1.0, source="E7 structure window"), pose_px=[lead["pose_px"][0] + 5, lead["pose_px"][1] - 2],
                           heading_deg=lead.get("heading_deg") or 0, shift_m=0.22, pts=ov["pts"], dist_m=ov["dist_m"], median_dist_m=0.0, on_wall_frac=0.85)
    ov["written_wall"] = time.time(); (st / "loc_overlay.json").write_text(json.dumps(ov)); time.sleep(9)
    leg = mn.js("return document.getElementById('loc_map_legend').innerText") or ""
    px2 = mn.js(PIX)
    step("E7 correction shown (Δx Δy Δθ, shift, after-correction on-wall share)", "E7 보정 후" in leg and "이동 0.22 m" in leg, legend=leg)
    step("after-correction points drawn (blue pixels)", px2 and px2["blue"] > 200, pixels=px2)
    mn.cmd("WebDriver:ExecuteScript", {"script": "document.getElementById('loc_map_box').scrollIntoView(); return 1;", "args": []}); time.sleep(1)
    mn.shot("p5_lidar_map_corrected")
    (st / "structure_path.json").write_text(json.dumps(dict(kind="LIVE2_STRUCTURE_PATH", leader=lead["id"], floor=lead["floor"], status="RUNNING",
                                                            phase="VERIFY_WHILE_MOVING", mode="SHADOW", started_wall=time.time(), t_confident=1.0, passes=[])))
    time.sleep(9)
    s2 = (mn.state("/api/state") or {}).get("live2", {}).get("stage") or {}
    txt = mn.js("return document.getElementById('loc_progress').innerText") or ""
    step("E4 CONFIDENT + structure E7 running -> stage VERIFYING ('E7 LiDAR 검증 중 — 아직 FIX 아님')", s2.get("stage") == "VERIFYING" and "아직 FIX 아님" in txt, stage=s2.get("stage"))
    (st / "structure_path.json").unlink()
    pair = dict(kind="LIVE2_E8_PAIR_STATE", policy="E8_PAIR", state="MORE_OBSERVATION_REQUIRED", why="a second independent plate is needed", group_id="PAIR_T_1",
                observations=[dict(observation_id="O1", ocr_lines=[["401", 0.98]], read=True, vlm=None),
                              dict(observation_id="O2", ocr_lines=[], read=False, vlm=None)],
                evaluation=dict(verdict="MORE_OBSERVATION_REQUIRED", why="1 usable independent observation(s) of 2",
                                per_observation=[dict(observation_id="O1", labels=[["401", 0.98]], usable=True,
                                                      provisional=dict(hypothesis=lead["id"], floor=lead["floor"], room="room_401", dist_m=3.1, target_px=[1100, 560]))]),
                e7=None, history=[], processed=["O1", "O2"])
    (st / "e8_pair_state.json").write_text(json.dumps(pair)); time.sleep(9)
    sc = mn.js("return document.getElementById('sem_card').innerText") or ""
    step("의미정보 측정 결과: 1차 (OCR 401, room_401, E3 위치·방향, 독립 관측) + 2차 판독 중 + 최소 1회 더 필요", "의미정보 측정 결과" in sc and "1차" in sc
         and "room_401" in sc and "독립 관측" in sc and "2차" in sc and "판독 중" in sc and "최소 1회 더 필요" in sc, card=sc[:500])
    mn.shot("p6_semantic_card")
    (st / "e8_pair_state.json").unlink(); (st / "loc_overlay.json").unlink()
    errs = mn.js("return window.__errs") or []
    step("no JS errors (r31 checks)", not errs, errs=errs)
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
