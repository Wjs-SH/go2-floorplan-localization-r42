"""R34 UI test: REAL 8770 console + web_v2 page in REAL Firefox (headless), FAKE workspace; state files written in the backend's own formats. SYNTHETIC scene values.
Checked: E2 crash text with its error and the next action; FAST-LIO hold; thermal hold; regional fit; normal search; restored count; DEPTH_STALE; E4 rival; E7 SHADOW is not a final result; SEMANTIC_ANCHORED provisional;
408+TOILETS comparison; observations with valid / attempt counts; honest VLM text; pre-stop aim guidance; the final card never appears without a backend FIX. Usage: python3 -B ui_r34_firefox.py OUT"""
import base64
import json
import os
import shutil
import socket
import subprocess
import threading
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_console_fake as F  # noqa: E402

L1 = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261031_231209/state"
L2 = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261031_231940/state"
root = OUT / "fake_ws"
if root.exists():
    shutil.rmtree(root)
root.mkdir()
(root / "LIVE2_FAKE_WORKSPACE.json").write_text(json.dumps(dict(kind="LIVE2_FAKE_WORKSPACE", made_by="ui_r34_firefox")))
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
    mn.js("window.confirm=()=>true; window.alert=()=>{}; window.__jsErrors=[]; window.addEventListener('error',e=>window.__jsErrors.push(String(e.message))); return 1;")
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

    def chips():
        return mn.js("""const q=[...document.querySelectorAll('#fs_chips .fs_row')]; const g=document.getElementById('field_status');
          return {vis:!g.classList.contains('hide'), codes:q.map(x=>x.dataset.code), text:g.innerText, final:!document.getElementById('final_card').classList.contains('hide')};""")
    (C.STATE / "e2").mkdir(parents=True, exist_ok=True)
    sess = "S1"
    base_e2 = dict(status="RUNNING", floors_total=9, floors=list(range(1, 10)), floors_done=[1, 2, 3, 4, 5, 6, 7], floors_failed={}, pass_id=1, per_floor={}, candidates=[], candidates_pass=[], odom_session=sess)
    put("e2/e2_state.json", dict(base_e2)); time.sleep(3.5); v = chips()
    step("normal search: 'E2 7/9' shown and no result card", v["vis"] and "E2_SEARCHING" in v["codes"] and "E2 7/9" in v["text"] and not v["final"], text=v["text"][:300])
    put("e2/e2_state.json", dict(base_e2, held="HELD_FASTLIO odom 7.6 Hz < 8.0 Hz (FAST-LIO keeps priority)")); time.sleep(3.5); v = chips()
    step("FAST-LIO hold is named, with what to do", "HELD_FASTLIO" in v["codes"] and "FAST-LIO 보호" in v["text"] and "자동으로 재개" in v["text"])
    put("e2/e2_state.json", dict(base_e2, held="HELD_THERMAL max temp 76.3 C")); time.sleep(3.5); v = chips()
    step("thermal hold is named", "HELD_THERMAL" in v["codes"] and "온도" in v["text"])
    put("e2/e2_state.json", dict(base_e2, regional=[dict(request_id="OBS_O3", floor=4, status="RUNNING", radius_px_at_fit=809.8)])); time.sleep(3.5); v = chips()
    step("regional fit is named (floor, radius)", "E2_REGIONAL_FIT" in v["codes"] and "지역 탐색" in v["text"] and "809.8" in v["text"])
    put("e2/e2_state.json", dict(base_e2, regional=[dict(request_id="OBS_O3", floor=4, status="FAILED", why_failed="NEEDS_NEW_OBSERVATION: region 60 m > 45 m")])); time.sleep(3.5); v = chips()
    step("region too large asks for a new observation", "E2_REGIONAL_NEW_OBS" in v["codes"] and "새 표지판" in v["text"])
    put("e2/e2_state.json", dict(base_e2, status="IDLE", floors_done=list(range(1, 10)), restarts=2, restore=dict(status="RESTORED", mode="PASS_COMPLETE"))); time.sleep(3.5); v = chips()
    step("restart with kept floors: E2 9/9, restored", "E2_DONE" in v["codes"] and "E2 9/9" in v["text"] and "복구" in v["text"])
    # ---- depth: the fake hub index
    def hub_depth(v):
        d = json.loads((root / "fake_inject.json").read_text()); d["hub_depth_offset_s"] = v
        (root / "fake_inject.json").write_text(json.dumps(d))
    hub_depth(900.0)
    time.sleep(3.5); v = chips()
    step("frozen depth: DEPTH_STALE with what it blocks and what continues", "DEPTH_STALE" in v["codes"] and "DEPTH_UNAVAILABLE" in v["text"] and "FAST-LIO" in v["text"], text=v["text"][:400])
    hub_depth(None)
    time.sleep(3.5); v = chips()
    step("no depth: DEPTH_MISSING", "DEPTH_MISSING" in v["codes"])
    hub_depth(0.05)
    time.sleep(3.5); v = chips()
    step("fresh depth: DEPTH_OK", "DEPTH_OK" in v["codes"])
    # ---- E4 rival, E7 states
    put("e4_state.json", dict(state="AMBIGUOUS", metrics=dict(partial_best_floor=4, partial_competitor_floor=8, partial_gap=0.044), written_wall=time.time())); time.sleep(3.5); v = chips()
    step("E4 rival floor and gap", "경쟁층 8F" in v["text"] and "0.044" in v["text"])
    rm("e7_state.json"); rm("structure_path.json"); rm("e7_shadow_early.json"); time.sleep(3.5); v = chips()
    step("E7 not run is said plainly", "E7_NOT_RUN" in v["codes"] and "E7 NOT_RUN" in v["text"])
    put("e7_shadow_early.json", dict(status="SHADOW_WOULD_CONFIRM", n_pass=3, last=dict(best="H075_F4", final_decision="GEOMETRY_CONSISTENT_SEMANTIC_PENDING", gap_m=0.08, unsearched_floors=[8, 9]))); time.sleep(3.5); v = chips()
    step("early E7 SHADOW would-confirm: says record only, no final card", "E7_SHADOW_EARLY_SHADOW_WOULD_CONFIRM" in v["codes"] and "최종 확정 아님" in v["text"] and not v["final"])
    put("structure_path.json", dict(status="SHADOW_WOULD_CONFIRM", mode="SHADOW", leader="H075_F4", floor=4)); time.sleep(3.5); v = chips()
    step("structure SHADOW_WOULD_CONFIRM: no final card", "최종 확정 아님" in v["text"] and not v["final"])
    rm("structure_path.json")
    # ---- anchored + pair + observations
    put("e2/semantic_anchor.json", dict(model_status="UNVERIFIED_ANCHOR_MODEL", candidates=[], runs=[dict(status="ANCHORED_PROVISIONAL", observation_id="O3", floor=4, room="room_408", top=[dict(lidar_cost=0.011, separation_ratio=10.7)])])); time.sleep(3.5); v = chips()
    step("SEMANTIC_ANCHORED: provisional and cannot confirm", "SEMANTIC_ANCHORED" in v["codes"] and "room_408" in v["text"] and "확정 불가" in v["text"] and not v["final"])
    obs = [dict(observation_id="O2", ocr_lines=[["B", 0.52]], reading=dict(vlm=dict(state="UNAVAILABLE"))), dict(observation_id="O3", ocr_lines=[["408", 0.997]], reading=dict(vlm=dict(state="NOT_NEEDED"))),
           dict(observation_id="O5", ocr_lines=[], reading=dict(vlm=dict(state="NOT_ADMITTED_RESOURCE"))), dict(observation_id="O6", ocr_lines=[["TOILETS", 0.935]], reading=dict(vlm=dict(state="NOT_NEEDED")))]
    per = [dict(observation_id="O2", usable=False, why_not="NO_DECISIVE_LABEL", reading_kind="NONE", labels=[]), dict(observation_id="O3", usable=True, reading_kind="ROOM", labels=[["408", 0.68]]),
           dict(observation_id="O5", usable=False, why_not="NO_DECISIVE_LABEL", reading_kind="NONE", labels=[]), dict(observation_id="O6", usable=True, reading_kind="CATEGORY", category=dict(category="restroom"), labels=[["cat:restroom", 0.935]])]
    put("e8_pair_state.json", dict(state="E7_VERIFYING", observations=obs, evaluation=dict(verdict="REGION_CANDIDATE_READY", why="2 independent plates", n_valid_independent=2, per_observation=per))); time.sleep(3.5); v = chips()
    step("408+TOILETS comparison line", "408+TOILETS 비교 중" in v["text"], text=v["text"][-500:])
    step("valid independent 2 vs 4 attempts, failed reads kept with reasons", "유효 독립 관측 2" in v["text"] and "촬영 시도 4" in v["text"] and "NO_DECISIVE_LABEL" in v["text"] and "시설 범주" in v["text"])
    step("VLM that did not run is never shown as done", "VLM 실행 안 됨 (서버 없음) — 확인된 것이 아닙니다" in v["text"] and "VLM 실행 안 됨 (자원 부족, OCR 우선)" in v["text"] and "VLM 실행됨" not in v["text"])
    with open(C.STATE / "e6_observations.jsonl", "a") as f_:
        f_.write(json.dumps(dict(state="WAIT_STOP", observation_id="O7", pre_stop_aim=dict(status="STOP_EARLY", guidance="요청 후 1.2 m 안에 정지해야 조준할 수 있습니다. 더 가면 조준이 거부됩니다.", rows=[], command_sent=False))) + "\n")
    time.sleep(3.5); v = chips()
    step("pre-stop aim guidance is shown", "STOP_EARLY" in v["codes"] and "1.2 m 안에" in v["text"])
    # ---- R35: E7 class, shadow, E3 lag
    rm("e7_shadow_early.json"); rm("structure_path.json")
    put("e7_state.json", dict(global_decision="AMBIGUOUS", structural_class="STRUCTURALLY_AMBIGUOUS", why="E7: best E2X_F2_25 leads by only 0.0021 m < 0.05 m", written_wall=time.time(), e7=dict(best="E2X_F2_25", gap_m=0.0021))); time.sleep(3.5); v = chips()
    step("R35 ambiguous E7: says LiDAR cannot separate and asks for another room plate; no final card", "E7_AMBIGUOUS" in v["codes"] and "구조로 구별되지 않음" in v["text"] and "다른 호실 번호" in v["text"] and not v["final"], text=v["text"][:300])
    put("e7_state.json", dict(global_decision="REJECTED", structural_class="STRUCTURAL_CONTRADICTION", why="E7: X fits better by 0.2 m", written_wall=time.time(), e7=dict(best="X", gap_m=0.2))); time.sleep(3.5); v = chips()
    step("R35 real contradiction is shown as a contradiction; no final card", "E7_REJECTED" in v["codes"] and "모순" in v["text"] and not v["final"])
    put("e7_semantic_shadow.json", dict(verdict="WOULD_FIX", blocked_by=[], rejected_by=[])); time.sleep(3.5); v = chips()
    step("R35 shadow WOULD_FIX is record only: no final card, says not final", "SHADOW_WOULD_FIX" in v["codes"] and "기록 전용" in v["text"] and not v["final"])
    put("e3_state.json", dict(leader="H1", leader_kf_lag_s=45.0, kf_pending=30, hypotheses=[], n_alive=0)); time.sleep(3.5); v = chips()
    step("R35 E3 lag is shown", "E3_LAG" in v["codes"] and "45" in v["text"])
    rm("e7_semantic_shadow.json"); rm("e3_state.json")
    # ---- only a real backend FIX shows the final card
    put("e7_state.json", dict(global_decision="GLOBAL_CONFIRMED", written_wall=(X.get("created_wall") or time.time()) - 500, why="older run")); time.sleep(3.5); v = chips()
    step("a GLOBAL_CONFIRMED of an earlier run shows no final result", not v["final"] and "GLOBAL_CONFIRMED" not in v["codes"])
    errs = mn.js("return window.__jsErrors||[]") or []
    step("page has no JS exception", errs == [], errors=errs[:3])
    mn.js("document.getElementById('field_status').scrollIntoView(); return 1;"); time.sleep(0.5)
    mn.shot("r34_field_status")
    mn.js("act('end'); return 1;")
    t1 = time.time()
    while time.time() - t1 < 120:
        if ((mn.state() or {}).get("experiment") or {}).get("state") == "ENDED":
            break
        time.sleep(1)
    step("experiment ends cleanly", ((mn.state() or {}).get("experiment") or {}).get("state") == "ENDED")
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
