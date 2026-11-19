"""R41: the page + console (real code, fake workspace) showing the REAL records of an integrated replay: e7_state, fix_overlay, fix_current, E3, E4, E7 long, E8 of that replay. Only their times are moved to now, as the console refuses records older than the experiment.
usage: ui_r41_firefox.py OUT BASELINE_DIR REPLAY_STATE_DIR"""
import base64, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPLAY = Path(sys.argv[3])
sys.argv = sys.argv[:3]
OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
T = HERE
sys.path.insert(0, str(T))
import shutil, subprocess
import test_console_fake as F  # noqa: E402
src = (HERE / "ui_r34_firefox.py").read_text()
head = src[:src.index("try:\n    mn = Mn(2834)")].replace('made_by="ui_r34_firefox"', 'made_by="ui_r41_firefox"')
exec(compile(head, "ui_r34_head", "exec"))
def rd(n):
    try:
        return json.loads((REPLAY / n).read_text())
    except (OSError, ValueError):
        return None
try:
    mn = Mn(2834)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1280, "height": 1500})
    mn.cmd("WebDriver:Navigate", {"url": f"http://127.0.0.1:{ws.port}/"}); time.sleep(4)
    mn.js("window.confirm=()=>true; window.alert=()=>{}; window.__jsErrors=[]; window.addEventListener('error',e=>window.__jsErrors.push(String(e.message))); return 1;")
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:1007.5,py:708.5,yaw_deg:-0.3}); return 1;"); time.sleep(1.5)
    mn.js("document.getElementById('inp').value='body'; document.getElementById('e6mode').value='AUTO'; document.getElementById('b_start').click(); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120:
        X = (mn.state() or {}).get("experiment") or {}
        if X.get("state") in ("RUNNING", "PARTIAL"):
            break
        time.sleep(1)
    step("automatic run started", X.get("state") in ("RUNNING", "PARTIAL"), state=X.get("state"))
    now = time.time()
    E7 = rd("e7_state.json"); FO = rd("fix_overlay.json"); FC = rd("fix_current.json")
    old = float(E7["written_wall"]); dt = now - old
    def mv(o):
        if isinstance(o, dict):
            return {k: mv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [mv(v) for v in o]
        if isinstance(o, float) and abs(o - old) < 5000:
            return o + dt
        return o
    for n in ("e2/e2_state.json", "e3_state.json", "e4_state.json", "auto_state.json", "e7_long.json", "loc_overlay.json", "e8_pair_state.json", "semantic_estimate.json", "e7_semantic_shadow.json", "vlm_state.json"):
        d = rd(n)
        if d is not None and n != "e7_state.json":
            put(n, mv(d))
    time.sleep(4)
    v = mn.js("const o=document.getElementById('res_open'); if(o) o.click(); return {btn:!!o};"); time.sleep(3)
    v = mn.js("return {vis:!document.getElementById('resdlg').classList.contains('hide'),title:document.getElementById('res_title').innerText,kv:document.getElementById('res_kv').innerText};")
    step("before the FIX: details open on request, say unconfirmed, floor and reason", v["vis"] and "위치 미확정" in v["title"] and "4F" in v["kv"], kv=v["kv"][:300])
    mn.shot("r41_details_before_fix")
    mn.js("resClose(); return 1;")
    put("fix_overlay.json", mv(FO)); put("fix_current.json", mv(FC) if FC else {}); put("e7_state.json", mv(E7)); time.sleep(6)
    st = (mn.state() or {}).get("final_fix") or {}
    v = mn.js("const c=document.getElementById('final_card'); return {pos:c.style.position,t:c.innerText,map:!!document.getElementById('fin_map'),errs:window.__jsErrors};")
    step("real FIX record: console final_fix.valid, snapshot accepted", st.get("valid") is True and st.get("fix_overlay_ok") is True, ff={k: st.get(k) for k in ("valid", "path", "fix_overlay_ok", "TTF_raw_s")})
    step("popup: title, floor, current and verified pose, before E7, plates, both TTF", v["pos"] == "fixed" and v["map"] and all(x in v["t"] for x in ("최종 초기 전역 위치추정 완료", "4F", "현재 위치", "보정 전 위치", "실제 최초 위치 확정시간", "수동조작 기준 보정", "SEMANTIC_2ROOM")), text=v["t"][:900])
    print("POPUP_TEXT", v["t"], flush=True)
    step("no JS errors", not v["errs"], errs=v["errs"])
    mn.shot("r41_final_popup")
    mn.js("act('end'); return 1;"); time.sleep(5)
finally:
    (OUT / "RESULT.json").write_text(json.dumps(RES, ensure_ascii=False, default=str, indent=0))
    print("RESULT", "OK" if RES["ok"] else "FAILED", sum(s["ok"] for s in RES["steps"]), "/", len(RES["steps"]), flush=True)
    try:
        ff.terminate()
    except Exception:
        pass
    try:
        ws.ctl("stop", "all", timeout=200)
    except Exception:
        pass
