"""R36 UI test: REAL 8770 console + web_v2 page in REAL Firefox (headless), FAKE workspace, state files in the backend's own formats. Nothing moves.
Checked: start refused without a mode; loc1 starts without SigLIP / E6 and hides the pan-tilt form; long-structure status lines; loc2 start; resume text during reading; reposition panel with its buttons; VLM chip; no final card without a FIX; no JS errors."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_console_fake as F  # noqa: E402
src = (T / "ui_r34_firefox.py").read_text()
# reuse the harness of ui_r34 up to the browser start (workspace, approval, Marionette client)
head = src[:src.index("try:\n    mn = Mn(2834)")].replace('made_by="ui_r34_firefox"', 'made_by="ui_r36_firefox"')
exec(compile(head, "ui_r34_head", "exec"))
try:
    mn = Mn(2834)
    mn.cmd("WebDriver:SetWindowRect", {"width": 1100, "height": 1600})
    base = f"http://127.0.0.1:{ws.port}/"
    mn.cmd("WebDriver:Navigate", {"url": base}); time.sleep(3)
    mn.js("window.__alerts=[]; window.confirm=()=>true; window.alert=(m)=>window.__alerts.push(String(m)); window.__jsErrors=[]; window.addEventListener('error',e=>window.__jsErrors.push(String(e.message))); return 1;")
    mn.js("act('gt_save',{building:'kci',floor:4,kind:'start',px:2319.7,py:709.0,yaw_deg:179.3}); return 1;"); time.sleep(1.5)
    v = mn.js("return {radios:document.querySelectorAll('input[name=locmode]').length,btn:document.getElementById('b_start').innerText,note:document.getElementById('auto_note').innerText,ptc:!document.getElementById('ptc_box').classList.contains('hide')};")
    step("start screen: no zone choice, one button, consent still offered", v["radios"] == 0 and "자동 초기 전역 위치추정 시작" in v["btn"] and v["ptc"], v=v)
    mn.js("window.confirm=()=>true; document.getElementById('inp').value='body'; document.getElementById('b_start').click(); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120:
        X = (mn.state() or {}).get("experiment") or {}
        if X.get("state") in ("RUNNING", "PARTIAL"):
            break
        time.sleep(1)
    EXP["id"] = X.get("exp_id")
    step("automatic run: one mode recorded, SigLIP on", X.get("loc_mode") == "AUTO_INITIAL_GLOBAL" and X.get("siglip") is True, state=X.get("state"))
    put("e7_long.json", dict(kind="LIVE2_E7_LONG", status="AMBIGUOUS_FLOORS", leader_floor=4, close_floors=[7, 8], next_gap_m=0.003, path_m=64.0, floors_exact=9, floors_total=9, leader_cost=0.044, authority="NONE", written_wall=time.time()))
    (C.STATE / "e3_state.json").exists()
    time.sleep(3.5)
    now = mn.js("const e=document.getElementById('now_status'); return e?e.innerText:'';")
    fs = mn.js("return document.getElementById('field_status').innerText;")
    fin = mn.js("return document.getElementById('final_card').classList.contains('hide');")
    step("loc1 ambiguous floors named, no final card", "7F" in fs.replace("[7, 8]", "7F") and "구조" in fs and fin, now=now[:160], fs=fs[:300])
    mn.js("act('end',{}); return 1;")
    t0 = time.time()
    while time.time() - t0 < 120:
        X = (mn.state() or {}).get("experiment") or {}
        if X.get("state") in ("ENDED", None) or not X.get("exp_id"):
            break
        time.sleep(1)
    step("loc1 ended", True, state=X.get("state"))
    for k in list(C.STATE.glob("*.json")):
        pass
    errs = mn.js("return window.__jsErrors;")
    step("no JS errors", not errs, errs=errs)
    # panel rendering, directly with backend-format data
    r = mn.js("""const M={phase:'REPOSITION_NEEDED',observation_id:'O1',experiment_id:'X',wall:Date.now()/1000,phase_since:Date.now()/1000,why:'x',
        guidance:{ok:true,text:'조이스틱으로 뒤로 약 1.0 m 움직인 뒤 정지하고 [재조준]을 누르세요.',moves:[{kind:'BACK',m:1.0}]},refused:{status:'OUTSIDE_VERIFIED_ENVELOPE',pan:61.88,tilt:150.12}};
        CONTROL=true; e6mRepos(M,{you_own:true}); const el=document.getElementById('e6m_repos'); return {vis:!el.classList.contains('hide'),t:el.innerText,b:[...el.querySelectorAll('button')].map(x=>x.innerText)};""")
    step("reposition panel: instruction and both buttons", r["vis"] and "뒤로 약 1.0 m" in r["t"] and "재조준" in r["b"] and any("건너뛰기" in x for x in r["b"]), r=r)
    r = mn.js("""const v=flowView({state:'RESUME_READY',observation_id:'O1',why:'PAIR_READING: photo stored'}); return v;""")
    step("after the photo the page says drive on while reading continues", "이동을 재개" in r[1] and "판독은 이동 중" in r[1], r=r)
    r = mn.js("return flowView({state:'PAIR_READING'})===null && flowView({state:'E7_VERIFYING'})===null;")
    step("no 'stay stopped for reading / E7' text left", r)
    r = mn.js("""const M={phase:'REPOSITION_NEEDED',observation_id:'O1',guidance:{ok:false,text:'이번 표지판은 건너뛰세요.'},refused:{}}; document.getElementById('e6m_repos').dataset.k=''; e6mRepos(M,{you_own:true}); return [...document.getElementById('e6m_repos').querySelectorAll('button')].map(x=>x.innerText);""")
    step("no way back inside: only skip offered", len(r) == 1 and "건너뛰기" in r[0], r=r)
    # ---- R37: statuses from backend-format records, in the real page functions
    NS = "const now=Date.now()/1000; const X1={exp_id:'X',created_wall:now-100,loc_mode:'AUTO_INITIAL_GLOBAL'}, X2=X1;"
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{final_verify:{status:'FINAL_VERIFICATION_STOP_REQUESTED',leader_floor:4}}},{experiment:X1,final_fix:{}},null);")
    step("final verification: the exact sentence", r[1] == "최종 위치 검증을 위해 정지해 주세요.", r=r)
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now+300},live2:{auto_state:{structure_class:'AMBIGUOUS',assessment:{why:'x'}}}},{experiment:X1,final_fix:{},op_clock:{ops_s:300,ops_paused:false}},null);")
    step("no FIX past the time: the status line only, exact text", r[1] == "현재 위치추정 진행 중 — 추가 구조정보 또는 장소정보가 필요합니다.", r=r)
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now+300},live2:{auto_state:{structure_class:'AMBIGUOUS',assessment:{why:'x'}}}},{experiment:X1,final_fix:{},op_clock:{ops_s:300,ops_paused:true}},null);")
    step("during E6 work no time notice", "추가 구조정보" not in r[1], r=r)
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{final_verify:{status:'SHADOW_WOULD_CONFIRM',why:'x'}}},{experiment:X1,final_fix:{}},null);")
    step("policy off is named", "SHADOW" in r[1], r=r)
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{auto_state:{needs_pantilt_consent:true}}},{experiment:X1,final_fix:{}},null);")
    step("no consent: says hardware consent is needed, structure continues", "하드웨어 사용 동의" in r[1] and "구조 기반 추정은 계속" in r[2], r=r)
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{auto_state:{structure_class:'DISCRIMINATIVE',assessment:{leader_floor:4}}}},{experiment:X1,final_fix:{}},null);")
    step("structure sufficient: floor named, provisional, no stop", "4F" in r[1] and "PROVISIONAL" in r[1] and "표지판 정지 없이" in r[2], r=r)
    SE = "semantic_estimate:{status:'PROVISIONAL',readings:[{label:'405'}],floors:[4],best:{id:'A',floor:4,pose_px:[100,200],heading_deg:10},alternatives:[{}],written_wall:now,age_s:1}"
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{" + SE + "}},{experiment:X2,final_fix:{}},null);")
    step("first reading: estimate named with room, floor, other candidates", "405" in r[1] and "4F" in r[1] and "다른 위치 후보 1" in r[2], r=r)
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{semantic_estimate:{status:'WAITING_CANDIDATE',readings:[{label:'405'}],floors:[4],why:'계산 중'}}},{experiment:X2,final_fix:{}},null);")
    step("reading before the fit: says the candidate is being computed", "위치 후보 계산 중" in r[1], r=r)
    r = mn.js("return nowStatus({meta:{server_wall:1},live2:{final_verify:{status:'VERIFIED_FIX'}}},{experiment:{created_wall:0},final_fix:{valid:true}},null);")
    step("final status only from the backend FIX record", "완료" in r[1], r=r)
    # ---- R39: no automatic dialog; details on request; the final popup waits for the pan-tilt
    RD = NS + """const hy=[{id:'H1',floor:4,status:'ACTIVE',score:0.3,pose_px:[1000,700],pose_m:[40,28],heading_deg:12,ppm:25}];
      const lg={status:'AMBIGUOUS_FLOORS',leader_floor:4,close_floors:[7,8],next_gap_m:0.002,leader_cost:0.024,path_m:87,floors_exact:9,floors_total:9,why:'4F leads, but [7, 8]F are within 0.05 m'};
      const st=()=>({vis:!document.getElementById('resdlg').classList.contains('hide'),title:document.getElementById('res_title').innerText,kv:document.getElementById('res_kv').innerText});"""
    for sec in (240, 300, 900):
        r = mn.js(RD + f"resClose(); renderResult({{meta:{{server_wall:now+{sec}}},live2:{{e3:{{hypotheses:hy}},e7_long:lg}}}},{{experiment:Object.assign({{state:'RUNNING'}},X1),final_fix:{{}}}}); return st();")
        step(f"{sec} s without a FIX: no dialog opens by itself", not r["vis"])
    r = mn.js(RD + "resDetail({meta:{server_wall:now+150},live2:{e3:{hypotheses:hy},e7_long:lg}},{experiment:Object.assign({state:'RUNNING'},X1),final_fix:{}}); return st();")
    step("details on request: says unconfirmed, floor, rivals, reason", r["vis"] and "위치 미확정" in r["title"] and all(x in r["kv"] for x in ("4F", "40.0", "12°", "0.024", "7F", "확정 불가 이유")), title=r["title"])
    r = mn.js(RD + "renderResult({meta:{server_wall:now+220},live2:{}},{experiment:Object.assign({state:'RUNNING'},X1),final_fix:{valid:true}}); return st();")
    step("a real FIX closes the details", not r["vis"])
    FP = NS + "const X3=Object.assign({state:'RUNNING',exp_id:'F1'},X1,{exp_id:'F1',state:'RUNNING'}); const fx=(p)=>({experiment:X3,final_fix:{valid:true,t_fix:now,TTF_raw_s:88.5,TTF_normalized_s:60.1,path:'SEMANTIC_2ROOM',used_observation_ids:['O1','O2'],presentation:p}}); const el=document.getElementById('final_card'); const sv=()=>({t:el.innerText,fixed:el.style.position==='fixed',pend:!!document.getElementById('final_pending')});"
    for ph, hw in (("AIMING", "PANTILT_AIMING"), ("WAIT_MANUAL_FOCUS", "WAIT_MANUAL_FOCUS"), ("CAPTURE_REVIEW", "CAPTURE_REVIEW"), ("REPOSITION_NEEDED", "REPOSITION_NEEDED"), ("RETURNING_HOME", "HOME_RETURNING")):
        r = mn.js(FP + f"renderFinal({{meta:{{server_wall:now}},live2:{{}}}},fx({{state:'FINAL_PRESENTATION_PENDING',phase:'{ph}',hw:'{hw}'}})); return sv();")
        step(f"FIX during {ph}: no popup, one line waits", r["pend"] and not r["fixed"] and "최종 초기" not in r["t"], t=r["t"][:80])
    r = mn.js(FP + "renderFinal({meta:{server_wall:now},live2:{}},fx({state:'FINAL_RESULT_VISIBLE',hw:'E6_IDLE'})); const a=sv(); document.getElementById('final_close').click(); return [a,sv()];")
    step("after HOME: full-screen popup with the required title and both TTF", r[0]["fixed"] and "최종 초기 전역 위치추정 완료" in r[0]["t"] and "88.5" in r[0]["t"] and "60.1" in r[0]["t"] and "SEMANTIC_2ROOM" in r[0]["t"], t=r[0]["t"][:200])
    step("[닫기] leaves the card in the page", not r[1]["fixed"] and "최종 초기" in r[1]["t"])
    mn.js("renderFinal({},{experiment:{},final_fix:{}}); return 1;")
    r = mn.js("renderTopK({live2:{auto_state:{structure_class:'AMBIGUOUS',assessment:{n_active:3,n_alive:20,exact_done:2,floors_total:9,coarse_done:9,floors:[{floor:4,score:0.3,status:'ACTIVE',n_kf:30},{floor:7,score:0.31,status:'ACTIVE',n_kf:12}]},topk_hist:[{t:100,floor:4,n_active:3,n_alive:20,gap:0.01}]},e7_long:{floors:[{floor:4,cost:0.024,stage:'refresh',exact_done:true}]},e2:{refresh:{n_passes:2}}}}); return document.getElementById('topk_box').innerText;")
    step("Top-K panel: counts, floors, whole-path cost, change", all(x in r for x in ("활성 3", "4F", "7F", "0.024", "반복 탐색 2", "+100s")), r=r[:200])
    # ---- R40
    r = mn.js(NS + "return nowStatus({meta:{server_wall:now},live2:{semantic_estimate:{status:'MULTI_PLACE',readings:[{label:'407'},{label:'405'}],floors:[4]},e7_semantic_shadow:{blocked_by:['NO_FLOOR_EVIDENCE'],rejected_by:[]}}},{experiment:X1,final_fix:{}},null);")
    step("two rooms, no floor evidence: the sentence says exactly that", "2개 호실 관측 완료 — 위치 후보 생성 완료 — 층 식별 근거 추가 필요" in r[2], r=r)
    r = mn.js(FP + "renderFinal({meta:{server_wall:now},live2:{}},Object.assign(fx({state:'FINAL_RESULT_VISIBLE',hw:'E6_IDLE'}),{})); const f0=sv().t; renderFinal({},{experiment:{},final_fix:{}}); return f0;")
    step("popup names the first candidate row", "최초 위치 후보" in r, t=r[:120])
    errs = mn.js("return window.__jsErrors;")
    step("no JS errors after R37 checks", not errs, errs=errs)
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
