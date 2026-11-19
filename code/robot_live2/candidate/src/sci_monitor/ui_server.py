#!/usr/bin/env python3
"""SCI E5 operator UI (read-only). Serves the state published by e5_node in /dev/shm/sci_e5.

There is NO endpoint that moves the robot or the pan/tilt. STOP_REQUEST and RELEASE are notifications for the operator,
who stops/resumes the Go2 with the joystick. The UI keeps running when the E5 node or SigLIP is down (shows the fault).

  python ui_server.py --port 8095      ->  http://<robot>:8095/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from flask import Flask, Response, jsonify, send_file

SHM = Path(os.environ.get("SCI_SHM", "/dev/shm/sci_e5"))
app = Flask(__name__)


def status():
    try:
        s = json.loads((SHM / "status.json").read_text())
        s["age_s"] = round(time.time() - s.get("now", 0), 1)
        s["node"] = "OK" if s["age_s"] < 5 else "STALE"
        return s
    except Exception as e:
        return dict(node="DOWN", error=f"{type(e).__name__}: {e}")


@app.route("/status")
def r_status():
    return jsonify(status())


@app.route("/frame.jpg")
def r_frame():
    p = SHM / "frame.jpg"
    return send_file(str(p), mimetype="image/jpeg", max_age=0) if p.exists() else ("no frame", 404)


@app.route("/last_stop.jpg")
def r_last_stop():
    p = SHM / "last_stop.jpg"
    return send_file(str(p), mimetype="image/jpeg", max_age=0) if p.exists() else ("no stop yet", 404)


@app.route("/stream.mjpg")
def r_stream():
    def gen():
        last = 0
        while True:
            p = SHM / "frame.jpg"
            try:
                m = p.stat().st_mtime
                if m != last:
                    last = m
                    b = p.read_bytes()
                    yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n"
            except FileNotFoundError:
                pass
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SCI E5 Console</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--fg:#e6e8eb;--mut:#8a93a3;--ok:#2fbf71;--warn:#f0b429;--bad:#e5484d;--stop:#ff2d2d;--acc:#4ea1ff}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f5f6f8;--panel:#fff;--fg:#14171c;--mut:#5b6472}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;gap:10px;align-items:center;padding:8px 16px;background:var(--panel);flex-wrap:wrap}
header b{font-size:16px}.pill{padding:2px 8px;border-radius:10px;font-size:12px;background:#333;color:#fff}
.ok{background:var(--ok)}.warn{background:var(--warn);color:#000}.bad{background:var(--bad)}
#banner{display:none;padding:14px 16px;font-size:26px;font-weight:800;text-align:center;color:#fff;background:var(--stop);animation:bl 1s infinite}
#banner.release{background:var(--ok);animation:none}#banner.wait{background:#c26a00;animation:none}
@keyframes bl{50%{opacity:.65}}
main{display:grid;grid-template-columns:minmax(0,640px) minmax(0,1fr);gap:12px;padding:12px 16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border-radius:8px;padding:10px;min-width:0}
.card h3{margin:0 0 6px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
img{width:100%;border-radius:6px;display:block}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:3px 4px;border-bottom:1px solid #2a2f3a;text-align:left;vertical-align:top}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:13px}.kv span:nth-child(odd){color:var(--mut)}
.st{font-weight:700}.STOP_REQUEST{color:var(--stop)}.VERIFYING{color:#ffb000}.REJECTED_POSTER,.REJECTED_GLARE,.REJECTED_ELEVATOR{color:#c77dff}
.REJECTED_DISTANCE,.REJECTED_ANGLE,.REJECTED_PANTILT{color:#ff9f43}.ALREADY_OBSERVED{color:var(--ok)}
.scroll{max-height:320px;overflow:auto}details summary{cursor:pointer;color:var(--mut)}
.tl span{display:inline-block;margin:2px;padding:2px 6px;border-radius:4px;background:#2a2f3a;font-size:12px}
.tl span.cur{background:var(--acc);color:#000}
</style></head><body>
<header><b>SCI E5 · 표지판 관측 이벤트</b><span id="p_node" class="pill">node</span><span id="p_sig" class="pill">SigLIP</span>
<span id="p_odom" class="pill">odom</span><span id="p_depth" class="pill">depth</span><span id="p_e4" class="pill">E4</span>
<span id="p_pt" class="pill">pan/tilt</span><span id="p_res" class="pill">res</span><span id="p_deg" class="pill">load</span><span style="margin-left:auto;color:var(--mut)" id="run"></span></header>
<div id="banner"></div>
<main>
 <section class="card"><h3>전방 카메라 (후보·판단)</h3><img id="cam" src="/stream.mjpg" alt="camera">
  <div class="kv" style="margin-top:8px" id="scene"></div></section>
 <section style="display:flex;flex-direction:column;gap:12px;min-width:0">
  <div class="card"><h3>관측 이벤트</h3><div id="fsm"></div></div>
  <div class="card"><h3>판독 · E6 / E7</h3><div id="read" class="scroll"></div></div>
  <div class="card"><h3>마지막 정지 요청 화면</h3><img id="last" src="/last_stop.jpg" alt="no stop yet" onerror="this.style.display='none'"></div>
 </section>
 <section class="card" style="grid-column:1/-1"><h3>P2 표지 트랙 — 정지 시점 예측(반응시간 반영)</h3><div class="scroll"><table id="tracks"></table></div>
  <div style="color:var(--mut);font-size:12px">WAIT_READABLE = 표지 확인됨, 정지 위치에서 판독 조건(사각·거리·팬틸트·글자 크기) 충족을 기다리는 중 · MISSED = 조건 충족 전에 지나침</div></section>
 <section class="card" style="grid-column:1/-1"><h3>후보 (디버그: 계층별 거부 사유)</h3><div class="scroll"><table id="cands"></table></div></section>
 <section class="card" style="grid-column:1/-1"><h3>자원 · 지연 · 장애</h3><div class="kv" id="res"></div><details><summary>장애 기록</summary><pre id="faults" style="white-space:pre-wrap"></pre></details></section>
</main>
<script>
const $=id=>document.getElementById(id);
const pill=(id,t,c)=>{const e=$(id);e.textContent=t;e.className='pill '+c};
const f1=v=>v==null?'–':(typeof v=='number'?v.toFixed(2):v);
const ORDER=["STOP_REQUEST","WAIT_STABLE","STOP_CONFIRMED","AIM","CAPTURE","CAPTURED","RELEASE","READ_ASYNC","LOCALIZE_ASYNC","FIX_CONFIRMED"];
let lastStop=0;
async function tick(){
 let s; try{s=await (await fetch('/status')).json()}catch(e){pill('p_node','UI↔node ERR','bad');return}
 pill('p_node','node '+(s.node||'?'),s.node=='OK'?'ok':'bad');
 if(s.node=='DOWN'){ $('banner').style.display='block';$('banner').className='';$('banner').textContent='E5 NODE DOWN — '+(s.error||'');return;}
 pill('p_sig','SigLIP '+s.siglip,s.siglip=='OK'?'ok':'bad');
 const h=s.source_health||{};pill('p_odom','odom '+(h.odom||'?')+(h.odom_hz?(' '+h.odom_hz+' Hz'):'')+(h.odom_topic?(' '+h.odom_topic):''),(h.odom=='OK'||h.odom=='REPLAY')?'ok':'bad');
 pill('p_deg',s.degraded?'과부하: 장면 0.5 Hz':'부하 정상',s.degraded?'warn':'ok');
 pill('p_depth','depth '+(h.depth||'?'),(h.depth=='OK'||h.depth=='REPLAY')?'ok':'bad');
 const e4=s.e4||{};pill('p_e4',e4.status||'E4 ?',e4.available?'ok':'warn');
 const pc=s.pantilt_console||{};pill('p_pt','pan/tilt '+(pc.reachable?'console OK':'console OFF')+(s.pantilt_execute_enabled?' · EXEC':' · DRY-RUN'),pc.reachable?(s.pantilt_execute_enabled?'ok':'warn'):'warn');
 const r=s.resources||{};pill('p_res',`CPU ${f1(r.cpu_pct)}% GPU ${f1(r.gpu_pct)}% RAM ${r.mem_used_mb||'?'}MB`,(r.mem_avail_mb||0)>2500?'ok':'bad');
 $('run').textContent=(s.source||'')+' · '+(s.run||'').split('/').pop()+' · stage '+(s.active_stage||'');
 const sc=s.scene||{};const tm=sc.timing||{};
 $('scene').innerHTML=`<span>장면 t</span><span>${f1(sc.t)} s (#${sc.n_scene||0}, 건너뜀 ${sc.n_skipped||0})</span><span>속도</span><span>${f1(sc.speed)} m/s</span>
 <span>처리</span><span>전체 ${f1(tm.scene_ms)} ms · 후보 ${f1(tm.proposals_ms)} · SigLIP ${f1(tm.siglip_infer_ms)} ms</span><span>상대기준</span><span>${f1(sc.thr)}</span>
 <span>정지 수</span><span>${Object.entries(s.n_stops||{}).map(([k,v])=>k+':'+v).join(' · ')}</span>`;
 const f=s.fsm; const b=$('banner');
 if(f){const st=f.state; b.style.display='block';
   if(['STOP_REQUEST','WAIT_STABLE'].includes(st)){b.className='';b.textContent=`STOP_REQUEST #${f.id} — 로봇을 정지하세요 (조이스틱)`;}
   else if(st=='AIM'){b.className='wait';b.textContent=`AIMING #${f.id} — 팬틸트 조준 중, 움직이지 마세요`;}
   else if(['STOP_CONFIRMED','AIM','CAPTURE'].includes(st)){b.className='wait';b.textContent=`STOP_CONFIRMED #${f.id} — ${st} 진행 중, 움직이지 마세요`;}
   else if(['CAPTURED','RELEASE','READ_ASYNC','LOCALIZE_ASYNC','FIX_CONFIRMED'].includes(st)){b.className='release';b.textContent=`RELEASE #${f.id} — 촬영 완료, 주행 재개 가능 (판독은 백그라운드)`;}
   else if(st=='FAILED'){b.className='wait';b.textContent=`FAILED #${f.id} — ${f.log[f.log.length-1].reason}`;}
   const aim=(f.result||{}).aim||{};const pl=aim.plan||{};
   $('fsm').innerHTML=`<div class="tl">${ORDER.map(x=>`<span class="${x==st?'cur':''}">${x}</span>`).join('')}${st=='FAILED'?'<span class="cur" style="background:var(--bad)">FAILED</span>':''}</div>
   <div class="kv"><span>팬틸트 목표</span><span>pan ${f1(pl.pan_cmd)}° · tilt ${f1(pl.tilt_cmd)}° · ${pl.reachable?'범위 안':'범위 밖'}</span>
   <span>팬틸트 실제</span><span>${aim.mode=='EXECUTE'?(aim.ok?'콘솔 완료(명령각; 실제 서보각 미측정)':'실패 '+aim.why):'미구동 ('+(aim.why||aim.mode||'-')+')'}</span>
   <span>촬영</span><span>${(f.result||{}).capture_source||'–'} ${(f.result||{}).quality? 'sharp '+f1(f.result.quality.sharp):''}</span>
   <span>소요</span><span>${(f.result||{}).durations_s?Object.entries(f.result.durations_s).map(([k,v])=>k+' '+v).join(' · '):'–'}</span></div>
   <details><summary>전이 기록</summary><table>${f.log.map(x=>`<tr><td>${x.state}</td><td>${x.reason}</td><td>${new Date(x.t*1000).toLocaleTimeString()}</td></tr>`).join('')}</table></details>`;
 } else {b.style.display='none';$('fsm').textContent='SEARCHING — 정지 요청 없음';}
 $('read').innerHTML=(s.events||[]).slice().reverse().map(e=>{const rd=e.reading||{};
   return `<div style="margin-bottom:6px"><b>#${e.id}</b> t=${f1(e.t)}s · ${e.state} · 촬영 ${e.capture_source||'–'}<br>
   판독: <b>${rd.status||'대기'}</b> ${rd.sign_type||''} ${(rd.labels||[]).map(l=>l.label+'('+f1(l.p)+')').join(', ')} <span style="color:var(--mut)">OCR ${(rd.ocr_text||[]).join(' ')}</span><br>
   VLM: ${rd.vlm_answer?JSON.stringify(rd.vlm_answer):(rd.vlm_note||'–')}<br>E6: ${rd.e6?rd.e6.status+' '+((rd.e6.floors||[]).map(x=>x.floor+':'+f1(x.p)).join(' ')):'–'} · E7: ${rd.e7?rd.e7.status+' ('+(rd.e7.why||'')+')':'–'}</div>`}).join('')||'없음';
 $('tracks').innerHTML='<tr><th>spot</th><th>상태</th><th>사유</th><th>예측 사각°</th><th>예측 거리 m</th><th>방위°</th><th>pan/tilt</th><th>글자 px</th></tr>'+
  (s.p2_tracks||[]).map(t=>{const p=t.pred||{};return `<tr><td>${t.spot}</td><td class="st ${t.state=='STOP_REQUEST'?'STOP_REQUEST':''}">${t.state}</td><td style="font-size:11px">${(t.why||[]).join(', ')}</td>
  <td>${f1(p.obliquity)}</td><td>${f1(p.range)}</td><td>${f1(p.bearing)}</td><td>${p.pan!=null?f1(p.pan)+'/'+f1(p.tilt):'–'}${p.reachable===false?' ✗':''}</td><td>${f1(p.text_px)}</td></tr>`}).join('')+
  `<tr><td colspan=8 style="color:var(--mut)">놓친 표지(MISSED) 누계: ${s.p2_missed||0}</td></tr>`;
 $('cands').innerHTML='<tr><th>spot</th><th>상태</th><th>SigLIP</th><th>평균</th><th>거리 m</th><th>예측사각°</th><th>pan/tilt</th><th>글자px</th><th>포화</th><th>사유</th></tr>'+
  (sc.regions||[]).map(c=>`<tr><td>${c.spot??'–'}</td><td class="st ${c.display}">${c.display}</td><td>${f1(c.sim)}</td><td>${f1(c.msim)}</td><td>${f1(c.z)}</td>
  <td>${f1(c.obliquity)}</td><td>${c.pantilt&&c.pantilt.pan!=null?f1(c.pantilt.pan)+'/'+f1(c.pantilt.tilt):'–'}${c.pantilt&&!c.pantilt.ok?' ✗':''}</td><td>${f1(c.text_px)}</td><td>${f1(c.glare&&c.glare.sat)}</td>
  <td style="font-size:11px">${(c.reasons||[]).join(', ')}</td></tr>`).join('');
 $('res').innerHTML=`<span>RAM 사용/가용</span><span>${r.mem_used_mb} / ${r.mem_avail_mb} MB · swap ${r.swap_used_mb} MB</span><span>CPU·GPU</span><span>${f1(r.cpu_pct)}% · ${f1(r.gpu_pct)}% · load ${f1(r.load1)}</span>
 <span>온도</span><span>${Object.entries(r.temps||{}).slice(0,4).map(([k,v])=>k+' '+v.toFixed(0)+'°').join(' · ')}</span><span>프로세스 RSS</span><span>${Object.entries(r.rss_mb||{}).map(([k,v])=>k+' '+f1(v)+'MB').join(' · ')}</span>
 <span>상태 나이</span><span>${s.age_s}s</span>`;
 $('faults').textContent=(s.faults||[]).map(x=>new Date(x.t*1000).toLocaleTimeString()+' '+x.kind+' '+x.detail).join('\n')||'없음';
 if(s.fsm&&s.fsm.id!==lastStop){lastStop=s.fsm.id;const im=$('last');im.style.display='block';im.src='/last_stop.jpg?'+Date.now();}
}
setInterval(tick,500);tick();
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, threaded=True)
