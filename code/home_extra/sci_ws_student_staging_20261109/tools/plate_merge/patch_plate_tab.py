#!/usr/bin/env python3
"""patch_plate_tab.py — 본 홈페이지(LIVE2 콘솔 화면 web_v2/index.html)의 "운용" 탭 안에 명판 촬영(케이스 1) 화면을 끼워 넣는다.

탭을 따로 만들지 않는다. 실험이 도는 화면(run_live)의 안내 띠들 바로 아래에 카드 하나로 들어가고, 그 안에 촬영 도구(plate_site, 기본 포트 8791)를
?embed=1 로 띄운다(높이는 내용에 맞춰 늘어나 따로 스크롤되지 않는다). 기존 화면 코드는 한 글자도 바꾸지 않고 표시(PLATE_MERGE) 사이에 세 덩어리만 더한다:
  (1) 실험 화면의 카드  (2) 준비 화면 맨 위의 "팬틸트가 아직 연결돼 있습니다" 띠  (3) 독립된 <script>
콘솔 서버는 요청마다 이 파일을 다시 읽으므로 서버를 다시 띄울 필요가 없다(화면을 새로 고치면 된다).

  python3 patch_plate_tab.py <index.html> --check            바꿀 수 있는지만 본다 (아무것도 안 씀)
  python3 patch_plate_tab.py <index.html> --out <새 파일>     끼워 넣은 결과를 새 파일로 (원본은 그대로)
  python3 patch_plate_tab.py <index.html> --revert --out <새 파일>   끼워 넣은 것을 빼낸 결과 (원본과 바이트까지 같아야 한다)
이미 끼워져 있으면(예전의 탭 방식 포함) 빼낸 뒤 다시 넣는다. 이 도구는 입력 파일을 직접 덮어쓰지 않는다."""
import argparse, hashlib, re, sys

B, E = "<!-- PLATE_MERGE_BEGIN -->", "<!-- PLATE_MERGE_END -->"
A_PANEL = '    <div id="settle_panel"></div>\n'                    # 이 줄 바로 뒤에 카드 (실험이 도는 화면 안)
A_IDLE = '  <div id="run_idle">\n'                                # 이 줄 바로 뒤에 띠 (준비 화면 맨 위)
A_SCR = "</body>"                                                # </body> 바로 앞에 독립 스크립트

PANEL = B + """
    <div id="plate_panel" class="card" style="margin-bottom:10px;padding:10px 10px 6px">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 8px">
        <h3 style="margin:0">명판 촬영 — 신호음이 나면 안내대로 이동해 촬영</h3>
        <span id="plate_stat" class="pill p-neu">확인 전</span>
        <span class="spacer"></span>
        <button class="ghost sm" id="plate_reload">다시 불러오기</button>
      </div>
      <iframe id="plate_frame" data-port="8791" title="명판 촬영" allow="autoplay" scrolling="no"
              style="display:block;width:100%;height:640px;border:0;border-radius:8px;background:#111827"></iframe>
    </div>
""" + E + "\n"

IDLE = B + """
    <div id="plate_idle" class="note warn hide" style="margin:0 0 10px">실험은 끝났지만 팬틸트가 아직 연결돼 있습니다.
      <button class="sm" id="plate_idle_off" style="margin-left:8px">정면으로 돌리고 연결 끊기</button></div>
""" + E + "\n"

SCR = B + """
<script>
/* 운용 탭 안의 명판 촬영 카드. 화면이 열리면 촬영 도구(iframe 의 data-port)를 한 번 불러오고, 도구가 보내는 상태로 상태 표시·카드 높이·알림을 바꾼다.
   기존 화면의 변수는 쓰지 않는다(알림 띠 toast 만 있으면 빌려 쓴다). 여기서 오류가 나도 위 스크립트에는 영향이 없다. */
(function(){
  var fr=document.getElementById("plate_frame"); if(!fr) return;
  var port=fr.getAttribute("data-port")||"8791", origin=location.protocol+"//"+location.hostname+":"+port;
  var stat=document.getElementById("plate_stat"), idle=document.getElementById("plate_idle"), live=document.getElementById("run_live"), app=document.getElementById("app");
  var loaded=false, last=0, since=0, ptOn=false;
  function vis(el){ return !!el&&!el.classList.contains("hide"); }
  function set(txt,cls){ if(stat){ stat.textContent=txt; stat.className="pill "+cls; } }
  function load(force){ if(loaded&&!force) return; loaded=true; last=0; since=Date.now(); set("촬영 화면 불러오는 중","p-neu"); fr.src=origin+"/?embed=1&t="+Date.now(); }
  var rl=document.getElementById("plate_reload"); if(rl) rl.onclick=function(){ load(true); };
  var off=document.getElementById("plate_idle_off"); if(off) off.onclick=function(){ try{ fr.contentWindow.postMessage({src:"live2_main",cmd:"disconnect"},origin); }catch(e){} };
  window.addEventListener("message",function(e){ var d=e.data; if(e.origin!==origin||!d||d.src!=="plate_site") return;
    if(d.toast){ if(typeof window.toast==="function") window.toast(String(d.toast),d.ok===false); return; }
    last=Date.now(); ptOn=!!d.pt;
    if(typeof d.h==="number"&&d.h>200) fr.style.height=Math.ceil(d.h)+"px";
    if(d.block) set(d.plate?"'팬틸트 카메라 사용'을 체크하지 않고 시작한 실험 — 팬틸트 사용 불가":"기존 자동 촬영이 팬틸트를 쓰는 실험 — 여기서는 사용 불가","p-warn");
    else if(d.phase==="stop") set("정지 지점 — 조준·촬영","p-live");
    else if(d.target) set("표지 기억됨"+(typeof d.remaining==="number"?" · 남은 거리 "+d.remaining.toFixed(1)+" m":""),"p-info");
    else if(d.pt) set("팬틸트 연결됨 · 표지 찾는 중","p-live");
    else set("표지 찾는 중 · 팬틸트 연결 전","p-neu"); });
  setInterval(function(){
    if(!loaded&&vis(app)) load(false);                               /* 로그인해 화면이 열리면 한 번 불러온다 */
    var stale=loaded&&Date.now()-Math.max(last,since)>6000;
    if(stale){ set(vis(live)?"촬영 도구를 불러오는 중 — 실험을 시작하면 함께 뜹니다":"촬영 도구 응답 없음 — 실험을 시작하면 함께 뜹니다","p-bad"); ptOn=false;
      if(vis(live)&&Date.now()-since>8000) load(true); }           /* 실험이 도는 동안에는 뜰 때까지 8초마다 다시 불러온다 */
    if(idle) idle.classList.toggle("hide",!(ptOn&&!vis(live)));     /* 실험이 끝났는데 팬틸트가 연결돼 있으면 준비 화면에 알린다 */
  },1000);
})();
/* E6 = 명판 촬영 방식 (2026-11-15). 이 화면의 기존 코드는 예전 자동 촬영 프로그램만을 E6 로 알아서, 명판 촬영 방식으로 시작한 실험에서도
   "팬틸트(E6) 꺼짐 … 움직이지 않습니다", "팬틸트 동의 없이 시작해 촬영할 수 없습니다", "로봇을 정지해 주세요", "촬영이 완료되었습니다" 를 보여 준다.
   그리기 직전에 상태 자료를 이 방식에 맞게 고쳐 넘기고, 문구를 만드는 함수 네 개의 결과를 이 방식의 말로 바꾼다. 표시만 바꾼다(서버·기록·판정은 그대로).
   이 방식인지는 콘솔이 남긴 E6 모듈 기록의 사유("명판 촬영 …")로 안다. 여기서 오류가 나면 원래 표시로 돌아간다. */
(function(){
  var ACTV=["STARTING","RUNNING","PARTIAL","PAUSED","STOPPING"];
  var WHY="SigLIP 신호 → 정지 지점까지 이동 → 아래 명판 촬영 카드에서 조준·촬영";
  var STEP={stopreq:"표지 인식·이동",stopok:"정지",aim:"조준(카드)",focus:"초점",cap:"촬영(카드)",home:"정면 복귀",resume:"이동 가능"}, step0=null;
  function isPlate(cs){ try{ var X=(cs&&cs.experiment)||{}, w=((((cs&&cs.modules)||{}).e6||{}).exp||{}).why;
    return ACTV.indexOf(X.state)>=0&&/명판 촬영/.test(Array.isArray(w)?w.join(" "):String(w||"")); }catch(e){ return false; } }
  function steps(on){ try{ if(typeof FSTEPS==="undefined") return;
    if(!step0){ step0={}; FSTEPS.forEach(function(x){ step0[x[0]]=x[1]; }); }
    FSTEPS.forEach(function(x){ x[1]=(on&&STEP[x[0]])||step0[x[0]]; }); }catch(e){} }
  function adapt(d,cs){ var on=isPlate(cs); P.on=on; steps(on);
    if(on&&d){ var L=d.live2=d.live2||{};
      L.e6_bridge={state:"PLATE_CASE1",alive:true,why:WHY,blocked:null,n_observations:((L.e8_pair||{}).observations||[]).length||null};
      L.e6_mode={mode:"명판 촬영",ko:"E6 명판 촬영 방식 — 아래 카드에서 조준·촬영"};
      if(L.auto_state) L.auto_state.needs_pantilt_consent=false; }
    return on; }
  /* 녹화 시간 알림 (2026-11-15): 실험을 시작한 뒤 5분이 지나면 알림음을 내고, 그 뒤에도 실험을 끝내지 않으면 1분마다 다시 낸다(모든 실험).
     경과 시간은 서버 시각 - 실험 시작 시각. 소리는 이 화면의 알림음(🔊 단추로 켠 것)을 쓰고, 화면 위쪽에 알림 띠도 띄운다. */
  var REC={id:null,k:-1,LIMIT_S:300,EVERY_S:60,n:0};
  function rec(d,cs){ var X=(cs&&cs.experiment)||{};
    if(["RUNNING","PARTIAL","PAUSED"].indexOf(X.state)<0||!X.created_wall){ REC.id=null; REC.k=-1; return false; }
    if(REC.id!==X.exp_id){ REC.id=X.exp_id; REC.k=-1; }
    var now=((d||{}).meta||{}).server_wall||Date.now()/1000, el=now-X.created_wall;
    if(!(el>=REC.LIMIT_S)) return false;
    var k=Math.floor((el-REC.LIMIT_S)/REC.EVERY_S);
    if(k===REC.k) return false;
    REC.k=k; REC.n++;
    try{ if(typeof tone==="function") tone([[880,0.18],[660,0.18],[880,0.18],[660,0.32]]); }catch(e){}
    try{ if(navigator.vibrate) navigator.vibrate([300,150,300]); }catch(e){}
    try{ if(typeof window.toast==="function") window.toast("녹화 "+Math.floor(el/60)+"분 경과 — 실험을 종료하세요",true); }catch(e){}
    return true; }
  var P=window.__plateE6={on:false,adapt:adapt,rec:rec,REC:REC};
  var _render=window.render, _e6Row=window.e6Row, _sem=window.semState, _flow=window.flowView;
  if(typeof _render==="function") window.render=function(d,cs){ try{ adapt(d,cs); }catch(e){ P.on=false; } try{ rec(d,cs); }catch(e){} return _render.apply(this,arguments); };
  if(typeof _e6Row==="function") window.e6Row=function(d){
    try{ var B=((d&&d.live2)||{}).e6_bridge||{};
      if(B.state==="PLATE_CASE1") return "명판 촬영 방식 동작 중"+(B.n_observations?" · 넘긴 사진 "+B.n_observations+"장":"")+' <span class="small mut">'+WHY+"</span>"; }catch(e){}
    return _e6Row.apply(this,arguments); };
  if(typeof _sem==="function") window.semState=function(d){ var r=_sem.apply(this,arguments);
    try{ if(P.on&&r&&r[0]==="stop") return ["stop","표지판 후보가 발견되었습니다. 아래 명판 촬영 카드의 안내대로 이동하세요.",
      String(r[2]||"").replace(" — 로봇은 조이스틱으로 정지","")+" — 정지 지점(표지 약 1 m 앞)에서 멈춘 뒤 카드에서 [조준 + 촬영]"]; }catch(e){}
    return r; };
  if(typeof _flow==="function") window.flowView=function(F){ var r=_flow.apply(this,arguments);
    try{ if(P.on&&F&&r){ var id=String(F.observation_id||"");
      if(F.state==="STOP_CONFIRMED"||F.state==="E6_OFF") return ["info","정지 확인됨 — 명판 촬영 카드의 안내를 따르세요 (정지 지점이면 [조준 + 촬영])",id+" · E6 명판 촬영 방식"];
      if(F.state==="RESUME_READY") return ["info","이동할 수 있습니다 — 조준·촬영은 아래 명판 촬영 카드에서 합니다",id+" · E6 명판 촬영 방식"]; } }catch(e){}
    return r; };
})();
</script>
""" + E + "\n"

STRIP = re.compile(re.escape(B) + r".*?" + re.escape(E) + r"\n?", re.S)


def strip(s):
    """끼워 넣은 덩어리를 모두 뺀다(예전의 탭 방식 덩어리 포함). 줄 단위 덩어리는 넣을 때 붙인 줄바꿈 하나까지 뺀다;
    예전 탭 단추 덩어리는 같은 줄의 다음 단추 앞에 있어 뒤에 줄바꿈이 없으므로 그대로 빠진다."""
    return STRIP.sub("", s)


def apply(s):
    s = strip(s)
    for name, a in (("실험 화면의 카드 자리", A_PANEL), ("준비 화면의 띠 자리", A_IDLE), ("스크립트 자리", A_SCR)):
        if s.count(a) != 1:
            raise SystemExit("중단: %s 를 찾을 수 없거나 여러 곳입니다 (%d곳) — 화면 파일이 예상과 다릅니다" % (name, s.count(a)))
    for must in ('<div id="run_live" class="hide">', 'id="app"', 'class="pill', ".spacer{", ".hide{", "function toast("):
        if must not in s:
            raise SystemExit("중단: 화면 파일에 %r 가 없습니다 — 화면 구조가 예상과 다릅니다" % must)
    if not (s.index('<div id="run_live" class="hide">') < s.index(A_PANEL) and s.index(A_IDLE) < s.index('<div id="run_live" class="hide">')):
        raise SystemExit("중단: 카드·띠 자리의 순서가 예상과 다릅니다")
    for clash in ('id="plate_', 'data-t="plate"'):
        if clash in s:
            raise SystemExit("중단: 화면 파일에 이미 %r 가 있습니다" % clash)
    s = s.replace(A_PANEL, A_PANEL + PANEL)
    s = s.replace(A_IDLE, A_IDLE + IDLE)
    s = s.replace(A_SCR, SCR + A_SCR)
    return s


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out"); ap.add_argument("--check", action="store_true"); ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    raw = open(a.src, "rb").read(); s = raw.decode("utf-8")
    new = strip(s) if a.revert else apply(s)
    nb = new.encode("utf-8")
    assert strip(new) == strip(s), "빼낸 결과가 원래 내용과 다릅니다"
    print("입력  %s  sha256 %s  %d바이트%s" % (a.src, sha(raw)[:16], len(raw), "  (이미 끼워져 있음)" if B in s else ""))
    print("결과  sha256 %s  %d바이트  (%+d)" % (sha(nb)[:16], len(nb), len(nb) - len(raw)))
    if a.check:
        print("점검만 했습니다 — 아무것도 쓰지 않았습니다"); return
    if not a.out:
        raise SystemExit("--out 이 필요합니다 (입력 파일은 직접 덮어쓰지 않습니다)")
    import os
    if os.path.realpath(a.out) == os.path.realpath(a.src):
        raise SystemExit("--out 이 입력 파일과 같습니다")
    open(a.out, "wb").write(nb); print("씀   ", a.out)


if __name__ == "__main__":
    main()
