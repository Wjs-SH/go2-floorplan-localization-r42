#!/usr/bin/env python3
"""test_e6_plate_page.py — 본 홈페이지가 명판 촬영 방식의 실험에서 E6 를 "명판 촬영 방식"으로 보여 주는지 노트북에서 확인한다.
운영 콘솔·로봇을 쓰지 않는다. 가짜 콘솔(이 스크립트, 18771)이 화면 파일만 주고, 머리 없는 Chrome 에서 화면의 함수에 상태 자료를 직접 넣어 본다.
  python3 test_e6_plate_page.py <index.html (운영판 사본 또는 원본)>"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import patch_plate_tab as P
import cdp_min as CDP

SRC = open(sys.argv[1], encoding="utf-8").read()
PAGES = {"/merged": P.apply(SRC).replace('data-port="8791"', 'data-port="8799"'), "/plain": P.strip(SRC)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _s(self, code, body, ct="application/json"):
        b = body.encode(); self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in PAGES: return self._s(200, PAGES[p], "text/html; charset=utf-8")
        if p == "/api/session": return self._s(200, json.dumps(dict(logged_in=True, csrf="x", user="recs", mode="user", control_network_ok=True, remember=False, kind="password")))
        return self._s(200 if p.startswith("/api/") else 404, "{}")
    def do_POST(self): self._s(200, "{}")


srv = ThreadingHTTPServer(("127.0.0.1", 18771), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:18771"
N = dict(ok=0, bad=0)
def check(name, cond, info=""):
    N["ok" if cond else "bad"] += 1; print(("  통과  " if cond else "  실패  ") + name + ("" if cond else "  " + str(info)[:500]))

PLATE_WHY = "명판 촬영(케이스 1) 방식 — E6 자동 촬영 꺼짐, 팬틸트는 명판 촬영 탭에서 조준·촬영"
NOAPPR_WHY = "승인 B 없음(NO_APPROVAL_RECORD) — E6 팬틸트·초점·촬영 꺼짐, 정지는 관측 전용"
PROBE = """(function(state, why){
  const cs={experiment:{state:state,exp_id:"X_T"},modules:{e6:{alive:false,exp:{state:"OFF",why:[why]}}}};
  const d={live2:{e6_bridge:{state:"OFF",alive:false},auto_state:{needs_pantilt_consent:true},e6_mode:{mode:"E2_E5_OBSERVATIONAL",ko:"관측 전용"},e8_pair:{observations:[{},{}]}},
           e5:{stop_request_ui:{active:true,items:[{sim:0.12,depth_m:3.2}]}}};
  const has=!!window.__plateE6, on=has?window.__plateE6.adapt(d,cs):null, B=d.live2.e6_bridge;
  return JSON.stringify({has:has,on:on,b:B,mode:d.live2.e6_mode,npc:d.live2.auto_state.needs_pantilt_consent,row:e6Row(d),sem:semState(d),
    warn:!B.alive&&["OFF","REFUSED",undefined,null].includes(B.state),
    f_resume:flowView({state:"RESUME_READY",observation_id:"O1",why:"observe-only mode (E6 off): you may drive on"}),
    f_off:flowView({state:"E6_OFF",observation_id:"O1",why:"x"}), f_conf:flowView({state:"STOP_CONFIRMED",observation_id:"O1",why:"x"}),
    f_hold:flowView({state:"HOLD_ERROR",observation_id:"O1",why:"odometry not OK"}),
    steps:FSTEPS.map(function(x){return x[1];})});
})(%s, %s)"""
def probe(T, state, why):
    return json.loads(T.js(PROBE % (json.dumps(state), json.dumps(why))))

C = CDP.Chrome()
try:
    T = C.page()
    print("[1] 끼우지 않은 화면 (기준)")
    T.go(BASE + "/plain", 2.5)
    base = probe(T, "RUNNING", PLATE_WHY)
    check("기준 화면에는 맞춤 코드가 없고 예전 표시가 나온다", base["has"] is False and "꺼짐" in base["row"] and "정지해 주세요" in base["sem"][1] and "촬영이 완료" in base["f_resume"][1] and base["warn"] is True, base)
    steps0 = base["steps"]
    time.sleep(2.5)
    kinds = lambda errs: sorted(set((e.split("\n")[0] + " @ " + (e.split("\n")[1].strip().split(" ")[1] if "\n" in e else "")) for e in errs))
    plain_err = kinds(T.errors)          # 가짜 콘솔은 빈 자료를 주므로 원래 화면도 준비 화면·이력에서 오류를 낸다 -> 이것을 기준으로 삼는다

    print("[2] 끼운 화면 — 명판 촬영 방식으로 시작한 실험")
    T.go(BASE + "/merged", 2.5)
    check("맞춤 코드가 화면에 올라옴", T.js("typeof window.__plateE6.adapt") == "function")
    r = probe(T, "RUNNING", PLATE_WHY)
    check("명판 촬영 방식으로 인식", r["on"] is True)
    check("E6 줄: '명판 촬영 방식 동작 중' + 넘긴 사진 수", "명판 촬영 방식 동작 중" in r["row"] and "넘긴 사진 2장" in r["row"] and "꺼짐" not in r["row"], r["row"])
    check("'팬틸트(E6) 꺼짐 … 움직이지 않습니다' 경고가 나오지 않는 조건", r["warn"] is False and r["b"]["alive"] is True and r["b"]["state"] == "PLATE_CASE1", r["b"])
    check("'팬틸트 동의 없이 시작해 촬영할 수 없습니다' 표시가 꺼짐", r["npc"] is False)
    check("E6 방식 표시가 '명판 촬영'", r["mode"]["mode"] == "명판 촬영" and "명판 촬영 방식" in r["mode"]["ko"], r["mode"])
    check("SigLIP 신호 때: '정지해 주세요' 대신 카드 안내대로 이동", r["sem"][0] == "stop" and "카드의 안내대로 이동" in r["sem"][1] and "정지 지점" in r["sem"][2] and "조이스틱으로 정지" not in r["sem"][2]
          and "유사도 0.120" in r["sem"][2], r["sem"])
    check("정지 뒤 문구: '촬영이 완료되었습니다' 대신 카드에서 조준·촬영", "촬영이 완료" not in r["f_resume"][1] and "명판 촬영 카드" in r["f_resume"][1] and "O1" in r["f_resume"][2], r["f_resume"])
    check("'E6 하드웨어 승인 없음 — 구동 차단' 대신 카드 안내", "승인 없음" not in r["f_off"][1] and "명판 촬영 카드" in r["f_off"][1] and "명판 촬영 카드" in r["f_conf"][1], (r["f_off"], r["f_conf"]))
    check("오류 유지 문구(위치 끊김 등)는 그대로", r["f_hold"] == base["f_hold"], r["f_hold"])
    check("진행 단계 이름이 이 방식의 말로 바뀜", r["steps"] != steps0 and "조준(카드)" in r["steps"] and "촬영(카드)" in r["steps"] and "자동 조준" not in r["steps"] and len(r["steps"]) == len(steps0), r["steps"])

    print("[3] 끼운 화면 — 이 방식이 아닌 경우는 예전 표시 그대로")
    r = probe(T, "RUNNING", NOAPPR_WHY)
    check("동의 없이 시작한 실험: 예전 표시(꺼짐·관측 전용) 그대로", r["on"] is False and r["row"] == base["row"] and r["sem"] == base["sem"] and r["f_resume"] == base["f_resume"] and r["f_off"] == base["f_off"]
          and r["npc"] is True and r["warn"] is True and r["mode"]["mode"] == "E2_E5_OBSERVATIONAL", r)
    check("진행 단계 이름도 원래대로 돌아옴", r["steps"] == steps0, r["steps"])
    r = probe(T, "ENDED", PLATE_WHY)
    check("실험이 끝난 뒤에는 적용하지 않음", r["on"] is False and r["row"] == base["row"])
    r = probe(T, "STARTING", PLATE_WHY)
    check("시작 중에도 적용(카드가 뜨기 전부터)", r["on"] is True)
    time.sleep(2.5)
    check("1초마다 도는 화면 갱신에서 새로 생긴 오류 없음(원래 화면과 같은 오류만)", kinds(T.errors) == plain_err, (kinds(T.errors), plain_err))

    print("[4] 녹화 시간 알림: 5분에 한 번, 그 뒤 1분마다")
    T.js("window.__t=[]; window.__m=[]; window.tone=function(q){ window.__t.push(q.length); }; window.toast=function(m,bad){ window.__m.push(m); }; 0")
    REC = "(function(state,id,el){ return window.__plateE6.rec({meta:{server_wall:10000+el}},{experiment:{state:state,exp_id:id,created_wall:10000}}); })(%s,%s,%s)"
    seq = [("RUNNING", "X_A", 120, False), ("RUNNING", "X_A", 299.5, False), ("RUNNING", "X_A", 300.2, True), ("RUNNING", "X_A", 301, False), ("RUNNING", "X_A", 359, False),
           ("RUNNING", "X_A", 360.1, True), ("PAUSED", "X_A", 395, False), ("PAUSED", "X_A", 421, True), ("PARTIAL", "X_A", 600, True), ("PARTIAL", "X_A", 601, False),
           ("STOPPING", "X_A", 700, False), ("ENDED", "X_A", 720, False), ("STARTING", "X_B", 400, False), ("RUNNING", "X_B", 60, False), ("RUNNING", "X_B", 310, True), ("RUNNING", "X_B", 315, False)]
    # 화면이 1초마다 빈 자료로 다시 그리며 알림 상태를 초기화하므로(시험용 가짜 콘솔), 한 번의 실행 안에서 순서대로 넣는다
    got = json.loads(T.js("JSON.stringify(%s.map(function(q){ return window.__plateE6.rec({meta:{server_wall:10000+q[2]}},{experiment:{state:q[0],exp_id:q[1],created_wall:10000}}); }))"
                          % json.dumps([[s, i, e] for s, i, e, _ in seq])))
    check("5분 전에는 조용하고, 5분·6분·7분…에 한 번씩만 울림(일시정지 중에도), 끝나면 멈추고 다음 실험에서 다시 셈", got == [w for *_, w in seq], list(zip([e for _, _, e, _ in seq], got)))
    msgs = json.loads(T.js("JSON.stringify(window.__m)")); tones = json.loads(T.js("JSON.stringify(window.__t)"))
    check("울릴 때마다 알림음과 알림 띠('녹화 N분 경과 — 실험을 종료하세요')", len(tones) == 5 and msgs == ["녹화 5분 경과 — 실험을 종료하세요", "녹화 6분 경과 — 실험을 종료하세요", "녹화 7분 경과 — 실험을 종료하세요",
                                                                                 "녹화 10분 경과 — 실험을 종료하세요", "녹화 5분 경과 — 실험을 종료하세요"], (tones, msgs))
    T.js("window.tone=function(){ throw new Error('x'); }; 0")
    check("알림음 함수가 실패해도 화면 그리기를 막지 않음", T.js(REC % ('"RUNNING"', '"X_C"', "305")) is True)
    print("\n모두 통과: %d개" % N["ok"] if not N["bad"] else "\n실패 %d개 (통과 %d개)" % (N["bad"], N["ok"]))
finally:
    C.close()
sys.exit(1 if N["bad"] else 0)
