#!/usr/bin/env python3
"""test_merge.py — 운용 탭 안에 명판 촬영 카드를 끼운 본 홈페이지를 노트북에서 확인한다 (운영 콘솔·로봇을 쓰지 않는다).

가짜 콘솔(이 스크립트, 18770)이 화면 파일과 최소한의 /api 응답을 주고, 머리 없는 Chrome 을 직접 조작해 탭을 눌러 본다.
촬영 사이트는 시뮬 모드(plate_site/run_sim.sh, 8792)가 떠 있어야 한다.
  python3 test_merge.py <운영판 원본 index.html> [--sim-port 8792] [--shot <png>]"""
import argparse, json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import patch_plate_tab as P
import cdp_min as CDP

ap = argparse.ArgumentParser(); ap.add_argument("orig"); ap.add_argument("--sim-port", type=int, default=8792); ap.add_argument("--port", type=int, default=18770); ap.add_argument("--shot")
a = ap.parse_args()
ORIG = open(a.orig, encoding="utf-8").read(); MERGED = P.apply(ORIG)
PAGES = {"/orig": ORIG, "/merged": MERGED.replace('data-port="8791"', 'data-port="%d"' % a.sim_port), "/down": MERGED.replace('data-port="8791"', 'data-port="8799"')}


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _s(self, code, body, ct="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in PAGES: return self._s(200, PAGES[p], "text/html; charset=utf-8")
        if p == "/api/session": return self._s(200, json.dumps(dict(logged_in=True, csrf="x", user="recs", mode="user", control_network_ok=True, remember=False, kind="password")))
        if p.startswith("/api/"): return self._s(200, "{}")
        return self._s(404, "{}")
    def do_POST(self): self._s(200, "{}")


srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE, SIM = "http://127.0.0.1:%d" % a.port, "http://127.0.0.1:%d" % a.sim_port
ok = True
def check(name, cond, info=""):
    global ok; ok = ok and bool(cond); print(("  통과  " if cond else "  실패  ") + name + ("" if cond else "  " + str(info)[:300]))
def sim_state(): return json.load(urllib.request.urlopen(SIM + "/api/state", timeout=3))
def sim_post(path, body): urllib.request.urlopen(urllib.request.Request(SIM + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3).read()
try: sim_state()
except Exception as e: sys.exit("시뮬 사이트(%s)가 떠 있지 않습니다: %s" % (SIM, e))
TABS = "[...document.querySelectorAll('#tabs button')].map(b=>b.dataset.t)"
ON = "[...document.querySelectorAll('.tab.on')].map(s=>s.id).join(',')"
def each(tab, names):
    out = {}
    for t in names:
        tab.js("document.querySelector('#tabs button[data-t=\"%s\"]').click()" % t); time.sleep(0.2); out[t] = tab.js(ON)
    return out
def wait_for(fn, sec):
    t0 = time.time(); v = None
    while time.time() - t0 < sec:
        v = fn()
        if v: return v
        time.sleep(0.3)
    return v

C = CDP.Chrome()
J = lambda i: "document.getElementById('%s')" % i
try:
    T = C.page()
    print("[1] 원본 화면 (기준)")
    T.go(BASE + "/orig", 2.0); o_tabs = T.js(TABS); o_each = each(T, o_tabs); o_err = list(T.errors)
    check("원본이 가짜 콘솔에서 열리고 탭 5개", o_tabs == ["run", "cand", "sem", "eval", "rec"] and T.js("typeof goTab") == "function", o_tabs)

    print("[2] 운용 탭 안에 끼운 화면")
    sim_post("/api/case", {"case": 2})
    T.go(BASE + "/merged", 2.0)
    check("탭은 그대로 5개 — 따로 만든 탭 없음", T.js(TABS) == o_tabs and not T.js("!!%s" % J("t_plate")), T.js(TABS))
    check("명판 촬영 카드가 운용 탭의 실험 화면 안, 안내 띠 바로 아래에 있음",
          T.js("%s.parentElement.id" % J("plate_panel")) == "run_live" and T.js("%s.previousElementSibling.id" % J("plate_panel")) == "settle_panel"
          and T.js("!!%s.closest('#t_run')" % J("plate_panel")))
    src = wait_for(lambda: T.js("%s.getAttribute('src')" % J("plate_frame")), 6)
    check("화면이 열리면 촬영 도구를 불러옴 (같은 호스트의 도구 포트 + embed=1)", str(src or "").startswith(SIM + "/?embed=1"), src)
    stat = wait_for(lambda: (lambda s: s if ("표지" in s or "정지" in s or "팬틸트" in s) else None)(T.js("%s.textContent" % J("plate_stat"))), 8)
    check("촬영 도구가 보낸 상태가 카드에 표시됨", stat, T.js("%s.textContent" % J("plate_stat")))
    print("        상태 표시:", stat)
    check("준비 화면(실험 없음)에서는 카드가 보이지 않음", T.js("%s.getBoundingClientRect().height" % J("plate_panel")) == 0)
    T.js("(function(){var s=document.createElement('style');s.id='t_force';s.textContent='#run_live.hide{display:block!important}#run_idle{display:none!important}';document.head.appendChild(s)})()")
    time.sleep(1.5)                                             # 실험이 도는 화면처럼 보이게 해서(시험 전용) 카드 안을 본다
    hpx = T.js("%s.getBoundingClientRect().height" % J("plate_frame"))
    check("카드 높이가 내용에 맞춰짐 (안에서 따로 스크롤되지 않음)", hpx > 300 and T.js("%s.style.height" % J("plate_frame")) != "640px" and T.js("%s.getAttribute('scrolling')" % J("plate_frame")) == "no", hpx)
    F = T.frame(SIM + "/?embed=1")
    check("카드 안의 촬영 화면을 들여다봄", F is not None)
    if F:
        check("끼움 모드(html.embed)", "embed" in (F("document.documentElement.className") or ""))
        check("내용이 잘리지 않음 (촬영 화면 높이 ≤ 카드 높이)", F("Math.ceil(document.body.getBoundingClientRect().height)") <= hpx + 2, (F("Math.ceil(document.body.getBoundingClientRect().height)"), hpx))
        check("케이스 선택 단추가 숨겨짐", F("getComputedStyle(document.getElementById('case1').parentElement).display") == "none")
        check("'몸 들림'(케이스 2 용) 칸이 숨겨짐", F("getComputedStyle(document.getElementById('n_pitch').parentElement).display") == "none")
        check("정지 지점 거리 선택(케이스 1 용)은 보임", F("getComputedStyle(document.getElementById('aheadrow')).display") != "none")
        check("촬영 화면이 상태를 받고 있음", "연결 끊김" not in F("document.getElementById('c_exp').textContent"), F("document.getElementById('c_exp').textContent"))
        check("안내 문구가 나옴", (F("document.getElementById('g_title').textContent") or "–") != "–", F("document.getElementById('g_title').textContent"))
        check("어두운 카드 색", F("getComputedStyle(document.querySelector('.card')).backgroundColor") == "rgb(31, 41, 55)", F("getComputedStyle(document.querySelector('.card')).backgroundColor"))
        check("[조준 + 촬영]·[팬틸트 연결]·[소리 켜기] 단추가 있음", F("['aim','pt_conn','snd'].every(i=>!!document.getElementById(i))"))
        F("document.getElementById('snd').click()"); time.sleep(0.4)
        check("[소리 켜기]를 누르면 켜짐 표시", F("document.getElementById('snd').textContent") == "소리 켜짐", F("document.getElementById('snd').textContent"))
        T.js("%s.textContent=''" % J("toast"))                     # 고정 카메라 화면을 눌러 본다(표지 기억 시도) -> 결과 문구가 본 화면의 알림 띠로 와야 한다
        F("(function(){const c=document.getElementById('ov'),r=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('click',{clientX:r.left+r.width/2,clientY:r.top+r.height/2,bubbles:true}));})()")
        msg = wait_for(lambda: T.js("%s.textContent" % J("toast")), 5)
        check("촬영 화면의 알림이 본 화면의 알림 띠로 나옴", bool(msg), msg)
        print("        알림:", msg)
        sim_post("/api/forget", {}); time.sleep(0.5)
    c = wait_for(lambda: sim_state()["case"] == 1, 4)
    check("케이스 2 상태였어도 케이스 1 로 되돌림", c, sim_state()["case"])
    if a.shot:
        time.sleep(1.0); T.shot(a.shot); print("        그림:", a.shot)
    T.js("(function(){var s=document.getElementById('t_force'); if(s) s.remove();})()")
    m_each = each(T, o_tabs)
    check("기존 탭 5개가 원본과 같게 열림", m_each == o_each, (m_each, o_each))
    msg = lambda e: str(e).split("\n")[0]                      # 줄 번호·주소는 두 화면이 다르므로 오류 문구만 비교
    extra = sorted({msg(e) for e in T.errors} - {msg(e) for e in o_err})
    check("원본에 없던 스크립트 오류 없음", not extra, extra)
    print("        (가짜 응답 때문에 원본에서도 나는 오류 %d개, 끼운 화면 %d개)" % (len(o_err), len(T.errors)))

    print("[3] 실험이 끝났는데 팬틸트가 연결돼 있을 때 (준비 화면의 띠)")
    T.js("document.querySelector('#tabs button[data-t=\"run\"]').click()")
    check("팬틸트가 연결돼 있지 않으면 띠가 없음", T.js("%s.classList.contains('hide')" % J("plate_idle")))
    F3 = T.frame(SIM + "/?embed=1")                             # 카드 안의 [팬틸트 연결]: 확인 창 대신 두 번 누르기
    F3("document.getElementById('pt_conn').click()"); time.sleep(1.5)
    check("카드 안 [팬틸트 연결] 한 번만 누르면 연결하지 않고 다시 묻는다", not sim_state()["pt"].get("connected") and not sim_state()["pt"].get("starting")
          and "한 번 더" in F3("document.getElementById('pt_conn').textContent"), F3("document.getElementById('pt_conn').textContent"))
    F3("document.getElementById('pt_conn').click()")
    on = wait_for(lambda: sim_state()["pt"].get("connected"), 15)
    check("6초 안에 한 번 더 누르면 연결된다", on, sim_state()["pt"])
    shown = wait_for(lambda: not T.js("%s.classList.contains('hide')" % J("plate_idle")), 6)
    check("팬틸트가 연결돼 있으면 준비 화면 맨 위에 띠가 뜸", on and shown and T.js("%s.getBoundingClientRect().height" % J("plate_idle")) > 0, (on, shown))
    T.js("%s.click()" % J("plate_idle_off"))
    offd = wait_for(lambda: not sim_state()["pt"].get("connected"), 20)
    check("[정면으로 돌리고 연결 끊기]를 누르면 연결이 끊김", offd, sim_state()["pt"])
    gone = wait_for(lambda: T.js("%s.classList.contains('hide')" % J("plate_idle")), 6)
    check("끊기면 띠가 사라짐", gone)

    print("[4] 촬영 도구가 꺼져 있을 때")
    T.go(BASE + "/down", 2.0)
    d = wait_for(lambda: (lambda s: s if "응답 없음" in s else None)(T.js("%s.textContent" % J("plate_stat"))), 12)
    check("응답 없음 표시", d and "p-bad" in T.js("%s.className" % J("plate_stat")), T.js("%s.textContent" % J("plate_stat")))
    src1 = T.js("%s.getAttribute('src')" % J("plate_frame"))
    def retried():
        T.js("%s.classList.remove('hide')" % J("run_live"))                  # 실험이 도는 화면인 것처럼 (시험 전용)
        return T.js("%s.getAttribute('src')" % J("plate_frame")) != src1
    check("실험이 도는 동안에는 촬영 도구가 뜰 때까지 다시 불러옴", wait_for(retried, 14))
    T.js("%s.classList.add('hide')" % J("run_live"))
    check("그래도 탭 5개는 원본과 같게 열림", each(T, o_tabs) == o_each)

    print("[5] 지난 실험 재생으로 표지 기억 → 카드의 상태 표시")
    ds = [d["id"] for d in sim_state()["sim"]["datasets"]]
    pick = next((d for d in ds if d.endswith("103935")), ds[0])
    sim_post("/api/sim", {"op": "dataset", "id": pick}); sim_post("/api/sim", {"op": "next"}); sim_post("/api/sim", {"op": "speed", "v": 2}); sim_post("/api/sim", {"op": "play"})
    T.go(BASE + "/merged", 2.0)
    st5 = wait_for(lambda: (lambda s: s if ("표지 기억됨" in s or "정지 지점" in s) else None)(T.js("%s.textContent" % J("plate_stat"))), 25)
    check("표지가 기억되면 상태 표시가 알림 (%s)" % pick[-6:], st5, T.js("%s.textContent" % J("plate_stat")))
    print("        상태 표시:", st5)
    sim_post("/api/sim", {"op": "pause"})
    if a.shot:
        T.js("(function(){var s=document.createElement('style');s.textContent='#run_live.hide{display:block!important}#run_idle{display:none!important}';document.head.appendChild(s)})()")
        time.sleep(1.5); T.js("%s.scrollIntoView()" % J("plate_panel")); time.sleep(0.5)
        T.shot(a.shot.replace(".png", "_target.png")); print("        그림:", a.shot.replace(".png", "_target.png"))
    sim_post("/api/forget", {})

    print("[6] 촬영 도구를 직접 열 때 (끼움 모드 아님)")
    T.go(SIM + "/", 2.0)
    check("예전 화면 그대로: 케이스 선택 단추가 보임", "embed" not in (T.js("document.documentElement.className") or "") and T.js("getComputedStyle(document.getElementById('case1').parentElement).display") != "none")
finally:
    C.close()
print("모두 통과" if ok else "실패 있음"); sys.exit(0 if ok else 1)
