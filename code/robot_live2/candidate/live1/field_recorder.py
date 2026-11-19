#!/usr/bin/env python3
"""LIVE1 field evaluation recorder (FIELD_DEVELOPMENT_EVALUATION). FILE-ONLY: no ROS, no model, no device.

Runs beside the live components and never touches their inference path:
  - every 0.2 s reads lio_state.json (FAST-LIO odom, raw frame) -> appends NEW odom samples to odom_trace.jsonl
    (stamp, x, y, yaw, session) so motion (moving / stationary) can be derived per scene afterwards
  - every 0.2 s reads the E5 status.json; when the scene counter changes appends a compact scene_index.jsonl record
    (wall, scene t/stamp, n_scene, thr, would_stop counts) and copies frame_inference.jpg (annotated view) once per scene
The authoritative per-region data and the RAW scene images are written by e5_node itself (scenes.jsonl + --save-frames
events/scene_*.jpg in the run dir); this recorder only adds the motion trace and the operator-visible annotated view.

Optional mark server (--port, default 8771): a SEPARATE evaluation page (not the operations panel). POST /mark only
appends a human note (e.g. SIGN_VISIBLE_NOW) with the wall time and the latest scene id to marks.jsonl. It sends nothing
to the robot and changes no component.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live1_common as C  # noqa: E402

MARK_KINDS = ("SIGN_VISIBLE_NOW", "DETECTED_CORRECT_NOW", "FALSE_ALARM_NOW", "POSTER_OR_NONLOCAL_NOW", "ROBOT_STOPPED", "ROBOT_MOVING", "NOTE")

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIVE1 평가 마킹</title><style>body{margin:0;background:#15181d;color:#e8eaee;font:15px system-ui,sans-serif;padding:12px}
h1{font-size:17px}button{font-size:17px;padding:14px 10px;margin:4px;border-radius:8px;border:0;min-width:46%;cursor:pointer}
.g{background:#2fb36d}.r{background:#e05252;color:#fff}.y{background:#e0a526}.b{background:#4a78d0;color:#fff}.n{background:#555;color:#fff}
#log{font:12px monospace;white-space:pre-wrap;color:#aab}.note{color:#f0c060;font-size:13px}img{max-width:100%;border-radius:6px}</style></head><body>
<h1>FIELD_DEVELOPMENT_EVALUATION · 평가 마킹 (운용 웹과 분리)</h1>
<div class="note">이 페이지는 기록만 남깁니다. 로봇·FAST-LIO·SigLIP에 아무 명령도 보내지 않습니다. 정답 판정(TP/FP/MISSED)은 주행 후 검토 화면에서 사람이 확정합니다.</div>
<img id="im" src="/img/inference.jpg"><div id="sc"></div>
<button class="b" onclick="mk('SIGN_VISIBLE_NOW')">실제 표지판 지금 보임</button><button class="g" onclick="mk('DETECTED_CORRECT_NOW')">맞게 검출됨 (현재)</button>
<button class="r" onclick="mk('FALSE_ALARM_NOW')">오검출 (현재)</button><button class="y" onclick="mk('POSTER_OR_NONLOCAL_NOW')">포스터·비위치 문자</button>
<button class="n" onclick="mk('ROBOT_STOPPED')">로봇 정지함</button><button class="n" onclick="mk('ROBOT_MOVING')">로봇 이동 시작</button>
<div><input id="nt" placeholder="메모 (호실명 등)" style="width:70%;font-size:15px;padding:8px"><button class="n" style="min-width:20%" onclick="mk('NOTE')">메모</button></div>
<div id="log"></div>
<script>
async function mk(k){const r=await fetch('/mark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k,note:document.getElementById('nt').value,client_wall:Date.now()/1000})});
const j=await r.json();document.getElementById('log').textContent=new Date().toLocaleTimeString()+' 기록됨 '+k+' scene '+j.scene_n+'\\n'+document.getElementById('log').textContent;document.getElementById('nt').value='';}
setInterval(async()=>{document.getElementById('im').src='/img/inference.jpg?t='+Date.now();try{const s=await (await fetch('/state')).json();
document.getElementById('sc').textContent=`scene ${s.scene_n} · 원본 age ${s.source_age_s} s · 이동 ${s.speed_mps} m/s · 마크 ${s.marks}`;}catch(e){}},1000);
</script></body></html>"""


class Recorder:
    def __init__(self, state, shm, out):
        self.state, self.shm, self.out = Path(state), Path(shm), Path(out)
        (self.out / "inference_views").mkdir(parents=True, exist_ok=True)
        self.last_odom_stamp = None
        self.last_scene_n = None
        self.lock = threading.Lock()
        self.latest = dict(scene_n=None, stamp=None, speed_mps=None, marks=0)
        self.recent = []                                   # (stamp, x, y)

    def tick(self):
        d = C.read_json(self.state / "lio_state.json")
        if d:
            o = (d.get("odom") or {})
            lr = o.get("last_raw")
            if lr and lr["stamp"] != self.last_odom_stamp:
                self.last_odom_stamp = lr["stamp"]
                rec = dict(wall=time.time(), stamp=lr["stamp"], x=lr["x"], y=lr["y"], yaw_deg=lr["yaw_deg"], session=o.get("session"),
                           frame="RAW_FASTLIO_camera_init")
                with open(self.out / "odom_trace.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                self.recent = [r for r in self.recent if lr["stamp"] - r[0] <= 1.0] + [(lr["stamp"], lr["x"], lr["y"])]
                if len(self.recent) >= 2 and self.recent[-1][0] > self.recent[0][0]:
                    a, b = self.recent[0], self.recent[-1]
                    self.latest["speed_mps"] = round(((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 / (b[0] - a[0]), 3)
        s = C.read_json(self.shm / "status.json")
        if s:
            sc = s.get("scene") or {}
            n = sc.get("n_scene")
            if n is not None and n != self.last_scene_n:
                self.last_scene_n = n
                ws = (s.get("sem1_would_stop_private") or {}).get("counts")
                rec = dict(wall=time.time(), run_id=s.get("run_id"), n_scene=n, t=sc.get("t"), stamp=sc.get("stamp"), thr=sc.get("thr"),
                           n_regions=len(sc.get("regions") or []), would_stop_counts=ws, speed_mps_odom=self.latest["speed_mps"],
                           odom_stamp=self.last_odom_stamp, pose_mode=(s.get("source_health") or {}).get("pose_mode"))
                src = self.shm / "frame_inference.jpg"
                if src.exists():
                    dst = self.out / "inference_views" / f"scene_{n:05d}.jpg"
                    shutil.copyfile(src, dst)
                    rec["inference_view"] = str(dst.relative_to(self.out))
                with open(self.out / "scene_index.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                with self.lock:
                    self.latest.update(scene_n=n, stamp=sc.get("stamp"), run_id=s.get("run_id"))

    def mark(self, body):
        k = body.get("kind")
        if k not in MARK_KINDS:
            return None
        with self.lock:
            rec = dict(wall=time.time(), client_wall=body.get("client_wall"), kind=k, note=str(body.get("note") or "")[:300],
                       scene_n=self.latest.get("scene_n"), scene_stamp=self.latest.get("stamp"), run_id=self.latest.get("run_id"),
                       odom_stamp=self.last_odom_stamp, speed_mps=self.latest.get("speed_mps"),
                       evaluation="FIELD_DEVELOPMENT_EVALUATION", human_note=True)
            with open(self.out / "marks.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.latest["marks"] += 1
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(C.STATE))
    ap.add_argument("--e5-shm", default=str(C.RUNTIME / "e5_shm"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-s", type=float, default=3600)
    a = ap.parse_args()
    R = Recorder(a.state, a.e5_shm, a.out)
    stop = [False]

    def loop():
        t0 = time.monotonic()
        while not stop[0] and time.monotonic() - t0 < a.max_s:
            try:
                R.tick()
            except Exception as e:                     # never die on a half-written file
                print("tick", type(e).__name__, e, flush=True)
            time.sleep(0.2)
        stop[0] = True
    th = threading.Thread(target=loop, daemon=True)
    th.start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ct):
            self.send_response(code)
            self.send_header("Content-Type", ct)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "/state":
                s = dict(R.latest)
                st = s.get("stamp")
                s["source_age_s"] = None if st is None else round(time.time() - float(st), 2)
                return self._send(200, json.dumps(s).encode(), "application/json")
            if p == "/img/inference.jpg":
                try:
                    return self._send(200, (Path(a.e5_shm) / "frame_inference.jpg").read_bytes(), "image/jpeg")
                except OSError:
                    return self._send(404, b"DATA_UNAVAILABLE", "text/plain")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/mark":
                return self._send(405, b"evaluation page: only /mark", "text/plain")
            n = min(int(self.headers.get("Content-Length") or 0), 4096)
            try:
                rec = R.mark(json.loads(self.rfile.read(n) or b"{}"))
            except ValueError:
                rec = None
            if rec is None:
                return self._send(400, b"bad mark", "text/plain")
            return self._send(200, json.dumps(rec, ensure_ascii=False).encode(), "application/json")

    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.timeout = 0.5
    print(f"LIVE1_FIELD_RECORDER out={a.out} mark_page={a.host}:{a.port}", flush=True)
    try:
        while not stop[0]:
            srv.handle_request()
    except KeyboardInterrupt:
        pass
    stop[0] = True
    srv.server_close()


if __name__ == "__main__":
    main()
