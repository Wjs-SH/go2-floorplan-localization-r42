#!/usr/bin/env python3
"""fake_console.py — 시험용 가짜 조준 콘솔 (장치 없음). 사용자 콘솔의 r1 계약 가운데 사이트가 쓰는 부분만 흉내 낸다:
/state /aim /home /request /capture /pt.mjpg. 조준 계산은 같은 보정 식(aimgeo.Geo)으로 하고, 범위를 넘으면 진짜 콘솔처럼 거부한다(strict)."""
import argparse, hashlib, json, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aimgeo

REC, A = {}, {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def js(self, o, code=200):
        b = json.dumps(o, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path); q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        if url.path == "/state":
            return self.js(dict(protocol="r1-aim-1", device_state="IDLE", status="fake"))
        if url.path == "/request":
            return self.js(REC.get(q.get("id"), dict(status="UNKNOWN")))
        if url.path == "/pt.mjpg" and A.get("render"):       # 시뮬: 마지막으로 조준한 방향의 '실시간' 화면
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            try:
                while True:
                    last = A.get("last") or A["home_px"]
                    ok, jpg = cv2.imencode(".jpg", A["render"](*last, width=800), [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b = jpg.tobytes()
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n" + ("Content-Length: %d\r\n\r\n" % len(b)).encode() + b + b"\r\n")
                    time.sleep(0.5)
            except Exception:                              # noqa: BLE001  보는 쪽이 닫으면 끝
                return
        if url.path == "/home":
            REC[q["req"]] = dict(request_id=q["req"], kind="home", status="COMPLETED", device_state="IDLE", reason_code="home_done")
            A["moves"].append(("home",)); A["last"] = None
            return self.js(REC[q["req"]])
        if url.path == "/aim":
            g = A["geo"]; u, v, z = int(float(q["u"])), int(float(q["v"])), float(q["z"])
            pan, tilt = g.pixel_servo(u, v, z)
            inside = g.pan_lim[0] <= pan <= g.pan_lim[1] and g.tilt_lim[0] <= tilt <= g.tilt_lim[1]
            if not inside:
                rec = dict(request_id=q["req"], kind="aim", status="FAILED", device_state="IDLE", reason_code="target_out_of_pantilt_range", plan=dict(refused="target_out_of_pantilt_range", pan=pan, tilt=tilt))
            else:
                if A.get("render"):
                    img = A["render"](u, v, z, pan, tilt); A["last"] = (u, v, z, pan, tilt)
                else:
                    img = np.full((1080, 1920, 3), 90, np.uint8); cv2.putText(img, "fake pan %.1f tilt %.1f" % (pan, tilt), (60, 540), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
                cid = "cap-%s-s%d" % (q["req"], len(REC)); path = Path(A["out"]) / (cid + ".jpg"); cv2.imwrite(str(path), img); data = path.read_bytes()
                rec = dict(request_id=q["req"], kind="aim", status="COMPLETED", device_state="IDLE", reason_code="captured", plan=dict(refused=None, pan=pan, tilt=tilt, u=u, v=v, dist=z),
                           capture=dict(capture_id=cid, path=str(path), sha256=hashlib.sha256(data).hexdigest(), width=1920, height=1080))
                A["moves"].append(("aim", round(pan, 2), round(tilt, 2)))
            REC[q["req"]] = rec
            return self.js(dict(rec, status="ACCEPTED"))
        self.js({}, 404)


def start(port, out, pan_lim=aimgeo.PAN_LIM, tilt_lim=aimgeo.TILT_LIM, render=None):
    """render(u, v, z, pan, tilt, width=1920) -> BGR 사진: 시뮬 모드에서 가짜 팬틸트 사진을 만들어 주는 함수(없으면 회색 판)"""
    Path(out).mkdir(parents=True, exist_ok=True)
    g = aimgeo.Geo(pan_lim=pan_lim, tilt_lim=tilt_lim)
    hu, hv, hz, _ = g.virtual_pixel(g.home[0], g.home[1])
    A.update(geo=g, out=str(out), moves=[], render=render, last=None, home_px=(hu, hv, hz, g.home[0], g.home[1]))
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=18089); ap.add_argument("--out", default="/tmp/fake_console_out")
    a = ap.parse_args(); start(a.port, a.out); print("fake console on", a.port)
    while True:
        time.sleep(60)
