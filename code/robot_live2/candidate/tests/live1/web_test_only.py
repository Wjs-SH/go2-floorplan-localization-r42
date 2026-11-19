#!/usr/bin/env python3
"""TEST_ONLY web check: build fixture state (synthetic odom walk + a RECORDED SEM1 status from 03:15, which must show as
STALE / OLD), let `live1_ctl.sh start web --test-only` serve it, GET every route, take a headless-Firefox screenshot
(own temporary profile, loopback only), then stop the web through the ctl. No ROS, no model, no device."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

L = Path(os.environ.get("LIVE1_ROOT", Path(__file__).resolve().parents[3]))
sys.dont_write_bytecode = True                           # never leave a .pyc in the follow-up baseline tree
sys.path.insert(0, str(L / "candidate" / "live1"))
sys.path.insert(0, str(L.parent / "sem1_followup_20261018_041435" / "candidate" / "tools"))
import live1_common as C  # noqa: E402
import lio_monitor as LM  # noqa: E402

SEM1_SHM = Path("/home/unitree/sci_ws/repairs/e5sem1_20261018_025424/runtime/live_shm")


def fixtures():
    st, shm = L / "runtime" / "state_TEST_ONLY", L / "runtime" / "e5_shm_TEST_ONLY"
    for d in (st, shm):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    t = [time.monotonic() - 30]
    od = LM.OdomStats(wall=lambda: time.time(), mono=lambda: t[0])
    acc = LM.Accumulator(stride=1)
    import numpy as np
    for i in range(300):                                 # synthetic L-shaped corridor walk (TEST_ONLY)
        t[0] += 0.1
        x, y = (0.03 * i, 0.0) if i < 150 else (4.5, 0.03 * (i - 150))
        yaw = 0.0 if i < 150 else math.pi / 2
        od.on_odom(time.time() - 30 + 0.1 * i, x, y, 0.0, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2), "camera_init", "base_link")
        wall = np.stack([np.linspace(-1, 6, 200), np.full(200, 1.2), np.zeros(200)], 1)
        wall2 = np.stack([np.full(200, 5.7), np.linspace(-1, 6, 200), np.zeros(200)], 1)
        acc.add_xyz(np.concatenate([wall, wall2, wall * [1, -1, 1]]).astype(np.float32), z_ref=0.0)
    t[0] = time.monotonic()
    s = od.summary()
    C.atomic_json(st / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="TEST_ONLY synthetic odom", odom=s,
                                              track_raw=[(a[0], a[1]) for a in od.track], accumulation=acc.summary(),
                                              frame_note="TEST_ONLY synthetic"))
    C.atomic_bytes(st / "lio_map.png", acc.render(list(od.track)))
    for f in ("status.json", "frame.jpg", "frame_inference.jpg"):
        shutil.copy2(SEM1_SHM / f, shm / f)              # copy2 keeps the 03:15 mtime -> must be STALE
    return st, shm


def main():
    out = L / "evidence" / "web_TEST_ONLY"
    out.mkdir(parents=True, exist_ok=True)
    fixtures()
    ctl = str(L / "scripts" / "live1_ctl.sh")
    r = subprocess.run([ctl, "start", "web", "--test-only", "--run-s", "300", "--host", "127.0.0.1"], capture_output=True, text=True)
    print(r.stdout[-800:])
    base = "http://127.0.0.1:8770"
    res = {}
    for _ in range(40):
        try:
            urllib.request.urlopen(base + "/api/state", timeout=2)
            break
        except Exception:
            time.sleep(0.25)
    for path in ("/", "/api/state", "/img/live.jpg", "/img/inference.jpg", "/img/map.png"):
        try:
            with urllib.request.urlopen(base + path, timeout=5) as h:
                body = h.read()
                res[path] = dict(code=h.status, bytes=len(body), mode=h.headers.get("X-LIVE1-MODE"))
                if path == "/api/state":
                    snap = json.loads(body)
                    (out / "api_state_TEST_ONLY.json").write_text(json.dumps(snap, indent=1, ensure_ascii=False, default=str))
        except Exception as e:
            res[path] = dict(error=str(e))
    req = urllib.request.Request(base + "/api/state", data=b"{}", method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        res["POST"] = "ACCEPTED (BAD)"
    except urllib.error.HTTPError as e:
        res["POST"] = e.code
    shot = screenshot(base, out / "web_TEST_ONLY.png")
    r2 = subprocess.run([ctl, "stop", "web"], capture_output=True, text=True)
    print(r2.stdout[-400:])
    summary = dict(routes=res, lio_status=snap["lio"]["status"], e5_status=snap["e5"]["status"],
                   e5_source_old=snap["e5"]["scene"]["source_old"], test_only=snap["meta"]["test_only"], screenshot=shot,
                   stop=r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else None)
    (out / "web_TEST_ONLY_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


def screenshot(base, png):
    import base64
    import signal
    from followup_ui_roundtrip import Marionette
    prof = Path(tempfile.mkdtemp(prefix="live1_ff_", dir=str(L / "runtime")))
    (prof / "user.js").write_text('user_pref("marionette.port", 28830);\nuser_pref("browser.shell.checkDefaultBrowser", false);\n'
                                  'user_pref("datareporting.policy.dataSubmissionEnabled", false);\nuser_pref("toolkit.telemetry.enabled", false);\n')
    p = subprocess.Popen(["firefox", "--headless", "--marionette", "-no-remote", "-profile", str(prof), "--window-size=1400,1800"],
                         env=dict(os.environ, MOZ_HEADLESS="1", MOZ_HEADLESS_WIDTH="1400", MOZ_HEADLESS_HEIGHT="1800"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        m = Marionette(28830)
        m.cmd("WebDriver:NewSession", {"capabilities": {}})
        m.cmd("WebDriver:SetWindowRect", {"width": 1400, "height": 1800})
        m.cmd("WebDriver:Navigate", {"url": base + "/"})
        time.sleep(3.0)
        b64 = m.cmd("WebDriver:TakeScreenshot", {"full": True})["value"]
        png.write_bytes(base64.b64decode(b64))
        txt = m.cmd("WebDriver:ExecuteScript", {"script": "return document.getElementById('banner').textContent+' | '+"
                                                "document.getElementById('stopui').textContent", "args": []})["value"]
        try:
            m.cmd("WebDriver:DeleteSession")
        except Exception:
            pass
        return dict(file=str(png), banner_and_stop=txt)
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(10)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
