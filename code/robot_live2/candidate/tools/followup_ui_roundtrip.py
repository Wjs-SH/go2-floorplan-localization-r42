#!/usr/bin/env python3
"""Execute the REAL label UI JavaScript in a private headless Firefox (Marionette on 127.0.0.1, temporary profile,
own process only) to create TEST_ONLY records (one sign-like + one poster-like item, a note with comma / quote /
newline / Korean), export CSV + JSON exactly as the UI download buttons would, and re-import the JSON.
Records are TEST_ONLY and human_verified=false. No ROS, no device, no network beyond loopback."""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PORT = 28829


class Marionette:
    def __init__(self, port):
        t0 = time.time()
        while True:
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                if time.time() - t0 > 60:
                    raise
                time.sleep(0.5)
        self.buf, self.mid = b"", 0
        self._read()                                          # handshake

    def _read(self):
        while b":" not in self.buf:
            self.buf += self.s.recv(65536)
        n, rest = self.buf.split(b":", 1)
        n = int(n)
        while len(rest) < n:
            rest += self.s.recv(65536)
        msg, self.buf = rest[:n], rest[n:]
        return json.loads(msg)

    def cmd(self, name, params=None):
        self.mid += 1
        b = json.dumps([0, self.mid, name, params or {}]).encode()
        self.s.sendall(str(len(b)).encode() + b":" + b)
        while True:
            r = self._read()
            if r[0] == 1 and r[1] == self.mid:
                if r[2]:
                    raise RuntimeError(r[2])
                return r[3]


SCRIPT = r"""
const S1=window.SEM1;const out={};
// item 0 of POSITIVE group, frame 0; item 0 of OTHER group (poster-like), frame 0
S1.select(0,0,0);let b=S1.newBox(100.5,120.25,160.75,170.5);
S1.setB(b,'visual_type','ROOM_PLATE');S1.setB(b,'localization_role','LOCAL_ANCHOR');S1.setB(b,'text_readability','PARTIAL');
S1.setB(b,'physical_target_id','TEST-S1F-001');S1.setB(b,'visit_id','v1');S1.setB(b,'test_only',true);
S1.setB(b,'note','쉼표, "따옴표"\n줄바꿈 한글 테스트');S1.setB(b,'target_in_model_crop','Y');
S1.select(2,0,0);let c=S1.newBox(30,40,90,120);S1.setB(c,'visual_type','POSTER');S1.setB(c,'localization_role','NONLOCAL_TEXT');
S1.setB(c,'physical_target_id','TEST-POSTER-001');S1.setB(c,'test_only',true);S1.setB(c,'polygon',[[31,41],[89,42],[88,119],[30,118]]);
out.csv=S1.buildCSV();out.json=S1.buildJSON();
const before=JSON.stringify(S1.state());S1.importJSON(JSON.parse(out.json));out.reimport_equal=JSON.stringify(S1.state())===before;
out.items=S1.DOC.items.length;return out;
"""


def main():
    ui, outdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prof = Path(tempfile.mkdtemp(prefix="sem1_ff_", dir=str(outdir.parent)))
    (prof / "user.js").write_text(f'user_pref("marionette.port", {PORT});\nuser_pref("browser.shell.checkDefaultBrowser", false);\n'
                                  'user_pref("datareporting.policy.dataSubmissionEnabled", false);\nuser_pref("toolkit.telemetry.enabled", false);\n')
    env = dict(os.environ, MOZ_HEADLESS="1")
    p = subprocess.Popen(["firefox", "--headless", "--marionette", "-no-remote", "-profile", str(prof)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        m = Marionette(PORT)
        m.cmd("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"acceptInsecureCerts": True}}})
        m.cmd("WebDriver:Navigate", {"url": ui.as_uri()})
        time.sleep(1.0)
        r = m.cmd("WebDriver:ExecuteScript", {"script": SCRIPT, "args": []})["value"]
        with open(outdir / "TEST_ONLY_HUMAN_BBOX.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(r["csv"])
        (outdir / "TEST_ONLY_sem1_labels_state.json").write_text(r["json"], encoding="utf-8")
        print(json.dumps(dict(reimport_equal=r["reimport_equal"], items=r["items"], csv_bytes=len(r["csv"]), firefox_pid=p.pid)))
        try:
            m.cmd("WebDriver:DeleteSession")
        except Exception:
            pass
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(10)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
