#!/usr/bin/env python3
"""cdp_min.py — 머리 없는 Chrome 을 표준 라이브러리만으로 조작하는 최소 도구 (원격 디버깅 프로토콜, WebSocket 직접 구현).
시험 전용. Chrome 을 새 사용자 폴더로 띄우고, 화면 이동·스크립트 실행·화면 찍기·오류 수집만 한다."""
import base64, json, os, shutil, socket, struct, subprocess, tempfile, time, urllib.request


class WS:
    def __init__(self, url):
        assert url.startswith("ws://"); host, path = url[5:].split("/", 1); h, p = host.split(":")
        self.s = socket.create_connection((h, int(p)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall(("GET /%s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (path, host, key)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        assert b" 101 " in buf.split(b"\r\n")[0], buf[:200]
        self.buf = buf.split(b"\r\n\r\n", 1)[1]

    def _read(self, n):
        while len(self.buf) < n:
            c = self.s.recv(65536)
            if not c: raise ConnectionError("닫힘")
            self.buf += c
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text):
        b = text.encode(); m = os.urandom(4); n = len(b)
        hdr = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.s.sendall(hdr + m + bytes(c ^ m[i % 4] for i, c in enumerate(b)))

    def recv(self):
        data = b""
        while True:
            h = self._read(2); fin, op, n = h[0] & 0x80, h[0] & 0x0F, h[1] & 0x7F
            if n == 126: n = struct.unpack(">H", self._read(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._read(8))[0]
            p = self._read(n)
            if op == 9: continue                       # ping: 시험은 짧으므로 응답 생략
            if op == 8: raise ConnectionError("닫힘")
            data += p
            if fin: return data.decode("utf-8", "replace")


class Tab:
    def __init__(self, ws_url):
        self.ws = WS(ws_url); self.n = 0; self.errors = []
        self.call("Runtime.enable"); self.call("Page.enable")

    def call(self, method, **params):
        self.n += 1; i = self.n; self.ws.send(json.dumps(dict(id=i, method=method, params=params)))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == i:
                if "error" in m: raise RuntimeError("%s: %s" % (method, m["error"]))
                return m.get("result", {})
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]; self.errors.append((d.get("exception") or {}).get("description") or d.get("text"))

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"): raise RuntimeError(str(r["exceptionDetails"].get("exception", {}).get("description") or r["exceptionDetails"])[:300])
        return r["result"].get("value")

    def frame(self, url_prefix, tries=40):
        """이 화면 안의 iframe(같은 사이트라 따로 뜨지 않는 것)에서 스크립트를 돌리는 함수를 돌려준다. 격리된 실행 환경이라 DOM 만 본다."""
        def walk(t):
            yield t["frame"]
            for c in t.get("childFrames", []):
                yield from walk(c)
        for _ in range(tries):
            f = [x for x in walk(self.call("Page.getFrameTree")["frameTree"]) if x.get("url", "").startswith(url_prefix)]
            if f:
                ctx = self.call("Page.createIsolatedWorld", frameId=f[0]["id"])["executionContextId"]
                def js(expr):
                    r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True, contextId=ctx)
                    if r.get("exceptionDetails"): raise RuntimeError(str(r["exceptionDetails"])[:300])
                    return r["result"].get("value")
                return js
            time.sleep(0.25)
        return None

    def go(self, url, wait=1.5):
        self.errors = []; self.call("Page.navigate", url=url); time.sleep(wait)

    def shot(self, path):
        open(path, "wb").write(base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"]))


class Chrome:
    def __init__(self, port=9333, size="1200,1000"):
        self.port = port; self.dir = tempfile.mkdtemp(prefix="cdp_")
        self.p = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", "--remote-debugging-port=%d" % port,
                                   "--user-data-dir=" + self.dir, "--window-size=" + size, "--autoplay-policy=no-user-gesture-required", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try: self.targets(); break
            except Exception: time.sleep(0.25)

    def targets(self):
        return json.load(urllib.request.urlopen("http://127.0.0.1:%d/json/list" % self.port, timeout=3))

    def page(self):
        t = [x for x in self.targets() if x["type"] == "page"][0]; return Tab(t["webSocketDebuggerUrl"])

    def frame(self, url_prefix, tries=40):
        for _ in range(tries):
            t = [x for x in self.targets() if x["type"] in ("iframe", "page") and x["url"].startswith(url_prefix)]
            if t: return Tab(t[0]["webSocketDebuggerUrl"])
            time.sleep(0.25)
        return None

    def close(self):
        self.p.terminate()
        try: self.p.wait(5)
        except Exception: self.p.kill()
        shutil.rmtree(self.dir, ignore_errors=True)
