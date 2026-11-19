#!/usr/bin/env python3
"""Qwen3-VL-2B-Instruct (official Qwen GGUF) via llama.cpp `llama-server` built in ~/sci_ws/external/llama.cpp.

The server is a separate process (selective loading): started on demand only when the resource guard admits it,
and stopped after `idle_s` without requests. OpenAI-compatible /v1/chat/completions with a base64 image.
"""
from __future__ import annotations

import base64
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2

WS = Path.home() / "sci_ws"
LLAMA = WS / "external" / "llama.cpp" / "build" / "bin" / "llama-server"
MODEL = WS / "models" / "Qwen3-VL-2B-Instruct-GGUF" / os.environ.get("SCI_VLM_GGUF", "Qwen3VL-2B-Instruct-Q8_0.gguf")
MMPROJ = WS / "models" / "Qwen3-VL-2B-Instruct-GGUF" / "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
PORT = int(os.environ.get("SCI_VLM_PORT", "8097"))

PROMPT = ("You see a photo taken by a robot in a university corridor. Classify the main sign in the CENTER of the image "
          "and read it. Answer ONLY with JSON: {\"sign_type\": one of [\"room_number_plate\", \"lab_or_office_name_sign\", "
          "\"floor_directory\", \"elevator_sign_or_button\", \"poster_or_notice\", \"other\"], \"room_number\": the room number "
          "printed on the sign or \"\" if none, \"text\": other readable text}. Numbers on posters, elevator buttons or "
          "floor directories are NOT room numbers.")


class VlmServer:
    def __init__(self, ngl=99, ctx=4096, log=None):
        self.ngl, self.ctx, self.p, self.log = ngl, ctx, None, log or str(WS / "logs" / "llama_server.log")
        self.url = f"http://127.0.0.1:{PORT}"
        self.last_use, self.load_s = 0.0, None

    @staticmethod
    def available():
        missing = [str(p) for p in (LLAMA, MODEL, MMPROJ) if not p.exists()]
        return (not missing), missing

    def healthy(self, timeout=1.0):
        try:
            return json.loads(urllib.request.urlopen(self.url + "/health", timeout=timeout).read()).get("status") == "ok"
        except Exception:
            return False

    def start(self, wait_s=180):
        if self.healthy():
            return True
        ok, missing = self.available()
        if not ok:
            raise RuntimeError(f"VLM files missing: {missing}")
        t0 = time.time()
        cmd = [str(LLAMA), "-m", str(MODEL), "--mmproj", str(MMPROJ), "--host", "127.0.0.1", "--port", str(PORT),
               "-ngl", str(self.ngl), "-c", str(self.ctx), "-np", "1", "--no-webui", "-t", "2"]
        self.p = subprocess.Popen(cmd, stdout=open(self.log, "ab"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        while time.time() - t0 < wait_s:
            if self.p.poll() is not None:
                raise RuntimeError(f"llama-server exited rc={self.p.returncode} (see {self.log})")
            if self.healthy():
                self.load_s = time.time() - t0
                return True
            time.sleep(1.0)
        raise RuntimeError("llama-server start timeout")

    def stop(self):
        if self.p is not None and self.p.poll() is None:
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
        self.p = None

    def ask(self, bgr, prompt=PROMPT, max_side=768, max_tokens=96, timeout=120):
        h, w = bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        b64 = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()
        body = dict(messages=[dict(role="user", content=[dict(type="image_url", image_url=dict(url=f"data:image/jpeg;base64,{b64}")),
                                                         dict(type="text", text=prompt)])],
                    temperature=0.0, max_tokens=max_tokens)
        t0 = time.time()
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        self.last_use = time.time()
        txt = d["choices"][0]["message"]["content"]
        parsed = None
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
        return dict(raw=txt, parsed=parsed, sec=time.time() - t0, timings=d.get("timings"), image_px=list(img.shape[:2]))
