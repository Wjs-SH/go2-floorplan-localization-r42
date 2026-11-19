#!/usr/bin/env python3
"""vlm_ondemand.py (R36): the VLM server actually runs. Until R35 the [VLM] box only armed a client for an endpoint nothing ever started, so every field reading said 'VLM endpoint down'.

Same model and server as the student's reader (Qwen3-VL-2B Q8_0 + mmproj through llama.cpp llama-server, sci_text_reading/vlm.py). Started when E8 first needs a fallback, stopped after IDLE_S without requests,
always a child of this process (dies with it). Measured on this Jetson 2026-11-02 (VLM_MEASUREMENT.md): load 11 s, one image 2-6 s, about 3.7 GB; SigLIP p50 285 -> 383 ms while it infers, so the two share the GPU.
What it refuses: memory, temperature or FAST-LIO rate outside the limits below, each with its reason shown on the tablet. FAST-LIO comes first: the server is not started, and a running one is stopped, when the rate drops.
A VLM number is never accepted by itself: student fusion marks it VLM_ONLY_UNCONFIRMED (not decisive). In the same measurement the VLM invented '202' and '101' on blurry photos."""
import atexit
import json
import queue
import os
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

WS = Path.home() / "sci_ws"
LLAMA = WS / "external" / "llama.cpp" / "build" / "bin" / "llama-server"
MODEL_DIR = WS / "models" / "Qwen3-VL-2B-Instruct-GGUF"
MODEL, MMPROJ = MODEL_DIR / os.environ.get("SCI_VLM_GGUF", "Qwen3VL-2B-Instruct-Q8_0.gguf"), MODEL_DIR / "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
PORT = int(os.environ.get("SCI_VLM_PORT", "8097"))
MIN_MEM_MB, MAX_TEMP_C, MIN_ODOM_HZ, IDLE_S, LOAD_TIMEOUT_S = 5400, 78.0, 8.5, 90.0, 120.0   # R41, measured: the server takes 3650 MB at any -c (runs/r41/vlm_mem.py), load 4-10 s. With E2 at budget 0 and SigLIP resident the board holds 5680-6050 MB (load5 +602..+816), so 6200 (3650 + the keeper 2500) was never met and a read plate waited 250 s. 5400 = what the board has with E2 given back. Below 2500 MB the keeper stops SigLIP in order (its limit, unchanged), which returns about 2900 MB, and resumes it when the VLM has stopped


def mem_available_mb():
    try:
        return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) // 1024
    except (OSError, StopIteration, ValueError):
        return 0


def max_temp_c():
    import glob
    t = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            v = int(open(p).read()) / 1000.0
            if 0 < v < 150:
                t.append(v)
        except (OSError, ValueError):
            pass
    return max(t) if t else None


def _die_with_parent():
    os.setsid()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)      # PR_SET_PDEATHSIG
    except OSError:
        pass
    os.nice(5)
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 8)))  # e34 is pinned to core 5: not inherited. R37: R36 used the E2 cores: field load 58 s and one 60 s timeout (idle 8-11 s)
    except (AttributeError, OSError, ValueError):
        pass


class OnDemandVLM:
    def __init__(self, state_dir=None, clock=time.time, popen=subprocess.Popen, healthy=None):
        self.state_dir = None if state_dir is None else Path(state_dir)
        self.clock, self.popen, self.p, self.lock = clock, popen, None, threading.Lock()
        self._healthy = healthy
        self.url = f"http://127.0.0.1:{PORT}"
        self.d = dict(kind="LIVE2_VLM_STATE", state="OFF", why="not needed yet", n_asked=0, n_load=0, load_s=None, last_s=None, model=MODEL.name)
        self.last_use = 0.0
        self._write()
        atexit.register(self.stop, "PROCESS_EXIT")

    def _write(self):
        if self.state_dir is None:
            return
        try:
            tmp = self.state_dir / f".vlm_state.json.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(dict(self.d, written_wall=self.clock()), ensure_ascii=False))
            os.replace(tmp, self.state_dir / "vlm_state.json")
            with open(self.state_dir / "vlm_events.jsonl", "a") as f:
                f.write(json.dumps(dict(self.d, wall=self.clock()), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _set(self, state, why=None, **kw):
        self.d.update(state=state, why=why, **kw)
        self._write()

    def healthy(self, timeout=1.0):
        if self._healthy is not None:
            return self._healthy()
        try:
            return json.loads(urllib.request.urlopen(self.url + "/health", timeout=timeout).read()).get("status") == "ok"
        except Exception:                      # noqa: BLE001
            return False

    def _odom_hz(self):
        try:
            return ((json.loads((self.state_dir / "lio_state.json").read_text()).get("odom") or {}).get("hz_10s"))
        except (OSError, ValueError, AttributeError):
            return None

    def admit(self):
        """-> (ok, why). Checked before every request; a server that is already running does not need the load margin again."""
        missing = [str(p) for p in (LLAMA, MODEL, MMPROJ) if not p.exists()]
        if missing:
            return False, f"VLM files missing: {missing}"
        running = self.p is not None and self.p.poll() is None
        mem = mem_available_mb()
        if not running and mem < MIN_MEM_MB:
            return False, f"MemAvailable {mem} MB < {MIN_MEM_MB} MB: VLM not started (FAST-LIO first)"
        t = max_temp_c()
        if t is not None and t >= MAX_TEMP_C:
            return False, f"temperature {t:.0f} C >= {MAX_TEMP_C:.0f} C"
        hz = self._odom_hz()
        if hz is not None and float(hz) < MIN_ODOM_HZ:
            return False, f"FAST-LIO {hz} Hz < {MIN_ODOM_HZ} Hz: VLM held"
        return True, f"MemAvailable {mem} MB, {t} C, FAST-LIO {hz} Hz"

    def _popen_in_owner(self, cmd, log):
        """R41: PR_SET_PDEATHSIG fires when the THREAD that forked ends, not the process. The reading thread of e8_pair ends when its queue is empty, so the server died a moment after every answer (log: 'cleaning up before exit' seconds after the reply; field X20261103_001810 rc=-2), every next photo needed a new load, and a load needs 6200 MB: 4 of 6 field photos never got the VLM. The server now belongs to a thread that lives as long as this process."""
        if getattr(self, "_owner", None) is None or not self._owner.is_alive():
            self._owner_q = queue.Queue()
            self._owner = threading.Thread(target=self._owner_loop, daemon=True)
            self._owner.start()
        box = queue.Queue(1)
        self._owner_q.put((cmd, log, box))
        r = box.get()
        if isinstance(r, Exception):
            raise r
        return r

    def _owner_loop(self):
        while True:
            cmd, log, box = self._owner_q.get()
            try:
                box.put(self.popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent))
            except Exception as e:                 # noqa: BLE001
                box.put(e)

    def _start(self):
        if self.p is not None and self.p.poll() is None and self.healthy():
            return True
        t0 = self.clock()
        self._set("LOADING", "VLM server starting", n_load=self.d["n_load"] + 1)
        cmd = [str(LLAMA), "-m", str(MODEL), "--mmproj", str(MMPROJ), "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "4096", "-np", "1", "--no-webui", "-t", "2"]
        log = open(WS / "logs" / "llama_server.log", "ab")
        self.p = self._popen_in_owner(cmd, log)
        while self.clock() - t0 < LOAD_TIMEOUT_S:
            if self.p.poll() is not None:
                self._set("FAILED", f"llama-server exited rc={self.p.returncode}")
                return False
            if self.healthy():
                self._set("READY", "VLM server ready", load_s=round(self.clock() - t0, 1))
                threading.Thread(target=self._idle_watch, daemon=True).start()
                return True
            time.sleep(0.5)
        self.stop("LOAD_TIMEOUT")
        self._set("FAILED", f"no health within {LOAD_TIMEOUT_S:.0f} s")
        return False

    def ask(self, img):
        import e6_semantic  # noqa: F401  (sci_text_reading on sys.path)
        from vlm import VlmServer, PROMPT
        with self.lock:
            if not self._start():
                raise RuntimeError(self.d.get("why") or "VLM server not available")
            self._set("ASKING", None)
            srv = VlmServer.__new__(VlmServer)
            srv.url = self.url
            t0 = self.clock()
            try:
                r = VlmServer.ask(srv, img, PROMPT)
            finally:
                self.last_use = self.clock()
            self._set("READY", "idle (stops after {:.0f} s)".format(IDLE_S), n_asked=self.d["n_asked"] + 1, last_s=round(self.clock() - t0, 2))
            return dict(r, model=MODEL.name)

    def _idle_watch(self):
        while self.p is not None and self.p.poll() is None:
            time.sleep(5.0)
            hz = self._odom_hz()
            if self.lock.locked():
                continue
            if self.clock() - self.last_use > IDLE_S and self.last_use:
                self.stop("IDLE")
                return
            if hz is not None and float(hz) < MIN_ODOM_HZ - 0.5:
                self.stop(f"FAST-LIO {hz} Hz")
                return

    def stop(self, why="STOP"):
        p, self.p = self.p, None
        if p is not None and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except OSError:
                    pass
            self._set("OFF", f"stopped: {why}")
