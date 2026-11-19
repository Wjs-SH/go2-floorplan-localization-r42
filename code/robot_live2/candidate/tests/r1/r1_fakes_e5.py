"""SIMULATED test doubles for R1 E5 tests (no device, no ROS, no model, no network).

FakeConsole implements r1-aim-1 with the JSON shapes of R1/reports/_parts/W-B_protocol_notes.md (single slot,
idempotent req, CANCELLING until the simulated hardware step ends, capture by capture_id, quarantine of late results).
LegacyConsole mimics the pre-R1 console (job counter, plain-text replies) to show PROTOCOL_MISMATCH refusal.
"""
import argparse
import json
import os
import threading
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np

PROTO = "r1-aim-1"


def jpeg(sharp=True, seed=0, w=320, h=180):
    rng = np.random.RandomState(seed)
    if sharp:
        img = (rng.rand(h, w, 3) * 255).astype(np.uint8)            # high Laplacian variance
    else:
        img = np.full((h, w, 3), 120, np.uint8)                      # flat = blurred / unreadable
    return cv2.imencode(".jpg", img)[1].tobytes()


class FakeConsole:
    def __init__(self, job_s=0.0, sharp=True, clock=time.monotonic, same_capture_on_retry=False, identical_bytes=False):
        self.job_s, self.sharp, self.clock = job_s, sharp, clock
        self.same_capture_on_retry, self.identical_bytes = same_capture_on_retry, identical_bytes
        self.lock = threading.Lock()
        self.reqs, self.order, self.captures = {}, [], {}
        self.active, self.device_state, self.seq, self.calls = None, "IDLE", 0, []
        self.hw_end_override = {}                                    # rid -> extra hardware seconds after cancel
        self.n_manual = 0

    # ------------------------------------------------------------------ simulated hardware progress
    def _advance(self):
        now = self.clock()
        r = self.reqs.get(self.active) if self.active else None
        if r is None:
            return
        if r["status"] == "ACCEPTED":
            r["status"], r["t_start"] = "RUNNING", dict(mono_ns=time.monotonic_ns(), wall=time.time())
            r["_t0"] = now
            self.device_state = "BUSY"
        end = r["_t0"] + r["_dur"]
        if now >= end:
            self.device_state = "IDLE"
            r["hw_done"], r["device_state"] = True, "IDLE"
            r["t_hw_end"] = r["t_device_idle"] = dict(mono_ns=time.monotonic_ns(), wall=time.time())
            cap = self._make_capture(r) if r["kind"] in ("aim", "focus") else None
            if r["status"] == "CANCELLING":
                r["status"], r["quarantine"] = "CANCELLED", dict(result="late", capture=cap)
            else:
                r["status"] = "COMPLETED"
                r["capture"], r["reason_code"] = cap, ("captured" if cap else "home_done")
            self.active = None

    def _make_capture(self, r):
        if self.same_capture_on_retry and self.captures:          # 'same full.jpg re-downloaded': no new exposure
            first = next(iter(self.captures.values()))
            rec = dict(first["rec"], capture_id=f"cap-{r['request_id']}-s{first['rec']['frame_seq']}", request_id=r["request_id"])
            self.captures[rec["capture_id"]] = dict(rec=rec, data=first["data"])
            return rec
        self.seq += 1
        cid = f"cap-{r['request_id']}-s{self.seq}"
        data = jpeg(self.sharp, seed=0 if self.identical_bytes else self.seq)
        rec = dict(capture_id=cid, request_id=r["request_id"], path=f"/fake/{cid}.jpg", frame_seq=self.seq,
                   frame_rx_mono_ns=time.monotonic_ns(), source_t=None,
                   source_t_null_reason="camera driver gives no exposure timestamp; rx time only",
                   saved_wall=time.time(), sha256="x", width=320, height=180)
        self.captures[cid] = dict(rec=rec, data=data)
        return rec

    def _rec(self, r, **kw):
        return dict({k: v for k, v in r.items() if not k.startswith("_")}, protocol=PROTO, **kw)

    def submit(self, kind, q):
        rid, owner = q.get("req"), q.get("owner")
        if not rid:
            self.n_manual += 1
            rid, owner = f"manual-{kind}-{self.n_manual}-ab", f"manual:{self.n_manual}"
        if rid in self.reqs:
            return self._rec(self.reqs[rid], duplicate=True)
        r = dict(request_id=rid, owner=owner, kind=kind, params=q, status="ACCEPTED", device_state=None, hw_done=False,
                 capture=None, focus=None, reason_code=None, t_accept=dict(mono_ns=time.monotonic_ns(), wall=time.time()),
                 t_start=None, t_cancel_requested=None, t_logical_expired=None, t_hw_end=None, t_device_idle=None,
                 quarantine=None, result=None, _dur=self.job_s, _t0=None)
        self.reqs[rid] = r
        if self.active is not None or self.device_state != "IDLE":
            r.update(status="REJECTED_BUSY", hw_done=True, reason_code="device_busy")
            return self._rec(r, duplicate=False)
        self.active = rid
        self._advance()
        return self._rec(r, duplicate=False)

    # ------------------------------------------------------------------ HTTP-like entry (injected http_get)
    def http_get(self, url, timeout):
        u = urllib.parse.urlparse(url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        with self.lock:
            self.calls.append((u.path, q))
            self._advance()
            if u.path == "/state":
                return json.dumps(dict(protocol=PROTO, device_state=self.device_state, active_request=self.active, queued=[],
                                       status="fake", log=[], checks=0)).encode()
            if u.path in ("/aim", "/home", "/focus"):
                return json.dumps(self.submit(u.path[1:], q)).encode()
            if u.path == "/request":
                r = self.reqs.get(q.get("id"))
                if r is None:
                    return json.dumps(dict(protocol=PROTO, status="UNKNOWN", reason_code="unknown_request",
                                           device_state=self.device_state)).encode()
                return json.dumps(self._rec(r)).encode()
            if u.path == "/cancel":
                r = self.reqs.get(q.get("id"))
                if r is None:
                    return json.dumps(dict(protocol=PROTO, status="UNKNOWN")).encode()
                if r["status"] == "ACCEPTED":
                    r.update(status="CANCELLED", hw_done=True)
                    self.active = None
                elif r["status"] == "RUNNING":
                    r.update(status="CANCELLING", hw_done=False, t_cancel_requested=dict(mono_ns=time.monotonic_ns()))
                    if q.get("expired"):
                        r["t_logical_expired"] = dict(mono_ns=time.monotonic_ns())
                    r["_dur"] += self.hw_end_override.get(r["request_id"], 0.0)
                    self.device_state = "CANCELLING"
                return json.dumps(self._rec(r)).encode()
            if u.path == "/capture":
                c = self.captures.get(q.get("capture_id"))
                if c is None:
                    raise IOError("404")
                return c["data"]
            if u.path == "/full.jpg":
                raise AssertionError("R1 client must never fetch the global full.jpg")
            raise IOError(f"404 {u.path}")

    def manual(self, kind="focus"):
        with self.lock:
            return self.submit(kind, {})


class LegacyConsole:
    """Pre-R1 console: /state with a job counter, plain-text /aim reply, global full.jpg."""

    def __init__(self):
        self.job, self.calls = 0, []

    def http_get(self, url, timeout):
        u = urllib.parse.urlparse(url)
        self.calls.append(u.path)
        if u.path == "/state":
            return json.dumps(dict(status="대기", job=self.job, log=[])).encode()
        if u.path == "/aim":
            self.job += 1
            return "조준 시작".encode()
        if u.path == "/full.jpg":
            return jpeg(True)
        raise IOError("404")


class FakeMon:
    def __init__(self):
        self.last, self.watch = {}, {}


class FakeBuf:
    def __init__(self):
        self.items = []

    def newest(self):
        return self.items[-1] if self.items else None


class FakeL:
    floor_up = None


class FakeFrontend:
    def __init__(self, scene_delay_s=0.0):
        self.buf, self.L, self.vlm, self.pool = FakeBuf(), FakeL(), None, None
        self.scene_delay_s = scene_delay_s
        self.frames = []

    def add_frame(self, fid, t, bgr):
        self.buf.items = (self.buf.items + [dict(id=fid, tb=t, th=t, bgr=bgr)])[-30:]

    def prefetch(self, t):
        pass

    def scene(self, t, dimg, pose, allow_stop=True):
        time.sleep(self.scene_delay_s)
        base = self.buf.newest()
        return dict(t=t, base=base, regs=[], vp=None, thr=0.1, legacy_ev=None, timing=dict(scene_ms=self.scene_delay_s * 1e3),
                    regions=[])


class FakeSource:
    def __init__(self):
        self.lock = threading.Lock()
        self.odom_hist, self.alive, self.last_odom_wall, self.rx_mono_ns = [], True, time.time(), {}
        self.queue = []

    def poll(self):
        out, self.queue = self.queue, []
        return out

    def depth(self, stamp):
        return np.zeros((480, 640), np.uint16), stamp

    def pose(self, stamp, max_gap=0.3):
        if not self.odom_hist:
            return None, None
        o = min(self.odom_hist, key=lambda o: abs(o[0] - stamp))
        return (o[1], o[2], o[3]), o[0]

    def health(self):
        return dict(source="fake")

    def close(self):
        pass


def args(tmp, **kw):
    ws = Path(os.environ["SCI_WS"])
    a = dict(source="ros", dataset=None, realtime=False, t_start=0.0, t_end=None, odom_topic="/fake",
             policy=str(ws / "config" / "e5_policy.json"), stage="A3", p2_policy=str(ws / "config" / "e5_policy_p2.yaml"),
             ui_scale=1.0, max_temp_c=85.0, min_mem_mb=1500.0, e4_file=None, pantilt_execute=True, assume_stopped=False,
             warmup=3.0, max_lag=1.5, workers=1, save_frames=False, run_name="r1test", diagnostic_proxy=False)
    a.update(kw)
    return argparse.Namespace(**a)


def a3_stop(box=(300, 220, 340, 260), z=2.0):
    return dict(t=1.0, stage="A3", spot=7, box=list(box), tile=list(box), z=z, sim=0.2, msim=0.2, nscene=3, nobs=3,
                xyz=[2.0, 0.5, 1.5], box_frame=dict(frame_id=5, header_stamp=100.0, box=list(box)))
