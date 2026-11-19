"""R1: single device owner + r1-aim-1 request table for the pan/tilt aim console (pure stdlib, no hardware imports).

All motion/focus/HOME/diagnostic work runs on ONE worker thread, strictly one request at a time. Each request has its
own request_id + owner; completion is reported per request (no global job counter). Hardware access goes through an
injected adapter, so this module is tested with fakes and never opens a serial port or camera.

Hardware adapter interface (aim_console.R1Hardware in the field, FakeHardware in tests):
    plan_aim(params) -> dict(pan=, tilt=, clipped=, refused=None|str, dist=, dist_src=)
    move(pan, tilt, cancel)            servo motion incl. settle; may return early when cancel is set
    home(cancel)                       HOME pose
    focus(cancel, params) -> dict      r1_focus result dict (status/reason/...)
    frame_seq() -> int                 latest camera frame sequence number
    capture_fresh(after_seq, timeout_s) -> None | dict(jpeg=bytes, seq=, rx_mono_ns=, w=, h=)
    stop_actuators()                   best-effort F0 (focus motor stop). NOT a servo power cut.

Time semantics recorded per request (all monotonic ns of this boot + wall):
    t_cancel_requested  logical request to stop (client timeout / operator)   -> status CANCELLING
    t_hw_end            the hardware step really returned                      -> hw_done = True
    t_device_idle       the owner is free for the next request                 -> device_state IDLE
A cancelled request's late result is quarantined: it is never reported as its capture and never given to another id.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import os
import re
import threading
import time
import traceback

PROTOCOL = "r1-aim-1"           # == sci_common.r1_contracts.AIM_PROTOCOL (kept literal: console runs without SCI src)


class S:  # request status (== r1_contracts.ReqStatus)
    ACCEPTED, RUNNING, COMPLETED, FAILED = "ACCEPTED", "RUNNING", "COMPLETED", "FAILED"
    CANCELLING, CANCELLED, REJECTED_BUSY, UNKNOWN = "CANCELLING", "CANCELLED", "REJECTED_BUSY", "UNKNOWN"


TERMINAL = {S.COMPLETED, S.FAILED, S.CANCELLED, S.REJECTED_BUSY}


class D:  # device state (== r1_contracts.DeviceState)
    IDLE, BUSY, CANCELLING, UNKNOWN = "IDLE", "BUSY", "CANCELLING", "UNKNOWN"


FRESH_CAPTURE_UNAVAILABLE = "FRESH_CAPTURE_UNAVAILABLE"
SAFE_ID = re.compile(r"[^A-Za-z0-9_.:-]")


def _now():
    return dict(mono_ns=time.monotonic_ns(), wall=time.time())


class Cancelled(Exception):
    """Raised at a cooperative cancel point."""


class CancelToken:
    def __init__(self):
        self._ev = threading.Event()

    def set(self):
        self._ev.set()

    def is_set(self):
        return self._ev.is_set()

    def check(self):
        if self._ev.is_set():
            raise Cancelled()


class DeviceOwner:
    def __init__(self, hw, capture_dir, fresh_timeout_s=2.0, max_records=500, log=print):
        self.hw, self.capture_dir, self.fresh_timeout_s, self.log = hw, capture_dir, fresh_timeout_s, log
        os.makedirs(capture_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.records = collections.OrderedDict()   # request_id -> record (bounded, terminal ones evicted first)
        self.queue = collections.deque()
        self.tokens = {}
        self.device_state = D.IDLE
        self.active = None
        self.max_records = max_records
        self._manual = itertools.count(1)
        self.last_capture_id = None
        self._stop = False
        self.worker = threading.Thread(target=self._run, name="r1-device-owner", daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------ public API (thread-safe)
    def manual_id(self, kind):
        n = next(self._manual)
        return f"manual-{kind}-{n}-{os.urandom(3).hex()}", f"manual:{n}"

    def submit(self, kind, request_id, owner, params=None, fn=None):
        """Accept a device job if the owner is free, else REJECTED_BUSY. Same request_id again = idempotent."""
        if not request_id or not owner:
            return dict(protocol=PROTOCOL, status="PROTOCOL_ERROR", reason_code="request_id_and_owner_required")
        request_id = SAFE_ID.sub("_", str(request_id))[:160]
        with self.lock:
            if request_id in self.records:
                rec = self._view(self.records[request_id])
                rec["duplicate"] = True
                return rec
            rec = dict(protocol=PROTOCOL, request_id=request_id, owner=str(owner), kind=kind, params=params or {},
                       status=None, device_state=None, hw_done=False, capture=None, focus=None, reason_code=None,
                       t_accept=_now(), t_start=None, t_cancel_requested=None, t_logical_expired=None,
                       t_hw_end=None, t_device_idle=None, quarantine=None, result=None)
            busy = self.device_state != D.IDLE or self.active is not None or self.queue
            if busy:
                rec.update(status=S.REJECTED_BUSY, hw_done=True,
                           reason_code=f"device_{self.device_state.lower()}" if self.device_state != D.IDLE else "device_busy")
                self._store(rec)
                out = self._view(rec)
                out["duplicate"] = False
                return out
            rec["status"] = S.ACCEPTED
            rec["_fn"] = fn
            self._store(rec)
            self.tokens[request_id] = CancelToken()
            self.queue.append(request_id)
            self.cv.notify_all()
            out = self._view(rec)
            out["duplicate"] = False
            return out

    def cancel(self, request_id, why="client_cancel", logical_expiry=False):
        with self.lock:
            rec = self.records.get(request_id)
            if rec is None:
                return dict(protocol=PROTOCOL, request_id=request_id, status=S.UNKNOWN, reason_code="unknown_request")
            if rec["status"] in TERMINAL:
                return self._view(rec)
            t = _now()
            rec["t_cancel_requested"] = rec["t_cancel_requested"] or t
            if logical_expiry:
                rec["t_logical_expired"] = rec["t_logical_expired"] or t
            if rec["status"] == S.ACCEPTED:                  # never started: no hardware step happened
                self.queue.remove(request_id)
                rec.update(status=S.CANCELLED, hw_done=True, reason_code=why, t_hw_end=t)
            else:                                            # running: cooperative cancel, wait for the hw step
                rec.update(status=S.CANCELLING, reason_code=why)
                self.device_state = D.CANCELLING
                self.tokens[request_id].set()
            return self._view(rec)

    def get(self, request_id):
        with self.lock:
            rec = self.records.get(request_id)
            if rec is None:
                return dict(protocol=PROTOCOL, request_id=request_id, status=S.UNKNOWN, reason_code="unknown_request",
                            device_state=self.device_state)
            return self._view(rec)

    def state(self):
        with self.lock:
            return dict(protocol=PROTOCOL, device_state=self.device_state, active_request=self.active,
                        queued=list(self.queue), last_capture_id=self.last_capture_id)

    def capture_path(self, capture_id):
        cid = SAFE_ID.sub("_", str(capture_id))
        p = os.path.join(self.capture_dir, f"{cid}.jpg")
        return p if os.path.isfile(p) else None

    def wait_idle(self, timeout=5.0):
        t0 = time.monotonic()
        with self.lock:
            while (self.active is not None or self.queue or self.device_state not in (D.IDLE, D.UNKNOWN)) \
                    and time.monotonic() - t0 < timeout:
                self.cv.wait(0.02)
            return self.device_state

    def reset_unknown(self, operator_note):
        """Only after a supervised physical check: UNKNOWN -> IDLE. Logged; never automatic."""
        with self.lock:
            if self.device_state == D.UNKNOWN and self.active is None:
                self.device_state = D.IDLE
                self.log(f"device UNKNOWN -> IDLE by operator: {operator_note}")
                self.cv.notify_all()
                return True
            return False

    def close(self):
        with self.lock:
            self._stop = True
            self.cv.notify_all()

    # ------------------------------------------------------------------ internals
    def _store(self, rec):
        self.records[rec["request_id"]] = rec
        while len(self.records) > self.max_records:
            for k, r in self.records.items():
                if r["status"] in TERMINAL:
                    del self.records[k]
                    self.tokens.pop(k, None)
                    break
            else:
                break

    @staticmethod
    def _view(rec):
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        return json.loads(json.dumps(out, default=str))

    def _run(self):
        while True:
            with self.lock:
                while not self.queue and not self._stop:
                    self.cv.wait(0.5)
                if self._stop:
                    return
                rid = self.queue.popleft()
                rec = self.records[rid]
                token = self.tokens[rid]
                rec.update(status=S.RUNNING, t_start=_now())
                self.active, self.device_state = rid, D.BUSY
            result, err, cancelled = None, None, False
            try:
                result = self._execute(rec, token)
            except Cancelled:
                cancelled = True
            except Exception as e:  # noqa: BLE001  hardware errors: device state becomes UNKNOWN
                err = f"{type(e).__name__}: {e}"
                self.log(f"r1 owner: {rec['kind']} {rid} error {err}\n{traceback.format_exc()}")
                try:
                    self.hw.stop_actuators()
                except Exception:  # noqa: BLE001
                    pass
            t_end = _now()
            with self.lock:
                rec["t_hw_end"], rec["hw_done"] = t_end, True
                if token.is_set() or cancelled:
                    # late completion of a cancelled request: quarantine, never its (or anyone's) result
                    if result:
                        rec["quarantine"] = dict(result=result, capture=result.get("capture"))
                    rec.update(status=S.CANCELLED, reason_code=rec.get("reason_code") or "cancelled")
                elif err:
                    rec.update(status=S.FAILED, reason_code="hardware_error", error=err)
                else:
                    ok = bool(result and result.get("ok"))
                    rec.update(status=S.COMPLETED if ok else S.FAILED, result={k: v for k, v in (result or {}).items()
                                                                               if k not in ("capture", "focus")},
                               capture=(result or {}).get("capture"), focus=(result or {}).get("focus"),
                               reason_code=(result or {}).get("reason_code"))
                    if rec["capture"]:
                        self.last_capture_id = rec["capture"]["capture_id"]
                self.active = None
                self.device_state = D.UNKNOWN if err else D.IDLE
                rec["t_device_idle"] = None if err else _now()
                rec["device_state"] = self.device_state
                self.cv.notify_all()

    def _execute(self, rec, token):
        kind, p, hw = rec["kind"], rec["params"], self.hw
        token.check()
        if kind == "aim":
            plan = hw.plan_aim(p)
            rec["plan"] = plan
            if plan.get("refused"):
                return dict(ok=False, reason_code=plan["refused"])
            hw.move(plan["pan"], plan["tilt"], token)
            token.check()
            foc = None
            if p.get("focus", True):
                foc = hw.focus(token, p)
                token.check()
            return self._capture(rec, token, focus=foc)
        if kind == "focus":
            foc = hw.focus(token, p)
            token.check()
            return self._capture(rec, token, focus=foc)
        if kind == "home":
            hw.home(token)
            return dict(ok=True, reason_code="home_done")
        if kind == "diag":
            fn = rec.get("_fn")
            out = fn(token) if fn else None
            return dict(ok=True, reason_code="diag_done", detail=out if isinstance(out, (str, int, float, dict)) else None)
        return dict(ok=False, reason_code=f"unknown_kind:{kind}")

    def _capture(self, rec, token, focus):
        """New exposure after all motion/focus ended: frame seq must advance past the settle point."""
        seq0 = self.hw.frame_seq()
        fr = self.hw.capture_fresh(seq0, self.fresh_timeout_s)
        token.check()
        if not fr or fr.get("seq") is None or fr["seq"] <= seq0 or not fr.get("jpeg"):
            return dict(ok=False, reason_code=FRESH_CAPTURE_UNAVAILABLE, focus=focus,
                        detail=dict(seq_after_settle=seq0, got_seq=(fr or {}).get("seq")))
        cid = SAFE_ID.sub("_", f"cap-{rec['request_id']}-s{fr['seq']}")
        path = os.path.join(self.capture_dir, f"{cid}.jpg")
        if os.path.exists(path):                                  # immutable: never overwrite an earlier capture
            return dict(ok=False, reason_code="capture_id_collision", focus=focus)
        data = fr["jpeg"]
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        cap = dict(capture_id=cid, request_id=rec["request_id"], path=path, frame_seq=fr["seq"],
                   frame_rx_mono_ns=fr.get("rx_mono_ns"), source_t=None,
                   source_t_null_reason="camera driver gives no exposure timestamp; rx time only",
                   saved_wall=time.time(), sha256=hashlib.sha256(data).hexdigest(), width=fr.get("w"), height=fr.get("h"))
        with open(os.path.join(self.capture_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
            json.dump(cap, f, ensure_ascii=False, indent=1)
        focus_ok = focus is None or focus.get("status") == "AF_CONVERGED"
        return dict(ok=True, capture=cap, focus=focus,
                    reason_code="captured" if focus_ok else f"captured_focus_{(focus or {}).get('status')}")


# ---------------------------------------------------------------------- HTTP routing (no sockets here)
def _q1(q, k, default=None):
    v = q.get(k)
    return v[0] if v else default


def route(owner: DeviceOwner, path: str, q: dict):
    """Handle r1-aim-1 paths. Returns (http_code, content_type, body_bytes) or None if not an r1 path.
    /aim and /focus without req are manual (browser) requests and get a generated manual id + owner."""
    js = lambda o, code=200: (code, "application/json", json.dumps(o, ensure_ascii=False, default=str).encode())  # noqa: E731
    if path == "/request":
        return js(owner.get(_q1(q, "id", "")))
    if path == "/cancel":
        return js(owner.cancel(_q1(q, "id", ""), why=_q1(q, "why", "client_cancel"),
                               logical_expiry=_q1(q, "expired", "0") == "1"))
    if path == "/capture":
        p = owner.capture_path(_q1(q, "capture_id", ""))
        if not p:
            return js(dict(protocol=PROTOCOL, status="NOT_FOUND"), 404)
        with open(p, "rb") as f:
            return 200, "image/jpeg", f.read()
    if path in ("/aim", "/focus", "/home"):
        kind = path[1:]
        req, own = _q1(q, "req"), _q1(q, "owner")
        if not req:
            req, own = owner.manual_id(kind)
        params = {}
        if kind == "aim":
            params = dict(u=int(float(_q1(q, "u", "320"))), v=int(float(_q1(q, "v", "240"))),
                          z=float(_q1(q, "z")) if _q1(q, "z") else None, focus=_q1(q, "focus", "1") != "0",
                          strict=_q1(q, "strict", "0") == "1")
        elif kind == "focus":
            params = dict(focus=True)
        return js(owner.submit(kind, req, own, params))
    return None
