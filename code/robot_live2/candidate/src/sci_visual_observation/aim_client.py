"""R1 aim-console client speaking r1-aim-1 (see sci_common/r1_contracts.AIM_PROTOCOL_SPEC).

- every request carries a request_id bound to run/event/attempt and an owner string; success is decided ONLY by the
  /request record of that request_id (never by a global job counter, never by the global full.jpg)
- a console whose /state does not announce r1-aim-1 is refused with PROTOCOL_MISMATCH (no legacy fallback)
- a logical deadline sends /cancel; the device is considered idle only when the console reports the request
  terminal AND device_state IDLE (hw_done). Otherwise the result says device_state UNKNOWN/CANCELLING.
- the HTTP transport is injected (http_get(url, timeout) -> bytes) so tests never touch a real port.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sci_common"))
import r1_contracts as C  # noqa: E402


def urllib_get(url, timeout):
    return urllib.request.urlopen(url, timeout=timeout).read()


class AimClient:
    def __init__(self, console, http_get=urllib_get, clock=time.monotonic, sleep=time.sleep):
        self.console = console.rstrip("/")
        self.http_get, self.clock, self.sleep = http_get, clock, sleep

    # --------------------------------------------------------------------------------------- transport
    def _get_json(self, path, params=None, timeout=2.0):
        url = self.console + path + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            raw = self.http_get(url, timeout)
        except Exception as e:                      # unreachable / timeout -> explicit error record
            return dict(_error=f"{type(e).__name__}: {e}", _url=url)
        try:
            d = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
            return d if isinstance(d, dict) else dict(_error="non-object reply", _raw=str(raw)[:200])
        except (ValueError, UnicodeDecodeError):
            return dict(_error="non-JSON reply (legacy console?)", _raw=str(raw)[:200], _url=url)

    @staticmethod
    def new_request_id(run_id, event_id, attempt):
        return f"{run_id}:ev{int(event_id):03d}:a{int(attempt)}:{uuid.uuid4().hex[:8]}"

    def probe(self):
        st = self._get_json("/state", timeout=1.5)
        if "_error" in st:
            return dict(ok=False, reason_code="CONSOLE_UNREACHABLE", detail=st)
        if st.get("protocol") != C.AIM_PROTOCOL:
            return dict(ok=False, reason_code=C.PROTOCOL_MISMATCH, detail=dict(protocol=st.get("protocol")))
        return dict(ok=True, state=st)

    # ------------------------------------------------------------------------------------------- jobs
    def run(self, kind, params, run_id, event_id, attempt, deadline_s, poll_s=0.25, cancel_grace_s=5.0):
        """Submit one device job ('aim' or 'home') and follow it to a terminal state or the logical deadline.
        Returns a dict with request_id, status, hw_done, device_state, capture, focus, reason_code, logical_expired."""
        rid = self.new_request_id(run_id, event_id, attempt)
        owner = f"e5:{run_id}:ev{int(event_id):03d}:a{int(attempt)}"
        base = dict(request_id=rid, owner=owner, kind=kind, attempt_id=int(attempt), t_submit=C.stamp(),
                    logical_expired=False, hw_done=False, device_state=C.DeviceState.UNKNOWN, capture=None, focus=None)
        pr = self.probe()
        if not pr["ok"]:
            return dict(base, status=C.ReqStatus.FAILED, reason_code=pr["reason_code"], detail=pr.get("detail"),
                        device_state=C.DeviceState.IDLE if pr["reason_code"] == C.PROTOCOL_MISMATCH else C.DeviceState.UNKNOWN,
                        hw_done=pr["reason_code"] == C.PROTOCOL_MISMATCH, submitted=False)
        q = dict(params, req=rid, owner=owner)
        acc = self._get_json("/" + kind, q, timeout=5.0)
        if acc.get("protocol") != C.AIM_PROTOCOL or acc.get("request_id") != rid:
            # nothing is assumed about the device: the request may or may not have started
            return dict(base, status=C.ReqStatus.UNKNOWN, reason_code=C.PROTOCOL_MISMATCH if "_error" not in acc else "SUBMIT_ERROR",
                        detail=acc, submitted=None)
        if acc.get("status") == C.ReqStatus.REJECTED_BUSY:
            return dict(base, status=C.ReqStatus.REJECTED_BUSY, reason_code="CONSOLE_BUSY", device_state=acc.get("device_state") or C.DeviceState.BUSY,
                        hw_done=True, submitted=False, detail=acc)
        t0 = self.clock()
        rec = acc
        while True:
            rec = self._get_json("/request", dict(id=rid))
            if rec.get("request_id") == rid and rec.get("status") in C.REQ_TERMINAL:
                return self._final(base, rec, submitted=True)
            if self.clock() - t0 >= deadline_s:
                break
            self.sleep(poll_s)
        # logical deadline: ask for cancellation, then wait a bounded grace for the HARDWARE step to end
        self._get_json("/cancel", dict(id=rid, why="e5_logical_deadline", expired=1))
        t1 = self.clock()
        while self.clock() - t1 < cancel_grace_s:
            rec = self._get_json("/request", dict(id=rid))
            if rec.get("request_id") == rid and rec.get("status") in C.REQ_TERMINAL:
                out = self._final(base, rec, submitted=True)
                out.update(logical_expired=True, reason_code="LOGICAL_TIMEOUT_" + str(rec.get("status")))
                return out
            self.sleep(poll_s)
        return dict(base, status=rec.get("status", C.ReqStatus.UNKNOWN) if rec.get("request_id") == rid else C.ReqStatus.UNKNOWN,
                    logical_expired=True, hw_done=False,
                    device_state=(C.DeviceState.UNKNOWN if rec.get("device_state") in (None, C.DeviceState.IDLE)
                                  else rec.get("device_state")),
                    reason_code="LOGICAL_TIMEOUT_DEVICE_NOT_CONFIRMED", submitted=True, detail=rec)

    @staticmethod
    def _final(base, rec, submitted):
        idle = bool(rec.get("hw_done")) and rec.get("device_state") == C.DeviceState.IDLE
        cap = rec.get("capture")
        if cap is not None and cap.get("request_id", rec.get("request_id")) != rec.get("request_id"):
            cap = None                               # capture of another request is never accepted
        return dict(base, status=rec["status"], hw_done=bool(rec.get("hw_done")), device_state=rec.get("device_state", C.DeviceState.UNKNOWN),
                    device_idle_confirmed=idle, capture=cap, focus=rec.get("focus"), reason_code=rec.get("reason_code") or rec["status"],
                    submitted=submitted, record=rec)

    def fetch_capture(self, capture_id, timeout=5.0):
        """Immutable bytes of exactly this capture. Never the global full.jpg."""
        if not capture_id:
            return None
        try:
            return self.http_get(self.console + "/capture?" + urllib.parse.urlencode(dict(capture_id=capture_id)), timeout)
        except Exception:
            return None
