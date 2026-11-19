"""R1 E5 event store: one durable record per observation event, independent of the FSM being active.

Single writer (the E5 event loop thread). Every change is an atomic JSON replace of the event record plus an
append-only line in history.jsonl. States never move backward; late/duplicate results are kept as history only.
Reader results attach ONLY to the event whose event_key AND capture_id match (no 'last event' fallback).
Hot memory (for the UI) is bounded; the disk record stays exact.
"""
from __future__ import annotations

import json
import sys
import threading
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sci_common"))
import r1_contracts as C  # noqa: E402

# monotone rank: a transition to a lower rank is refused (logged as rejected history)
RANK = {"STOP_REQUEST": 0, "WAIT_BASE_STOP": 1, "BASE_STOPPED": 2, "WAIT_IMAGE_STABLE": 3, "AIM": 4, "CAPTURE": 5,
        C.CaptureStatus.RAW_CAPTURE_SAVED: 6, C.CaptureStatus.CAPTURE_QUALITY_OK: 7, C.CaptureStatus.CAPTURE_QUALITY_FAIL: 7,
        C.CaptureStatus.CAPTURE_PROXY: 7, C.CaptureStatus.FRESH_CAPTURE_UNAVAILABLE: 7, C.CaptureStatus.NO_CAPTURE: 7,
        "FAILED": 8, "HOME": 8, C.ReleaseStatus.RELEASE_BLOCKED_ACTUATOR_UNKNOWN: 8,
        C.ReleaseStatus.RELEASE_READY_OBSERVATION_FAILED: 9, C.ReleaseStatus.RELEASE_READY_CAPTURE_OK: 9,
        "READ_ASYNC": 10, "READ_DONE": 11, "CLOSED": 12}
# RELEASE_BLOCKED may later resolve to a RELEASE_READY_* once the device is confirmed idle (rank 8 -> 9)
RELEASE_STATES = (C.ReleaseStatus.RELEASE_BLOCKED_ACTUATOR_UNKNOWN, C.ReleaseStatus.RELEASE_READY_OBSERVATION_FAILED,
                  C.ReleaseStatus.RELEASE_READY_CAPTURE_OK)


def safe_key(event_key: str) -> str:
    return event_key.replace("/", "_").replace("#", "__")


def trial_from_pointer(path) -> dict:
    """trial_current.json written by the live viewer (W-A). Absent -> nulls with reason (never synthesised)."""
    if path is None:
        return dict(**C.null("trial_id", "no trial pointer configured"), **C.null("generation_id", "no trial pointer configured"))
    d = C.read_json(path)
    if not d:
        return dict(**C.null("trial_id", f"trial pointer absent: {path}"), **C.null("generation_id", f"trial pointer absent: {path}"))
    return dict(trial_id=d.get("trial_id"), generation_id=d.get("generation_id"), exp_id=d.get("exp_id"), trial_state=d.get("state"))


class EventStore:
    def __init__(self, run_dir, run_id, trial_pointer=None, hot_max=20):
        self.dir = Path(run_dir) / "events_r1"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.trial_pointer, self.hot_max = run_id, trial_pointer, hot_max
        self.hist_path = self.dir / "history.jsonl"
        self._hot = OrderedDict()                 # event_key -> record (bounded)
        self._lock = threading.Lock()
        self._owner = None                        # thread ident of the single writer (set on first write)

    # ------------------------------------------------------------------------------------------- internals
    def _check_owner(self):
        me = threading.get_ident()
        if self._owner is None:
            self._owner = me
        elif self._owner != me:
            raise RuntimeError("EventStore: second writer thread (single-writer contract)")

    def path(self, key):
        return self.dir / f"{safe_key(key)}.json"

    def get(self, key):
        r = self._hot.get(key)
        if r is None:
            r = C.read_json(self.path(key))
        return r

    def _save(self, rec, hist):
        C.atomic_write_json(self.path(rec["event_key"]), rec)
        C.append_jsonl(self.hist_path, dict(event_key=rec["event_key"], **C.stamp(), **hist))
        self._hot[rec["event_key"]] = rec
        self._hot.move_to_end(rec["event_key"])
        while len(self._hot) > self.hot_max:     # hot memory bounded; disk record stays
            self._hot.popitem(last=False)

    # ---------------------------------------------------------------------------------------------- API
    def new_event(self, ev_num, stop, frame_meta=None, track_id=None):
        with self._lock:
            self._check_owner()
            key = f"{self.run_id}#ev{ev_num:03d}"
            rec = dict(schema_version=C.SCHEMA_VERSION, boot_id=C.BOOT_ID, run_id=self.run_id, event_id=ev_num,
                       event_key=key, track_id=track_id, **trial_from_pointer(self.trial_pointer),
                       state="STOP_REQUEST", terminal=False, reason_code="policy_pass", result_revision=0,
                       created=C.stamp(), frame=frame_meta or C.null("frame", "no frame metadata"),
                       stop={k: stop.get(k) for k in ("t", "stage", "spot", "box", "z", "sim", "nobs", "advance_m")},
                       request_id=None, attempt_id=None, capture_id=None, capture_status=None, release=None,
                       reading=None, reading_status=None, late_results=[])
            self._save(rec, dict(kind="NEW", state="STOP_REQUEST"))
            return rec

    def transition(self, key, state, reason, terminal=None, **fields):
        """Move event to `state` unless that would go backward. Returns True if applied."""
        with self._lock:
            self._check_owner()
            rec = self.get(key)
            if rec is None:
                return False
            cur = rec["state"]
            if RANK.get(state, 0) < RANK.get(cur, 0):
                if state in RELEASE_STATES:
                    # E5RT2: device release and reading are different axes. A reading may finish (READ_DONE) while HOME
                    # is still running; the later RELEASE must still be recorded (fields + history) without moving the
                    # event state backward. E5-RT1/R1 dropped it (state stayed READ_DONE, release=None).
                    rec.update(fields)
                    rec.update(release_state=state, release_reason=reason, result_revision=rec["result_revision"] + 1)
                    if terminal is not None:
                        rec["terminal"] = bool(terminal)
                    self._save(rec, dict(kind="RELEASE_RECORDED_STATE_KEPT", state_kept=cur, release=state, reason=reason))
                    return True
                C.append_jsonl(self.hist_path, dict(event_key=key, **C.stamp(), kind="REJECTED_BACKWARD", from_=cur,
                                                    to=state, reason=reason))
                return False
            if state in RELEASE_STATES:
                fields = dict(fields, release_state=state, release_reason=reason)
            rec.update(fields)
            rec.update(state=state, reason_code=reason, result_revision=rec["result_revision"] + 1)
            if terminal is not None:
                rec["terminal"] = bool(terminal)
            self._save(rec, dict(kind="TRANSITION", from_=cur, to=state, reason=reason, fields=sorted(fields)))
            return True

    def set_fields(self, key, **fields):
        with self._lock:
            self._check_owner()
            rec = self.get(key)
            if rec is None:
                return False
            rec.update(fields)
            rec["result_revision"] += 1
            self._save(rec, dict(kind="FIELDS", fields=sorted(fields)))
            return True

    def attach_result(self, res: dict) -> str:
        """Attach a reader result. Matching: exact event_key and exact capture_id. Returns an outcome code."""
        with self._lock:
            self._check_owner()
            key, cap = res.get("event_key"), res.get("capture_id")
            if not key or not cap:
                return "IDENTITY_MISSING"
            rec = self.get(key)
            if rec is None:
                return "NO_SUCH_EVENT"
            if rec.get("capture_id") != cap:
                C.append_jsonl(self.hist_path, dict(event_key=key, **C.stamp(), kind="RESULT_CAPTURE_MISMATCH",
                                                    got=cap, expected=rec.get("capture_id")))
                return "CAPTURE_MISMATCH"
            if rec.get("observation_id") and res.get("observation_id") and res["observation_id"] != rec["observation_id"]:
                C.append_jsonl(self.hist_path, dict(event_key=key, **C.stamp(), kind="RESULT_OBSERVATION_MISMATCH",
                                                    got=res.get("observation_id"), expected=rec.get("observation_id")))
                return "OBSERVATION_MISMATCH"                     # E5RT2: a reading of another hand-off is never attached
            if rec.get("payload_sha256") and res.get("payload_sha256") and res["payload_sha256"] != rec["payload_sha256"]:
                C.append_jsonl(self.hist_path, dict(event_key=key, **C.stamp(), kind="RESULT_PAYLOAD_MISMATCH",
                                                    got=res.get("payload_sha256"), expected=rec.get("payload_sha256")))
                return "PAYLOAD_MISMATCH"                         # SEM1 P02: reading of a different geometry revision
            new_rev = int(res.get("result_revision") or 0)
            old = rec.get("reading") or {}
            old_rev = int(old.get("result_revision") or -1)
            if new_rev <= old_rev:
                return "DUPLICATE_OR_STALE"
            if old.get("terminal") and not res.get("terminal"):
                rec["late_results"] = (rec.get("late_results") or []) + [dict(revision=new_rev, reason=res.get("reason_code"))]
                self._save(rec, dict(kind="LATE_RESULT_KEPT_AS_HISTORY", revision=new_rev))
                return "LATE_AFTER_TERMINAL"
            rec["reading"] = res
            rec["reading_status"] = res.get("reason_code")
            rec["result_revision"] += 1
            if res.get("terminal") and RANK.get(rec["state"], 0) <= RANK["READ_DONE"]:
                rec["state"] = "READ_DONE"
            self._save(rec, dict(kind="READING", revision=new_rev, reason=res.get("reason_code"), terminal=res.get("terminal")))
            return "ATTACHED"

    def scan_done(self, done_dir) -> dict:
        """Attach every reader result file in done_dir by its CONTENT identity (file names are not trusted)."""
        out = {}
        for p in sorted(Path(done_dir).glob("*.json")):
            try:
                res = json.loads(p.read_text())
            except (OSError, ValueError):
                out[p.name] = "UNREADABLE"
                continue
            if "event_key" not in res:
                out[p.name] = "LEGACY_NO_IDENTITY"          # legacy results are analysis-only, never attached
                continue
            out[p.name] = self.attach_result(res)
        return out

    def hot(self):
        return [dict(r) for r in self._hot.values()]
