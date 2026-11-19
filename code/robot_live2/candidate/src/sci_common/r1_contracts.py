"""R1 shared runtime contracts (pure stdlib; no device/ROS/model access).

Identity, clocks, atomic/append-only records, generation guard, capture/release states, reason codes and the
r1-aim-1 console protocol shared by E5 (client), aim_console (server), reader and the live viewer.
Fields that do not apply to a record are null with a `<field>_null_reason`, never synthesised.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

SCHEMA_VERSION = "r1.1"

# ---------------------------------------------------------------------------------------------------------- clocks
CLOCK_MONO = "monotonic_ns(boot)"      # deadlines: same boot only
CLOCK_WALL = "wall_unix_s"             # logs only
CLOCK_ROS = "ros_header_s"             # sensor header stamps (sync with wall NOT verified)
CLOCK_BROWSER = "browser_ms"           # never subtracted from robot clocks without calibration


def mono_ns() -> int:
    return time.monotonic_ns()


def boot_id() -> str:
    """Kernel boot id; monotonic values are only comparable within one boot_id."""
    try:
        with open("/proc/sys/kernel/random/boot_id", "r") as f:
            return f.read().strip()
    except OSError:
        return "UNKNOWN_BOOT"


BOOT_ID = boot_id()


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def stamp(**extra) -> dict:
    """Common time block: monotonic for deadlines + wall for logs, both labelled."""
    return dict(boot_id=BOOT_ID, mono_ns=mono_ns(), wall=time.time(), **extra)


def null(field: str, reason: str) -> dict:
    return {field: None, f"{field}_null_reason": reason}


# --------------------------------------------------------------------------------------------- atomic records
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path) -> threading.Lock:
    k = str(Path(path).resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(k, threading.Lock())


def atomic_write_json(path, obj) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace (atomic on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = json.dumps(obj, ensure_ascii=False, default=str, indent=1).encode()
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, rec) -> None:
    """Append-only history with an explicit per-file lock (single process); one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ------------------------------------------------------------------------------------------ trial generation
TRIAL_FILE = "trial_current.json"     # written by the live viewer at [start]/[stop]/[cancel]; read by everyone else


def trial_record(trial_id, exp_id, generation_id, state, **kw) -> dict:
    return dict(schema_version=SCHEMA_VERSION, trial_id=trial_id, exp_id=exp_id, generation_id=int(generation_id),
                state=state, **stamp(), **kw)


class GenerationGuard:
    """Results carry (trial_id, generation_id) captured at compute start. They may be published as *current* only if
    the pointer still names the same generation both before compute and immediately before publish."""

    def __init__(self, pointer_path):
        self.pointer_path = Path(pointer_path)
        self._lock = threading.Lock()

    def current(self) -> dict | None:
        return read_json(self.pointer_path)

    def token(self) -> dict | None:
        cur = self.current()
        if not cur:
            return None
        return dict(trial_id=cur.get("trial_id"), generation_id=cur.get("generation_id"))

    def still_current(self, token) -> bool:
        cur = self.current()
        return bool(token and cur and cur.get("trial_id") == token.get("trial_id")
                    and cur.get("generation_id") == token.get("generation_id") and cur.get("state") != "CANCELLED")

    def publish_if_current(self, token, publish_fn, discard_fn=None):
        """Check-and-publish under one lock (single process). Returns True if published."""
        with self._lock:
            if self.still_current(token):
                publish_fn()
                return True
        if discard_fn:
            discard_fn()
        return False


# ------------------------------------------------------------------------------------------- event / capture
class CaptureStatus:
    RAW_CAPTURE_SAVED = "RAW_CAPTURE_SAVED"
    CAPTURE_QUALITY_OK = "CAPTURE_QUALITY_OK"
    CAPTURE_QUALITY_FAIL = "CAPTURE_QUALITY_FAIL"
    CAPTURE_PROXY = "CAPTURE_PROXY"                       # RealSense diagnostic proxy, pantilt_ok=False, never auto-consumed
    FRESH_CAPTURE_UNAVAILABLE = "FRESH_CAPTURE_UNAVAILABLE"
    NO_CAPTURE = "NO_CAPTURE"


class SemanticStatus:
    SEMANTIC_ACCEPTED = "SEMANTIC_ACCEPTED"
    SEMANTIC_REJECTED = "SEMANTIC_REJECTED"


class ReleaseStatus:
    RELEASE_READY_CAPTURE_OK = "RELEASE_READY_CAPTURE_OK"
    RELEASE_READY_OBSERVATION_FAILED = "RELEASE_READY_OBSERVATION_FAILED"
    RELEASE_BLOCKED_ACTUATOR_UNKNOWN = "RELEASE_BLOCKED_ACTUATOR_UNKNOWN"


class ReadReason:
    CAPTURE_QUALITY_FAIL = "CAPTURE_QUALITY_FAIL"
    NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
    PARSE_REJECTED = "PARSE_REJECTED"
    OCR_ONLY = "OCR_ONLY"
    VLM_NOT_NEEDED = "VLM_NOT_NEEDED"
    DEFERRED_RESOURCE = "DEFERRED_RESOURCE"
    SKIPPED_RESOURCE = "SKIPPED_RESOURCE"
    VLM_TIMEOUT = "VLM_TIMEOUT"
    READING_CONFLICT = "READING_CONFLICT"
    CANCELLED = "CANCELLED"
    INPUT_INVALID = "INPUT_INVALID"
    OCR_ENGINE_ERROR = "OCR_ENGINE_ERROR"              # E5RT2: OCR raised (bounded retries, then terminal; never infinite)
    LEGACY_NO_ROOM_LABEL = "NO_ROOM_LABEL"            # legacy value: analysis-only, never produced by R1 paths


TERMINAL_READ = {ReadReason.OCR_ENGINE_ERROR, ReadReason.CAPTURE_QUALITY_FAIL, ReadReason.NO_TEXT_DETECTED, ReadReason.PARSE_REJECTED,
                 ReadReason.OCR_ONLY, ReadReason.VLM_NOT_NEEDED, ReadReason.SKIPPED_RESOURCE, ReadReason.VLM_TIMEOUT,
                 ReadReason.READING_CONFLICT, ReadReason.CANCELLED, ReadReason.INPUT_INVALID}


class GeoReason:
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    MAP_CONTRACT_UNPROVEN = "MAP_CONTRACT_UNPROVEN"
    NOT_WIRED = "NOT_WIRED"
    GEOMETRY_UNAVAILABLE = "GEOMETRY_UNAVAILABLE"
    NO_FEASIBLE_STOP_WINDOW = "NO_FEASIBLE_STOP_WINDOW"
    IO_FAILURE = "IO_FAILURE"
    RESOURCE_LIMITED = "RESOURCE_LIMITED"


class FocusStatus:
    FOCUS_PROFILE_UNVERIFIED = "FOCUS_PROFILE_UNVERIFIED"
    AF_CONVERGED = "AF_CONVERGED"
    AF_NOT_CONVERGED = "AF_NOT_CONVERGED"
    AF_BUDGET_EXHAUSTED = "AF_BUDGET_EXHAUSTED"
    AF_TARGET_LOST = "AF_TARGET_LOST"
    AF_LOW_EXPOSURE = "AF_LOW_EXPOSURE"
    AF_BUDGET_INFEASIBLE = "AF_BUDGET_INFEASIBLE"


# ----------------------------------------------------------------------------------------- r1-aim-1 protocol
AIM_PROTOCOL = "r1-aim-1"


class ReqStatus:
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    REJECTED_BUSY = "REJECTED_BUSY"
    UNKNOWN = "UNKNOWN"


REQ_TERMINAL = {ReqStatus.COMPLETED, ReqStatus.FAILED, ReqStatus.CANCELLED, ReqStatus.REJECTED_BUSY}


class DeviceState:
    IDLE = "IDLE"
    BUSY = "BUSY"
    CANCELLING = "CANCELLING"
    UNKNOWN = "UNKNOWN"


PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"

AIM_PROTOCOL_SPEC = """
r1-aim-1 (HTTP GET, JSON replies; every reply carries "protocol": "r1-aim-1"):
  /state                         -> {protocol, device_state, active_request, ...legacy fields}
  /aim?u&v&z&focus&strict&req&owner
                                 -> {protocol, request_id, owner, status: ACCEPTED|REJECTED_BUSY, duplicate: bool}
                                    same req again = idempotent (no second motor sequence), returns current record
  /home?req&owner                -> same as /aim (HOME is a device job owned by the same serial owner)
  /request?id                    -> {protocol, request_id, owner, kind, status, device_state, hw_done: bool,
                                     t_accept/t_start/t_end (mono_ns + wall), capture: {capture_id, path, frame_seq,
                                     frame_rx_mono_ns, source_t_null_reason, saved_wall, sha256, width, height} | null,
                                     focus: {...} | null, reason_code}
  /cancel?id                     -> logical cancel; status CANCELLING until the hardware step really ended, then CANCELLED
  /capture?capture_id            -> immutable JPEG bytes of that capture (never the global full.jpg)
A console without "protocol": "r1-aim-1" in /state is a legacy console: automatic observation is refused with
PROTOCOL_MISMATCH (read-only /state display stays compatible). No job-counter fallback exists.
"""
