"""E5RT2: immutable observation snapshots + the asynchronous scene worker.

ObservationSnapshot binds everything one scene computation is based on: the frame buffer content (frozen copy of the
item list; the arrays themselves are never mutated), the aligned depth image and its stamp, the odometry pose and its
stamp, the odometry session, the boot id, the scene generation and the run.  The scene worker computes on the
snapshot only; the result goes back to the single P2/event owner (the main loop), which decides with
`result_admission` whether the result may still add evidence and whether it may still issue a STOP.

SceneWorker: one scene in flight + the newest pending one (e5rt.scheduler.LatestSlot). An older pending scene is
dropped with a counted reason (latest-first, no backlog).  A logical deadline marks a running scene as overdue; that
does NOT mean the worker stopped (Python threads cannot be killed; concurrent.futures cancel() does not stop a running
call either), so a late result is still received, recorded as late and only used diagnostically.
"""
from __future__ import annotations

import itertools
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from .scheduler import LatestSlot

# result admission reasons (typed, never silently mapped to a normal negative)
OK = "OK"
STALE_SESSION = "STALE_ODOM_SESSION"
STALE_BOOT = "STALE_BOOT"
STALE_GENERATION = "STALE_GENERATION"
LATE_RESULT = "LATE_RESULT_NO_STOP"
RESOLUTION_CHANGED = "RESOLUTION_CHANGED"
NONFINITE_POSE = "NONFINITE_POSE"

_ids = itertools.count(1)


class FrozenFrames:
    """Read-only view of LER.Frames at snapshot time (same lookup rules, independent of later add/pop)."""

    def __init__(self, buf):
        self.keep = getattr(buf, "keep", 4.0)
        self.items = list(getattr(buf, "items", []) or [])
        cls = type(buf)
        self._nearest_tb, self._nearest_th = getattr(cls, "nearest_tb", None), getattr(cls, "nearest_th", None)

    def nearest_tb(self, t):
        return self._nearest_tb(self, t)

    def nearest_th(self, t):
        return self._nearest_th(self, t)

    def newest(self):
        return self.items[-1] if self.items else None

    def add(self, *a, **k):                                   # a frozen view never grows
        raise RuntimeError("FrozenFrames is read-only")


@dataclass(frozen=True)
class ObservationSnapshot:
    snapshot_id: int
    run_id: str
    boot_id: str
    generation: int
    odom_session: Optional[int]
    scene_t: float                       # scene time (s since source origin)
    stamp: float                         # source clock stamp of the scene time
    newest_frame_id: Optional[int]
    newest_header_stamp: Optional[float]
    depth_stamp: Optional[float]
    pose: Optional[tuple]
    pose_stamp: Optional[float]
    image_size: Optional[tuple]          # (w, h) of the newest frame
    created_mono: float
    created_wall: float
    trial_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(snapshot_id=self.snapshot_id, run_id=self.run_id, boot_id=self.boot_id, generation=self.generation,
                    odom_session=self.odom_session, scene_t=self.scene_t, stamp=self.stamp,
                    newest_frame_id=self.newest_frame_id, newest_header_stamp=self.newest_header_stamp,
                    depth_stamp=self.depth_stamp, pose=None if self.pose is None else list(self.pose), pose_stamp=self.pose_stamp,
                    image_size=None if self.image_size is None else list(self.image_size), created_mono=self.created_mono,
                    created_wall=self.created_wall, trial_id=self.trial_id, **({"extra": dict(self.extra)} if self.extra else {}))


def make_snapshot(run_id, boot_id, generation, odom_session, scene_t, stamp, buf, depth_stamp, pose, pose_stamp,
                  trial_id=None, clock=time.monotonic, wall=time.time, **extra):
    newest = buf.newest() if buf is not None else None
    size = None
    if newest is not None and newest.get("bgr") is not None:
        h, w = newest["bgr"].shape[:2]
        size = (int(w), int(h))
    return ObservationSnapshot(snapshot_id=next(_ids), run_id=run_id, boot_id=boot_id, generation=int(generation),
                               odom_session=odom_session, scene_t=float(scene_t), stamp=float(stamp),
                               newest_frame_id=None if newest is None else int(newest["id"]),
                               newest_header_stamp=None if newest is None else float(newest["th"]),
                               depth_stamp=depth_stamp, pose=None if pose is None else tuple(float(v) for v in pose),
                               pose_stamp=pose_stamp, image_size=size, created_mono=clock(), created_wall=wall(),
                               trial_id=trial_id, extra=dict(extra))


def result_admission(snap, current, now_mono, max_stop_age_s=1.5):
    """Owner-side check of a finished scene. current = dict(boot_id, generation, odom_session, image_size).
    Returns dict(evidence_ok, stop_ok, reasons, age_s). evidence_ok: may update world-anchored tracks (they carry the
    snapshot pose, which is correct for THAT time). stop_ok: may still propose a STOP from the scene itself (only for a
    fresh result; P2 STOPs come from the timing tick with the latest odometry anyway)."""
    reasons = []
    if snap.boot_id != current.get("boot_id"):
        reasons.append(STALE_BOOT)
    if snap.generation != current.get("generation"):
        reasons.append(STALE_GENERATION)
    cs = current.get("odom_session")
    if snap.odom_session is not None and cs is not None and snap.odom_session != cs:
        reasons.append(STALE_SESSION)
    cur_size = current.get("image_size")
    if snap.image_size is not None and cur_size is not None and tuple(snap.image_size) != tuple(cur_size):
        reasons.append(RESOLUTION_CHANGED)
    if snap.pose is None or not all(math.isfinite(v) for v in snap.pose):
        reasons.append(NONFINITE_POSE)
    age = float(now_mono - snap.created_mono)
    evidence_ok = not reasons
    stop_ok = evidence_ok and age <= max_stop_age_s
    if evidence_ok and not stop_ok:
        reasons.append(LATE_RESULT)
    return dict(evidence_ok=evidence_ok, stop_ok=stop_ok, reasons=reasons or [OK], age_s=age)


class SceneWorker:
    """Runs fn(job) for the newest submitted job in a worker thread. Results are queued for the owner thread.
    fn must not touch owner state (FSM, P2 tracks, event store)."""

    def __init__(self, fn, name="e5-scene-worker", deadline_s=10.0, clock=time.monotonic):
        self.fn, self.deadline_s, self.clock = fn, float(deadline_s), clock
        self.slot = LatestSlot()
        self.results = queue.Queue()
        self.counters = dict(submitted=0, superseded=0, completed=0, errors=0, overdue=0, late_completed=0)
        self._running = None                       # dict(id, t0) of the job in flight
        self._overdue_ids = set()
        self._stop = False
        self._lk = threading.Lock()
        self._th = threading.Thread(target=self._run, name=name, daemon=True)
        self._th.start()

    def submit(self, job):
        """job: dict with 'id'. Replaces a not-yet-started pending job (counted as superseded)."""
        self.slot.put(job, why="superseded_by_newer_scene")
        with self._lk:
            self.counters["submitted"] += 1
            self.counters["superseded"] = self.slot.n_dropped
        return True

    def busy(self):
        return self.slot.depth() > 0

    def in_flight(self):
        with self._lk:
            return None if self._running is None else dict(self._running)

    def check_deadline(self):
        """Owner side: mark the running job as logically overdue (the thread keeps running; its result will be late)."""
        with self._lk:
            r = self._running
            if r is not None and r["id"] not in self._overdue_ids and self.clock() - r["t0"] > self.deadline_s:
                self._overdue_ids.add(r["id"])
                self.counters["overdue"] += 1
                return dict(r, overdue_s=self.clock() - r["t0"])
        return None

    def _run(self):
        while not self._stop:
            job = self.slot.take(timeout=0.2)
            if job is None:
                continue
            t0 = self.clock()
            with self._lk:
                self._running = dict(id=job.get("id"), t0=t0)
            out = dict(id=job.get("id"), job=job, t_start=t0)
            try:
                out["result"] = self.fn(job)
                out["ok"] = True
            except BaseException as e:             # noqa: BLE001  the owner decides what an error means
                out.update(ok=False, error=e, error_repr=repr(e), trace=traceback.format_exc()[-1500:])
            out["t_end"] = self.clock()
            with self._lk:
                self._running = None
                late = job.get("id") in self._overdue_ids
                self._overdue_ids.discard(job.get("id"))
                self.counters["completed" if out["ok"] else "errors"] += 1
                if late:
                    self.counters["late_completed"] += 1
            out["late"] = late
            self.results.put(out)
            self.slot.done(job.get("id"))

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def stats(self):
        with self._lk:
            return dict(self.counters, depth=self.slot.depth(), dropped_log=list(self.slot.dropped[-5:]),
                        in_flight=None if self._running is None else dict(self._running, age_s=self.clock() - self._running["t0"]))

    def close(self, join_s=2.0):
        self._stop = True
        self._th.join(timeout=join_s)
        return not self._th.is_alive()
