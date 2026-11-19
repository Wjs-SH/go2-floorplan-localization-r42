"""R1 execution helpers for the E5 node.

DeviceWorker  : the single owner of pan/tilt/focus/HOME work on the E5 side. One job at a time, no queueing of a
                second physical job (submit returns False while busy). Results go to a result queue that only the
                FSM-owner thread drains. A Future-style timeout is never treated as 'device stopped'.
LatestSlot    : bounded scene work = newest pending item (1) + item in progress (1). Older pending items are dropped
                with a counted reason (never silently).
"""
from __future__ import annotations

import queue
import threading
import time
import traceback


class DeviceWorker:
    def __init__(self, name="e5-device"):
        self.results = queue.Queue()
        self._lock = threading.Lock()
        self._busy_key = None
        self._q = queue.Queue(maxsize=1)
        self._th = threading.Thread(target=self._run, name=name, daemon=True)
        self._stop = False
        self._th.start()

    @property
    def busy(self):
        with self._lock:
            return self._busy_key is not None

    @property
    def busy_key(self):
        with self._lock:
            return self._busy_key

    def submit(self, key, fn, *a, **k) -> bool:
        """Start a physical job for event `key`. Refused (False) while any job is pending or running."""
        with self._lock:
            if self._busy_key is not None:
                return False
            self._busy_key = key
        self._q.put((key, fn, a, k))
        return True

    def _run(self):
        while not self._stop:
            try:
                key, fn, a, k = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.monotonic_ns()
            try:
                res = fn(*a, **k)
                self.results.put(dict(key=key, ok=True, result=res, t_start_ns=t0, t_end_ns=time.monotonic_ns()))
            except Exception as e:                    # the job itself decides device state; an exception = UNKNOWN
                self.results.put(dict(key=key, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:],
                                      t_start_ns=t0, t_end_ns=time.monotonic_ns()))
            finally:
                with self._lock:
                    self._busy_key = None

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._stop = True


class LatestSlot:
    """Pending slot of size 1 (newest wins) next to the item in progress; drops are counted with a reason."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None
        self.in_progress = None
        self.dropped = 0
        self.drop_log = []                            # bounded

    def put(self, item, why="superseded_by_newer"):
        with self._lock:
            if self._pending is not None:
                self.dropped += 1
                self.drop_log = (self.drop_log + [dict(dropped=self._pending.get("id"), why=why, t=time.time())])[-50:]
            self._pending = item

    def take(self):
        with self._lock:
            item, self._pending = self._pending, None
            self.in_progress = item
            return item

    def done(self):
        with self._lock:
            self.in_progress = None

    def depth(self):
        with self._lock:
            return int(self._pending is not None) + int(self.in_progress is not None)
