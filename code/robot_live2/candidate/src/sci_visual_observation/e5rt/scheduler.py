"""SigLIP verification scheduling for tracked targets (pure Python).

Decides per scene which regions really need a fresh image-encoder forward and which can show the last result of the
same track. A cache hit is DISPLAY ONLY: it is returned with cached=True and the evidence layer never counts it.

Fresh verification is required when:
  NEW            track never verified (or no track association)
  INVALIDATED    model / prompt / preprocess revision, camera, odom session or generation changed
  IMPROVED_SCALE crop grew by >= scale_gain (closer view = new information)
  IMPROVED_SHARP sharpness grew by >= sharp_gain (same metric, same resolution)
  APPEARANCE     crop signature changed (box aspect / brightness jump: occlusion, re-entry, ID swap suspicion)
  AGED           last fresh verification older than max_age_s
Otherwise CACHED_FRESH. A known negative gets a short cooldown that expires (re-checked on change or age).
Budget: at most `max_fresh` fresh REGIONS per scene; a region can generate multiple neural crops. Continuously
waiting work gains priority over new arrivals. A separate total neural-crop budget is still required. Everything not served is recorded with a reason.
"""
from __future__ import annotations

import math
import threading
import time

NEW, INVALIDATED, IMPROVED_SCALE, IMPROVED_SHARP, APPEARANCE, AGED, CACHED_FRESH = (
    "NEW", "INVALIDATED", "IMPROVED_SCALE", "IMPROVED_SHARP", "APPEARANCE", "AGED", "CACHED_FRESH")
FRESH_REASONS = (NEW, INVALIDATED, IMPROVED_SCALE, IMPROVED_SHARP, APPEARANCE, AGED)


def crop_signature(box, sharp=None, mean_gray=None):
    w, h = max(1e-6, box[2] - box[0]), max(1e-6, box[3] - box[1])
    return dict(w=float(w), h=float(h), aspect=float(w / h), sharp=None if sharp is None else float(sharp),
                mean=None if mean_gray is None else float(mean_gray))


class VerifyScheduler:
    def __init__(self, revision, max_fresh=8, max_age_s=2.0, scale_gain=1.25, sharp_gain=1.5, aspect_jump=1.35,
                 mean_jump=40.0, neg_cooldown_s=3.0, starve_s=3.0, clock=time.monotonic):
        self.revision = dict(revision)                 # model/prompt/preprocess/camera/odom_session/generation
        self.max_fresh, self.max_age_s = max_fresh, max_age_s
        self.scale_gain, self.sharp_gain, self.aspect_jump, self.mean_jump = scale_gain, sharp_gain, aspect_jump, mean_jump
        self.neg_cooldown_s, self.starve_s = neg_cooldown_s, starve_s
        self.clock = clock
        self.cache = {}                                # track_id -> dict(t, sig, result, revision, first_wait)
        self._pending_since = {}                       # continuously waiting requests, not last fresh time
        self.counters = dict(fresh=0, cached=0, deferred_budget=0, invalidated=0)
        self.drops = []                                # bounded log of not-served requests

    def set_revision(self, **kw):
        changed = {k: v for k, v in kw.items() if self.revision.get(k) != v}
        if changed:
            self.revision.update(changed)
            self.counters["invalidated"] += len(self.cache)
            self.cache.clear()                         # nothing computed under the old revision is reusable
            self._pending_since.clear()
        return changed

    def need(self, track_id, sig, now):
        if track_id is None:
            return NEW
        c = self.cache.get(track_id)
        if c is None:
            return NEW
        if c["revision"] != self.revision:
            return INVALIDATED
        o = c["sig"]
        if sig["w"] * sig["h"] >= self.scale_gain ** 2 * o["w"] * o["h"]:
            return IMPROVED_SCALE
        if sig.get("sharp") is not None and o.get("sharp") and sig["sharp"] >= self.sharp_gain * o["sharp"]:
            return IMPROVED_SHARP
        a = max(sig["aspect"], o["aspect"]) / max(1e-6, min(sig["aspect"], o["aspect"]))
        if a >= self.aspect_jump or (sig.get("mean") is not None and o.get("mean") is not None
                                     and abs(sig["mean"] - o["mean"]) >= self.mean_jump):
            return APPEARANCE
        age = now - c["t"]
        limit = self.neg_cooldown_s if c.get("negative") else self.max_age_s
        if age >= limit:
            return AGED
        return CACHED_FRESH

    def plan(self, items, now=None):
        """items: list of dict(key, track_id, sig, side). Returns (fresh_keys, cached_keys, decisions{key: reason})."""
        now = self.clock() if now is None else now
        dec = {it["key"]: self.need(it.get("track_id"), it["sig"], now) for it in items}
        want = [it for it in items if dec[it["key"]] in FRESH_REASONS]

        wanted_ids = {it.get("track_id") for it in want if it.get("track_id") is not None}
        self._pending_since = {k: v for k, v in self._pending_since.items() if k in wanted_ids}
        for it in want:
            if it.get("track_id") is not None:
                self._pending_since.setdefault(it["track_id"], now)

        def prio(it):
            since = self._pending_since.get(it.get("track_id"), now)
            waited = max(0.0, now - since)
            # A continuously waiting recheck outranks a NEW arrival after starvation budget.
            return (0 if waited >= self.starve_s else 1, since,
                    0 if dec[it["key"]] in (NEW, INVALIDATED) else 1)
        want.sort(key=prio)
        fresh = [it["key"] for it in want[:self.max_fresh]]
        for it in want[self.max_fresh:]:
            dec[it["key"]] = "DEFERRED_BUDGET:" + dec[it["key"]]
            self.counters["deferred_budget"] += 1
            self.drops.append(dict(t=now, key=it["key"], track_id=it.get("track_id"), why="budget"))
        self.drops = self.drops[-500:]
        cached = [it["key"] for it in items if dec[it["key"]] == CACHED_FRESH]
        self.counters["fresh"] += len(fresh)
        self.counters["cached"] += len(cached)
        return fresh, cached, dec

    def store(self, track_id, sig, result, now=None, negative=False):
        if track_id is None:
            return
        now = self.clock() if now is None else now
        self.cache[track_id] = dict(t=now, sig=dict(sig), result=dict(result), revision=dict(self.revision), negative=negative)
        self._pending_since.pop(track_id, None)

    def cached_result(self, track_id):
        c = self.cache.get(track_id)
        return None if c is None else dict(c["result"], cached=True, cached_age_s=None)

    def forget(self, track_id):
        self.cache.pop(track_id, None)
        self._pending_since.pop(track_id, None)


class LatestSlot:
    """In-flight 1 + pending 1. A newer request replaces the pending one; the replaced one is recorded as dropped."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.pending, self.in_flight = None, None
        self.dropped = []
        self.n_put = self.n_taken = self.n_dropped = 0          # E5RT2: n_dropped is the unbounded count (dropped log is capped)

    def put(self, item, why="superseded"):
        with self.lock:
            if self.pending is not None:
                self.dropped.append(dict(t=time.monotonic(), id=self.pending.get("id"), why=why))
                self.dropped = self.dropped[-500:]
                self.n_dropped += 1
            self.pending = item
            self.n_put += 1
            self.cv.notify()

    def take(self, timeout=None):
        with self.lock:
            available = self.cv.wait_for(lambda: self.pending is not None and self.in_flight is None, timeout)
            if not available:
                return None
            it, self.pending = self.pending, None
            if it is not None:
                self.in_flight = it
                self.n_taken += 1
            return it

    def done(self, item_id=None):
        with self.lock:
            if self.in_flight is None:
                return False
            if item_id is not None and self.in_flight.get("id") != item_id:
                return False
            self.in_flight = None
            self.cv.notify_all()
            return True

    def depth(self):
        with self.lock:
            return int(self.pending is not None) + int(self.in_flight is not None)
