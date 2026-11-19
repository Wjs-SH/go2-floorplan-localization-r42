#!/usr/bin/env python3
"""e2_region_ledger.py (R34, 2026-11-01): the persistent record of regional E2 requests.

Field X20261101_193139: one request (OBS_..._O003) crashed the worker five times, because the only memory of it was a set in the
crashed process. Here every request has one durable entry, so a restart neither repeats finished work nor loops on a request that
keeps killing the worker.

Idempotency key = run_id + odom_session + request_id + map_version + OCR observation_id.
States: PENDING -> RUNNING -> COMPLETED | RETRY_WAIT | FAILED_RETRYABLE | FAILED_FINAL.
  * RETRY_WAIT       the fit could not start or the calibration refused it (robot at rest, FAST-LIO hold): retried, bounded.
  * FAILED_RETRYABLE the worker died during the fit (found at the next start): retried, at most MAX_INTERRUPTED times.
  * FAILED_FINAL     no more attempts; `why` says what to do (usually: take a new observation).
One writer: the E2 coordinator. The requester (E8 / E6E7Link) only appends to e2_region_requests.jsonl via submit_request."""
import hashlib
import json
import os
import time
from pathlib import Path

PENDING, RUNNING, RETRY_WAIT, COMPLETED, FAILED_RETRYABLE, FAILED_FINAL = (
    "PENDING", "RUNNING", "RETRY_WAIT", "COMPLETED", "FAILED_RETRYABLE", "FAILED_FINAL")
OPEN_STATES = (PENDING, RETRY_WAIT, FAILED_RETRYABLE)
MAX_ATTEMPTS = 8          # fits that started and were refused (CalibrationUnavailable, too few scan points)
MAX_INTERRUPTED = 3       # worker deaths during one request's fit: after this it is final, never a crash loop
MAX_WAIT_S = 600.0        # robot at rest / FAST-LIO hold: waiting costs no attempt, but it is bounded in time
RETRY_S = 15.0
RESTART_RETRY_S = 5.0


def atomic_durable(path, obj, default=str):
    """tempfile -> fsync -> os.replace -> fsync(dir): a crash leaves the previous complete file or the new complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=default))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def submit_request(e2_dir, req):
    """Requester side: durable append (the worker reads the whole list, a newer request cannot hide an older one), then the
    single-file form older tools read."""
    e2_dir = Path(e2_dir)
    e2_dir.mkdir(parents=True, exist_ok=True)
    with open(e2_dir / "e2_region_requests.jsonl", "a") as f:
        f.write(json.dumps(req, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    atomic_durable(e2_dir / "e2_region_request.json", req)


def read_requests(e2_dir):
    """All submitted requests, oldest first. A torn last line (crash during append) is skipped, the single file still has it."""
    e2_dir, out, seen = Path(e2_dir), [], set()
    try:
        for ln in open(e2_dir / "e2_region_requests.jsonl"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("request_id") and r["request_id"] not in seen:
                seen.add(r["request_id"]); out.append(r)
    except OSError:
        pass
    try:
        r = json.loads((e2_dir / "e2_region_request.json").read_text())
        if r.get("request_id") and r["request_id"] not in seen:
            out.append(r)
    except (OSError, ValueError):
        pass
    return out


def candidates_hash(cands):
    keys = [[c.get("floor"), c.get("px"), c.get("py"), c.get("phi_deg"), c.get("ppm"), c.get("merit")] for c in cands or []]
    return hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.d = dict(kind="LIVE2_E2_REGION_LEDGER", identity=None, entries={}, seq=0, discarded=[])

    # ---- persistence
    def load(self, identity):
        """Same run / odometry session / maps / calibration -> entries kept. Otherwise a fresh ledger, and the reason is recorded."""
        try:
            old = json.loads(self.path.read_text())
        except OSError:
            old = None
        except ValueError:
            self.d.update(identity=identity, discarded=[dict(wall=self.clock(), why="ledger file unreadable (corrupt): started fresh")])
            self.save()
            return "LEDGER_CORRUPT"
        if not old:
            self.d["identity"] = identity
            self.save()
            return "NEW"
        diff = [k for k in sorted(set(identity) | set(old.get("identity") or {})) if (old.get("identity") or {}).get(k) != identity.get(k)]
        if diff:
            self.d.update(identity=identity, discarded=[dict(wall=self.clock(), why=f"identity changed: {diff}", n=len(old.get("entries") or {}))])
            self.save()
            return "IDENTITY_CHANGED:" + ",".join(diff)
        self.d = old
        return "RESTORED"

    def save(self):
        atomic_durable(self.path, self.d)

    # ---- identity of one request
    def key(self, rq):
        i, R = self.d.get("identity") or {}, rq.get("region") or {}
        raw = "|".join(str(x) for x in (i.get("run_id"), rq.get("odom_session"), rq.get("request_id"), i.get("maps"), R.get("observation_id") or R.get("group_id")))
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def ingest(self, rq):
        """Idempotent: the same request again changes nothing. A request of another odometry session never becomes runnable."""
        k = self.key(rq)
        e = self.d["entries"].get(k)
        if e is not None:
            return e
        self.d["seq"] = int(self.d.get("seq") or 0) + 1
        i = self.d.get("identity") or {}
        e = dict(key=k, seq=self.d["seq"], request_id=rq.get("request_id"), request=rq, state=PENDING, attempts=0, interrupted=0, not_before=0.0,
                 first_seen_wall=self.clock(), updated_wall=self.clock(), why=None, candidates_hash=None, n_candidates=None, delivery=None)
        if rq.get("odom_session") != i.get("odom_session"):
            e.update(state=FAILED_FINAL, why=f"DISCARDED: request of another odometry session ({rq.get('odom_session')})")
        self.d["entries"][k] = e
        self.save()
        return e

    def get(self, request_id):
        return next((e for e in self.d["entries"].values() if e.get("request_id") == request_id), None)

    def ready(self, now=None):
        """The newest runnable entry (the newest observation has the smallest odometry drift), else None."""
        now = self.clock() if now is None else now
        ok = [e for e in self.d["entries"].values() if e["state"] in OPEN_STATES and now >= float(e.get("not_before") or 0)]
        return max(ok, key=lambda e: e["seq"]) if ok else None

    # ---- transitions
    def _up(self, e, **k):
        e.update(k, updated_wall=self.clock())
        self.save()
        return e

    def start(self, e):
        return self._up(e, state=RUNNING, attempts=int(e.get("attempts") or 0) + 1, started_wall=self.clock())

    def wait(self, e, why, retry_s=RETRY_S):
        """Could not start now (rest / hold). Costs no attempt; bounded by MAX_WAIT_S from the first sighting."""
        if self.clock() - float(e.get("first_seen_wall") or self.clock()) > MAX_WAIT_S:
            return self._up(e, state=FAILED_FINAL, why=f"NEEDS_NEW_OBSERVATION: waited > {MAX_WAIT_S:.0f} s ({why})")
        return self._up(e, state=RETRY_WAIT, not_before=self.clock() + retry_s, why=why)

    def retry(self, e, why, retry_s=RETRY_S):
        """The fit started and was refused, or it was stopped by the guards."""
        if int(e.get("attempts") or 0) >= MAX_ATTEMPTS:
            return self._up(e, state=FAILED_FINAL, why=f"{why} ({e.get('attempts')} attempts)")
        return self._up(e, state=RETRY_WAIT, not_before=self.clock() + retry_s, why=why)

    def complete(self, e, cands):
        return self._up(e, state=COMPLETED, candidates_hash=candidates_hash(cands), n_candidates=len(cands or []), why=None,
                        finished_wall=self.clock(), delivery=e.get("delivery") or "WAITING_E3")

    def fail(self, e, why):
        return self._up(e, state=FAILED_FINAL, why=why, finished_wall=self.clock())

    def interrupted(self, e, why):
        n = int(e.get("interrupted") or 0) + 1
        if n >= MAX_INTERRUPTED:
            return self._up(e, state=FAILED_FINAL, interrupted=n, why=f"{why}: stopped after {n} interruptions (no crash loop)", finished_wall=self.clock())
        return self._up(e, state=FAILED_RETRYABLE, interrupted=n, why=why, not_before=self.clock() + RESTART_RETRY_S)

    def recover_after_restart(self, state_regional):
        """Entries left RUNNING by a dead worker. If the candidates were already saved in e2_state (the worker died after saving
        them and before the ledger update), the entry is COMPLETED and the fit is NOT repeated."""
        res = []
        for e in self.d["entries"].values():
            if e["state"] != RUNNING:
                continue
            done = [r for r in state_regional or [] if r.get("request_id") == e["request_id"] and r.get("status") == "DONE"]
            if done:
                self._up(e, state=COMPLETED, n_candidates=done[-1].get("n"), candidates_hash=done[-1].get("candidates_hash"),
                         finished_wall=done[-1].get("finished_wall"), delivery=e.get("delivery") or "WAITING_E3", why="reconciled from e2_state after restart")
                res.append((e["request_id"], "RECONCILED_COMPLETED"))
            else:
                self.interrupted(e, "worker ended during the regional fit")
                res.append((e["request_id"], e["state"]))
        return res

    def merge_ack(self, ack):
        """E3 intake acknowledgement {request_id: {delivered_wall, added, merged, hypotheses}}."""
        changed = False
        for rid, a in (ack or {}).items():
            e = self.get(rid)
            if e is not None and e.get("state") == COMPLETED and e.get("delivery") != "DELIVERED":
                e.update(delivery="DELIVERED", delivered=a, updated_wall=self.clock())
                changed = True
        if changed:
            self.save()
        return changed

    def summary(self):
        keys = ("request_id", "seq", "state", "attempts", "interrupted", "why", "n_candidates", "candidates_hash", "delivery", "not_before", "updated_wall")
        return [{k: e.get(k) for k in keys} for e in sorted(self.d["entries"].values(), key=lambda e: e["seq"])]
