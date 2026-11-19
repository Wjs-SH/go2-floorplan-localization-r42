"""Revisioned atomic file writes (candidate).

Student fix (kept): every writer uses its own temp name (pid+thread) in the same directory, then os.replace -> no
FileNotFoundError race between the scene loop and the event thread (10-17 19:40 E5 exit).
Remaining gap closed here: two writers can still publish snapshots OUT OF ORDER (a slow thread replaces a newer
status.json with an older snapshot). RevisionedWriter assigns a monotonic revision when the snapshot is TAKEN and
refuses to publish a snapshot older than the last published one for that path (stale overwrite -> dropped + counted).
Temp files are created in the target directory, fsync is optional, failures remove the temp file.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
from pathlib import Path


def atomic_write_unique(path, data, fsync=False):
    """Student semantics: per-writer temp name + os.replace (same directory)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class RevisionedWriter:
    def __init__(self, fsync=False):
        self._rev = itertools.count(1)
        self._lock = threading.Lock()
        self._published = {}                 # path -> last published revision
        self.fsync = fsync
        self.stale_dropped = 0
        self.errors = 0

    def take(self):
        """Call when the snapshot is BUILT (not when it is written). Revisions are global and strictly increasing."""
        with self._lock:
            return next(self._rev)

    def write(self, path, data, rev):
        """Publish `data` (bytes) taken at revision `rev`. Returns True if published, False if it was stale."""
        key = str(Path(path))
        with self._lock:                     # serialise per process: compare + replace must not interleave
            if self._published.get(key, 0) >= rev:
                self.stale_dropped += 1
                return False
            try:
                atomic_write_unique(path, data, fsync=self.fsync)
            except OSError:
                self.errors += 1
                raise
            self._published[key] = rev
            return True

    def write_json(self, path, obj, rev=None):
        rev = self.take() if rev is None else rev
        if isinstance(obj, dict):
            obj = dict(obj, _rev=rev, _rev_t=time.time())
        return self.write(path, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), rev)


def sweep_stale_tmp(directory, older_than_s=60.0, now=None):
    """Remove leftover '.<name>.<pid>.<tid>.tmp' files of dead writers (never touches non-tmp files)."""
    now = time.time() if now is None else now
    removed = []
    for p in Path(directory).glob(".*.tmp"):
        try:
            parts = p.name.split(".")
            pid = int(parts[-3])
        except (ValueError, IndexError):
            continue
        alive = os.path.exists(f"/proc/{pid}")
        if (not alive) and now - p.stat().st_mtime > older_than_s:
            try:
                p.unlink()
                removed.append(p.name)
            except OSError:
                pass
    return removed
