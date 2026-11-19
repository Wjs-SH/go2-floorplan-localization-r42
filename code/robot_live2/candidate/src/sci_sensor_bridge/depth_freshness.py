"""depth_freshness.py (R34, 2026-11-01): one rule for every reader of aligned depth.

Field X20261101_193139: depth stopped at about +60 s and the nearest-stamp lookup kept returning the same image for 1100 s (colour-depth
gap p50 538 s). Every plate distance after that was the old image's value. Rule: a depth image older or newer than the colour frame by more
than LIVE2_DEPTH_MAX_GAP_S is not returned. Unknown depth stays unknown; nothing downstream receives a substitute.

Limit 0.5 s: the good run X20261031_231940 had |colour-depth| p50 0.0, p90 0.133, p99 0.534 s (491 detections), so 0.5 s keeps ~99 % of
normal frames and refuses a frozen stream within half a second."""
import os

DEPTH_OK, DEPTH_STALE, DEPTH_MISSING = "DEPTH_OK", "DEPTH_STALE", "DEPTH_MISSING"
DEFAULT_MAX_GAP_S = 0.5


def max_gap_s():
    try:
        v = float(os.environ.get("LIVE2_DEPTH_MAX_GAP_S", DEFAULT_MAX_GAP_S))
    except ValueError:
        return DEFAULT_MAX_GAP_S
    return v if 0.0 < v <= 2.0 else DEFAULT_MAX_GAP_S


class Tracker:
    """Counts and the current verdict of depth lookups (one per reader)."""

    def __init__(self):
        self.ok, self.stale, self.missing = 0, 0, 0
        self.state, self.last_gap_s = DEPTH_MISSING, None

    def judge(self, frame_stamp, depth_stamp, limit=None):
        """-> True when depth_stamp may be used for frame_stamp."""
        limit = max_gap_s() if limit is None else limit
        if depth_stamp is None:
            self.missing += 1
            self.state, self.last_gap_s = DEPTH_MISSING, None
            return False
        gap = abs(float(depth_stamp) - float(frame_stamp))
        self.last_gap_s = round(gap, 3)
        if gap > limit:
            self.stale += 1
            self.state = DEPTH_STALE
            return False
        self.ok += 1
        self.state = DEPTH_OK
        return True

    def snapshot(self):
        return dict(state=self.state, last_gap_s=self.last_gap_s, ok=self.ok, stale=self.stale, missing=self.missing, max_gap_s=max_gap_s())


def frame_depth_state(frame, limit=None):
    """Verdict for one hub index frame record: (state, gap_s)."""
    limit = max_gap_s() if limit is None else limit
    ds = (frame or {}).get("depth_stamp")
    if ds is None or (frame or {}).get("stamp") is None:
        return DEPTH_MISSING, None
    gap = abs(float(ds) - float(frame["stamp"]))
    return (DEPTH_OK if gap <= limit else DEPTH_STALE), round(gap, 3)


def hub_depth_state(index, limit=None):
    """Verdict for a live hub index: the newest frames decide. Colour frames arriving with no usable depth -> not OK."""
    frames = (index or {}).get("frames") or []
    if not frames:
        return dict(state=DEPTH_MISSING, why="no hub frame", gap_s=None)
    state, gap = frame_depth_state(frames[-1], limit)
    return dict(state=state, gap_s=gap, why=None if state == DEPTH_OK else
                ("colour frames arrive without aligned depth" if state == DEPTH_MISSING else f"newest depth is {gap} s away from the colour frame"))
