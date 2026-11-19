"""Live image vs inference panel (candidate replacement of the student 'boxes from <=2.5 s ago' overlay).

R1 contract: a detection box is drawn on a frame only if it was computed on that frame.
Student 10-17 (KCI_LIVE_BOXES=1): boxes of the last scene (<= 2.5 s old) drawn on the newest frame with a note ->
while walking, the box is offset from the sign (it marks where the sign WAS).
E5-RT: the live image draws only boxes the tracker VERIFIED on this exact frame (LK update on this frame id), labelled
with track id, semantic age (time since the last fresh SigLIP result) and track quality. Anything else stays on the
inference panel (source frame + its own boxes + frame id + age). KCI_LIVE_BOXES=legacy reproduces the student overlay
for comparison only.
"""
from __future__ import annotations

import os
import time

LEGACY_MAX_AGE_S = 2.5


def live_overlay(latest_frame_id, tracker_live, sem_by_track, now_mono, legacy_meta=None, mode=None):
    """Pure: returns (boxes, note). boxes = [((x0,y0,x1,y1), label, thickness, kind)].
    tracker_live = tracks verified on latest_frame_id; sem_by_track = track_id -> dict(display, sim, t_mono)."""
    mode = mode or os.environ.get("KCI_LIVE_BOXES", "tracked")
    out = []
    if mode == "legacy" and legacy_meta and legacy_meta.get("boxes"):
        age = now_mono - legacy_meta.get("t_drawn_mono", now_mono)
        if latest_frame_id == legacy_meta.get("frame_id"):
            return list(legacy_meta["boxes"]), "same_frame"
        if age <= LEGACY_MAX_AGE_S:
            return [(b, lab + " (old)", th, "legacy_old_box") for b, lab, th in legacy_meta["boxes"]], f"LEGACY boxes from {age:.1f}s ago"
        return [], "legacy_box_too_old"
    if mode == "off":
        return [], "off"
    for tr in tracker_live:
        if tr.get("frame_id") != latest_frame_id:
            continue                                   # defensive: never draw a box of another frame
        s = sem_by_track.get(tr["id"])
        if s is None or s.get("display") not in ("VERIFIED", "VERIFYING", "STOP_REQUEST", "ALREADY_OBSERVED", "CONFIRMED", "TENTATIVE"):
            continue
        sem_age = None if s.get("t_mono") is None else now_mono - s["t_mono"]
        lab = f"trk{tr['id']} {s.get('display', '')[:4]} sem {sem_age:.1f}s q{tr.get('quality', 0):.1f}" if sem_age is not None \
            else f"trk{tr['id']} q{tr.get('quality', 0):.1f}"
        out.append((tuple(int(round(v)) for v in tr["box"]), lab, 4 if s.get("display") == "STOP_REQUEST" else 2, "tracked"))
    return out, f"tracked boxes verified on frame {latest_frame_id}"


def paint(im, boxes):
    import cv2
    for b, lab, th, kind in boxes:
        x0, y0, x1, y1 = b
        c = (0, 90, 255) if th >= 4 else ((160, 160, 160) if kind == "legacy_old_box" else (0, 150, 255))
        cv2.rectangle(im, (x0, y0), (x1, y1), c, th)
        cv2.putText(im, lab, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return im


def now_mono():
    return time.monotonic()
