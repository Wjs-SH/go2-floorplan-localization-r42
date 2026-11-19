"""Bounded sparse pyramidal-LK ROI tracker (CPU, OpenCV only).

Tracking is GEOMETRY evidence: it moves a box from the frame where it was detected to newer frames. It never creates
semantic evidence. A track that cannot be verified on the current frame is TRACK_LOST (its last box is not drawn as a
current detection).  Checks per update: LK status, forward-backward error, inlier count/fraction, bounded scale change,
box inside the image, frame gap (real header-time difference), textureless ROI (too few corners) -> UNCERTAIN.

All boxes are kept in FULL-RESOLUTION pixel coordinates of the frame they belong to. Tracking itself may run on a
downscaled grey image; the scale is derived from the actual array shapes (no 640x480 assumption).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

LIVE, LOST, UNCERTAIN = "LIVE", "TRACK_LOST", "UNCERTAIN_TEXTURELESS"


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class LKTracker:
    def __init__(self, track_scale=0.5, max_tracks=12, win=15, levels=3, fb_max_px=1.0, min_points=5,
                 min_inlier_frac=0.5, max_gap_s=0.6, max_scale_step=1.25, reseed_below=8, max_corners=24, clock=time.monotonic):
        self.s = float(track_scale)
        self.max_tracks, self.win, self.levels = max_tracks, (win, win), levels
        self.fb_max, self.min_points, self.min_frac = fb_max_px, min_points, min_inlier_frac
        self.max_gap_s, self.max_scale_step = max_gap_s, max_scale_step
        self.reseed_below, self.max_corners = reseed_below, max_corners
        self.clock = clock
        self.tracks = {}                      # id -> dict
        self.prev = None                      # (small grey, frame_id, header_t, full shape)
        self.stats = dict(updates=0, lost=0, cpu_ms=[], drops_capacity=0)

    # ---------------------------------------------------------------- helpers
    def _small(self, gray_full):
        if self.s == 1.0:
            return gray_full
        h, w = gray_full.shape[:2]
        return cv2.resize(gray_full, (max(1, int(round(w * self.s))), max(1, int(round(h * self.s)))), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _sx(full_shape, small):
        return small.shape[1] / float(full_shape[1]), small.shape[0] / float(full_shape[0])

    def _seed(self, small, box_full, full_shape):
        sx, sy = self._sx(full_shape, small)
        h, w = small.shape[:2]
        x0, y0 = int(max(0, np.floor(box_full[0] * sx))), int(max(0, np.floor(box_full[1] * sy)))
        x1, y1 = int(min(w, np.ceil(box_full[2] * sx))), int(min(h, np.ceil(box_full[3] * sy)))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return np.zeros((0, 1, 2), np.float32)
        mask = np.zeros_like(small)
        mask[y0:y1, x0:x1] = 255
        pts = cv2.goodFeaturesToTrack(small, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=2, mask=mask, blockSize=3)
        return np.zeros((0, 1, 2), np.float32) if pts is None else pts.astype(np.float32)

    # ---------------------------------------------------------------- API
    def seed(self, track_id, box_full, gray_full, frame_id, header_t, meta=None):
        """Create/refresh a track from a detection on THIS frame (box in full-res pixels of this frame)."""
        small = self._small(gray_full)
        pts = self._seed(small, box_full, gray_full.shape)
        if track_id not in self.tracks and len([t for t in self.tracks.values() if t["status"] != LOST]) >= self.max_tracks:
            self.stats["drops_capacity"] += 1
            return None
        tr = self.tracks.get(track_id) or dict(id=track_id, born_frame=frame_id, born_t=header_t, meta={})
        tr.update(box=[float(v) for v in box_full], pts=pts, frame_id=frame_id, header_t=header_t, seed_frame=frame_id,
                  status=LIVE if len(pts) >= self.min_points else UNCERTAIN, why=None if len(pts) >= self.min_points else "few_corners",
                  quality=1.0 if len(pts) >= self.min_points else 0.0, fb_med=0.0, n_pts=int(len(pts)), shape=gray_full.shape[:2])
        if meta:
            tr["meta"].update(meta)
        self.tracks[track_id] = tr
        if self.prev is None or self.prev[1] != frame_id:
            self.prev = (small, frame_id, header_t, gray_full.shape[:2])
        return tr

    def update(self, gray_full, frame_id, header_t):
        """Propagate every LIVE track from the previous frame to this one. One forward + one backward LK call."""
        t0 = time.perf_counter()
        small = self._small(gray_full)
        if self.prev is None or self.prev[1] == frame_id:
            self.prev = (small, frame_id, header_t, gray_full.shape[:2])
            return self.tracks
        p_small, p_fid, p_t, p_shape = self.prev
        gap = None if (header_t is None or p_t is None) else header_t - p_t
        live = [t for t in self.tracks.values() if t["status"] == LIVE and t["frame_id"] == p_fid]
        for tr in self.tracks.values():                   # tracks not on the previous frame cannot be propagated
            if tr["status"] == LIVE and tr["frame_id"] != p_fid:
                self._lose(tr, "not_on_previous_frame")
            elif tr["status"] == UNCERTAIN:
                self._lose(tr, "textureless_no_points")
        if live and (gap is None or not np.isfinite(gap) or gap < 0 or gap > self.max_gap_s):
            for tr in live:
                self._lose(tr, f"frame_gap_{gap if gap is None else round(float(gap), 3)}s")
            live = []
        if live and p_small.shape != small.shape:
            for tr in live:
                self._lose(tr, "resolution_changed")
            live = []
        if live:
            pts0 = np.concatenate([t["pts"] for t in live]).astype(np.float32)
            owner = np.concatenate([np.full(len(t["pts"]), i) for i, t in enumerate(live)])
            try:
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(p_small, small, pts0, None, winSize=self.win, maxLevel=self.levels)
                if p1 is None or st1 is None or p1.shape != pts0.shape or not np.isfinite(p1).all():
                    raise ValueError("missing_or_nonfinite_forward_flow")
                p0b, st2, _ = cv2.calcOpticalFlowPyrLK(small, p_small, p1, None, winSize=self.win, maxLevel=self.levels)
                if p0b is None or st2 is None or p0b.shape != pts0.shape or not np.isfinite(p0b).all():
                    raise ValueError("missing_or_nonfinite_backward_flow")
            except (cv2.error, ValueError, TypeError) as exc:
                for tr in live:
                    self._lose(tr, "LK_UNAVAILABLE:" + type(exc).__name__)
                self.prev = (small, frame_id, header_t, gray_full.shape[:2])
                self.stats["updates"] += 1
                self.stats["cpu_ms"] = (self.stats["cpu_ms"] + [(time.perf_counter()-t0)*1e3])[-500:]
                return self.tracks
            fb = np.linalg.norm((pts0 - p0b).reshape(-1, 2), axis=1)
            ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb <= self.fb_max)
            h, w = small.shape[:2]
            q1 = p1.reshape(-1, 2)
            ok &= (q1[:, 0] >= 0) & (q1[:, 0] < w) & (q1[:, 1] >= 0) & (q1[:, 1] < h)
            sx, sy = self._sx(gray_full.shape, small)
            for i, tr in enumerate(live):
                sel = owner == i
                good = ok & sel
                n_good, n_all = int(good.sum()), int(sel.sum())
                if n_good < self.min_points or n_good < self.min_frac * n_all:
                    self._lose(tr, f"lk_inliers_{n_good}/{n_all}")
                    continue
                a, b = pts0.reshape(-1, 2)[good], q1[good]
                d = np.median(b - a, axis=0)
                scale = 1.0
                if n_good >= 3:
                    ca, cb = a.mean(0), b.mean(0)
                    ra, rb = np.linalg.norm(a - ca, axis=1), np.linalg.norm(b - cb, axis=1)
                    m = ra > 1.0
                    if m.sum() >= 3:
                        scale = float(np.median(rb[m] / ra[m]))
                if not (1.0 / self.max_scale_step <= scale <= self.max_scale_step):
                    self._lose(tr, f"scale_jump_{scale:.2f}")
                    continue
                x0, y0, x1, y1 = tr["box"]
                cx, cy = (x0 + x1) / 2 + d[0] / sx, (y0 + y1) / 2 + d[1] / sy
                hw, hh = (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
                H, W = gray_full.shape[:2]
                if not (0 <= cx < W and 0 <= cy < H):
                    self._lose(tr, "left_view")
                    continue
                tr.update(box=[cx - hw, cy - hh, cx + hw, cy + hh], frame_id=frame_id, header_t=header_t,
                          pts=b.reshape(-1, 1, 2).astype(np.float32), n_pts=n_good,
                          fb_med=float(np.median(fb[good])), quality=round(n_good / max(1, n_all), 3), why=None)
                if n_good < self.reseed_below:            # refresh corners inside the moved box (same frame)
                    np_ = self._seed(small, tr["box"], gray_full.shape)
                    if len(np_) >= n_good:
                        tr["pts"], tr["n_pts"] = np_, int(len(np_))
        self.prev = (small, frame_id, header_t, gray_full.shape[:2])
        self.stats["updates"] += 1
        self.stats["cpu_ms"].append((time.perf_counter() - t0) * 1e3)
        self.stats["cpu_ms"] = self.stats["cpu_ms"][-500:]
        return self.tracks

    def _lose(self, tr, why):
        tr["status"], tr["why"] = LOST, why
        self.stats["lost"] += 1

    def live_on(self, frame_id):
        """Tracks whose box is VERIFIED on this exact frame (the only ones the live image may draw)."""
        return [t for t in self.tracks.values() if t["status"] == LIVE and t["frame_id"] == frame_id]

    def prune(self, keep_lost=32):
        lost = [k for k, t in self.tracks.items() if t["status"] == LOST]
        for k in lost[:-keep_lost] if len(lost) > keep_lost else []:
            self.tracks.pop(k, None)

    def associate(self, box_full, frame_id, min_iou=0.3):
        """Track id whose box on `frame_id` best overlaps `box_full` (IoU >= min_iou), else None."""
        best, bi = None, min_iou
        for k, t in self.tracks.items():
            if t["status"] == LIVE and t["frame_id"] == frame_id:
                v = iou(t["box"], box_full)
                if v >= bi:
                    best, bi = k, v
        return best
