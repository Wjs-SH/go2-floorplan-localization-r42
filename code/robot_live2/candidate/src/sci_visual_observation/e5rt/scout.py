"""Multi-rate candidate discovery (CPU). Downsampled scout for new candidates + periodic full-resolution rescue.

The legacy proposal function (live_fast.propose_fast) is passed in, unchanged. The scout runs it on a downscaled
copy and maps boxes back with the EXACT scale of the actual arrays (no 640x480 constant). Final SigLIP crops are
always taken from the original full-resolution frame. Every `rescue_every`-th call runs at full resolution so that
small targets that vanish at low resolution are not permanently missed. Proposals already covered by a live track can
be skipped (the tracker keeps them); a rescue call never skips. Pure apart from cv2.resize.
"""
from __future__ import annotations

import time

import cv2

from .tracker import iou


def map_box(box_small, small_shape, full_shape):
    sx = full_shape[1] / float(small_shape[1])
    sy = full_shape[0] / float(small_shape[0])
    return [box_small[0] * sx, box_small[1] * sy, box_small[2] * sx, box_small[3] * sy]


class Scout:
    def __init__(self, propose_fn, scale=0.5, rescue_every=3, skip_tracked_iou=0.5):
        self.propose, self.scale, self.rescue_every, self.skip_iou = propose_fn, float(scale), int(rescue_every), skip_tracked_iou
        self.n = 0
        self.log = []

    def step(self, bgr, tracked_boxes=()):
        """Returns (boxes_full_res, info). Boxes are lists [x0,y0,x1,y1,...extra] in full-res pixels."""
        t0 = time.perf_counter()
        self.n += 1
        rescue = self.rescue_every > 0 and (self.n % self.rescue_every == 0)
        if rescue or self.scale >= 0.999:
            props = [list(b) for b in self.propose(bgr)]
            src = "full_res_rescue" if rescue and self.scale < 0.999 else "full_res"
        else:
            h, w = bgr.shape[:2]
            small = cv2.resize(bgr, (max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale)))), interpolation=cv2.INTER_AREA)
            props = []
            for b in self.propose(small):
                fb = map_box(b[:4], small.shape, bgr.shape)
                props.append(fb + list(b[4:]))
            src = "scout_x%.2f" % self.scale
        skipped = 0
        if tracked_boxes and not rescue:
            keep = []
            for b in props:
                if any(iou(b[:4], t) >= self.skip_iou for t in tracked_boxes):
                    skipped += 1
                    continue
                keep.append(b)
            props = keep
        info = dict(source=src, n=len(props), skipped_tracked=skipped, ms=(time.perf_counter() - t0) * 1e3, call=self.n)
        self.log = (self.log + [info])[-200:]
        return props, info
