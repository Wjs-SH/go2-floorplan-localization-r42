"""Explicit per-scene timing/cause record for E5-RT (pure Python).

Clock domains are kept apart: header stamps (ROS/bag seconds) are never mixed with monotonic receive/processing times.
Missing values are written as None with a *_null_reason (NOT_LOGGED / NOT_AVAILABLE_IN_SOURCE), never as 0.
GPU time comes from CUDA events (device span); host time is wall time around the call including the final sync.
"""
from __future__ import annotations

import time

NOT_LOGGED = "NOT_LOGGED"
FIELDS = ("rx_mono_ns", "proposal_queue_ns", "proposal_start_ns", "proposal_end_ns", "track_start_ns", "track_end_ns",
          "crop_prep_start_ns", "crop_prep_end_ns", "siglip_queue_ns", "siglip_start_ns", "siglip_end_ns",
          "policy_start_ns", "policy_end_ns", "publish_ns", "browser_rx_ns", "browser_ack_ns")


class SceneTelemetry:
    def __init__(self, frame_id=None, camera_id=None, header_stamp=None, clock_domain=None, boot_id=None):
        self.r = dict(frame_id=frame_id, camera_id=camera_id, header_stamp=header_stamp, clock_domain=clock_domain,
                      boot_id=boot_id, exposure_stamp=None, exposure_stamp_null_reason="RealSense exposure time not exported")
        for f in FIELDS:
            self.r[f] = None
        self.extra = {}

    def mark(self, field, ns=None):
        assert field in FIELDS, field
        self.r[field] = time.monotonic_ns() if ns is None else int(ns)
        return self.r[field]

    def set(self, **kw):
        self.extra.update(kw)

    def record(self):
        out = dict(self.r)
        for f in FIELDS:
            if out[f] is None:
                out[f + "_null_reason"] = NOT_LOGGED
        ms = lambda a, b: None if out.get(a) is None or out.get(b) is None else round((out[b] - out[a]) / 1e6, 3)  # noqa: E731
        out["durations_ms"] = dict(proposal=ms("proposal_start_ns", "proposal_end_ns"),
                                   proposal_wait=ms("proposal_queue_ns", "proposal_start_ns"),
                                   track=ms("track_start_ns", "track_end_ns"), crop_prep=ms("crop_prep_start_ns", "crop_prep_end_ns"),
                                   siglip_queue=ms("siglip_queue_ns", "siglip_start_ns"), siglip_host=ms("siglip_start_ns", "siglip_end_ns"),
                                   policy=ms("policy_start_ns", "policy_end_ns"), rx_to_publish=ms("rx_mono_ns", "publish_ns"))
        out.update(self.extra)
        return out
