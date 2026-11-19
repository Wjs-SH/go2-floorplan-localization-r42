"""E5-RT shared contracts: cause codes, E4 status contract, ObservationCandidate schema. Pure Python."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---------------------------------------------------------------- why a candidate is not (yet) a STOP
INPUT_TOO_SMALL = "INPUT_TOO_SMALL"          # target crop below the readable/encodable size
LOW_QUALITY = "LOW_QUALITY"                  # blur/exposure/glare of the ROI too poor to count as evidence
PROPOSAL_MISS = "PROPOSAL_MISS"              # no proposal/track covered the target in this frame
SEMANTIC_MARGIN_LOW = "SEMANTIC_MARGIN_LOW"  # sign-vs-confuser margin / relative threshold not met
TEMPORAL_PENDING = "TEMPORAL_PENDING"        # evidence accumulating (not enough distinct observations yet)
STALE_RESULT = "STALE_RESULT"                # semantic result older than its validity window / other session
E4_UNAVAILABLE = "E4_UNAVAILABLE"            # no live E4 producer: E5 runs in explicit standalone mode
NO_FEASIBLE_VIEW = "NO_FEASIBLE_VIEW"        # no reachable/readable stop window (pan never 'fixes' obliquity)
RESOURCE_DEFERRED = "RESOURCE_DEFERRED"      # resource manager lowered rate / deferred work (reason logged)
TRIAL_GATE_BLOCKED = "TRIAL_GATE_BLOCKED"    # operator/trial gate forbids the action (not a semantic verdict)
GEOMETRY_UNAVAILABLE = "GEOMETRY_UNAVAILABLE"
TRACK_LOST = "TRACK_LOST"
CAUSES = (INPUT_TOO_SMALL, LOW_QUALITY, PROPOSAL_MISS, SEMANTIC_MARGIN_LOW, TEMPORAL_PENDING, STALE_RESULT,
          E4_UNAVAILABLE, NO_FEASIBLE_VIEW, RESOURCE_DEFERRED, TRIAL_GATE_BLOCKED, GEOMETRY_UNAVAILABLE, TRACK_LOST)

# ---------------------------------------------------------------- E4 (global ambiguity) contract
E4_STATES = ("UNAVAILABLE", "AMBIGUOUS", "SEARCH_INCOMPLETE", "INSUFFICIENT_GEOMETRY", "DOMINANT_UNVERIFIED",
             "NEED_RECOVERY", "CONFIDENT")
E4_STANDALONE_MODE = "E5_STANDALONE_E4_UNAVAILABLE"


def finite(*v):
    return all(x is not None and isinstance(x, (int, float)) and math.isfinite(x) for x in v)


@dataclass
class ObservationCandidate:
    """Hand-off from E5 to pan/tilt + reader. Every field says where it came from; unknown stays None + reason."""
    run_id: str
    generation: Optional[str]
    event_key: Optional[str]
    track_id: int
    source_frame_id: Optional[int]            # frame the semantic evidence was computed on
    source_header_stamp: Optional[float]
    latest_frame_id: Optional[int]            # frame the tracked ROI was last updated on
    latest_header_stamp: Optional[float]
    capture_id: Optional[str] = None
    crop_box_source: Optional[list] = None    # [x0,y0,x1,y1] in source-frame pixels (full resolution)
    roi_box_latest: Optional[list] = None     # tracked ROI in latest frame pixels
    image_size: Optional[list] = None         # [w, h] of the frames above (no 640x480 assumption)
    transform_version: Optional[str] = None
    semantic_t_mono: Optional[float] = None   # last FRESH (non-cached) SigLIP verification, monotonic s
    geometry_t_mono: Optional[float] = None
    target_type: Optional[str] = None
    margin: Optional[float] = None            # positive - strongest confuser (same model, same crop)
    raw_scores: dict = field(default_factory=dict)
    depth: dict = field(default_factory=dict)  # valid_ratio, median_m, mad_m, n, kind (optical_z|range), reason
    frame_transform: Optional[str] = None
    covariance: Optional[list] = None
    covariance_null_reason: Optional[str] = "no uncertainty model for the ROI position"
    stop_feasibility: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    # ---- E5RT2 identity / geometry fields (all optional; unknown = None + reason, never synthesised)
    observation_id: Optional[str] = None
    boot_id: Optional[str] = None
    odom_session: Optional[int] = None
    trial_id: Optional[str] = None
    snapshot_id: Optional[int] = None         # scene snapshot the semantic evidence came from
    request_id: Optional[str] = None          # pan/tilt request that used this observation (filled at submit)
    depth_frame: dict = field(default_factory=dict)        # depth image identity of the source scene (stamp, pose stamp)
    native_roi_source: Optional[list] = None  # region box (native resolution) vs crop_box_source = semantic crop (tile)
    calibration_version: Optional[str] = None
    relative_target: Optional[dict] = None    # target relative to the CURRENT pose (P2 view), with its source
    relative_target_null_reason: Optional[str] = None
    geometry_confirmed: bool = False
    geometry_reasons: list = field(default_factory=list)
    payload_sha256: Optional[str] = None          # SEM1 P02: geometry content fingerprint
    payload_revision: int = 0                     # increments when the content of the same entity changes

    def to_dict(self):
        return asdict(self)


def roi_depth_stats(dimg, box, depth_scale=0.001, z_min=0.2, z_max=10.0, min_valid=20):
    """Robust depth support of a ROI. Never fills invalid depth with a constant.
    Returns dict(kind='optical_z', valid_ratio, median_m, mad_m, n, ok, reason)."""
    import numpy as np
    if dimg is None:
        return dict(kind="optical_z", ok=False, reason="NO_DEPTH_IMAGE", valid_ratio=0.0, n=0, median_m=None, mad_m=None)
    h, w = dimg.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box[:4])
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return dict(kind="optical_z", ok=False, reason="ROI_OUTSIDE_IMAGE", valid_ratio=0.0, n=0, median_m=None, mad_m=None)
    z = dimg[y0:y1, x0:x1].astype(np.float32).ravel() * depth_scale
    ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    n, tot = int(ok.sum()), int(z.size)
    if n < min_valid:
        return dict(kind="optical_z", ok=False, reason="INSUFFICIENT_VALID_DEPTH", valid_ratio=n / max(tot, 1), n=n,
                    median_m=None, mad_m=None)
    zz = z[ok]
    med = float(np.median(zz))
    mad = float(np.median(np.abs(zz - med)))
    return dict(kind="optical_z", ok=True, reason=None, valid_ratio=n / tot, n=n, median_m=med, mad_m=mad)
