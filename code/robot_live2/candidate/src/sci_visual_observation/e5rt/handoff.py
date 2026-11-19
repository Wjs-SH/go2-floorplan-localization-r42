"""Build the pan/tilt + reader hand-off (ObservationCandidate) from the same track / frame / geometry (pure)."""
from __future__ import annotations

from . import contracts as C


def build_candidate(run_id, generation, track, sem, src_frame, latest_frame, depth_stats=None, event_key=None,
                    capture_id=None, image_size=None, transform_version=None, geometry_t_mono=None, feasibility=None,
                    target_type=None):
    """track: tracker dict (box on latest frame), sem: dict(box_source, sim, conf, raw, t_mono, frame_id, state),
    src_frame/latest_frame: frame meta dicts (frame_id, header_stamp). depth_stats from contracts.roi_depth_stats on the
    SOURCE frame's aligned depth (never filled with a constant)."""
    ds = dict(depth_stats or dict(ok=False, reason=C.GEOMETRY_UNAVAILABLE))
    conf = sem.get("conf")
    margin = None if conf is None or sem.get("sim") is None else float(sem["sim"] - conf)
    if src_frame and sem.get("frame_id") is not None and src_frame.get("frame_id") != sem.get("frame_id"):
        raise ValueError("semantic result and source frame differ (provenance mismatch)")
    tl = (track or {}).get("status")
    roi_latest = (track or {}).get("box") if tl == "LIVE" else None
    if roi_latest is not None:
        tf, lf = (track or {}).get("frame_id"), (latest_frame or {}).get("frame_id")
        if tf is None or lf is None or tf != lf:
            raise ValueError("TRACK_FRAME_MISMATCH: refusing to relabel an old ROI as the latest frame")
        if len(roi_latest) != 4 or not C.finite(*roi_latest) or roi_latest[2] <= roi_latest[0] or roi_latest[3] <= roi_latest[1]:
            raise ValueError("INVALID_TRACK_ROI")
        tt, lt = (track or {}).get("header_t"), (latest_frame or {}).get("header_stamp")
        if tt is not None and lt is not None and (not C.finite(tt, lt) or abs(tt-lt) > 1e-6):
            raise ValueError("TRACK_TIME_MISMATCH")
    for label, box in (("source", sem.get("box_source")),):
        if box is not None and (len(box) != 4 or not C.finite(*box) or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("INVALID_" + label.upper() + "_ROI")
    return C.ObservationCandidate(
        run_id=run_id, generation=generation, event_key=event_key, track_id=int((track or {}).get("id", -1)),
        source_frame_id=(src_frame or {}).get("frame_id"), source_header_stamp=(src_frame or {}).get("header_stamp"),
        latest_frame_id=(latest_frame or {}).get("frame_id") if roi_latest else None,
        latest_header_stamp=(latest_frame or {}).get("header_stamp") if roi_latest else None,
        capture_id=capture_id, crop_box_source=[float(v) for v in sem["box_source"]] if sem.get("box_source") else None,
        roi_box_latest=[float(v) for v in roi_latest] if roi_latest else None, image_size=image_size,
        transform_version=transform_version, semantic_t_mono=sem.get("t_mono"), geometry_t_mono=geometry_t_mono,
        target_type=target_type, margin=margin, raw_scores=dict(sem.get("raw") or {}, sim=sem.get("sim"), conf=conf),
        depth=ds, frame_transform="camera_color_optical (RealSense D435 colour)",
        stop_feasibility=feasibility or (C.GEOMETRY_UNAVAILABLE if not ds.get("ok") else None),
        evidence=dict(state=sem.get("state"), n_distinct=sem.get("n_distinct"), track_status=tl,
                      track_why=(track or {}).get("why")))
