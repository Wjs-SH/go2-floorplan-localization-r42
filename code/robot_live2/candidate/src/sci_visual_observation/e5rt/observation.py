"""E5RT2: one ObservationSnapshot-derived hand-off used by BOTH the pan/tilt planner request and the reader job.

- depth_profile: robust ROI depth (median/MAD) PLUS the risks median/MAD hide: two depth populations inside the ROI
  (sign + background), disagreement with an independent reference range (the tile plane / region range of the same
  scene) and the absence of a sign mask. A background-dominated ROI can have a perfectly stable median; that is why
  geometry is only CONFIRMED with an independent consistent reference + repeated verification.
- geometry_confirmation: aim confirmation is refused (fail-closed) unless the evidence above is consistent.
- observation_id: content hash of the hand-off (same id in the event record, the aim request and the reader job).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from . import contracts as C
from .handoff import build_candidate

REF_TOL_M = 0.30
MIN_VALID_FRAC = 0.5
MAX_MAD_M = 0.10
MIN_VERIFY = 2
MAX_PLANE_RMS_M = 0.03          # SEM1: plane-fit residual of the same scene (tile neighbourhood)
MIN_ROI_SUPPORT = 0.5           # SEM1: valid depth fraction inside the box (plane fit 'valid_in')
# SEM1 P01: the reference range comes from the SAME aligned depth image (plane fit / region z of the same scene).
# Agreement is a CONSISTENCY check, never an independent sensor confirmation.
REF_SOURCE = "same_aligned_depth_image(plane_fit_or_region_z)"


def depth_profile(dimg, box, ref_z=None, depth_scale=0.001, ref_source=REF_SOURCE):
    d = dict(C.roi_depth_stats(dimg, box, depth_scale=depth_scale))
    d.update(sign_mask=None, sign_mask_null_reason="NO_SIGN_MASK (ROI box only; background pixels are not excluded)",
             ref_z_m=None if ref_z is None else float(ref_z), multimodal=None, ref_consistent=None, uncertainty="HIGH",
             ref_source=ref_source, ref_independent=False)
    if not d.get("ok"):
        d["risks"] = [d.get("reason") or "NO_DEPTH"]
        return d
    h, w = dimg.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box[:4])
    z = dimg[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].astype(np.float32).ravel() * depth_scale
    z = z[np.isfinite(z) & (z > 0.2) & (z < 10.0)]
    q = np.percentile(z, np.arange(5, 96, 5))
    gaps = np.diff(q)
    k = int(np.argmax(gaps)) if len(gaps) else 0
    frac_below = (k + 1) * 0.05 + 0.05 if len(gaps) else 0.0
    multimodal = bool(len(gaps) and gaps[k] > REF_TOL_M and 0.15 <= frac_below <= 0.85)
    risks = []
    if multimodal:
        risks.append(f"MULTIMODAL_DEPTH:gap_{gaps[k]:.2f}m_at_q{int(round(frac_below * 100))}")
    if d["valid_ratio"] < MIN_VALID_FRAC:
        risks.append(f"LOW_VALID_FRACTION:{d['valid_ratio']:.2f}")
    if d["mad_m"] is not None and d["mad_m"] > MAX_MAD_M:
        risks.append(f"HIGH_MAD:{d['mad_m']:.2f}m")
    if not C.finite(d.get("median_m"), d.get("mad_m")):
        risks.append("NONFINITE_DEPTH_STATS")
    ref_ok = None
    if ref_z is not None and math.isfinite(float(ref_z)):
        ref_ok = abs(d["median_m"] - float(ref_z)) <= REF_TOL_M
        if not ref_ok:
            risks.append(f"REFERENCE_MISMATCH:roi_{d['median_m']:.2f}_vs_ref_{float(ref_z):.2f}")
    else:
        risks.append("NO_REFERENCE_RANGE")
    high_mad = d["mad_m"] is not None and d["mad_m"] > MAX_MAD_M
    d.update(multimodal=multimodal, ref_consistent=ref_ok, risks=risks, high_mad=bool(high_mad),
             uncertainty="HIGH" if (multimodal or ref_ok is False or d["valid_ratio"] < MIN_VALID_FRAC or high_mad
                                    or "NONFINITE_DEPTH_STATS" in risks) else ("MEDIUM" if ref_ok is None else "LOW"))
    return d


def geometry_confirmation(depth, n_verify, plane_ok, plane=None):
    """SEM1 P01 contract: the gate consumes EVERY risk depth_profile reports (HIGH_MAD, non-finite stats, HIGH
    uncertainty) plus the plane-fit residual and ROI depth support when known. Good statistics alone never confirm
    that the depth is the sign's depth (no sign mask)."""
    why = []
    if not depth.get("ok"):
        why.append("DEPTH_UNAVAILABLE:" + str(depth.get("reason")))
    if depth.get("multimodal"):
        why.append("MULTIMODAL_DEPTH")
    if depth.get("high_mad") or ((depth.get("mad_m") or 0) > MAX_MAD_M):
        why.append("HIGH_MAD")
    if "NONFINITE_DEPTH_STATS" in (depth.get("risks") or []) or (depth.get("ok") and not C.finite(depth.get("median_m"), depth.get("mad_m"))):
        why.append("NONFINITE_DEPTH_STATS")
    if depth.get("ref_consistent") is not True:
        why.append("REFERENCE_INCONSISTENT_OR_MISSING(same-source consistency check)")
    if (depth.get("valid_ratio") or 0) < MIN_VALID_FRAC:
        why.append("LOW_VALID_FRACTION")
    if depth.get("uncertainty") == "HIGH" and not why:
        why.append("UNCERTAINTY_HIGH")
    if not plane_ok:
        why.append("NO_PLANE_NORMAL")
    if plane:
        rms = plane.get("rms_m")
        if rms is not None and (not C.finite(rms) or rms > MAX_PLANE_RMS_M):
            why.append(f"PLANE_RESIDUAL_HIGH:{rms}")
        vi = plane.get("valid_in")
        if vi is not None and vi < MIN_ROI_SUPPORT:
            why.append(f"LOW_ROI_SUPPORT:{vi:.2f}")
    if (n_verify or 0) < MIN_VERIFY:
        why.append(f"VERIFIED_ONLY_{n_verify or 0}x")
    return not why, why


PAYLOAD_FIELDS = ("depth", "relative_target", "crop_box_source", "roi_box_latest", "latest_frame_id", "latest_header_stamp",
                  "native_roi_source", "calibration_version", "transform_version", "depth_frame", "image_size",
                  "geometry_confirmed", "geometry_reasons", "frame_transform", "source_frame_id", "source_header_stamp")


def payload_sha256(d):
    """SEM1 P02: fingerprint of the GEOMETRY CONTENT (any change of range, pose-relative target, calibration, ROI,
    depth frame -> new fingerprint). Distinct from the entity id (observation_id) which stays stable for the target."""
    keep = {k: d.get(k) for k in PAYLOAD_FIELDS}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def observation_id(d):
    keep = {k: d.get(k) for k in ("run_id", "event_key", "boot_id", "odom_session", "generation", "snapshot_id", "track_id",
                                  "source_frame_id", "source_header_stamp", "crop_box_source", "latest_frame_id",
                                  "roi_box_latest")}
    return "obs-" + hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_observation(run_id, generation, event_key, stop, track, latest_frame, depth_rec, sem_extra=None, boot_id=None,
                      odom_session=None, trial_id=None, relative_target=None, image_size=None, n_verify=None, plane_ok=False,
                      calibration_version=None, semantic_t_mono=None, geometry_t_mono=None, plane=None, prev=None):
    """stop: the STOP dict (box_frame = source frame meta + box), track: live-tracker dict or None, depth_rec: dict(dimg,
    depth_stamp, pose, pose_stamp) of the SOURCE scene or None. Never raises for missing inputs: they become reasons."""
    bf = dict(stop.get("box_frame") or {})
    box_src = bf.get("box") or stop.get("tile") or stop.get("box")
    sem = dict(box_source=box_src, sim=stop.get("sim"), conf=None, frame_id=bf.get("frame_id"), t_mono=semantic_t_mono,
               state="VERIFIED", n_distinct=n_verify, raw=dict(sem_extra or {}))
    if sem_extra and sem_extra.get("adv") is not None and stop.get("sim") is not None:
        sem["conf"] = float(stop["sim"]) - float(sem_extra["adv"])     # legacy adv = sim - confuser (same crop)
    src_frame = dict(frame_id=bf.get("frame_id"), header_stamp=bf.get("header_stamp"))
    track_note = None
    try:
        oc = build_candidate(run_id, generation, track, sem, src_frame, latest_frame, event_key=event_key,
                             image_size=image_size, geometry_t_mono=geometry_t_mono, target_type="room_sign_candidate")
    except ValueError as e:                                   # never relabel / never accept an invalid ROI
        track_note = f"TRACK_NOT_USED:{e}"
        oc = build_candidate(run_id, generation, None, sem, src_frame, latest_frame, event_key=event_key,
                             image_size=image_size, geometry_t_mono=geometry_t_mono, target_type="room_sign_candidate")
    ref_z = stop.get("z_tile") or (stop.get("features") or {}).get("z_tile") or stop.get("z")
    depth = depth_profile(depth_rec["dimg"], box_src, ref_z=ref_z) if (depth_rec and box_src) else \
        dict(ok=False, reason="SOURCE_DEPTH_UNAVAILABLE", uncertainty="HIGH", risks=["SOURCE_DEPTH_UNAVAILABLE"])
    ok, why = geometry_confirmation(depth, n_verify, plane_ok, plane=plane)
    oc.depth = depth
    oc.stop_feasibility = None if ok else "GEOMETRY_UNCONFIRMED"
    oc.boot_id, oc.odom_session, oc.trial_id = boot_id, odom_session, trial_id
    oc.snapshot_id = bf.get("snapshot_id")
    oc.depth_frame = dict(source_frame_id=bf.get("frame_id"), depth_stamp=(depth_rec or {}).get("depth_stamp"),
                          pose_stamp=(depth_rec or {}).get("pose_stamp"), kind="aligned_depth_to_color (same scene snapshot)")
    oc.native_roi_source = [float(v) for v in stop["box"]] if stop.get("box") else None
    oc.calibration_version = calibration_version
    oc.semantic_t_mono = semantic_t_mono
    if relative_target is not None:
        oc.relative_target = relative_target
    else:
        oc.relative_target_null_reason = "no world track / current pose for this stage"
    oc.geometry_confirmed, oc.geometry_reasons = ok, why
    if track_note:
        oc.evidence["track_note"] = track_note
    d = oc.to_dict()
    oc.observation_id = observation_id(d)          # entity identity (stable)
    oc.payload_sha256 = payload_sha256(d)          # geometry content fingerprint
    if prev is not None and getattr(prev, "observation_id", None) == oc.observation_id:
        oc.payload_revision = prev.payload_revision + (0 if prev.payload_sha256 == oc.payload_sha256 else 1)
    else:
        oc.payload_revision = 1
    return oc
