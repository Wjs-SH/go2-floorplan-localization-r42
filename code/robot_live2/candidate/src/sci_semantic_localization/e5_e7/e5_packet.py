#!/usr/bin/env python3
"""e5_packet.py — build the canonical E5_OBSERVATION_PACKET (schema e5_observation_packet/v1) from raw evidence.

Sources (copied into the trial folder, never edited):
  runner/events.jsonl            canonical live_event_runner stop records (t, stamp, picks, z, xyz, pose, image, retarget, aim)
  runner/siglip_log.jsonl        e5_runner_logged.py: every SigLIP request/response (regions, sim, adv, board, ceil, tiles)
  console/aim_log.jsonl          robot aim_console record (time, u, v, depth_m, dist_used_m, pan, tilt, clipped, aim_sec,
                                 focus, total_sec, image)
  console/<aim image>.jpg        pan/tilt capture (robot out/aim/aim_<ts>.jpg)
  reading/e5_reading.json        e5_read_offboard.py raw VLM reading
  calibration json               ~/pantilt_calib/configs/pantilt_calib_multi_20260928.json (intrinsics, servo, mount)

Geometry: the observation is the RealSense pixel/depth that was aimed at (retarget box + z when present, else the
stop pick), converted with geometry.rs_pixel_to_body -> body bearing/range. Pan/tilt angles are recorded and only
cross-checked (geometry.servo_to_rs_direction), never used as the bearing.
E6 consumes the written packet file directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

try:
    from . import geometry as G
except ImportError:
    import geometry as G  # type: ignore

SCHEMA = "e5_observation_packet/v1"
R1_SCHEMA = "e5_observation_packet/v1+r1.1"          # v1 fields kept; R1 adds identity/status/clock-domain blocks
CLOCK_DOMAINS = dict(stop_t="scene_rel_s (seconds since the E5 run's first RealSense frame; NOT unix)",
                     image_stamp="ros_header_s (RealSense colour header; sync with wall clock NOT verified)",
                     retarget_t="as recorded by E5 (ros_header_s for live runs, replay scene clock for replay)",
                     aim_time="console wall clock string YYYYmmdd_HHMMSS (1 s resolution)",
                     reading_request="wall_unix_s", reading_answer="wall_unix_s", created_unix="wall_unix_s")
IDENTITY_KEYS = ("schema_version", "run_id", "event_id", "event_key", "track_id", "request_id", "attempt_id", "capture_id",
                 "frame_id", "camera_id", "trial_id", "exp_id", "generation_id", "odom_session_id", "calibration_id",
                 "map_bundle_id", "session_id", "boot_id")


def _sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None


def _jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def build_packet(trial: Path, calib_path: Path, event_index: int = 0, aim_index: int | None = None,
                 floor_hint: str | None = None, reading: dict | None = None, identity: dict | None = None) -> dict:
    """reading: R1 reader result record (preferred over reading/e5_reading.json); identity: the E5 job (identity keys,
    capture_status, pantilt_ok, odom_frame). Without them the legacy v1 behaviour is kept and marked as such."""
    events = _jsonl(trial / "runner" / "events.jsonl")
    if not events:
        raise SystemExit("no runner/events.jsonl records")
    ev = events[event_index]
    aims = _jsonl(trial / "console" / "aim_log.jsonl")
    reading_p = trial / "reading" / "e5_reading.json"
    r1 = reading is not None
    if reading is None:
        reading = json.loads(reading_p.read_text(encoding="utf-8")) if reading_p.exists() else None
    sig = _jsonl(trial / "runner" / "siglip_log.jsonl")
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    intr = {k: calib["realsense_color"][k] for k in ("fx", "fy", "cx", "cy")}
    mount = calib.get("realsense_mount_deg", {"yaw": 0, "pitch": 0, "roll": 0})

    rt = ev.get("retarget") or {}
    if rt.get("box") is not None and rt.get("z") is not None:
        box, z, src = rt["box"], float(rt["z"]), "retarget(box,z) — the pixel actually sent to /aim"
        pose = rt.get("pose") or ev.get("pose")
    else:
        box, z, src = ev["picks"][0], float(ev["z"][0]), "stop pick[0]"
        pose = ev.get("pose")
    u, v = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    try:                                   # R1: invalid depth -> null observation + reason (E6 refuses it), never clamped
        p_body = G.rs_pixel_to_body(u, v, z, intr, mount)
        beta, rng = G.body_bearing_range(p_body)
        sb = G.bearing_sigma(0.5 * (box[2] - box[0]), intr)
        sr = G.range_sigma(z)
        geo_status = None
    except G.InvalidDepth as e:
        p_body, beta, rng, sb, sr = [float("nan")] * 3, None, None, None, None
        geo_status = dict(status=G.InvalidDepth.reason_code, why=str(e), depth_z_m=repr(z))

    aim = None
    if aims:
        if aim_index is not None:
            aim = aims[aim_index]
        else:  # NOTE: the LAST aim record of this trial (legacy comment said 'first after the stop'); cross-check only
            aim = aims[-1]
    cross = None
    if aim and "servo" in calib:
        d = G.servo_to_rs_direction(float(aim["pan"]), float(aim["tilt"]), calib["servo"])
        cross = dict(pan_cmd=aim["pan"], tilt_cmd=aim["tilt"], clipped=aim.get("clipped"),
                     servo_bearing_deg_axis_offsets_ignored=math.degrees(math.atan2(d[1], d[0])),
                     rs_bearing_deg=None if beta is None else math.degrees(beta),
                     note="commanded angles; actual servo angle not measured (uncertainty: servo deadband ~1 deg, "
                          "lash 5 deg approach, FIELD_GUIDE §4)")

    labels = (reading.get("labels") if r1 else reading.get("label_alternatives")) if reading else []
    labels = labels or []
    ident = identity or {}
    pk = dict(
        schema=SCHEMA,
        packet_id=f"{trial.name}_ev{event_index:03d}",
        created_unix=time.time(),
        trial_dir=str(trial),
        floor_hint=floor_hint,
        timestamps=dict(stop_t=ev.get("t"), image_stamp=ev.get("stamp"), retarget_t=rt.get("t"),
                        aim_time=aim.get("time") if aim else None,
                        reading_request=reading.get("t_request") if reading else None,
                        reading_answer=reading.get("t_answer") if reading else None),
        candidate=dict(source=src, box_px=box, depth_z_m=z, u=u, v=v, xyz_world_odom=ev.get("xyz"),
                       event_image=ev.get("image"), nobs=ev.get("nobs")),
        siglip=dict(n_scored_frames=len(sig), log=str(trial / "runner" / "siglip_log.jsonl") if sig else None,
                    prompts=sig[0]["prompts"] if sig else "NOT_LOGGED",
                    note=None if sig else "runner was not wrapped by e5_runner_logged.py: per-region scores unavailable"),
        observation=dict(frame_chain="RS pixel -> RS optical -> body axes (R_mount from calibration) ; map heading later in E6",
                         bearing_body_rad=beta, range_m=rng, bearing_sigma_rad=sb, range_sigma_m=sr,
                         geometry_status=geo_status,
                         p_body_m=[float(c) for c in p_body], intrinsics=intr, mount_deg=mount,
                         calibration_file=str(calib_path), calibration_sha256=_sha(calib_path)),
        odom=dict(pose_at_stop=ev.get("pose"), pose_at_aim=pose,
                  frame=ident.get("odom_frame") or "UNKNOWN",
                  frame_null_reason=None if ident.get("odom_frame") else "E5 job did not record the odom frame (legacy label was hard-coded)"),
        pantilt=cross,
        capture=dict(aim_record=aim, image=(aim or {}).get("image")),
        semantic=dict(reader=(reading or {}).get("engine"), ocr_engine=(reading or {}).get("ocr_engine", "NONE_INSTALLED"),
                      raw_answer=(reading or {}).get("raw_answer"), label_alternatives=labels,
                      final_label=(labels[0]["label"] if labels else None),
                      confidence=(labels[0]["p"] if labels else None),
                      reading_file=str(reading_p) if reading else None),
        timing=dict(aim_sec=(aim or {}).get("aim_sec"), focus=(aim or {}).get("focus"), total_aim_sec=(aim or {}).get("total_sec"),
                    reading_sec=(reading or {}).get("sec")),
        failure_reason=None if (labels and labels[0]["label"] not in ("__NONE__",)) else
        ((reading or {}).get("reason_code") or "NO_READABLE_LABEL" if r1 else "NO_READABLE_LABEL"),
        gt_used=False,
    )
    if r1 or identity is not None:
        pk["schema"] = R1_SCHEMA
        pk["identity"] = {k: ident.get(k) for k in IDENTITY_KEYS}
        pk["identity_null_reason"] = {k: "not provided by the E5 job" for k in IDENTITY_KEYS if ident.get(k) is None}
        pk["status"] = dict(capture_status=ident.get("capture_status"), pantilt_ok=bool(ident.get("pantilt_ok")),
                            proxy=ident.get("capture_status") == "CAPTURE_PROXY",
                            semantic_status=(reading or {}).get("semantic_status"),
                            reading_reason_code=(reading or {}).get("reason_code"),
                            reading_terminal=(reading or {}).get("terminal"),
                            e7="NOT_WIRED")
        if r1:
            pk["semantic"].update(reader="R1 ReaderCore (OCR-first)", ocr_engine=((reading.get("ocr") or {}).get("engine")),
                                  raw_answer=json.dumps(dict(ocr=(reading.get("ocr") or {}).get("raw_text"),
                                                             vlm=(reading.get("vlm") or {}).get("parsed")), ensure_ascii=False),
                                  reading_file=None)
        pk["timestamps_clock_domain"] = CLOCK_DOMAINS
        pk["timestamps"]["reading_request"] = ((reading or {}).get("first_seen") or {}).get("wall")
        pk["timestamps"]["reading_answer"] = ((reading or {}).get("published") or {}).get("wall")
    else:
        pk["timestamps_clock_domain"] = CLOCK_DOMAINS
        pk["identity"] = None
        pk["identity_null_reason"] = "legacy v1 call without an E5 job identity"
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--calib", default=str(Path.home() / "pantilt_calib" / "configs" / "pantilt_calib_multi_20260928.json"))
    ap.add_argument("--event", type=int, default=0)
    ap.add_argument("--aim", type=int, default=None)
    ap.add_argument("--floor-hint", default=None)
    a = ap.parse_args()
    trial = Path(a.trial)
    pk = build_packet(trial, Path(a.calib), a.event, a.aim, a.floor_hint)
    out = trial / "E5_OBSERVATION_PACKET.json"
    out.write_text(json.dumps(pk, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
