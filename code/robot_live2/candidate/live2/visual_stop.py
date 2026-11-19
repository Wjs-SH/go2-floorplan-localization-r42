"""visual_stop.py — LIVE2 operator STOP request for a VERIFIED semantic candidate when no odometry frame contract exists.

2026-10-20 field (X20261020_011107 t=32 s): a room plate passed EVERY E5 layer (P2 VERIFIED: tracking persistence, SigLIP,
hard negatives, geometry checks) but E5's WOULD_STOP was DEFER_GEOMETRY_UNAVAILABLE (identity pose without a frame
contract), so no STOP_REQUEST existed at all. Without a contract the stop-distance timing cannot be computed, but the
candidate itself is verified, so the operator is asked to stop (basis VISUAL_VERIFIED_NO_GEOMETRY).
Qualifies: P2 state VERIFIED, absolute SigLIP sim >= SIM_MIN (0.09, provisional-label study), depth <= the P2 range.
Cooldown: per spot 30 s, global 8 s. Each request is an E6 observation record carrying the raw odometry at detection.
Nothing is sent to the robot. With a verified contract E5's own WOULD_STOP path decides and this module stays silent.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sci_sensor_bridge"))
from depth_freshness import max_gap_s as DF_MAX_GAP_S  # noqa: E402

SIM_MIN, SPOT_COOLDOWN_S, MIN_GAP_S = 0.09, 30.0, 8.0
HOLD_S, STILL_M, STILL_WIN_S = 600.0, 0.10, 1.5   # 2026-10-22 field: N1 O004 / N2 O003, O004 were new STOP requests (new E5 spot id,
#   same or nearby target) while the robot was STILL STOPPED from the previous request. While the robot has not moved
#   (<= STILL_M over >= STILL_WIN_S of detection-frame odometry) within HOLD_S of the last request, a candidate is logged as
#   SUPPRESSED_ALREADY_STOPPED instead of a new operator STOP. Not enough odometry -> not suppressed (fail towards STOP).
#   2026-10-26: 120 -> 600 s — a manual-focus observation (operator turns the lens) can keep the robot still for minutes.


DUP_M, DUP_S = 1.0, 60.0     # R34: same plate = its position (real odometry + fresh depth) within 1 m of a request in the last 60 s (P2 dedup values)

# R36 (11-02 field: vending machine, cabinet, door locks stopped the robot; 54 stop candidates of 6 runs labelled by eye, see SIGLIP_COMPARISON.md).
# The R34 spot expiry stays: it is what finds the plates. These gates remove what it let through:
SIM_FLOOR = float(os.environ.get("LIVE2_E5_SIM_FLOOR", "0.10"))         # sim OR the spot mean must reach this (was sim >= 0.09). Every labelled non-plate below it, one plate (a hallway sign, mean 0.117) kept by the mean
MEAN_FLOOR = float(os.environ.get("LIVE2_E5_MEAN_FLOOR", "0.11"))
TOP_MIN_UP_M = float(os.environ.get("LIVE2_E5_TOP_MIN_UP_M", "1.30"))   # top of the box above the floor (fresh depth): door locks and low objects stay below, every labelled plate above
CAPTURED_M = 1.5                                                        # a plate already photographed in this run is not requested again, at any time
AIM_OVERSHOOT_M = 1.2                                                   # shortest stop measured (11-01: 1.17 m): beyond the envelope even then -> no stop request


def world_target(xyz, od):
    """Plate position from the detection-frame odometry and the region's camera-relative (forward, left) offset, or None. Used only to tell
    the same plate from another one; relative distances do not need the (missing) map frame contract."""
    if not xyz or (od or {}).get("x") is None or (od or {}).get("yaw") is None:
        return None
    c, s = math.cos(float(od["yaw"])), math.sin(float(od["yaw"]))
    return [round(float(od["x"]) + c * xyz[0] - s * xyz[1], 3), round(float(od["y"]) + s * xyz[0] + c * xyz[1], 3)]


def body_point(box, z, K, which="center"):
    """(fwd, left, up from floor) of the box centre or its top edge, with the E6 camera model. None when K is unknown."""
    import e6_core as G
    if not K:
        return None
    b = list(box)
    if which == "top":
        b[3] = b[1]
    return G.target_in_body(b, float(z), K)


def aim_expected_ok(box, z, K, env, overshoot=AIM_OVERSHOOT_M):
    """-> (ok, detail). Would the pan-tilt reach this target from the shortest measured stop along the current heading? Display of the same calculation exists in e6_tracker.pre_stop_aim. Unknown -> ok."""
    try:
        import e6_core as G
        import e6_precision_bridge as B
        f, l, up = body_point(box, z, K)
        f2 = f - overshoot
        if f2 <= 0.3:
            return False, dict(why="target passed before the shortest stop", fwd=round(f, 2), left=round(l, 2))
        u_, v_, z_ = G.body_to_pixel(f2, l, up, K)
        pc = G.calibrated_aim(u_, v_, z_)
        pa = B.path_check(pc["pan_cmd"], pc["tilt_cmd"], env or B.load_envelope())
        return bool(pc["inside_limits"] and pa["ok"]), dict(pan=pc["pan_cmd"], tilt=pc["tilt_cmd"], outside=pa["outside"], overshoot_m=overshoot)
    except Exception as ex:                # noqa: BLE001  fail towards the operator's request
        return True, dict(unknown=f"{type(ex).__name__}: {ex}")


class VisualStop:
    def __init__(self, run_dir, range_m=5.0, contract_ok=False, K_fn=None, state_dir=None, envelope=None):
        self.run, self.range_m, self.contract_ok = run_dir, float(range_m), contract_ok
        self.K_fn, self.state_dir, self.envelope = K_fn, state_dir, envelope
        self.n_gate = {}
        self.oid_target = {}
        self.last_wall, self.spot_wall, self.n, self.last = -1e18, {}, 0, []
        try:                                          # R42: the keeper restarts this process in the same run dir; numbering from 1 again repeated observation ids that e6_tracker had seen and dropped
            for _l in open(Path(run_dir) / "live2_stop_requests.jsonl"):
                _id = str((json.loads(_l) or {}).get("observation_id") or "")
                if "_O" in _id and _id.rsplit("_O", 1)[1].isdigit():
                    self.n = max(self.n, int(_id.rsplit("_O", 1)[1]))
        except (OSError, TypeError, ValueError, AttributeError):
            pass
        self.hist, self.n_suppressed = [], 0           # (stamp, x, y) of detection-frame odometry
        self.n_depth_refused, self.n_duplicate, self.targets = 0, 0, []

    def feed(self, rec, ts, det_line, now=None):
        """rec = E5 scene record (regions with dec), det_line = LIVE2_DETECTION_POSE line. -> observation dict or None."""
        if self.contract_ok or det_line is None:
            return None
        now = time.time() if now is None else now
        od = det_line.get("odom_raw") or {}
        dt = det_line.get("depth_dt_s")
        if dt is None or abs(float(dt)) > DF_MAX_GAP_S():     # R34: a distance from an old depth image never makes a stop request
            self.n_depth_refused += 1
            return None
        if ts is not None and od.get("x") is not None:
            self.hist = [h for h in self.hist if ts - h[0] <= 3.0 * STILL_WIN_S] + [(float(ts), float(od["x"]), float(od["y"]))]
        for r in sorted(rec.get("regions") or [], key=lambda q: -(q.get("sim") or -1)):   # R36: the strongest region of the scene, not the first listed
            p2 = (r.get("dec") or {}).get("P2") or {}
            sim, z, spot = r.get("sim") or -1, r.get("z"), r.get("spot")
            if p2.get("state") != "VERIFIED" or sim < SIM_MIN or z is None or z > self.range_m:
                continue
            if now - self.last_wall < MIN_GAP_S or now - self.spot_wall.get(spot, -1e18) < SPOT_COOLDOWN_S:
                continue
            why = self.gate(r, od, now)
            if why:
                self._note(why, r, ts, rec, now)
                continue
            if now - self.last_wall <= HOLD_S and self.still(ts):
                self.n_suppressed += 1
                self.spot_wall[spot] = now
                if self.run is not None:
                    with open(self.run / "live2_stop_suppressed.jsonl", "a") as f:     # separate file: e6_tracker tails requests
                        f.write(json.dumps(dict(kind="SUPPRESSED_ALREADY_STOPPED", target_track_id=spot, detection_timestamp=ts, sim=sim,
                                                depth_m=z, bbox=r.get("box"), image_file=f"events/scene_{rec.get('t', 0):07.1f}.jpg",
                                                wall=now, since_last_request_s=round(now - self.last_wall, 1)), default=str) + "\n")
                return None
            wt = world_target(r.get("xyz"), od)
            if self.duplicate(wt, now):
                self.n_duplicate += 1
                self.spot_wall[spot] = now
                if self.run is not None:
                    with open(self.run / "live2_stop_suppressed.jsonl", "a") as f:
                        f.write(json.dumps(dict(kind="SUPPRESSED_SAME_TARGET", target_track_id=spot, detection_timestamp=ts, sim=sim, depth_m=z, bbox=r.get("box"),
                                                world_target=wt, wall=now, within_m=DUP_M, within_s=DUP_S), default=str) + "\n")
                return None
            if wt is not None:
                self.targets.append((now, wt))
            self.n += 1
            self.last_wall = self.spot_wall[spot] = now
            self.oid_target[f"{getattr(self.run, 'name', 'run')}_O{self.n:03d}"] = wt
            obs = dict(kind="VISUAL_VERIFIED_STOP", source="AUTO_E5", basis="VISUAL_VERIFIED_NO_GEOMETRY",
                       observation_id=f"{getattr(self.run, 'name', 'run')}_O{self.n:03d}", target_track_id=spot, detection_timestamp=ts,
                       image_frame_id=det_line.get("frame_id"), image_file=f"events/scene_{rec.get('t', 0):07.1f}.jpg",
                       bbox=r.get("box"), depth_m=z, depth_dt_s=dt, sim=sim, scene_t=rec.get("t"), odom_pose_at_detection=det_line.get("odom_raw"),   # R37: without depth_dt_s the pre-stop aim guidance was always AIM_TARGET_UNCERTAIN (all 8 stops of 11-02)
                       odom_frame=det_line.get("odom_frame"), pose_contract="MISSING", wall=now, mono=time.monotonic(),
                       note="operator request only; nothing is sent to the robot. Stop-distance timing unavailable without a frame contract")
            self.last = (self.last + [obs])[-5:]
            if self.run is not None:
                with open(self.run / "live2_stop_requests.jsonl", "a") as f:
                    f.write(json.dumps(obs, default=str) + "\n")
            return obs
        return None

    def K(self):
        try:
            return self.K_fn() if self.K_fn else None
        except Exception:                  # noqa: BLE001
            return None

    def gate(self, r, od, now):
        """R36 -> reason not to ask the operator to stop, or None. Each reason is counted and logged; none of them moves anything."""
        if self.state_dir:                 # R39: E4's automatic choice: signs are used only where structure does not separate the floors. A photo in progress is not affected: this only withholds a NEW request
            import auto_fsm as AF
            why = AF.stop_gate(self.state_dir, now)
            if why:
                return why
        sim, msim, z, box = float(r.get("sim") or -1), r.get("msim"), r.get("z"), r.get("box")
        if sim < SIM_FLOOR and (msim is None or float(msim) < MEAN_FLOOR):
            return "SUPPRESSED_LOW_SCORE"
        K = self.K()
        if K and box:
            tp = body_point(box, z, K, "top")
            if tp is not None and tp[2] < TOP_MIN_UP_M:
                return "SUPPRESSED_BELOW_PLATE_HEIGHT"
        wt = world_target(r.get("xyz"), od)
        if wt is not None and any(((wt[0] - c[0]) ** 2 + (wt[1] - c[1]) ** 2) ** 0.5 <= CAPTURED_M for c in self.captured()):
            return "SUPPRESSED_ALREADY_PHOTOGRAPHED"
        if K and box:
            ok, det = aim_expected_ok(box, z, K, self.envelope, overshoot=AIM_OVERSHOOT_M)
            r["_aim_expected"] = det
            if not ok:
                return "SUPPRESSED_NOT_AIMABLE"
        return None

    def captured(self):
        """World positions of this run's plates with a stored photo (e6_precision_results.jsonl + the position recorded at the request)."""
        if not self.state_dir:
            return []
        out = []
        try:
            with open(Path(self.state_dir) / "e6_precision_results.jsonl") as f:
                for ln in f:
                    try:
                        d = json.loads(ln)
                    except ValueError:
                        continue
                    t = self.oid_target.get(d.get("observation_id"))
                    if t is not None and d.get("status") in ("CAPTURE_STORED", "CAPTURED"):
                        out.append(t)
        except OSError:
            pass
        return out

    def _note(self, why, r, ts, rec, now):
        self.n_gate[why] = self.n_gate.get(why, 0) + 1
        if self.run is not None:
            with open(self.run / "live2_stop_suppressed.jsonl", "a") as f:
                f.write(json.dumps(dict(kind=why, target_track_id=r.get("spot"), detection_timestamp=ts, sim=r.get("sim"), msim=r.get("msim"), depth_m=r.get("z"), bbox=r.get("box"),
                                        image_file=f"events/scene_{rec.get('t', 0):07.1f}.jpg", aim_expected=r.get("_aim_expected"), wall=now), default=str) + "\n")

    def duplicate(self, wt, now):
        return wt is not None and any(now - w <= DUP_S and ((wt[0] - p[0]) ** 2 + (wt[1] - p[1]) ** 2) ** 0.5 <= DUP_M for w, p in self.targets)

    def still(self, ts):
        if ts is None:
            return False
        w = [h for h in self.hist if ts - h[0] <= STILL_WIN_S * 2]
        if len(w) < 2 or w[-1][0] - w[0][0] < STILL_WIN_S:
            return False
        return max(((h[1] - w[-1][1]) ** 2 + (h[2] - w[-1][2]) ** 2) ** 0.5 for h in w) <= STILL_M

    def status(self):
        return dict(n=self.n, last=list(self.last), n_suppressed_already_stopped=self.n_suppressed, n_depth_refused=self.n_depth_refused, n_duplicate_target=self.n_duplicate, gates=dict(self.n_gate))
