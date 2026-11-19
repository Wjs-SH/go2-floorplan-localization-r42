"""e6_tracker.py — LIVE2 E6 observation state machine (file-only; runs inside e34_node). Never commands a device.

Request sources (both use the CURRENT robot sensors; the source is kept on every record):
  AUTO_E5            <e5 run>/live2_stop_requests.jsonl (verified semantic candidate, observe_node / visual_stop.py)
  OPERATOR_ASSISTED  <state>/e6_operator_requests.jsonl (operator picked a target on the live RGB; console resolves depth)
                     -> never counted as automatic E5 detection or E8 automatic success.
States: WAIT_STOP -> STOPPED_TARGET_READY (target + predicted pan/tilt at the stop pose) | EXPIRED_NO_STOP (45 s)
        | INVALID_INPUT (no odometry at detection / no depth / no camera K).
Pan-tilt, IMX477, OCR/VLM: BLOCKED_BY_HARDWARE_APPROVAL (approval B) — predicted angles are shown, nothing is sent.
"""
import json
import math
import os
import time
from pathlib import Path

import e6_core as G


def od_gap(r):
    o = r.get("odom_pose_at_detection") or {}
    return o.get("gap_s") if o.get("status") == "INTERPOLATED" else o.get("dt_s")


REDETECT_WIN_PX, REDETECT_MAX_DT_S = 80.0, 2.0
# R36 (11-02 field O007, O008, O002): the region nearest the prediction was taken as "the target" whatever it was: O007 sim 0.047 at 1.6 m where the plate was at 3.2 m; O008 a 76x199 px
# door region whose centre is the lock. The photos show the door lock centred and the plate at or above the top edge, and OCR read nothing.
REDETECT_MIN_SIM, REDETECT_Z_REL, REDETECT_UP_TOL_M = 0.085, 0.35, 0.30
PLATE_W_M, LOOSE_BOX = 0.30, 1.5       # a region wider or taller than LOOSE_BOX x this size at its depth is more than a plate: its centre is not the plate
PLATE_UP_M = 1.72                      # measured: tight boxes on room plates in the 11-02 runs, body height 1.70-1.83 m (camera model values)
TALL_BOX = 1.5


def aim_up(bbox, z, K, up):
    """Height to aim at for a box that holds more than the plate (plate + lock + transom): the plate's height, kept inside the box. Tight box: its own centre. -> (up, basis)"""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h / w < TALL_BOX:
        return up, "BOX_CENTRE"
    _, _, top = G.target_in_body([bbox[0], bbox[1], bbox[2], bbox[1]], z, K)
    _, _, bot = G.target_in_body([bbox[0], bbox[3], bbox[2], bbox[3]], z, K)
    return min(max(PLATE_UP_M, bot), top), "PLATE_HEIGHT_IN_TALL_BOX"


PRE_STOP_OVERSHOOT_M = (1.2, 1.9, 2.6)   # 11-01 field: robot travel from the stop request to rest 1.17-2.60 m (operator + walking gait), measured, not a promise


def pre_stop_aim(pose_det, tu, tv, up, K, depth_dt_s, z, envelope=None):
    """R34 (11-01 O001 / O004 / O007: pan 155.0 / 136.8 / 138.8 or 8.1 deg refused AFTER the operator had stopped). Predicts, BEFORE the stop, the servo command for each likely stopping
    distance along the current heading. Display guidance only: no command is sent and no limit is changed. Stale depth -> no number at all."""
    import e6_precision_bridge as B
    max_gap = float(os.environ.get("LIVE2_DEPTH_MAX_GAP_S", "0.5"))
    if depth_dt_s is None or abs(float(depth_dt_s)) > max_gap:
        return dict(status="AIM_TARGET_UNCERTAIN", why=f"depth not fresh (colour-depth gap {depth_dt_s}): no aim prediction", rows=[], command_sent=False)
    env = envelope or B.load_envelope()
    rows = []
    for s in PRE_STOP_OVERSHOOT_M:
        pose = (pose_det[0] + s * math.cos(pose_det[2]), pose_det[1] + s * math.sin(pose_det[2]), pose_det[2])
        pt = G.pan_tilt_prediction(pose, (tu, tv), up)
        try:
            u_, v_, z_ = G.body_to_pixel(pt["target_fwd_m"], pt["target_left_m"], up, K)
            pc = G.calibrated_aim(u_, v_, z_)
        except Exception as ex:                # noqa: BLE001
            rows.append(dict(overshoot_m=s, ok=False, why=f"{type(ex).__name__}: {ex}"))
            continue
        pan, tilt = pc.get("pan_cmd"), pc.get("tilt_cmd")
        e = B.PRED_ERR_ODOM * s + B.PRED_ERR_DEPTH * z + B.PRED_ERR_MOUNT
        deg = math.degrees(math.atan2(e, max(pt["range_m"], 0.1)))
        pa = B.path_check(pan, tilt, env) if pan is not None and tilt is not None else dict(ok=False)
        why = None if pa.get("ok") and deg <= B.PRED_MAX_DEG else ("OUTSIDE_VERIFIED_ENVELOPE" if not pa.get("ok") else f"UNCERTAINTY_{deg:.1f}_DEG")
        rows.append(dict(overshoot_m=s, pan_cmd=pan, tilt_cmd=tilt, range_m=pt["range_m"], uncertainty_deg=round(deg, 1), ok=why is None, why=why))
    good = [x["overshoot_m"] for x in rows if x["ok"]]
    if len(good) == len(rows):
        st, txt = "AIM_OK_AT_TYPICAL_STOP", "평소처럼 정지하면 조준할 수 있습니다."
    elif good:
        st, txt = "STOP_EARLY", f"요청 후 {max(good):.1f} m 안에 정지해야 조준할 수 있습니다. 더 가면 조준이 거부됩니다."
    else:
        st, txt = "AIM_EXPECTED_REFUSED", "이 위치에서 정지하면 조준이 거부될 것으로 예상됩니다. 물러나 표지판을 더 정면으로 보거나 건너뛰세요."
    return dict(status=st, guidance=txt, rows=rows, command_sent=False, basis="odometry + fresh depth; straight-line stop assumption; envelope unchanged")


def same_object(g, expect):
    """R36 -> why this E5 region is not the requested target, or None."""
    if not expect:
        return None
    if (g.get("sim") or 0) < REDETECT_MIN_SIM:
        return f"sim {g.get('sim')} < {REDETECT_MIN_SIM}"
    z0 = expect.get("z")
    if z0 and abs(float(g["z"]) - z0) > REDETECT_Z_REL * z0:
        return f"depth {g['z']} vs expected {z0:.2f} m"
    K, up0 = expect.get("K"), expect.get("up")
    if K and up0 is not None:
        b = g["box"]
        _, _, top = G.target_in_body([b[0], b[1], b[2], b[1]], float(g["z"]), K)
        _, _, bot = G.target_in_body([b[0], b[3], b[2], b[3]], float(g["z"]), K)
        if not (bot - REDETECT_UP_TOL_M <= up0 <= top + REDETECT_UP_TOL_M):
            return f"height {bot:.2f}..{top:.2f} m vs expected {up0:.2f} m"
    return None


def repredict(o, samples, K, e5_status=None):
    """R36: the same prediction as E6Tracker.step for the robot's CURRENT pose, after the operator repositioned it. R37: then a NEW look: the current E5 scene is searched for the same plate (same_object); an old redetection is never reused. -> new observation dict."""
    t, x, y, yaw = samples[-1][:4]
    pose = G.raw_to_planar(x, y, yaw)
    tu, tv = o["target_pose_in_odom"]["planar_uv"]
    up = o["target_pose_in_odom"]["up_m"]
    pt = G.pan_tilt_prediction(pose, (tu, tv), up)
    ts = o.get("source_timestamp")
    n = dict(o, odom_pose_at_stop=dict(t=t, x=x, y=y, yaw=yaw, planar=[round(v, 3) for v in pose]), redetection=dict(status="NOT_ATTEMPTED_AFTER_REPOSITION"), stopped_wall=time.time(),
             target_pose_at_stop=dict(fwd=pt["target_fwd_m"], left=pt["target_left_m"], range_m=pt["range_m"]), pan_target=pt["pan_deg"], tilt_target=pt["tilt_deg"], prediction=pt,
             detect_to_stop_path_m=None if ts is None else G.path_between(samples, float(ts), t))
    u_, v_, z_ = G.body_to_pixel(pt["target_fwd_m"], pt["target_left_m"], up, K)
    n["pantilt_command_prediction"] = dict(G.calibrated_aim(u_, v_, z_), predicted_pixel=[round(u_, 1), round(v_, 1)], predicted_depth_m=round(z_, 2), chain="reposition: odom -> body -> RealSense -> pan-tilt")
    if e5_status is not None:
        tr = E6Tracker.__new__(E6Tracker)
        tr.e5_status = Path(e5_status)
        n["redetection"] = tr.redetect(u_, v_, t, expect=dict(z=z_, up=up, K=K))
    return n


def reposition_guidance(o, K, env):
    """R36: how the operator can bring a refused target inside the verified envelope without widening it. Display only. -> dict(text, moves). Straight-back first, then turning towards the target."""
    import e6_precision_bridge as B
    ts = o.get("target_pose_at_stop") or {}
    f0, l0, up = ts.get("fwd"), ts.get("left"), (o.get("target_pose_in_odom") or {}).get("up_m")
    if f0 is None or l0 is None or up is None or not K:
        return dict(text="목표 위치를 계산할 수 없습니다. 이번 표지판은 건너뛰세요.", moves=[], ok=False)

    def ok(f, l):
        if f <= 0.3:
            return False
        try:
            pc = G.calibrated_aim(*G.body_to_pixel(f, l, up, K))
            return bool(pc["inside_limits"] and B.path_check(pc["pan_cmd"], pc["tilt_cmd"], env)["ok"])
        except Exception:                      # noqa: BLE001
            return False
    moves = []
    for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        if ok(f0 + d, l0):
            moves.append(dict(kind="BACK", m=d, ko=f"뒤로 약 {d:.1f} m 이동"))
            break
    for deg in (15, 30, 45, 60, 90):
        th = math.radians(deg if l0 > 0 else -deg)
        f2, l2 = f0 * math.cos(th) + l0 * math.sin(th), -f0 * math.sin(th) + l0 * math.cos(th)
        if ok(f2, l2):
            moves.append(dict(kind="TURN", deg=deg, side="LEFT" if l0 > 0 else "RIGHT", ko=f"{'왼쪽' if l0 > 0 else '오른쪽'}으로 약 {deg}° 회전"))
            break
    if not moves:
        return dict(text="이 자리에서 뒤로 가거나 돌아도 검증된 팬틸트 범위에 들어오지 않습니다. 이번 표지판은 건너뛰세요.", moves=[], ok=False)
    return dict(text="조이스틱으로 " + " 또는 ".join(m["ko"] for m in moves) + "한 뒤 정지하고 [재조준]을 누르세요.", moves=moves, ok=True, target_body=dict(fwd=f0, left=l0, up=up))


class E6Tracker:
    def __init__(self, state_dir, hub_dir, e5_status_path, clock=time.time, experiment_json=None):
        self.state = Path(state_dir)
        self.exp_json = Path(experiment_json) if experiment_json else self.state.parent / "ctl" / "experiment.json"
        self.n_ev = 0
        self._K = None
        self.hub = Path(hub_dir)
        self.e5_status = Path(e5_status_path)
        self.clock = clock
        self.obs = []                       # newest last
        self.seen = set()
        self.offsets = {}

    # ---------------- inputs
    def _tail(self, path):
        try:
            with open(path) as f:
                f.seek(self.offsets.get(str(path), 0))
                lines = f.readlines()
                self.offsets[str(path)] = f.tell()
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out

    def hub_index(self):
        try:
            return json.loads((self.hub / "index.json").read_text())
        except (OSError, ValueError):
            return None

    def requests(self):
        reqs = []
        try:
            run = json.loads(self.e5_status.read_text()).get("run")
        except (OSError, ValueError):
            run = None
        if run:
            reqs += [dict(r, source=r.get("source", "AUTO_E5")) for r in self._tail(Path(run) / "live2_stop_requests.jsonl")]
        reqs += [dict(r, source="OPERATOR_ASSISTED") for r in self._tail(self.state / "e6_operator_requests.jsonl")]
        return [r for r in reqs if r.get("observation_id") and r["observation_id"] not in self.seen]

    # ---------------- per observation
    def experiment_id(self):
        try:
            return json.loads(self.exp_json.read_text()).get("exp_id")
        except (OSError, ValueError):
            return None

    def new_observation(self, r, K, hyps, sem, samples=()):
        self.seen.add(r["observation_id"])
        now = self.clock()
        at_req = G.stop_state(list(samples), samples[-1][0] if samples else now, wall_now=now) if samples else dict(stopped=None, why="NO_ODOMETRY")
        o = dict(experiment_id=self.experiment_id(), observation_id=r["observation_id"], source=r["source"],
                 source_mode=r["source"], target_track_id=r.get("target_track_id"), source_frame_id=r.get("image_frame_id"),
                 source_timestamp=r.get("detection_timestamp"), created_timestamp=now, module="E6",
                 motion_at_request=at_req, pan_tilt_extrinsics=None,
                 detection_timestamp=r.get("detection_timestamp"), image_frame_id=r.get("image_frame_id"), image_file=r.get("image_file"),
                 bbox=r.get("bbox"), depth=r.get("depth_m"), odom_pose_at_detection=r.get("odom_pose_at_detection"),
                 odom_pose_at_stop=None, camera_extrinsics=G.CAMERA_EXTRINSICS, target_pose_in_odom=None, target_pose_at_stop=None,
                 pan_target=None, tilt_target=None, semantic_map_candidates=[], created_wall=self.clock(), state="WAIT_STOP", why=None,
                 observation_confidence=dict(sim=r.get("sim"), basis=r.get("basis"), source=r["source"]),
                 pantilt="BLOCKED_BY_HARDWARE_APPROVAL (approval B) — prediction only", ocr_vlm="BLOCKED_BY_HARDWARE_APPROVAL")
        o["uncertainty"] = dict(odom_interp_gap_s=od_gap(r), depth_dt_s=r.get("depth_dt_s"),
                                extrinsics=G.CAMERA_EXTRINSICS["status"], pan_tilt_extrinsics="UNKNOWN (not calibrated / not read)")
        od = r.get("odom_pose_at_detection") or {}
        if od.get("status") not in ("INTERPOLATED", "NEAREST") or not r.get("bbox") or not r.get("depth_m") or not K:
            o.update(state="INVALID_INPUT", why=f"odom {od.get('status')} depth {r.get('depth_m')} K {'ok' if K else 'missing'}")
            return o
        pose_det = G.raw_to_planar(od["x"], od["y"], od["yaw"])
        fwd, left, up = G.target_in_body(r["bbox"], float(r["depth_m"]), K)
        up_box = up
        up, up_basis = aim_up(r["bbox"], float(r["depth_m"]), K, up)
        o["aim_height"] = dict(box_centre_up_m=round(up_box, 3), aim_up_m=round(up, 3), basis=up_basis)
        tu, tv = G.body_to_planar(pose_det, fwd, left)
        xl, yl = G.odom_link_point(tu, tv)
        o["target_pose_in_odom"] = dict(planar_uv=[round(tu, 3), round(tv, 3)], up_m=round(up, 3), odom_link_xy=[round(xl, 3), round(yl, 3)],
                                        body_at_detection=dict(fwd=round(fwd, 2), left=round(left, 2), up=round(up, 2)),
                                        frame="planar (u fwd at FAST-LIO start, v left); extrinsics UNVERIFIED")
        cands = []
        for h in hyps[:6]:
            try:
                px, py = G.map_projection(h["rec"], xl, yl)
            except (KeyError, TypeError, ValueError):
                continue
            c = dict(hyp=h["id"], floor=h["floor"], target_px=[round(px, 1), round(py, 1)], score=h.get("score"))
            sm = sem.get(h["floor"]) if sem is not None else None
            if sm:
                best = min(sm["landmarks"], key=lambda lm: math.hypot(*(lm["center"] - [px, py])), default=None)
                if best is not None:
                    c.update(nearest_label=best["label"], nearest_kind=best["kind"],
                             nearest_dist_m=round(math.hypot(*(best["center"] - [px, py])) / h["rec"]["ppm"], 2))
            cands.append(c)
        o["semantic_map_candidates"] = cands
        try:
            o["pre_stop_aim"] = pre_stop_aim(pose_det, tu, tv, up, K, r.get("depth_dt_s"), float(r["depth_m"]))
        except Exception as ex:                # noqa: BLE001  guidance only: never disturbs the observation
            o["pre_stop_aim"] = dict(status="AIM_TARGET_UNCERTAIN", why=f"{type(ex).__name__}: {ex}", rows=[], command_sent=False)
        return o

    def step(self, o, samples, now):
        if o["state"] != "WAIT_STOP":
            return False
        if now - o["created_wall"] > G.WAIT_STOP_TIMEOUT_S:
            o.update(state="EXPIRED_NO_STOP", why=f"no stop within {G.WAIT_STOP_TIMEOUT_S:.0f} s")
            return True
        st = G.stop_state(samples, samples[-1][0] if samples else now, wall_now=now)
        o["stop_check"] = st
        if not st["stopped"]:
            return False
        t, x, y, yaw = samples[-1][:4]
        pose_stop = G.raw_to_planar(x, y, yaw)
        o["odom_pose_at_stop"] = dict(t=t, x=x, y=y, yaw=yaw, planar=[round(v, 3) for v in pose_stop])
        tu, tv = o["target_pose_in_odom"]["planar_uv"]
        pt = G.pan_tilt_prediction(pose_stop, (tu, tv), o["target_pose_in_odom"]["up_m"])
        ts = o.get("source_timestamp")
        already = bool((o.get("motion_at_request") or {}).get("stopped"))
        o.update(stop_basis="ALREADY_STATIONARY_AT_REQUEST" if already else "OPERATOR_STOP_AFTER_REQUEST",
                 detect_to_stop_path_m=None if ts is None else G.path_between(samples, float(ts), t),
                 detect_to_stop_s=None if ts is None else round(t - float(ts), 2))
        try:                                   # calibrated servo command via the student aim math (prediction only)
            K_ = self._K
            u_, v_, z_ = G.body_to_pixel(pt["target_fwd_m"], pt["target_left_m"], o["target_pose_in_odom"]["up_m"], K_)
            o["pantilt_command_prediction"] = dict(G.calibrated_aim(u_, v_, z_), predicted_pixel=[round(u_, 1), round(v_, 1)],
                                                   predicted_depth_m=round(z_, 2),
                                                   chain="odom(stop) -> body -> RealSense [UNVERIFIED mount] -> pan-tilt [student calibration]")
            o["redetection"] = self.redetect(u_, v_, t, expect=dict(z=z_, up=o["target_pose_in_odom"]["up_m"], K=K_))
        except Exception as ex:                # noqa: BLE001
            o["pantilt_command_prediction"] = dict(error=f"{type(ex).__name__}: {ex}")
        o.update(target_pose_at_stop=dict(fwd=pt["target_fwd_m"], left=pt["target_left_m"], range_m=pt["range_m"]),
                 pan_target=pt["pan_deg"], tilt_target=pt["tilt_deg"], prediction=pt, state="STOPPED_TARGET_READY",
                 why="pan/tilt predicted; not commanded", stopped_wall=now)
        return True

    def redetect(self, u, v, t_stop, win_px=REDETECT_WIN_PX, max_dt=REDETECT_MAX_DT_S, expect=None):
        """Closed loop at the stop: the E5 region in the CURRENT RealSense scene nearest the predicted pixel (within
        win_px, scene within max_dt of the stop). Its centre + measured depth go straight to the student calibration, so
        odometry / camera-mount error drops out of the aim. 2026-10-20 field replay: predicted vs real region median
        29 px (max 67 px) over 8 stops. None = target not re-detected (aim would rely on the prediction)."""
        try:
            st = json.loads(self.e5_status.read_text())
        except (OSError, ValueError):
            return dict(status="E5_STATUS_UNAVAILABLE")
        sc = st.get("scene") or {}
        stamp = ((sc.get("frame") or {}).get("header_stamp")) or sc.get("stamp")
        if stamp is None or abs(float(stamp) - t_stop) > max_dt:
            return dict(status="NO_FRESH_SCENE", scene_dt_s=None if stamp is None else round(float(stamp) - t_stop, 2))
        best, refused = None, []
        for g in sc.get("regions") or []:
            b = g.get("box") or []
            if len(b) < 4 or not g.get("z"):
                continue
            d = math.hypot((b[0] + b[2]) / 2 - u, (b[1] + b[3]) / 2 - v)
            if d > win_px:
                continue
            why = same_object(g, expect)
            if why:
                refused.append(dict(box=[round(x) for x in b], z=g["z"], sim=g.get("sim"), why=why))
                continue
            if best is None or d < best[0]:
                best = (d, g)
        if best is None:
            return dict(status="NOT_REDETECTED", scene_dt_s=round(float(stamp) - t_stop, 2), window_px=win_px, refused=refused[:5])
        d, g = best
        b = g["box"]
        uc, vc = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        px_basis = "BOX_CENTRE"
        if expect and expect.get("K"):                 # a region larger than a plate: keep the prediction, moved into the region
            lim = LOOSE_BOX * PLATE_W_M * expect["K"][0] / float(g["z"])
            if b[2] - b[0] > lim:
                uc, px_basis = min(max(u, b[0]), b[2]), "PREDICTION_IN_LOOSE_BOX"
            if b[3] - b[1] > lim:
                vc, px_basis = min(max(v, b[1]), b[3]), "PREDICTION_IN_LOOSE_BOX"
        return dict(status="REDETECTED", pixel=[round(uc, 1), round(vc, 1)], depth_m=g["z"], sim=g.get("sim"), box=b, pixel_basis=px_basis, refused=refused[:5],
                    offset_from_prediction_px=round(d, 1), scene_dt_s=round(float(stamp) - t_stop, 2),
                    pantilt_command=G.calibrated_aim(uc, vc, float(g["z"])),
                    chain="RealSense re-detection -> pan-tilt [student calibration]; odometry / mount not used")

    def tick(self, hyps=(), sem=None):
        """One E6 step. hyps = [{'id','floor','score','rec'}] (E3 alive, best first). Returns True if anything changed."""
        now = self.clock()
        idx = self.hub_index() or {}
        K = ((idx.get("caminfo") or {}).get("k"))
        self._K = K
        samples = [tuple(o_[:4]) for o_ in (idx.get("odom") or [])]
        changed = False
        for r in self.requests():
            o = self.new_observation(r, K, list(hyps), sem, samples)
            self.obs.append(o)
            self._log(o, "created")
            changed = True
        for o in self.obs:
            if self.step(o, samples, now):
                self._log(o, o["state"])
                changed = True
        self.obs = self.obs[-20:]
        return changed

    def _log(self, o, what):
        self.n_ev += 1
        with open(self.state / "e6_observations.jsonl", "a") as f:
            f.write(json.dumps(dict(o, event=what, event_type=what, event_id=f"{o['observation_id']}_E{self.n_ev:03d}",
                                    state=o.get("state"), reason=o.get("why"), wall=self.clock()), default=str) + "\n")

    def snapshot(self):
        act = [o for o in self.obs if o["state"] == "WAIT_STOP"]
        return dict(kind="LIVE2_E6_STATE", active=act[-1] if act else None, recent=self.obs[-5:],
                    n_total=len(self.seen), pantilt="BLOCKED_BY_HARDWARE_APPROVAL", ocr_vlm="BLOCKED_BY_HARDWARE_APPROVAL",
                    extrinsics=G.CAMERA_EXTRINSICS)
