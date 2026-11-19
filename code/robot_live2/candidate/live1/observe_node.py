#!/usr/bin/env python3
"""LIVE1 G1 observe module: the REAL e5_onboard entry point (SEM1 follow-up code) with --source ros_readonly in SHADOW.

  - SHADOW only: SCI_E5_OBSERVE_MODE=shadow -> observe on, request/arm off (e5rt.gates); no --pantilt-execute,
    SCI_PANTILT_APPROVED unset, console/pantilt URLs point to 127.0.0.1:1 (refused). Nothing here can actuate.
  - WOULD_STOP is the private counterfactual (e5rt.would_stop). The web shows it as STOP_REQUEST_UI (a sign for the
    operator), never as a command. Mode label: E5_STANDALONE (no E2/E3/E4 producer is connected).
  - Run limit LIVE1_RUN_S, counted from the main loop start (after model load); must be <= the approved limit.
  - Odometry: ros_readonly subscribes /Odometry if published. Without a VERIFIED frame contract (SCI_SEM1_ODOM_FRAME_CONTRACT)
    scenes use a FLAGGED identity pose (semantic search only) and odom_hist is empty -> geometry / P2 STOP unavailable.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

tree = Path(os.environ["SCI_WS"])
here = Path(__file__).resolve().parent
assert tree.resolve() == here.parent.resolve(), f"SCI_WS {tree} is not this LIVE1 candidate {here.parent}"   # sci_env overwrite guard
sys.path.insert(0, str(tree / "external" / "sci_event" / "e5_onboard"))
assert os.environ.get("SCI_E5_OBSERVE_MODE") == "shadow" and not os.environ.get("SCI_PANTILT_APPROVED"), "shadow only"
assert "--pantilt-execute" not in sys.argv, "no actuation in LIVE1 observe"
run_s = float(os.environ["LIVE1_RUN_S"])
cap = float(os.environ["LIVE1_RUN_S_APPROVED_MAX"])
assert 0 < run_s <= cap, f"run limit {run_s} exceeds approved {cap}"
import e5_onboard as OB  # noqa: E402

N = OB.N
for mod in ("e5_node", "e5_onboard", "sources", "planner", "e5rt"):
    m = sys.modules.get(mod)
    if m is not None and not str(Path(m.__file__).resolve()).startswith(str(tree.resolve()) + "/"):
        raise SystemExit(f"IMPORT_OUTSIDE_CANDIDATE {mod} {m.__file__}")
print("LIVE1_IMPORTS " + " ".join(f"{k}={sys.modules[k].__file__}" for k in ("e5_node", "e5_onboard", "sources") if k in sys.modules), flush=True)
_loop = OB.GatedNode.loop
LINK = None
if os.environ.get("LIVE2_E4_FEED"):                      # LIVE2: E4 -> E5 link (budget order only, see live2/e5_e4_link.py)
    sys.path.insert(0, str(tree / "live2"))
    import e5rt.budget as _B  # noqa: E402
    import e5_e4_link as _EL  # noqa: E402
    LINK = _EL.install(_B, os.environ["LIVE2_E4_FEED"], os.environ.get("LIVE1_HUB_DIR"))
    print(f"LIVE2_E4_LINK installed feed={os.environ['LIVE2_E4_FEED']}", flush=True)
    _pub = OB.GatedNode.publish_status

    # E4 -> SigLIP activation (live2/siglip_gate.py): default mode e4 = infer only while a FRESH E4 request is ACTIVE;
    # missing / stale / IDLE_OK -> the scene is skipped (model stays resident). LIVE2_SIGLIP_MODE=eval = stand-alone
    # evaluation (always infer, labelled EVAL_MODE, never counted as E4 integration); shadow = always infer + count.
    import siglip_gate as SGM  # noqa: E402
    sys.path.insert(0, str(here))
    from live1_common import max_temp as C_max_temp  # noqa: E402
    SREQ = Path(os.environ["LIVE2_E4_FEED"]).with_name("siglip_request.json")
    MODE = os.environ.get("LIVE2_SIGLIP_MODE", "e4")
    assert MODE in SGM.MODES, f"LIVE2_SIGLIP_MODE {MODE} not in {SGM.MODES}"
    SG = dict(mode=MODE, kind="EVAL_MODE_NOT_E4_INTEGRATION" if MODE == "eval" else "E4_DRIVEN" if MODE == "e4" else "SHADOW",
              scenes_infer=0, scenes_skipped=0, skipped_by={}, last=None, request_file=str(SREQ))
    _gate = OB.gate

    def _live2_gate():
        on, why = _gate()
        if not on:
            return on, why
        r = SGM.read_request(SREQ)
        infer, why2 = SGM.decide(r, MODE, temp_c=C_max_temp())
        SG["last"] = dict(infer=infer, why=why2, e4_state=(r or {}).get("e4_state"), seq=(r or {}).get("seq"), wall=time.time())
        if not infer:
            SG["scenes_skipped"] += 1
            k = why2.split(" ")[0]
            SG["skipped_by"][k] = SG["skipped_by"].get(k, 0) + 1
            return False, "LIVE2_" + why2
        SG["scenes_infer"] += 1
        return on, why
    OB.gate = _live2_gate

    def _publish_status(self, src, rec, force=False):
        self.status["live2_e4_link"] = dict(LINK.last)
        self.status["live2_siglip_gate"] = dict(SG)
        self.status["live2_stop_request"] = _VSTOP[0].status() if _VSTOP[0] else dict(n=0, last=[])
        return _pub(self, src, rec, force=force)
    OB.GatedNode.publish_status = _publish_status
MODE = "E5_WITH_LIVE2_E4" if LINK else "E5_STANDALONE"


def loop(self, src):
    t0 = time.monotonic()
    rr = sys.modules.get("shared_hub") or sys.modules.get("ros_readonly")
    if rr is not None and not str(Path(rr.__file__).resolve()).startswith(str(tree.resolve()) + "/"):
        raise SystemExit(f"IMPORT_OUTSIDE_CANDIDATE ros_readonly {rr.__file__}")
    print(f"LIVE1_SOURCE {type(src).__name__} {getattr(rr, '__file__', None)}", flush=True)
    print(f"LIVE1_LOOP_START mono={t0:.3f} run_s={run_s} mode={MODE}", flush=True)
    self.status["live1"] = dict(mode=MODE, e4="LIVE2_E4_FILE" if LINK else "NOT_CONNECTED", run_s=run_s, loop_start_wall=time.time())

    def stop():
        time.sleep(run_s)
        print(f"LIVE1_RUN_LIMIT reached after {time.monotonic() - t0:.1f}s", flush=True)
        N.STOP[0] = True
    threading.Thread(target=stop, daemon=True).start()
    return _loop(self, src)


OB.GatedNode.loop = loop


# ---------------------------------------------------------------- LIVE2 detection pose record (E6/E7 input, evaluation)
# Every scene with a candidate region (sim >= DET_SIM_MIN or any accepted decision) or a STOP gets one line in
# <run>/live2_detection_pose.jsonl: colour-frame stamp, RAW FAST-LIO odometry interpolated at that stamp (camera_init ->
# base_link as published, NOT a map pose; frame contract status recorded), depth stamp offset, region boxes / sims / depth /
# identity-frame xyz / decisions. With a verified contract later, E6/E7 can re-project these without re-running SigLIP.
DET_SIM_MIN = 0.078


sys.path.insert(0, str(tree / "live2"))
from det_pose import odom_at as _odom_at  # noqa: E402


# ---------------------------------------------------------------- LIVE2 operator STOP request without geometry contract
import visual_stop as VSM  # noqa: E402  (live2/visual_stop.py; reason and rules documented there)
_VSTOP = [None]


_K = [None]


def _hub_K():
    """camera K from the LIVE1 hub index (R36 E5 gates). None -> those gates do not apply."""
    if _K[0] is None:
        hub = os.environ.get("LIVE1_HUB_DIR")
        if hub:
            ci = (json.loads((Path(hub) / "index.json").read_text()).get("caminfo") or {})
            _K[0] = ci.get("k") or ci.get("K")
    return _K[0]


def _visual_stop(self, rec, regs, ts, det_line):
    try:
        if _VSTOP[0] is None:
            _VSTOP[0] = VSM.VisualStop(self.run, os.environ.get("SCI_P2_RANGE_MAX_M", "5.0"),
                                       contract_ok=bool(os.environ.get("SCI_SEM1_ODOM_FRAME_CONTRACT")), K_fn=_hub_K,
                                       state_dir=os.path.dirname(os.environ.get("LIVE2_E4_FEED") or "") or None)
        obs = _VSTOP[0].feed(rec, ts, det_line)
        if obs:
            print(f"LIVE2_VISUAL_STOP {obs['observation_id']} spot={obs['target_track_id']} sim={obs['sim']:.3f} z={obs['depth_m']}", flush=True)
    except Exception as e:                           # noqa: BLE001
        print(f"LIVE2_VISUAL_STOP_ERROR {type(e).__name__}: {e}", flush=True)


_handle = OB.GatedNode.handle_scene


def _handle_scene(self, src, res, t, stamp, *args, **kw):
    out = _handle(self, src, res, t, stamp, *args, **kw)
    try:
        rec = getattr(self, "last_scene", None) or {}
        regs = rec.get("regions") or []
        cand = [r for r in regs if (r.get("sim") or -1) >= DET_SIM_MIN or any(bool(v) for v in (r.get("dec") or {}).values())]
        stop = bool(rec.get("legacy_stop"))
        if cand or stop:
            fr = rec.get("frame") or {}
            ts = fr.get("header_stamp") or rec.get("stamp")
            with src.lock:
                odom = list(getattr(src, "odom", []))
            ds = rec.get("depth_stamp")
            line = dict(kind="LIVE2_DETECTION_POSE", wall=time.time(), scene_t=rec.get("t"), image_stamp=ts, frame_id=fr.get("frame_id"),
                        camera_id=fr.get("camera_id"), odom_raw=_odom_at(odom, ts), odom_frame="camera_init->base_link (FAST-LIO as published)",
                        odom_session=getattr(src, "odom_session", None),
                        pose_contract=os.environ.get("SCI_SEM1_ODOM_FRAME_CONTRACT") and "CONTRACT_FILE_SET" or "MISSING (identity pose in E5)",
                        depth_stamp=ds, depth_dt_s=None if ds is None or ts is None else round(ds - ts, 3),
                        e4=rec.get("e4"), legacy_stop=stop, thr=rec.get("threshold"),
                        regions=[{k: r.get(k) for k in ("box", "tile", "sim", "z", "xyz", "h", "spot", "display", "dec")} for r in cand][:12])
            with open(self.run / "live2_detection_pose.jsonl", "a") as f_:
                f_.write(json.dumps(line, default=str) + "\n")
        _visual_stop(self, rec, regs, ts if (cand or stop) else None, line if (cand or stop) else None)
    except Exception as e:                                   # noqa: BLE001  never break the E5 loop for the side record
        print(f"LIVE2_DETECTION_POSE_ERROR {type(e).__name__}: {e}", flush=True)
    return out


OB.GatedNode.handle_scene = _handle_scene
_orig_source = N.ReplaySource_or_Ros


class _TravelMeter:
    """R34: cumulative robot path from the hub's real odometry (only differences are used, so no frame contract is needed). None until known."""

    def __init__(self, src):
        self.src, self.t, self.xy, self.path = src, None, None, None

    def __call__(self):
        with self.src.lock:
            od = list(getattr(self.src, "odom", []))
        for o in od:
            if self.t is not None and o[0] <= self.t:
                continue
            xy = (float(o[1]), float(o[2]))
            self.path = 0.0 if self.xy is None else self.path + ((xy[0] - self.xy[0]) ** 2 + (xy[1] - self.xy[1]) ** 2) ** 0.5
            self.t, self.xy = o[0], xy
        return self.path


def _install_travel(src):
    m = float(os.environ.get("LIVE2_SPOT_EXPIRE_M", "0.7"))
    if m > 0:
        import frontend as FE                       # e5_node imports it inside a function; same module object
        FE.Frontend.travel_source = staticmethod(_TravelMeter(src))
        FE.Frontend.spot_expire_m = m


def _live1_source(a):
    """LIVE1 free drive: --source shared_raw reads the LIVE1 sensor hub ring (no ROS subscription in this process)."""
    hub = os.environ.get("LIVE1_HUB_DIR")
    if a.source == "shared_raw" and hub:
        from shared_hub import SharedHubSource
        idx = Path(hub) / "index.json"
        if not idx.exists() or time.time() - idx.stat().st_mtime > 2.0:
            raise SystemExit(f"LIVE1_HUB_NOT_LIVE {hub}")
        src = SharedHubSource(hub, visual_identity_pose=os.environ.get("SCI_SEM1_VISUAL_ONLY_IDENTITY", "1") == "1")
        _install_travel(src)
        return src
    return _orig_source(a)


N.ReplaySource_or_Ros = _live1_source
N.main()
