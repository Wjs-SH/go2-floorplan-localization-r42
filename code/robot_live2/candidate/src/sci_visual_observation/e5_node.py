#!/usr/bin/env python3
"""SCI E5 node: onboard sign-observation event system (no robot motion commands, ever).

  source (replay | ros) -> legacy front end (A0) -> SCI features -> stage policies A1..A4 -> event FSM
  -> status (/dev/shm/sci_e5/status.json + frame.jpg for the UI) + run logs (experiments/<run>/)

Event flow after STOP_REQUEST (UI notification only):
  WAIT_STABLE (odometry says the operator stopped the robot) -> STOP_CONFIRMED -> AIM (pan/tilt plan; actuation only
  with SCI_PANTILT_APPROVED=1 + --pantilt-execute, otherwise DRY_RUN/PENDING_APPROVAL) -> CAPTURE (IMX477 via console when
  approved, otherwise a clearly-labelled RealSense proxy crop) -> quality check (bounded retries) -> RELEASE (UI shows
  'resume allowed'; nothing is sent to the robot) -> READ_ASYNC job for the reader -> LOCALIZE_ASYNC.

Run:  python e5_node.py --source replay --dataset ~/sci_ws/datasets/bags/S-1F --run-name test
      python e5_node.py --source ros --odom-topic /utlidar/robot_odom
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
for p in (HERE, SRC / "sci_sensor_bridge", SRC / "sci_scheduler", SRC / "sci_pantilt"):
    sys.path.insert(0, str(p))

from features import Geometry, odom_speed, predicted_obliquity  # noqa: E402
from policy import EventFSM, load_cfg, make_stages  # noqa: E402
from timing import P2Policy, cam_basis, load_p2, p2_display  # noqa: E402
from resource_guard import ResourceMonitor  # noqa: E402
from planner import PanTiltPlanner  # noqa: E402
from event_store import EventStore  # noqa: E402
from device_worker import DeviceWorker, LatestSlot  # noqa: E402
from stop_policy import BodySample, StopMonitor, decision_reference, geometry_input_check  # noqa: E402
sys.path.insert(0, str(SRC / "sci_common"))
import r1_contracts as C  # noqa: E402

WS = Path(os.environ.get("SCI_WS", str(Path.home() / "sci_ws")))     # R1: the tree that is actually running
SHM = Path(os.environ.get("SCI_SHM", "/dev/shm/sci_e5"))
TRIAL_POINTER = os.environ.get("SCI_TRIAL_POINTER", "/dev/shm/rt_live/out/trial_current.json")   # <store_root>/out (W-A r1_trial.py)
DIAGNOSTIC_PROXY = os.environ.get("SCI_DIAGNOSTIC_PROXY", "0") == "1"   # RealSense proxy only in explicit diagnostic mode
HOME_AFTER = os.environ.get("SCI_HOME_AFTER", "1") == "1"
QUALITY_SHARP_MIN, QUALITY_SAT_MAX = 40.0, 0.08                          # unchanged legacy gate


def _safe(s):
    """Same rule as the reader (r1_jobs.safe): '#' -> '__', other non [alnum-_.] -> '_'. Legacy evNNN.json names are never used."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s).replace("#", "__"))


def code_identity():
    """sha256 of every module actually loaded from the SCI tree + env that changes behaviour (R1 §4.1 / audit P0-7)."""
    import hashlib
    mods = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if f and str(WS) in os.path.realpath(f) or (f and "sci_event" in f):
            try:
                mods[os.path.realpath(f)] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
            except OSError:
                pass
    env = {k: v for k, v in os.environ.items() if k.startswith(("SCI_", "KCI_", "CYCLONEDDS"))}
    try:
        cpus = [ln.split(":", 1)[1].strip() for ln in open("/proc/self/status") if ln.startswith("Cpus_allowed_list")][0]
    except (OSError, IndexError):
        cpus = None
    return dict(schema_version=C.SCHEMA_VERSION, **C.stamp(), sci_ws=str(WS), modules=mods, env=env, cpus_allowed=cpus,
                python=sys.executable)


def _guard_run_dir(run):
    """R1 tests: refuse any run directory outside the tree under test (no writes into the operational tree)."""
    if os.environ.get("SCI_R1_TEST") == "1":
        real = Path(os.path.realpath(run))
        root = Path(os.path.realpath(WS))
        if root not in real.parents:
            raise RuntimeError(f"R1 write guard: run dir {real} outside SCI_WS {root}")
        import pwd
        oper = Path(os.path.realpath(os.environ.get("SCI_OPERATIONAL_WS", os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "sci_ws"))))
        if oper in real.parents and "repairs" not in real.parts:
            raise RuntimeError(f"R1 write guard: run dir inside the operational tree: {real}")
COLORS = {"STOP_REQUEST": (0, 0, 255), "VERIFYING": (0, 200, 255), "CANDIDATE": (200, 200, 0), "SEARCHING": (120, 120, 120),
          "REJECTED_POSTER": (255, 0, 255), "REJECTED_GLARE": (255, 255, 255), "REJECTED_ELEVATOR": (255, 120, 0),
          "REJECTED_DISTANCE": (0, 140, 255), "REJECTED_ANGLE": (0, 140, 255), "REJECTED_PANTILT": (0, 140, 255),
          "ALREADY_OBSERVED": (0, 255, 0), "E4_SUPPRESSED": (180, 180, 180),
          "DEFERRED_BUDGET": (90, 90, 160), "CACHED_NO_NEW_EVIDENCE": (90, 90, 160)}


def atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def jdump(o):
    return json.dumps(o, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


class E4Feed:
    """Optional E4 ambiguity input: JSON file {stamp, level: AMBIGUOUS|CONFIDENT, margin}. No file / stale -> unavailable.
    No ambiguity value is ever synthesised here."""

    def __init__(self, path=None, stale_s=3.0):
        self.path, self.stale = (Path(path) if path else None), stale_s

    def get(self, now_stamp):
        if self.path is None:
            return dict(available=False, status="E4_UNAVAILABLE", why="no E4 producer configured (no live ambiguity output exists)")
        try:
            d = json.loads(self.path.read_text())
        except Exception as e:
            return dict(available=False, status="E4_UNAVAILABLE", why=f"read error {type(e).__name__}")
        if abs(now_stamp - float(d.get("stamp", 0))) > self.stale:
            return dict(available=False, status="E4_UNAVAILABLE", why="stale")
        return dict(available=True, status="E4_" + d.get("level", "?"), level=d.get("level"), margin=d.get("margin"))


class R1Event:
    """In-memory view of the active event (FSM single owner = the event loop). Durable truth = EventStore record."""
    ACTIVE = ("STOP_REQUEST", "WAIT_BASE_STOP", "BASE_STOPPED", "WAIT_IMAGE_STABLE", "AIM", "CAPTURE", "HOME")

    def __init__(self, rec, stop, clock):
        self.rec, self.key, self.id, self.stop, self.clock = rec, rec["event_key"], rec["event_id"], stop, clock
        self.state, self.log, self.result = "STOP_REQUEST", [dict(state="STOP_REQUEST", t=clock(), reason="policy_pass")], {}
        self.attempt, self.monitor, self.trial, self.prev_capture = 0, None, None, None
        self.capture_status, self.release = None, None
        self.device_used, self.device_idle_confirmed = False, True

    def to(self, state, reason, **kw):
        now = self.clock()
        self.log[-1]["t_exit"] = now
        self.log.append(dict(state=state, t=now, reason=reason, **kw))
        self.state = state

    @property
    def active(self):
        return self.state in self.ACTIVE


class Node:
    def __init__(self, a, deps=None):
        d = deps or {}
        self.a, self.deps = a, d
        self.run = WS / "experiments" / (a.run_name or time.strftime("e5_%Y%m%d_%H%M%S"))
        _guard_run_dir(self.run)
        self.run_id = self.run.name
        for sub in ("events", "captures", "jobs/pending", "jobs/done", "trials"):
            (self.run / sub).mkdir(parents=True, exist_ok=True)
        SHM.mkdir(parents=True, exist_ok=True)
        self.cfg = load_cfg(a.policy)
        (self.run / "policy_config.json").write_text(json.dumps(self.cfg, indent=1))
        (self.run / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))
        C.atomic_write_json(self.run / "code_identity.json", code_identity())
        self.geom = d.get("geom") or Geometry()
        self.stages = make_stages(self.cfg, self.geom)
        self.p2cfg = load_p2(a.p2_policy)
        (self.run / "policy_p2.json").write_text(json.dumps(self.p2cfg, indent=1))
        self.p2 = P2Policy(self.cfg, self.p2cfg, self.geom)
        self.p2_crops, self.p2_summary, self.last_tick, self.last_bgr = {}, [], -1e9, None
        self.active_stage = a.stage
        self.planner = d.get("planner") or PanTiltPlanner(self.geom, execute=a.pantilt_execute)
        self.e4 = E4Feed(a.e4_file)
        self.mon = d.get("monitor") or ResourceMonitor(period=1.0, watch_pids={"e5_node": os.getpid()})
        if d.get("monitor") is None:
            self.mon.start()
        self.logs = {k: open(self.run / f"{k}.jsonl", "a", encoding="utf-8") for k in ("scenes", "stops", "fsm", "faults")}
        self.status = dict(run=str(self.run), run_id=self.run_id, schema_version=C.SCHEMA_VERSION, boot_id=C.BOOT_ID,
                           started=time.time(), siglip="STARTING", source=a.source, events=[], faults=[])
        self.fsm, self.n_ev, self.legacy_events = None, 0, []
        self.frontend, self.siglip_down_since = None, None
        # ---- R1 runtime pieces
        self.clock = d.get("clock") or ((lambda: self._sim_clock) if a.source == "replay" else time.monotonic)
        self._sim_clock = 0.0
        self.store = EventStore(self.run, self.run_id, trial_pointer=d.get("trial_pointer", TRIAL_POINTER))
        self.device = d.get("device") or DeviceWorker()
        self.proposals = LatestSlot()                      # newest STOP proposal (main -> event loop)
        self.body_speed = d.get("body_speed")              # optional primary BASE_STOPPED evidence adapter (no ROS here)
        self.frame_meta = {}                               # seq -> frame identity (bounded)
        self.scene_frames = {}                             # frame_id -> bgr of recent SCENE base frames (bounded)
        self.done_seen = {}                                # done file -> mtime (attach loop)
        self.threaded = d.get("threaded", a.source == "ros")
        self._ev_thread, self._last_pub = None, 0.0
        # ---- E5RT2: scene path mode + owner-side bookkeeping
        self.scene_mode = scene_mode_for(a)
        self.scene_generation, self._last_odom_session = 0, None
        self.scene_worker = None
        self.tick_log = deque(maxlen=20000)               # (mono_s, rel_s) of every P2 timing tick (real caller)
        self.scene_apply_log = deque(maxlen=2000)         # admission records of finished scenes
        self.scene_depth = {}                             # source frame_id -> aligned depth of that scene (bounded)
        self._last_idle_odom_t = None
        self.status["scene_mode"] = self.scene_mode
        from e5rt.calibration import ThresholdMode
        self.threshold_mode = ThresholdMode()                # E5RT2: legacy_p90 (default) | calibrated (DEV artifact only)
        self.status["threshold_mode"] = self.threshold_mode.describe()
        if "frontend" in d:
            self.frontend = d["frontend"]
            self.status.update(siglip="FAKE" if d["frontend"] is not None else "DOWN")
        else:
            self._start_frontend()
        self.mon.watch["siglip_worker"] = getattr(getattr(self.frontend, "vlm", None), "pid", None) if self.frontend else None

    # ------------------------------------------------------------------ fault-tolerant SigLIP front end
    def _start_frontend(self):
        from frontend import Frontend
        try:
            os.environ["SCI_SIGLIP_LOG"] = str(self.run / "siglip_worker.log")
            t0 = time.time()
            # the client must be created by legacy build_events() AFTER it exports KCI_SIGN_PROMPTS (=roomstd);
            # creating it earlier silently falls back to the legacy "wall" prompt (bug found 2026-10-14, runs v1/v2)
            self.frontend = Frontend(siglip=None, workers=self.a.workers)
            self.status.update(siglip="OK", siglip_load_s=round(time.time() - t0, 1), siglip_prompts=self.frontend.vlm.prompts)
            (self.run / "siglip_prompts.json").write_text(json.dumps(dict(prompts=self.frontend.vlm.prompts,
                                                                          env=os.environ.get("KCI_SIGN_PROMPTS"), model=self.frontend.vlm.model), indent=1))
            self.siglip_down_since = None
        except Exception as e:
            self.fault("SIGLIP_START_FAILED", repr(e))
            self.status["siglip"] = "DOWN"
            self.siglip_down_since = time.time()

    def fault(self, kind, detail):
        rec = dict(t=time.time(), kind=kind, detail=str(detail)[:500])
        self.logs["faults"].write(jdump(rec) + "\n"); self.logs["faults"].flush()
        self.status["faults"] = (self.status["faults"] + [rec])[-10:]

    # ------------------------------------------------------------------ features
    def features(self, r, bgr, dimg, speed):
        """Sign-level features are measured on the best SigLIP tile (the sign), not the whole wall region."""
        g, b, z = self.geom, r.get("tile") or r["box"], r["z"]
        u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        pl = g.plane(dimg, b)
        zt = pl.get("z_med") if pl.get("ok") else z                  # range of the tile itself (falls back to region)
        f = dict(glare=g.glare(bgr, b), plane=pl, edge=g.edge_margin(b), speed=speed, z_tile=zt,
                 bearing=math.atan2(-(u - g.cx) / g.fx, 1.0), up=None, aim_px=[u, v])
        L = self.frontend.L
        if L.floor_up is not None:
            f["up"] = [float(x) for x in L.floor_up]
        f["pantilt"] = g.pantilt(u, v, zt) if zt else dict(ok=False, why="no_range", pan=None, tilt=None)
        f["text_px"] = g.readable_px(b, zt, pl.get("obliquity_deg", 0.0)) if zt else None
        adv = (speed or 0.0) * self.cfg["A3"]["stop_reaction_s"]
        f["obliquity_pred"] = predicted_obliquity(pl, b, g.fx, g.fy, g.cx, g.cy, zt, adv)
        return f

    # ------------------------------------------------------------------ frame identity
    def note_frame(self, src, st, seq):
        rx = (getattr(src, "rx_mono_ns", None) or {}).get(seq)
        m = dict(frame_id=int(seq), camera_id="d435_color", header_stamp=float(st),
                 clock_domain=C.CLOCK_ROS if self.a.source in ("ros", "ros_readonly", "shared_raw") else "replay_bag_stamp_s",
                 rx_mono_ns=rx, **({} if rx is not None else dict(rx_mono_ns_null_reason="source keeps no receive clock")))
        self.frame_meta[int(seq)] = m
        while len(self.frame_meta) > 200:                          # bounded
            self.frame_meta.pop(next(iter(self.frame_meta)))
        return m

    # ------------------------------------------------------------------ main loop (scene producer)
    def loop(self, src):
        if self.scene_mode == "async":
            return self.loop_async(src)
        return self.loop_sync(src)

    def loop_sync(self, src):
        """Original synchronous scene producer (DETERMINISTIC_REPLAY / comparison). P2 ticks and frame tracking only run
        between scenes here: a scene blocks them for its whole duration."""
        a, nxt, origin, t_first = self.a, None, None, None
        n_scene, n_skipped = 0, 0
        if self.threaded:
            self.start_event_thread(src)
        while src.alive and not STOP[0]:
            if self.frontend is None:
                if time.time() - (self.siglip_down_since or 0) > 10:
                    self._start_frontend()
                src.poll()
                self._event_inline(src, None)
                time.sleep(0.2)
                continue
            got = src.poll()
            if not got:
                self._event_inline(src, None)
                time.sleep(0.01)
                continue
            for st, seq, img in got:
                if t_first is None:
                    t_first, origin = st, st
                    nxt = a.warmup
                rel = st - origin
                self.note_frame(src, st, seq)
                self.frontend.add_frame(seq, rel, img)
                self.frontend.prefetch(nxt)
                self.last_bgr, self.last_frame = img, self.frame_meta.get(int(seq))
                self.track_frame(seq, st, img)
                self.p2_tick(src, st, rel)
            if rel <= nxt:
                self._event_inline(src, origin + rel)
                continue
            t = nxt
            nxt += self.scene_interval()
            if a.source == "ros" and rel - t > a.max_lag:            # latest-frame-first: drop stale scenes
                n_skipped += int((rel - t) // 1.0)
                t = math.floor(rel) - 0.0 if rel - math.floor(rel) > 0.1 else rel - 0.1
                nxt = t + 1.0
            stamp = origin + t
            dimg, dst = src.depth(stamp)
            pose, pst = src.pose(stamp)
            if dimg is None or pose is None:
                self.fault("SCENE_SKIPPED", f"t={t:.1f} depth={'ok' if dimg is not None else getattr(getattr(src, 'depth_gate', None), 'state', 'missing')} pose={'ok' if pose is not None else 'missing'}")
                continue
            try:
                t_sc0 = time.monotonic_ns()
                res = self.frontend.scene(t, dimg, pose, allow_stop=True)
                res.setdefault("timing", {})["scene_mono_ns"] = [t_sc0, time.monotonic_ns()]
            except Exception as e:
                self.fault("SIGLIP_DOWN" if "SigLIP" in repr(e) or "Worker" in type(e).__name__ else "SCENE_ERROR", repr(e))
                if "Worker" in type(e).__name__:
                    self.status["siglip"] = "DOWN"; self.siglip_down_since = time.time()
                    try:
                        self.frontend.vlm.close()
                    except Exception:
                        pass
                    self.frontend = None
                else:
                    traceback.print_exc()
                continue
            n_scene += 1
            self.handle_scene(src, res, t, stamp, dimg, pose, img, n_scene, n_skipped, origin=origin, depth_stamp=dst, pose_stamp=pst)
            self._event_inline(src, stamp)
        self._event_inline(src, None)
        self.publish_status(src, None, force=True)
        self.stop_event_thread()

    # ------------------------------------------------------------------ E5RT2: asynchronous scene path (REALTIME)
    def loop_async(self, src):
        """Owner loop: frames, frame tracking, P2 timing ticks and the event loop never wait for a scene. The scene
        (proposals + SigLIP + legacy decide) runs on ONE worker thread on an immutable ObservationSnapshot; at most one
        scene is in flight and one (the newest) pending. Finished scenes are applied here, by the single owner of the
        policies/FSM, after the admission check (boot / generation / odom session / resolution / age)."""
        from e5rt.snapshot import FrozenFrames, SceneWorker, make_snapshot
        a = self.a
        nxt, origin, t_first, rel = None, None, None, None
        n_skipped = 0
        self._n_scene = 0
        if self.threaded:
            self.start_event_thread(src)
        self.scene_worker = SceneWorker(self._scene_job, deadline_s=float(getattr(a, "scene_deadline_s", 10.0) or 10.0))
        try:
            while src.alive and not STOP[0]:
                self._apply_scene_results(src)
                ov = self.scene_worker.check_deadline()
                if ov:
                    self.fault("SCENE_LOGICAL_DEADLINE", f"scene {ov['id']} running {ov['overdue_s']:.1f}s; worker NOT assumed "
                                                         f"stopped, its late result is diagnostic only")
                self._check_generation(src)
                if self.frontend is None:
                    if not self.scene_worker.busy() and time.time() - (self.siglip_down_since or 0) > 10:
                        self._start_frontend()
                    src.poll()
                    self._event_inline(src, None)
                    time.sleep(0.2)
                    continue
                got = src.poll()
                if not got:
                    self._idle_tick(src, origin)
                    self._event_inline(src, None)
                    time.sleep(0.005)
                    continue
                for st, seq, img in got:
                    if t_first is None:
                        t_first, origin = st, st
                        nxt = a.warmup
                    rel = st - origin
                    self.note_frame(src, st, seq)
                    self.frontend.add_frame(seq, rel, img)
                    self.frontend.prefetch(nxt)
                    self.last_bgr, self.last_frame = img, self.frame_meta.get(int(seq))
                    self.track_frame(seq, st, img)
                    self.p2_tick(src, st, rel)
                if rel <= nxt:
                    self._event_inline(src, origin + rel)
                    continue
                t = nxt
                nxt += self.scene_interval()
                if rel - t > a.max_lag:                          # latest-first: a scene for an old time is not queued
                    n_skipped += int((rel - t) // 1.0)
                    t = math.floor(rel) - 0.0 if rel - math.floor(rel) > 0.1 else rel - 0.1
                    nxt = t + 1.0
                stamp = origin + t
                dimg, dst = src.depth(stamp)
                pose, pst = src.pose(stamp)
                if dimg is None or pose is None:
                    self.fault("SCENE_SKIPPED", f"t={t:.1f} depth={'ok' if dimg is not None else getattr(getattr(src, 'depth_gate', None), 'state', 'missing')} pose={'ok' if pose is not None else 'missing'}")
                    continue
                fe = self.frontend
                snap = make_snapshot(self.run_id, C.BOOT_ID, self.scene_generation, getattr(src, "odom_session", None), t, stamp,
                                     fe.buf, dst, pose, pst)
                self.scene_worker.submit(dict(id=snap.snapshot_id, snap=snap, fe=fe, buf=FrozenFrames(fe.buf), t=t, stamp=stamp,
                                              dimg=dimg, pose=pose, img=img, origin=origin, depth_stamp=dst, pose_stamp=pst,
                                              n_skipped=n_skipped))
                self._event_inline(src, stamp)
            t_wait = time.monotonic() + min(30.0, self.scene_worker.deadline_s + 5.0)
            while self.scene_worker.busy() and time.monotonic() < t_wait:      # finish the scene in flight (bounded)
                time.sleep(0.02)
            self._apply_scene_results(src)
        finally:
            self.status["scene_worker"] = self.scene_worker.stats()
            self.scene_worker.close()
        self._event_inline(src, None)
        self.publish_status(src, None, force=True)
        self.stop_event_thread()

    @staticmethod
    def _scene_accepts_buf(fe):
        import inspect
        try:
            return "buf" in inspect.signature(fe.scene).parameters
        except (TypeError, ValueError):
            return False

    def _scene_job(self, job):
        """Scene worker thread: computes on the snapshot only (no FSM / policy / store access here)."""
        fe = job["fe"]
        t_sc0 = time.monotonic_ns()
        if self._scene_accepts_buf(fe):
            res = fe.scene(job["t"], job["dimg"], job["pose"], allow_stop=True, buf=job["buf"])
        else:
            res = fe.scene(job["t"], job["dimg"], job["pose"], allow_stop=True)
        res.setdefault("timing", {})["scene_mono_ns"] = [t_sc0, time.monotonic_ns()]
        return res

    def _current_identity(self, src):
        size = None
        if getattr(self, "last_bgr", None) is not None:
            h, w = self.last_bgr.shape[:2]
            size = (int(w), int(h))
        return dict(boot_id=C.BOOT_ID, generation=self.scene_generation, odom_session=getattr(src, "odom_session", None),
                    image_size=size)

    def _apply_scene_results(self, src):
        from e5rt.snapshot import result_admission
        if self.scene_worker is None:
            return 0
        n = 0
        for out in self.scene_worker.drain():
            job, snap = out["job"], out["job"]["snap"]
            if not out["ok"]:
                e = out["error"]
                self.fault("SIGLIP_DOWN" if "SigLIP" in repr(e) or "Worker" in type(e).__name__ else "SCENE_ERROR", out["error_repr"])
                if "Worker" in type(e).__name__:
                    self.status["siglip"] = "DOWN"; self.siglip_down_since = time.time()
                    fe = job.get("fe")
                    try:
                        fe.vlm.close()
                    except Exception:
                        pass
                    if self.frontend is fe:
                        self.frontend = None
                    self.scene_generation += 1                      # nothing computed by the old worker is reused
                else:
                    print(out.get("trace", ""), file=sys.stderr)
                continue
            res = out["result"]
            adm = result_admission(snap, self._current_identity(src), time.monotonic(),
                                   max_stop_age_s=float(getattr(self.a, "max_stop_age_s", 1.5) or 1.5))
            if out.get("late"):
                adm["evidence_ok"] = adm["stop_ok"] = False
                adm["reasons"] = [r for r in adm["reasons"] if r != "OK"] + ["LATE_AFTER_LOGICAL_DEADLINE"]
            adm.update(snapshot_id=snap.snapshot_id, worker_ms=(out["t_end"] - out["t_start"]) * 1e3,
                       queue_ms=(out["t_start"] - snap.created_mono) * 1e3, applied_mono=time.monotonic(),
                       snapshot=snap.to_dict())
            res.setdefault("timing", {})["e5rt2_admission"] = {k: adm[k] for k in ("evidence_ok", "stop_ok", "reasons", "age_s",
                                                                                 "worker_ms", "queue_ms", "snapshot_id")}
            self.scene_apply_log.append({k: v for k, v in adm.items() if k != "snapshot"})
            if not adm["evidence_ok"]:
                self.fault("SCENE_RESULT_NOT_APPLIED", f"snapshot {snap.snapshot_id}: {adm['reasons']} (diagnostic record only)")
                rec = dict(t=job["t"], stamp=job["stamp"], diagnostic_only=True, admission=adm, timing=res.get("timing"),
                           n_regions=len(res.get("regions") or []))
                self.logs["scenes"].write(jdump(rec) + "\n"); self.logs["scenes"].flush()
                continue
            self._n_scene += 1
            n += 1
            self.handle_scene(src, res, job["t"], job["stamp"], job["dimg"], job["pose"], job["img"], self._n_scene,
                              job["n_skipped"], origin=job["origin"], depth_stamp=job["depth_stamp"], pose_stamp=job["pose_stamp"],
                              admission=adm)
        return n

    def _check_generation(self, src):
        """A new odometry session (reset / restart) invalidates world-anchored tracks and in-flight scenes."""
        s = getattr(src, "odom_session", None)
        if s is None:
            return
        if self._last_odom_session is not None and s != self._last_odom_session:
            self.scene_generation += 1
            self.on_generation_change(f"odom_session {self._last_odom_session}->{s}")
        self._last_odom_session = s

    def on_generation_change(self, why):
        n = 0
        for tr in self.p2.tracks.values():
            if tr.get("state") in ("VERIFIED", "WAIT_READABLE"):
                tr["state"], tr["stale_why"] = "STALE_ODOM_SESSION", why
                n += 1
        self.fault("SCENE_GENERATION_CHANGED", f"{why}: generation={self.scene_generation}, world tracks invalidated={n}")
        self.status["generation"] = dict(scene_generation=self.scene_generation, why=why, invalidated_tracks=n)

    def _idle_tick(self, src, origin):
        """No new camera frame: the P2 timing layer may still run on NEWER odometry (no image evidence is added)."""
        if origin is None or getattr(self, "last_frame", None) is None or not self.threaded:
            return
        hist = src.odom_hist or []
        if not hist:
            return
        ot = hist[-1][0]
        if self._last_idle_odom_t is not None and ot <= self._last_idle_odom_t:
            return
        self._last_idle_odom_t = ot
        fs = self.last_frame.get("header_stamp")
        if fs is None or ot <= fs:
            return
        self.p2_tick(src, fs, ot - origin)

    # ------------------------------------------------------------------ owner hooks (overridden by the wrapper)
    def track_frame(self, seq, st, img):
        """Per camera frame, on the owner thread: geometry-only ROI tracking (no semantic evidence)."""
        return None

    def after_scene(self, res, decisions, p2recs, act, fmeta, dimg):
        return None

    def reader_job_extra(self, f):
        return {}

    def scene_frame_meta(self, res, origin):
        base = res.get("base") or {}
        fid = base.get("id")
        if fid is None:
            return C.null("frame_id", "scene base frame has no id")
        m = dict(self.frame_meta.get(int(fid)) or dict(frame_id=int(fid), header_stamp=None,
                                                         header_stamp_null_reason="frame meta evicted"))
        if base.get("tb") is not None and origin is not None:
            m["scene_base_rel_s"] = float(base["tb"])
        return m

    def handle_scene(self, src, res, t, stamp, dimg, pose, img, n_scene, n_skipped, origin=None, depth_stamp=None, pose_stamp=None,
                     admission=None):
        """E5RT2: admission (async path) = owner-side check of a finished scene; it may add world-anchored evidence, but
        the P2 STOP is decided by the timing tick with the latest odometry, and a stage STOP from the scene itself is only
        proposed while the result is fresh (admission.stop_ok)."""
        async_scene = admission is not None
        stop_ok = admission is None or bool(admission.get("stop_ok"))
        tm_ = getattr(self, "threshold_mode", None)
        cal = tm_.apply(res) if tm_ is not None else dict(state="legacy_p90", stop_allowed=True)
        stop_ok = stop_ok and cal["stop_allowed"]
        bgr = res["base"]["bgr"]
        fmeta = self.scene_frame_meta(res, origin)
        if async_scene:
            fmeta["snapshot_id"] = admission.get("snapshot_id")
        if fmeta.get("frame_id") is not None:
            self.scene_frames[fmeta["frame_id"]] = bgr
            while len(self.scene_frames) > 8:                      # bounded evidence ring
                self.scene_frames.pop(next(iter(self.scene_frames)))
            if dimg is not None:
                self.scene_depth[fmeta["frame_id"]] = dict(dimg=dimg, depth_stamp=depth_stamp, pose=list(pose), pose_stamp=pose_stamp)
                while len(self.scene_depth) > 8:
                    self.scene_depth.pop(next(iter(self.scene_depth)))
        lin, yawr = odom_speed(src.odom_hist, stamp) if src.odom_hist else (None, None)
        feats = [self.features(r, bgr, dimg, lin) for r in res["regions"]]
        e4 = self.e4.get(stamp)
        busy = self.event_busy()
        decisions, stops = {}, {}
        for name, sp in self.stages.items():
            d, s = sp.step(t, res["regions"], feats, res["thr"], e4, busy=(busy or not stop_ok) if name == self.active_stage else False)
            decisions[name], stops[name] = d, s
        p2recs, p2stop, p2summ = self.p2.step(t, tuple(pose), lin, res["regions"], feats, res["thr"], e4,
                                              busy=busy if self.active_stage == "P2" else False, decide=not async_scene)
        if not async_scene:
            self.p2_summary = p2summ
        decisions["P2"], stops["P2"] = [p2_display(r) for r in p2recs], p2stop
        for r, rec, ft in zip(res["regions"], p2recs, feats):  # keep the last in-view crop of each verified sign (proxy capture)
            if rec["state"] == "VERIFIED" and r["spot"] is not None:
                x0, y0, x1, y1 = (int(round(v)) for v in (r.get("tile") or r["box"]))
                w, h = x1 - x0, y1 - y0
                c = bgr[max(0, y0 - h):y1 + h, max(0, x0 - w):x1 + w]
                if c.size:
                    self.p2_crops[r["spot"]] = c.copy()
                tr = self.p2.tracks.get(r["spot"])
                if tr is not None:                         # frame lineage of the track box (R1 §7.1)
                    tr["box_frame"] = dict(fmeta, box=r.get("tile") or r["box"], scene_t=t)
                    tr["plane_last"] = {k: v for k, v in (ft.get("plane") or {}).items() if k != "normal"}   # SEM1 P01 input
        self.last_tick = t
        if res["legacy_ev"]:
            ev = dict(res["legacy_ev"], stage="A0", stamp=stamp, pose=list(pose))
            self.legacy_events.append(ev)
            self.logs["stops"].write(jdump(ev) + "\n")
        for name, s in stops.items():
            if s:
                self.logs["stops"].write(jdump(dict(s, stamp=stamp, pose=list(pose), frame=fmeta)) + "\n")
        self.logs["stops"].flush()
        act = stops.get(self.active_stage) if stop_ok else None
        if act:
            if not act.get("box_frame"):   # E5RT2: a P2 STOP fired inside step() carries box_frame=None (set after) -> this scene
                act["box_frame"] = dict(fmeta, box=act.get("tile") or act.get("box"))
            self.propose_event(act, stamp, pose, bgr, dimg, latest_frame=fmeta)
        self.after_scene(res, decisions, p2recs, act, fmeta, dimg)
        # scene log (everything needed to audit and re-score offline)
        rec = dict(t=t, stamp=stamp, pose=list(pose), thr=res["thr"], timing=res["timing"], speed=lin, yaw_rate=yawr, e4=e4,
                   frame=fmeta, depth_stamp=depth_stamp, pose_stamp=pose_stamp,
                   legacy_stop=bool(res["legacy_ev"]), n_scene=n_scene, n_skipped=n_skipped, p2_tracks=self.p2_summary,
                   scene_mode=self.scene_mode, threshold=dict(state=cal["state"], thr_used=cal.get("thr_used"),
                                                               thr_legacy=cal.get("legacy_thr"), why=cal.get("why")), admission=None if admission is None else {k: admission.get(k) for k in (
                       "snapshot_id", "evidence_ok", "stop_ok", "reasons", "age_s", "worker_ms", "queue_ms", "applied_mono")},
                   regions=[dict(r, feat=f,
                                 dec={k: decisions[k][i] for k in decisions}) for i, (r, f) in enumerate(zip(res["regions"], feats))])
        self.logs["scenes"].write(jdump(rec) + "\n"); self.logs["scenes"].flush()
        if self.a.save_frames:
            cv2.imwrite(str(self.run / "events" / f"scene_{t:07.1f}.jpg"), bgr)
        self.last_scene = rec
        self.draw(bgr, res, decisions[self.active_stage], t, act, frame=fmeta)
        self.publish_status(src, rec, force=True)

    def scene_interval(self):
        """Overload guard (measured limits: 90.9 C tj and <1 GB free under continuous VLM). Above the limits the SigLIP
        scene rate is halved; the P2 timing tick and FAST-LIO are not touched. Every change is logged."""
        r = self.mon.last or {}
        hot = max((r.get("temps") or {}).values() or [0])
        low = (r.get("mem_avail_mb") or 1e9) < self.a.min_mem_mb
        degrade = hot >= self.a.max_temp_c or low
        if degrade != getattr(self, "_degraded", False):
            self._degraded = degrade
            self.fault("OVERLOAD_DEGRADE" if degrade else "OVERLOAD_RECOVER", f"tj={hot:.1f}C avail={r.get('mem_avail_mb')}MB")
            self.status["degraded"] = degrade
        return 2.0 if degrade else 1.0

    def p2_tick(self, src, stamp, rel):
        """P2 timing layer between SigLIP scenes (rate limited to tick_hz; no image work).
        R1 §7.4: the prediction starts from the LATEST valid odom at decision time (frame age applied once, no future
        samples); unsynced clocks / odom reset / stale odom -> DEFER (no STOP)."""
        if rel - self.last_tick < 1.0 / max(self.p2cfg["timing"].get("tick_hz", 10.0), 0.1):
            return
        self.last_tick = rel
        self.tick_log.append((time.monotonic(), float(rel)))
        hist = list(src.odom_hist or [])
        now_stamp = max(stamp, hist[-1][0]) if hist else stamp
        ref = decision_reference(hist, now_stamp, stamp)
        ok, why = geometry_input_check(stamp, pose_stamp=stamp, need_depth=False, odom_hist=hist[-50:])
        defer = None if (ok and ref["ok"]) else (why + ([ref["why"]] if not ref["ok"] else []))
        if ref["ok"]:
            pose = ref["pose"]
        else:
            pose, _ = src.pose(stamp)
            if pose is None:
                return
        lin, _ = odom_speed(hist, now_stamp) if hist else (None, None)
        busy = self.event_busy() or not self.can_start_physical()
        stop, summ = self.p2.tick(rel, tuple(pose), lin, self.e4.get(stamp), busy=busy if self.active_stage == "P2" else False,
                                  defer_why=defer)
        self.p2_summary = summ
        if defer:
            self.status["p2_defer"] = dict(why=defer, t=time.time())
        if stop:
            stop["reference"] = dict(source="latest_odom_at_decision", stamp=ref.get("stamp"), frame_stamp=stamp,
                                     frame_age_s=ref.get("frame_age_s"))
            self.logs["stops"].write(jdump(dict(stop, stamp=stamp, pose=list(pose), tick=True, latest_frame=getattr(self, "last_frame", None))) + "\n")
            self.logs["stops"].flush()
            if self.active_stage == "P2" and not busy and self.last_bgr is not None:
                self.propose_event(stop, stamp, pose, self.last_bgr, None, latest_frame=getattr(self, "last_frame", None))

    # ------------------------------------------------------------------ physical-action arm (UI heartbeat; overridden)
    def can_start_physical(self):
        return True

    def event_busy(self):
        f = self.fsm
        tm_ = getattr(self, "threshold_mode", None)
        if tm_ is not None and tm_.state == "UNCALIBRATED":       # E5RT2: tentative semantics never request a STOP
            return True
        return (f is not None and f.active) or self.device.busy

    # ------------------------------------------------------------------ event loop (single FSM owner)
    def propose_event(self, stop, stamp, pose, bgr, dimg, latest_frame=None):
        """Called by the scene producer. Never starts an event itself: the event loop owns the FSM."""
        if self.event_busy():
            self.fault("STOP_PROPOSAL_DROPPED", "event or device job active (no new STOP during an observation)")
            return False
        self.proposals.put(dict(id=stop.get("spot"), stop=stop, stamp=stamp, pose=list(pose), bgr=bgr, latest_frame=latest_frame))
        if not self.threaded:
            pass                                             # consumed by the next inline event step
        return True

    def start_event_thread(self, src):
        import threading
        self._ev_stop = False

        def run():
            while not self._ev_stop and not STOP[0]:
                try:
                    self.event_step(src, None)
                except Exception as e:                       # the event loop must not die silently
                    self.fault("EVENT_LOOP_ERROR", repr(e)); traceback.print_exc()
                time.sleep(0.05)
        self._ev_thread = threading.Thread(target=run, name="e5-event-loop", daemon=True)
        self._ev_thread.start()

    def stop_event_thread(self):
        self._ev_stop = True
        if self._ev_thread is not None:
            self._ev_thread.join(timeout=2.0)

    def _event_inline(self, src, stamp):
        if stamp is not None:
            self._sim_clock = stamp
        if not self.threaded:
            self.event_step(src, stamp)

    def event_step(self, src, stamp):
        """One iteration of the event loop: start proposals, advance the active event, consume device results,
        attach reader results to their exact events, publish status. Never blocks on the device."""
        if stamp is not None:
            self._sim_clock = stamp
        p = self.proposals.take()
        if p is not None:
            if self.event_busy():
                self.fault("STOP_PROPOSAL_DROPPED", "event became active")
            elif not self.can_start_physical():
                self.fault("STOP_PROPOSAL_DROPPED", "operator arm/heartbeat missing: no new STOP")
            else:
                self.start_event(p["stop"], p["stamp"], p["pose"], p["bgr"], None, latest_frame=p.get("latest_frame"))
        self.proposals.done()
        for r in self.device.drain():
            self.on_device_result(src, r)
        self.tick_event(src, stamp)
        attached = self.attach_readings()
        if attached or time.time() - self._last_pub > 0.25:
            self.publish_status(src, None, force=True)
            self._last_pub = time.time()

    def tick_event(self, src, stamp):
        f = self.fsm
        if f is None or not f.active:
            return
        now = self.clock()
        if f.state in ("STOP_REQUEST", "WAIT_BASE_STOP", "WAIT_IMAGE_STABLE", "BASE_STOPPED"):
            if f.monitor is None:
                f.monitor = StopMonitor(now, cfg=dict(wait_base_stop_timeout_s=self.cfg["event"]["wait_stable_timeout_s"],
                                                      aux_lin_mps=self.cfg["event"]["stable_lin_mps"],
                                                      aux_yaw_dps=self.cfg["event"]["stable_yaw_dps"],
                                                      aux_hold_s=self.cfg["event"]["stable_hold_s"]))
                self._ev_to(f, "WAIT_BASE_STOP", "waiting_for_operator_stop")
            now_stamp = stamp if stamp is not None else (self._sim_clock if self.a.source == "replay" else time.time())
            lin, yawr = odom_speed(src.odom_hist, now_stamp) if src.odom_hist else (None, None)
            aux_fresh = (time.time() - getattr(src, "last_odom_wall", time.time())) <= 0.5
            if self.a.assume_stopped:                        # offline/UI tests only: operator stop is SIMULATED
                lin, yawr, aux_fresh = 0.0, 0.0, True
            prim = self.body_speed(now) if self.body_speed else None
            prev = f.monitor.state
            st = f.monitor.update(now, primary=prim, aux_lin=lin, aux_yaw=yawr, aux_fresh=aux_fresh)
            if st != prev:
                basis = f.monitor.log[-1].get("basis")
                if st == "BASE_STOPPED":
                    self._ev_to(f, "BASE_STOPPED", "base_stopped", basis=basis, lin=lin, yaw_rate=yawr, simulated_stop=self.a.assume_stopped)
                    st = f.monitor.update(now, primary=prim, aux_lin=lin, aux_yaw=yawr, aux_fresh=aux_fresh)
                if st == "WAIT_IMAGE_STABLE" and f.state != "WAIT_IMAGE_STABLE":
                    self._ev_to(f, "WAIT_IMAGE_STABLE", "body_optical_stability_before_aim")
                elif st == "BASE_STOP_TIMEOUT":
                    self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "stop_not_confirmed_timeout")
                    return
                elif st == "IMAGE_STABLE_TIMEOUT":
                    self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "image_not_stable_timeout")
                    return
            if f.monitor.state == "IMAGE_STABLE":
                if not self.can_start_physical():
                    self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "operator_arm_lost_before_aim")
                    return
                self.aim_and_capture(src, f, now_stamp)

    def _ev_to(self, f, state, reason, **kw):
        f.to(state, reason, **kw)
        self.fsm_log(state, reason, **kw)
        self.store.transition(f.key, state, reason)

    # ------------------------------------------------------------------ event start / evidence
    def start_event(self, stop, stamp, pose, bgr, dimg, latest_frame=None):
        self.n_ev += 1
        self._sim_clock = stamp if self.a.source == "replay" else self._sim_clock
        rec = self.store.new_event(self.n_ev, stop, frame_meta=dict(source=stop.get("box_frame"), latest=latest_frame),
                                   track_id=stop.get("spot"))
        self.fsm = f = R1Event(rec, dict(stop, stamp=stamp, pose=list(pose)), self.clock)
        f.trial = self.run / "trials" / f"ev{self.n_ev:03d}"
        for sub in ("runner", "console", "reading"):
            (f.trial / sub).mkdir(parents=True, exist_ok=True)
        ev_info = self.save_stop_evidence(f, stop, bgr, latest_frame)
        f.legacy_rec = dict(t=stop["t"], stamp=stamp, picks=[stop["box"]], z=[stop["z"]], xyz=stop["xyz"], pose=list(pose),
                            image=ev_info.get("source_image"), nobs=stop["nobs"], cur_sim=stop["sim"], stage=stop["stage"],
                            event_key=f.key)
        self.store.set_fields(f.key, evidence=ev_info, reference=stop.get("reference"))
        self.fsm_log("STOP_REQUEST", "policy_pass", stop={k: stop[k] for k in ("box", "z", "sim", "msim", "nscene", "spot")})
        return f

    def save_stop_evidence(self, f, stop, latest_bgr, latest_frame):
        """(a) the frame the box was decided on + box, (b) the newest frame at the request - separately, with times.
        A missing source frame is recorded as missing; the newest frame is NEVER drawn as if it were the source."""
        out = dict(latest_frame=latest_frame, latest_image=None, source_frame=stop.get("box_frame"), source_image=None)
        bf = stop.get("box_frame") or {}
        src_img = self.scene_frames.get(bf.get("frame_id")) if bf.get("frame_id") is not None else None
        if src_img is not None:
            im = src_img.copy()
            x0, y0, x1, y1 = (int(round(v)) for v in (bf.get("box") or stop["box"]))
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(im, f"SOURCE frame {bf.get('frame_id')} stamp {bf.get('header_stamp')}", (6, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            p = f.trial / "runner" / "stop_source_frame_with_box.jpg"
            cv2.imwrite(str(p), im)
            out["source_image"] = str(p)
            atomic_write(SHM / "last_stop.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        else:
            out["source_image_null_reason"] = "SOURCE_FRAME_UNAVAILABLE (evicted or never retained); not substituted"
        if latest_bgr is not None:
            im = latest_bgr.copy()
            cv2.putText(im, f"LATEST at request: frame {(latest_frame or {}).get('frame_id')} (no detection box)", (6, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            p = f.trial / "runner" / "stop_latest_frame_no_box.jpg"
            cv2.imwrite(str(p), im)
            out["latest_image"] = str(p)
        return out

    def fsm_log(self, state, reason, **kw):
        f = self.fsm
        rec = dict(ev=self.n_ev, event_key=getattr(f, "key", None), state=state, reason=reason, wall=time.time(),
                   mono_ns=time.monotonic_ns(), **kw)
        self.logs["fsm"].write(jdump(rec) + "\n"); self.logs["fsm"].flush()

    # ------------------------------------------------------------------ aim / capture (device work is asynchronous)
    def aim_and_capture(self, src, f, now_stamp):
        pose2, _ = src.pose(now_stamp)
        stop = f.stop
        if stop.get("stage") == "P2" and pose2 is not None and stop["spot"] in self.p2.tracks:
            # P2: aim at the world-anchored sign from the current pose (it may have left the RealSense view)
            tr = self.p2.tracks[stop["spot"]]
            if self.a.assume_stopped and stop.get("pose"):
                x0, y0, w0 = stop["pose"]
                adv = stop.get("advance_m") or 0.0
                pose2 = (x0 + adv * math.cos(w0), y0 + adv * math.sin(w0), w0)
                self.fsm_log("AIM", "SIMULATED_STOP_POSE", pose=list(pose2))
            basis = getattr(self.p2, "last_basis", cam_basis(None))
            v = self.p2.view_from(tr, tuple(pose2), basis)
            plan = self.planner.plan_point(v["p_cam"])
            if not plan.get("reachable") and stop.get("predicted", {}).get("reachable") and not self.a.assume_stopped:
                plan["why"] = "ROBOT_OVERSHOT: reachable at the predicted stop pose, not at the actual pose"
                self.fsm_log("AIM", "ROBOT_OVERSHOT", actual=v.get("bearing"), predicted=stop["predicted"].get("bearing"))
            u, v_px, z = (plan.get("u"), plan.get("v"), plan.get("z")) if plan.get("url") else (None, None, None)
            return self._aim_capture_with_plan(f, plan, None, u, v_px, z, src, now_stamp, view=v, pose2=pose2)
        b, z = stop.get("tile") or stop["box"], (stop.get("features") or {}).get("z_tile") or stop["z"]
        u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        plan = self.planner.plan(u, v, z)
        return self._aim_capture_with_plan(f, plan, b, u, v, z, src, now_stamp, pose2=pose2)

    def _aim_capture_with_plan(self, f, plan, b, u, v, z, src, now_stamp, rt=None, pose2=None, view=None):
        """Event-loop side: decide, then hand the physical job to the single device worker. Never waits for it."""
        if not plan.get("url"):
            plan.setdefault("reachable", False)
        f.plan, f.aim_ctx = plan, dict(b=b, u=u, v=v, z=z, pose2=list(pose2) if pose2 else None,
                                       view=None if view is None else {k: x for k, x in view.items()})
        self._ev_to(f, "AIM", "image_stable")
        if not plan.get("reachable"):
            why = "robot_overshot" if str(plan.get("why", "")).startswith("ROBOT_OVERSHOT") else "pantilt_out_of_range"
            f.result["aim"] = dict(mode="REFUSED", ok=False, why=why)
            self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, why)       # no device command was sent
            return
        if not self.planner.execute:
            f.result["aim"] = dict(mode="DRY_RUN", ok=False, why="PENDING_APPROVAL")
            if DIAGNOSTIC_PROXY or getattr(self.a, "diagnostic_proxy", False):
                return self._proxy_capture(f)
            self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "pantilt_not_approved_no_capture")
            return
        self._submit_aim(f)

    def _submit_aim(self, f):
        f.attempt += 1
        f.device_idle_confirmed = False
        timeout = float(os.environ.get("SCI_AIM_TIMEOUT_S", "30"))
        ok = self.device.submit(f.key, self._device_aim_job, f.plan, f.id, f.attempt, timeout)
        if not ok:                                             # someone else owns the device -> no parallel aim
            f.device_idle_confirmed = False
            self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "device_busy_no_parallel_aim")
            return
        self.store.set_fields(f.key, attempt_id=f.attempt)
        self._ev_to(f, "CAPTURE", f"device_job_submitted_attempt{f.attempt}")

    def _device_aim_job(self, plan, ev_id, attempt, timeout):
        """Runs in the device worker thread: one r1-aim-1 request + capture fetch by capture_id. No FSM changes here."""
        r = self.planner.aim(plan, run_id=self.run_id, event_id=ev_id, attempt=attempt, timeout_s=timeout)
        data = None
        if r.get("ok") and r.get("capture"):
            data = self.planner.fetch_capture(r["capture"].get("capture_id"))
        return dict(kind="aim", aim=r, image_bytes=data)

    def _device_home_job(self, ev_id, attempt, timeout=30.0):
        r = self.planner.aim(dict(reachable=True), run_id=self.run_id, event_id=ev_id, attempt=attempt, timeout_s=timeout, kind="home")
        return dict(kind="home", aim=r)

    def on_device_result(self, src, r):
        f = self.fsm
        if f is None or r["key"] != f.key:                     # late result of an old event: history only
            self.fault("LATE_DEVICE_RESULT_QUARANTINED", f"key={r['key']} active={getattr(f, 'key', None)}")
            try:
                self.store.set_fields(r["key"], late_device_result=dict(ok=r.get("ok"), error=r.get("error")))
            except Exception:
                pass
            return
        if not r["ok"]:                                        # exception: device state unknown
            f.device_used, f.device_idle_confirmed = True, False
            self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "device_job_exception:" + str(r.get("error"))[:120])
            return
        res = r["result"]
        aim = res["aim"]
        if aim.get("submitted") is not False:                 # True or None (unknown whether it started) = we commanded it
            f.device_used = True
        f.device_idle_confirmed = bool(aim.get("device_idle_confirmed"))
        if res["kind"] == "home":
            f.result["home"] = {k: aim.get(k) for k in ("request_id", "status", "reason_code", "device_state", "hw_done")}
            self.store.set_fields(f.key, home=f.result["home"])
            self._release(f)
            return
        f.result["aim"] = {k: aim.get(k) for k in ("mode", "ok", "why", "request_id", "status", "reason_code", "device_state",
                                                   "hw_done", "logical_expired", "focus", "attempt_id")}
        self.store.set_fields(f.key, request_id=aim.get("request_id"), aim=f.result["aim"])
        self.fsm_log("AIM", aim.get("mode"), ok=aim.get("ok"), why=aim.get("why"), request_id=aim.get("request_id"))
        cap = aim.get("capture")
        data = res.get("image_bytes")
        if not aim.get("ok") or cap is None:
            self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, f"aim_failed:{aim.get('reason_code')}")
            return
        prev = f.prev_capture
        if prev is not None and (cap.get("capture_id") == prev.get("capture_id") or
                                 (cap.get("frame_seq") is not None and prev.get("frame_seq") is not None
                                  and cap["frame_seq"] <= prev["frame_seq"])):
            self._finish_observation(f, C.CaptureStatus.FRESH_CAPTURE_UNAVAILABLE, "retry_returned_no_new_exposure")
            return
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        q = self.capture_quality(img)
        path = f.trial / "console" / f"capture_{cap.get('capture_id')}.jpg"
        if data:
            path.write_bytes(data)                                 # unmodified bytes of THIS capture_id
        import hashlib
        cap_rec = dict(cap, saved_path=str(path) if data else None, bytes_sha256=hashlib.sha256(data).hexdigest() if data else None,
                       request_id=aim.get("request_id"), pantilt_ok=True, quality=q,
                       servo_settled_before_exposure=(cap.get("servo_settled_before_exposure")
                                                      if "servo_settled_before_exposure" in cap else None),
                       image_stable_at_exposure="UNVERIFIED_BY_CONSOLE" if "servo_settled_before_exposure" not in cap else
                       ("OK" if cap["servo_settled_before_exposure"] else "NOT_SETTLED"))
        f.prev_capture = cap
        f.result.update(capture=cap_rec["saved_path"], capture_id=cap.get("capture_id"), capture_source="IMX477_PANTILT", quality=q)
        self.store.set_fields(f.key, capture_id=cap.get("capture_id"), capture=cap_rec)
        self.store.transition(f.key, C.CaptureStatus.RAW_CAPTURE_SAVED, "capture_saved") if data else None
        self.fsm_log("CAPTURE", "IMX477_PANTILT", attempt=f.attempt, quality=q, capture_id=cap.get("capture_id"))
        good = data is not None and q["ok"] and cap_rec["image_stable_at_exposure"] != "NOT_SETTLED"
        if good:
            self._finish_observation(f, C.CaptureStatus.CAPTURE_QUALITY_OK, "quality_ok", cap_rec=cap_rec)
            return
        if f.attempt <= self.cfg["event"]["capture_retries"] and f.device_idle_confirmed and self.can_start_physical():
            self.fsm_log("CAPTURE", "quality_fail_retry_with_new_exposure", attempt=f.attempt)
            self._submit_aim(f)                                      # a new request = a new exposure (new capture_id)
            return
        self._finish_observation(f, C.CaptureStatus.CAPTURE_QUALITY_FAIL,
                                 "quality_fail" if data else "capture_bytes_unavailable", cap_rec=cap_rec)

    def _proxy_capture(self, f):
        """Explicit diagnostic mode only: RealSense crop, labelled CAPTURE_PROXY, pantilt_ok=False, never auto-consumed."""
        stop = f.stop
        img = self.p2_crops.get(stop.get("spot"))
        if img is None:
            self._finish_observation(f, C.CaptureStatus.NO_CAPTURE, "proxy_crop_unavailable")
            return
        cid = C.new_id("proxy")
        path = f.trial / "console" / f"capture_{cid}_REALSENSE_PROXY.jpg"
        cv2.imwrite(str(path), img)
        q = self.capture_quality(img)
        cap_rec = dict(capture_id=cid, saved_path=str(path), pantilt_ok=False, source="REALSENSE_PROXY_LAST_VIEW_NO_PANTILT",
                       quality=q, diagnostic_only=True)
        f.result.update(capture=str(path), capture_id=cid, capture_source=cap_rec["source"], quality=q)
        self.store.set_fields(f.key, capture_id=cid, capture=cap_rec)
        self._finish_observation(f, C.CaptureStatus.CAPTURE_PROXY, "diagnostic_proxy", cap_rec=cap_rec)

    @staticmethod
    def capture_quality(img):
        if img is None or img.size == 0:
            return dict(ok=False, why="empty", metric="laplacian_var_cv32f_full_gray", version="e5_node.capture_quality.v1")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        sat = float((g >= 250).mean())
        return dict(ok=bool(sharp >= QUALITY_SHARP_MIN and sat <= QUALITY_SAT_MAX), sharp=sharp, sat=sat, h=int(g.shape[0]),
                    w=int(g.shape[1]), metric="laplacian_var_cv32f_full_gray", roi="full_image", threshold=QUALITY_SHARP_MIN,
                    version="e5_node.capture_quality.v1", comparable_with_console_focus_metric=False)

    def _finish_observation(self, f, cap_status, reason, cap_rec=None):
        """Observation part done (success or failure). Then the device is returned to a safe state (HOME via the
        same single owner) and the RELEASE status is decided from CONFIRMED device state only."""
        f.capture_status = cap_status
        self.store.transition(f.key, cap_status, reason, capture_status=cap_status)
        self.fsm_log(cap_status, reason)
        if cap_status in (C.CaptureStatus.CAPTURE_QUALITY_OK, C.CaptureStatus.CAPTURE_QUALITY_FAIL, C.CaptureStatus.CAPTURE_PROXY):
            self.enqueue_reader_job(f, cap_status, cap_rec)
        T = {x["state"]: x["t"] for x in f.log}
        f.result["durations_s"] = dict(wait_base_stop=(T.get("BASE_STOPPED", 0) - T["WAIT_BASE_STOP"]) if "BASE_STOPPED" in T and "WAIT_BASE_STOP" in T else None,
                                       clock="replay scene clock" if self.a.source == "replay" else "monotonic")
        (f.trial / "runner" / "events.jsonl").write_text(jdump(f.legacy_rec) + "\n")
        if f.device_used and HOME_AFTER and f.device_idle_confirmed and self.can_start_physical():
            f.attempt += 1
            if self.device.submit(f.key, self._device_home_job, f.id, f.attempt):
                self._ev_to(f, "HOME", "return_to_safe_posture")
                return
        self._release(f)

    def _release(self, f):
        if f.device_used and not f.device_idle_confirmed:
            rel = C.ReleaseStatus.RELEASE_BLOCKED_ACTUATOR_UNKNOWN
        elif f.capture_status == C.CaptureStatus.CAPTURE_QUALITY_OK:
            rel = C.ReleaseStatus.RELEASE_READY_CAPTURE_OK
        else:
            rel = C.ReleaseStatus.RELEASE_READY_OBSERVATION_FAILED
        f.release = rel
        f.to(rel, "device_state_confirmed" if rel != C.ReleaseStatus.RELEASE_BLOCKED_ACTUATOR_UNKNOWN else "device_state_unknown")
        self.fsm_log(rel, f.log[-1]["reason"])
        self.store.transition(f.key, rel, f.log[-1]["reason"], release=rel,
                              terminal=rel != C.ReleaseStatus.RELEASE_BLOCKED_ACTUATOR_UNKNOWN)
        (f.trial / "fsm_log.json").write_text(jdump(f.log))

    def enqueue_reader_job(self, f, cap_status, cap_rec):
        cap_rec = cap_rec or {}
        rec = self.store.get(f.key) or {}
        img = cap_rec.get("saved_path")
        import hashlib
        sha = hashlib.sha256(Path(img).read_bytes()).hexdigest() if img and Path(img).exists() else None
        job = dict(schema_version=C.SCHEMA_VERSION, run_id=self.run_id, event_id=f.id, event_key=f.key, ev=f.id,
                   trial=str(f.trial), capture_id=cap_rec.get("capture_id"), request_id=cap_rec.get("request_id"),
                   capture_status=cap_status, pantilt_ok=bool(cap_rec.get("pantilt_ok")), image=img, image_sha256=sha,
                   capture_source=f.result.get("capture_source"), quality=cap_rec.get("quality"),
                   target_roi=None, target_roi_verified=False,
                   target_roi_null_reason="IMX477 target ROI not verified (no IMX477 intrinsics); use full image",
                   odom_frame=(f"base(delta {os.environ.get('SCI_ODOM_DELTA_DEG', '45.3')} deg, fastlio_to_base_2d)"
                               if os.environ.get("SCI_ODOM_BASE") == "1" else "raw_odom_topic_frame"),
                   trial_id=rec.get("trial_id"), generation_id=rec.get("generation_id"),
                   **({} if rec.get("trial_id") else dict(trial_id_null_reason=rec.get("trial_id_null_reason"))),
                   created=time.time(), created_mono_ns=time.monotonic_ns(), boot_id=C.BOOT_ID,
                   stop={k: f.stop.get(k) for k in ("t", "stage", "spot", "box", "z", "xyz", "sim", "stamp", "pose", "box_frame")})
        job.update(self.reader_job_extra(f) or {})
        C.atomic_write_json(self.run / "jobs" / "pending" / f"{_safe(f.key)}__{_safe(job['capture_id'])}.json", job)
        self.store.set_fields(f.key, reader_job=dict(enqueued=C.stamp(), capture_status=cap_status))

    def attach_readings(self):
        """Independent of FSM activity: every reader result is attached to the exact event_key + capture_id."""
        done = self.run / "jobs" / "done"
        changed = False
        for p in done.glob("*.json"):
            try:
                m = p.stat().st_mtime_ns
            except OSError:
                continue
            if self.done_seen.get(p.name) == m:
                continue
            self.done_seen[p.name] = m
            changed = True
        if changed:
            out = self.store.scan_done(done)
            self.status["reader_attach"] = out
        self.update_event_status()
        return changed

    def update_event_status(self):
        evs = []
        for r in self.store.hot():
            evs.append(dict(id=r["event_id"], event_key=r["event_key"], run_id=r["run_id"], state=r["state"], terminal=r["terminal"],
                            capture_status=r.get("capture_status"), capture_id=r.get("capture_id"), release=r.get("release"),
                            reading=r.get("reading"), reading_status=r.get("reading_status"), t=(r.get("stop") or {}).get("t"),
                            box=(r.get("stop") or {}).get("box"), z=(r.get("stop") or {}).get("z"),
                            trial_id=r.get("trial_id"), evidence=r.get("evidence"),
                            aim=(r.get("aim") or {}).get("status"), observation_id=r.get("observation_id"),
                            payload_sha256=r.get("payload_sha256"), payload_revision=r.get("payload_revision"), capture_source=(r.get("capture") or {}).get("source", "IMX477_PANTILT" if r.get("capture") else None)))
        self.status["events"] = evs[-20:]

    # ------------------------------------------------------------------ UI outputs
    def draw(self, bgr, res, dec, t, act, frame=None):
        """Boxes are drawn on THEIR OWN scene base frame (never on a newer frame)."""
        im = bgr.copy()
        for i, (r, d) in enumerate(zip(res["regions"], dec)):
            x0, y0, x1, y1 = (int(round(v)) for v in r["box"])
            c = COLORS.get(d["display"], (200, 200, 200))
            cv2.rectangle(im, (x0, y0), (x1, y1), c, 3 if d["display"] == "STOP_REQUEST" else 1)
            lab = f"#{r['spot'] if r['spot'] is not None else '-'} {r['sim']:.3f} {d['display'].replace('REJECTED_', 'X_')}"
            cv2.putText(im, lab, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
        if act:
            cv2.rectangle(im, (0, 0), (639, 28), (0, 0, 255), -1)
            cv2.putText(im, f"STOP_REQUEST  z={act['z']:.1f}m  sim={act['sim']:.3f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(im, f"t={t:.1f}s frame={(frame or {}).get('frame_id')}", (470, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if self.a.ui_scale != 1.0:                       # UI frame cost control (JPEG encode only once per scene)
            im = cv2.resize(im, None, fx=self.a.ui_scale, fy=self.a.ui_scale, interpolation=cv2.INTER_AREA)
        atomic_write(SHM / "frame.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        if act:
            atomic_write(SHM / "last_stop.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())

    def publish_status(self, src, rec, force=False):
        s = self.status
        s.update(now=time.time(), source_health=src.health(), resources=self.mon.last, active_stage=self.active_stage,
                 pantilt_console=self.planner.console_state(0.2) if int(time.time()) % 5 == 0 else s.get("pantilt_console"),
                 pantilt_execute_enabled=self.planner.execute, e4=self.e4.get(time.time() if self.a.source == "ros" else (rec or {}).get("stamp", 0)),
                 fsm=None if self.fsm is None else dict(id=self.fsm.id, event_key=getattr(self.fsm, "key", None), state=self.fsm.state,
                                                        log=self.fsm.log[-12:], result=self.fsm.result),
                 device=dict(busy=self.device.busy, busy_key=self.device.busy_key),
                 queues=dict(stop_proposals_dropped=self.proposals.dropped, frame_meta=len(self.frame_meta),
                             scene_frames=len(self.scene_frames)),
                 n_stops=dict({k: len(v.events) for k, v in self.stages.items()}, A0=len(self.legacy_events), P2=len(self.p2.events)),
                 p2_tracks=self.p2_summary, p2_missed=len(self.p2.missed))
        if rec is not None:
            s["scene"] = dict(t=rec["t"], stamp=rec["stamp"], timing=rec["timing"], speed=rec["speed"], thr=rec["thr"],
                              n_scene=rec["n_scene"], n_skipped=rec["n_skipped"],
                              regions=[dict(box=r["box"], spot=r["spot"], sim=r["sim"], msim=r["msim"], z=r["z"],
                                            obliquity=r["feat"]["plane"].get("obliquity_deg"), pantilt=r["feat"]["pantilt"],
                                            text_px=r["feat"]["text_px"], glare=r["feat"]["glare"],
                                            display=r["dec"][self.active_stage]["display"],
                                            state=r["dec"][self.active_stage]["state"], reasons=r["dec"][self.active_stage]["reasons"])
                                       for r in rec["regions"]])
        s["status_mono_ns"] = time.monotonic_ns()
        atomic_write(SHM / "status.json", jdump(s).encode())


STOP = [False]


def scene_mode_for(a):
    """E5RT2: 'sync' keeps the original blocking scene loop (deterministic replay); 'async' = realtime path.
    Env SCI_E5RT2_SCENE_MODE overrides the argument. auto: async for live ROS input or --realtime replay."""
    m = os.environ.get("SCI_E5RT2_SCENE_MODE", "").strip() or getattr(a, "scene_mode", "auto") or "auto"
    if m == "auto":
        return "async" if (getattr(a, "source", None) in ("ros", "ros_readonly", "shared_raw") or getattr(a, "realtime", False)) else "sync"
    if m not in ("sync", "async"):
        raise ValueError(f"scene mode {m!r}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["replay", "ros", "ros_readonly", "shared_raw"], default="replay",
                    help="SEM1: ros_readonly = approved allowlisted read-only receiver; shared_raw = verified shared producer only")
    ap.add_argument("--dataset", default=str(WS / "datasets" / "bags" / "S-1F"))
    ap.add_argument("--realtime", action="store_true", help="replay at recorded speed (otherwise as fast as possible)")
    ap.add_argument("--t-start", type=float, default=0.0)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--odom-topic", default="/utlidar/robot_odom")
    ap.add_argument("--policy", default=str(WS / "config" / "e5_policy.json"))
    ap.add_argument("--stage", default="P2", choices=["A1", "A2", "A3", "A4", "P2"], help="stage that drives the UI/FSM")
    ap.add_argument("--p2-policy", default=str(WS / "config" / "e5_policy_p2.yaml"))
    ap.add_argument("--ui-scale", type=float, default=1.0, help="annotated UI frame scale (0.5 halves encode cost)")
    ap.add_argument("--max-temp-c", type=float, default=85.0, help="overload guard: halve SigLIP scene rate above this tj")
    ap.add_argument("--min-mem-mb", type=float, default=1500.0, help="overload guard: halve scene rate below this MemAvailable")
    ap.add_argument("--e4-file", default=None)
    ap.add_argument("--pantilt-execute", action="store_true", help="also requires env SCI_PANTILT_APPROVED=1")
    ap.add_argument("--assume-stopped", action="store_true", help="OFFLINE/UI TEST ONLY: simulate operator stop")
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--max-lag", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--scene-mode", choices=["auto", "sync", "async"], default="auto",
                    help="E5RT2: async = scene on a worker thread, P2/tracking/event never wait (auto: ros or --realtime)")
    ap.add_argument("--scene-deadline-s", type=float, default=10.0, help="E5RT2: logical deadline of one scene (late = diagnostic)")
    ap.add_argument("--max-stop-age-s", type=float, default=1.5, help="E5RT2: a scene older than this proposes no STOP itself")
    ap.add_argument("--diagnostic-proxy", action="store_true",
                    help="DIAGNOSTIC ONLY: RealSense crop as CAPTURE_PROXY (pantilt_ok=false, never auto-consumed)")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: STOP.__setitem__(0, True))
    node = Node(a)
    src = ReplaySource_or_Ros(a)
    try:
        node.loop(src)
    finally:
        src.close()
        if node.frontend:
            node.frontend.vlm.close()
            if node.frontend.pool:
                node.frontend.pool.terminate()
        summary = dict(run=str(node.run), stops={k: len(v.events) for k, v in node.stages.items()}, legacy_A0=len(node.legacy_events))
        (node.run / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))


def ReplaySource_or_Ros(a):
    from sources import ReplaySource, RosSource
    if a.source == "replay":
        return ReplaySource(a.dataset, realtime=a.realtime, t_start=a.t_start, t_end=a.t_end)
    return RosSource(odom_topic=a.odom_topic)


if __name__ == "__main__":
    main()
