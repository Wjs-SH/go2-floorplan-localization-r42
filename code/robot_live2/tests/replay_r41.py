#!/usr/bin/env python3
"""R41 INTEGRATED REPLAY (OFFLINE, not a field result). The production processes run as ctl starts them: e2_worker.py (4 children, repeat search, nice 10, cores 0,1,6,7) and e34_node.py (E3, E4, auto_fsm, E7, E8, OCR, VLM, core 5). Only the sensors and the operator are replaced.
Input, at the recorded times, every stamp shifted by one constant: real store samples, far points, FAST-LIO rate series, odometry ring, the E6 outcome records with the real stored photos. OCR and VLM run for real on those photos (--no-vlm: recorded situation without). The real VisualStop judges the recorded E5 scenes against the live auto_state.
GT never enters a process: it is read only by the summary in another step.
usage: replay_r41.py RUN TAG [--code DIR] [--no-vlm] [--until-s S] [--drop OBS,OBS]"""
import argparse, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
S = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("tag")
ap.add_argument("--code", default=str(S / "candidate")); ap.add_argument("--no-vlm", action="store_true"); ap.add_argument("--until-s", type=float, default=None); ap.add_argument("--drop", default="")
ap.add_argument("--procs", default="4"); ap.add_argument("--live-lio", action="store_true", help="load test: the FAST-LIO rate comes from the real FAST-LIO running beside this replay, so that the protection really reacts")
a = ap.parse_args()
CODE = Path(a.code)
X = OPR / "experiments" / a.run
EXP = json.loads((X / "experiment.json").read_text())
T0 = float(EXP["created_wall"])
ROOT = Path.home() / "sci_ws_alt_imu_20261029_204740/runs/r41" / a.tag
STORE = Path("/dev/shm") / f"r41_{a.tag}"
for p in (ROOT, STORE):
    shutil.rmtree(p, ignore_errors=True)
STATE = ROOT / "state"; HUB = ROOT / "hub"; CTL = ROOT / "ctl"
for p in (STATE / "e2", STATE / "e6_captures", HUB, CTL, ROOT / "e5", STATE / "control"):
    p.mkdir(parents=True)
sys.path[:0] = [str(CODE / "live2"), str(CODE / "external/sci_live/realtime")]
from live_store import LiveStore  # noqa: E402
START_AT = 25.0                    # recorded time at which this replay begins
SHIFT = time.time() - (T0 + START_AT)
def sh(o):
    """every epoch time inside a record moves by the same constant; nothing else changes"""
    if isinstance(o, dict):
        return {k: sh(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sh(v) for v in o]
    if isinstance(o, float) and abs(o - T0) < 50000.0:
        return o + SHIFT
    return o
def jl(p):
    out = []
    try:
        for ln in open(p):
            try:
                out.append(json.loads(ln.replace("Infinity", "1e9")))
            except ValueError:
                pass
    except OSError:
        pass
    return out
def atomic(p, obj):
    p = Path(p); tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(obj, ensure_ascii=False, default=float)); os.replace(tmp, p)
# ---------- events in time order
ev = []
import csv
rows = list(csv.DictReader(open(X / "archive/live2_store/odom_link.csv")))
far = np.load(X / "archive/live2_store/far_points.npz")
for r in rows:
    ev.append((float(r["t_sec"]), "sample", r))
for r in jl(X / "state/lio_series.jsonl"):
    ev.append((float(r["wall"]), "lio", r))
trace = jl(X / "archive/field_eval/odom_trace.jsonl")
for r in trace:
    ev.append((float(r["wall"]), "odom", r))
stored = {}
for r in jl(X / "state/e8_observations.jsonl"):
    if r.get("event") in ("STORED", "REJECTED") and r.get("observation_id") not in stored:
        stored[r["observation_id"]] = float(r["wall"])
DROP = {x for x in a.drop.split(",") if x}
for r in jl(X / "state/e6_precision_results.jsonl"):
    if r["observation_id"][-4:] not in DROP:
        ev.append((stored.get(r["observation_id"], float(r.get("capture_wall") or r["wall"])), "e6", r))
scenes = jl(X / "archive/observe_run/scenes.jsonl"); det = jl(X / "archive/observe_run/live2_detection_pose.jsonl")
for s_, d_ in zip(scenes, det):
    ev.append((float(s_["stamp"]), "scene", (s_, d_)))
ev.sort(key=lambda e: e[0])
end_rec = T0 + (a.until_s if a.until_s else float(EXP["ended_wall"]) - T0 - 8.0)
# ---------- production processes, the arguments of ctl.py
atomic(CTL / "experiment.json", sh(dict(exp_id=a.run, state="RUNNING", loc_mode="AUTO_INITIAL_GLOBAL", siglip=True, created_wall=T0, modules=dict(e6=dict(state="RUNNING")), replay="OFFLINE_REPLAY_NOT_FIELD")))
sess = json.loads((X / "archive/live2_store/session.json").read_text())
store = LiveStore(str(STORE), keep_s=900.0, reset=True)
atomic(STORE / "session.json", sh(sess))
atomic(STATE / "control/pause.json", dict(paused=False))
atomic(HUB / "index.json", dict(wall=time.time(), odom=[], caminfo=json.loads((X / "archive/hub_index.json/index.json").read_text()).get("caminfo")))
VENV = str(Path.home() / "sci_ws/external/sci_live/venv_gpu/bin/python")
env2 = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES_="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
env2["CUDA_VISIBLE_DEVICES"] = ""
env3 = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
if not a.no_vlm:
    env3["SCI_E6_VLM"] = "1"
log2, log3 = open(ROOT / "e2.out", "ab"), open(ROOT / "e34.out", "ab")
P2 = subprocess.Popen(["nice", "-n", "10", "taskset", "-c", "0,1,6,7", VENV, "-I", "-B", str(CODE / "live2/e2_worker.py"), "--building", "kci", "--store", str(STORE), "--out", str(STATE / "e2"), "--cache-dir", str(OPR / "runtime/e2_cache"), "--procs", a.procs, "--device", "cpu", "--refresh-s", "90"], stdout=log2, stderr=log2, env=env2, start_new_session=True)
P3 = subprocess.Popen(["nice", "-n", "5", "taskset", "-c", "5", "/usr/bin/python3", "-s", str(CODE / "live2/e34_node.py"), "--store", str(STORE), "--state", str(STATE), "--building", "kci", "--max-s", "3600", "--hub", str(HUB), "--e5-status", str(ROOT / "e5/status.json")], stdout=log3, stderr=log3, env=env3, start_new_session=True)
def finish(*_):
    for p in (P3, P2):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
    for p in (P3, P2):
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
signal.signal(signal.SIGTERM, lambda *_: (finish(), sys.exit(3)))
# real E5 stop gate against the live auto_state
VS = None
try:
    sys.path.insert(0, str(CODE / "src/sci_sensor_bridge"))
    import visual_stop as VSM, e6_precision_bridge as B
    K = json.loads((X / "archive/hub_index.json/index.json").read_text()).get("caminfo", {}).get("k")
    VS = VSM.VisualStop(ROOT / "e5", contract_ok=False, K_fn=lambda: K, state_dir=STATE, envelope=B.load_envelope())
except Exception as ex:                       # noqa: BLE001
    print("E5 gate replay unavailable:", type(ex).__name__, ex, flush=True)
n = 0; track = []; odom = []; hz = None; samp = None; log = open(ROOT / "replay_timeline.jsonl", "a"); t_e3 = 0.0; last_odom = None
def tl(**k):
    log.write(json.dumps(dict(k, t=round(time.time() - SHIFT - T0, 1)), ensure_ascii=False, default=float) + "\n"); log.flush()
def e3_sample():
    d = None
    try:
        d = json.loads((STATE / "e3_state.json").read_text())
    except (OSError, ValueError):
        return
    H = sorted((h for h in d.get("hypotheses") or [] if h.get("status") in ("ACTIVE", "SUSPENDED")), key=lambda h: h.get("score") if h.get("score") is not None else 9e9)
    tl(kind="e3", odom=last_odom, leader=d.get("leader"), kf_pending=d.get("kf_pending"), lag=d.get("leader_kf_lag_s"), top=[{k: h.get(k) for k in ("id", "floor", "score", "status", "n_kf", "ppm", "heading_deg", "recent", "provenance", "global_pose")} for h in H[:12]])
tl(kind="start", shift=SHIFT, code=str(CODE), vlm=not a.no_vlm)
try:
    for t, kind, r in ev:
        if t < T0 + START_AT or t > end_rec:
            continue
        while True:
            now = time.time()
            if now >= t + SHIFT:
                break
            atomic(STATE / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="REPLAY", written_wall=now, odom=dict(n=len(trace), session=0, hz_10s=hz, last_rx_age_s=0.05, frame_id="camera_init", child_frame_id="base_link", last_raw=None if last_odom is None else dict(stamp=now, x=last_odom[0], y=last_odom[1], yaw_deg=last_odom[2])), track_raw=track[-1500:]))
            if samp:
                atomic(STATE / "e2_feed_state.json", dict(samples=n, last=dict(samp, wall=now), session=dict(odom_session_id=sess["odom_session_id"]), store=str(STORE), source="REPLAY"))
            if now - t_e3 >= 5.0:
                t_e3 = now; e3_sample()
            if P2.poll() is not None or P3.poll() is not None:
                raise RuntimeError(f"production process died e2={P2.poll()} e34={P3.poll()}")
            time.sleep(min(0.1, max(0.0, t + SHIFT - now)))
        if kind == "sample":
            with np.load(X / "archive/live2_store/samples" / r["sample_id"] / "data_short.npz") as z:
                xy, role = z["a1_3_1_all_candidate_xy"], z["a1_3_1_all_candidate_role"]
            fp = far[r["sample_id"]] if r["sample_id"] in far.files else np.zeros((0, 2), np.float32)
            sid = store.add(t + SHIFT, float(r["fastlio_x"]), float(r["fastlio_y"]), float(r["fastlio_yaw_rad"]), xy, role, fp)
            n += 1; samp = dict(sid=sid, stamp=t + SHIFT, n_cells=int(len(xy)), n_far=int(len(fp)))
        elif kind == "lio":
            q = sh(r); hz = q.get("hz")
            if a.live_lio:
                try:
                    q = json.loads(Path(OPR / "runtime/state/lio_series.jsonl").read_text().splitlines()[-1]); hz = q.get("hz")
                except (OSError, ValueError, IndexError):
                    pass
            with open(STATE / "lio_series.jsonl", "a") as f:
                f.write(json.dumps(q, default=float) + "\n")
        elif kind == "odom":
            last_odom = (r["x"], r["y"], r["yaw_deg"]); track.append((r["x"], r["y"]))
            odom.append([t + SHIFT, r["x"], r["y"], np.radians(r["yaw_deg"])]); odom = odom[-160:]
            atomic(HUB / "index.json", dict(wall=time.time(), odom=odom, odom_session=sess["odom_session_id"], caminfo=json.loads((X / "archive/hub_index.json/index.json").read_text()).get("caminfo")))
        elif kind == "e6":
            q = sh(r)
            if q.get("capture_file"):
                src = X / "state/e6_captures" / Path(q["capture_file"]).name
                dst = STATE / "e6_captures" / src.name
                shutil.copy(src, dst); q["capture_file"] = str(dst)
            with open(STATE / "e6_precision_results.jsonl", "a") as f:
                f.write(json.dumps(q, ensure_ascii=False, default=float) + "\n")
            tl(kind="e6_inject", obs=q["observation_id"][-4:], status=q.get("status"))
        elif kind == "scene" and VS is not None:
            s_, d_ = r
            o_ = VS.feed(sh(s_), s_["stamp"] + SHIFT, sh(d_), now=time.time())
            if o_:
                tl(kind="e5_stop_request", obs=o_["observation_id"][-4:], sim=o_.get("sim"), z=o_.get("depth_m"))
    tl(kind="end_of_record", e5_gates=VS.status().get("gates") if VS else None, n_e5=VS.status().get("n") if VS else None)
    e3_sample()
    time.sleep(20.0)                       # the last E7 / reading may still be running
    e3_sample()
finally:
    finish()
    log.close()
print("REPLAY_DONE", a.run, a.tag, flush=True)
