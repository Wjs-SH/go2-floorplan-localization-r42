#!/usr/bin/env python3
"""fake_module.py — TEST child for a LIVE2 FAKE workspace (marker LIVE2_FAKE_WORKSPACE.json). ctl launches it in place of
the real module, under the REAL live1_supervisor, so start / stop / ownership / approval paths are exercised without ROS,
GPU, sensors or the robot. It only writes FAKE-labelled liveness files inside the fake workspace.

fake_inject.json (workspace root) may contain {"exit": {"<module>": rc}} -> the child exits with rc after 1 s.
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path



def _fake_depth(root, now):
    """UI tests: fake_inject {"hub_depth_offset_s": 900} (frozen) or null (no depth). Default: fresh."""
    try:
        inj = json.loads((root / "fake_inject.json").read_text())
    except (OSError, ValueError):
        inj = {}
    if "hub_depth_offset_s" not in inj:
        return now
    return None if inj["hub_depth_offset_s"] is None else now - float(inj["hub_depth_offset_s"])


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    if not (root / "LIVE2_FAKE_WORKSPACE.json").exists() or (root / "LIVE_FIELD2_WORKSPACE.json").exists():
        sys.exit("REFUSED: fake_module runs only inside a FAKE workspace")
    state, hub, shm = root / "runtime" / "state", root / "runtime" / "fake_hub", root / "runtime" / "e5_shm"
    inj = json.loads((root / "fake_inject.json").read_text()) if (root / "fake_inject.json").exists() else {}
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    m, t0, n = a.module, time.time(), 0
    print(f"FAKE {m} started pid {os.getpid()}", flush=True)
    if m == "fastlio":
        time.sleep(0.5)
        print("IMU Initial Done\n[INFO] [laser_mapping]: Initialize the map kdtree (FAKE)", flush=True)
    while not stop[0]:
        rc = (inj.get("exit") or {}).get(m)
        if rc is not None and time.time() - t0 > 1.0:
            print(f"FAKE {m} injected exit rc={rc}", flush=True)
            sys.exit(int(rc))
        now = time.time()
        n += 1
        if m == "lio2":
            x, y = 3 * math.cos(now / 20), 3 * math.sin(now / 20)
            atomic_json(state / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="FAKE", written_wall=now,
                                                       odom=dict(n=n, hz_10s=10.0, last_rx_age_s=0.05, frame_id="FAKE", child_frame_id="FAKE",
                                                                 last_raw=dict(stamp=now, x=x, y=y, yaw_deg=0.0)),
                                                       track_raw=[[x, y]], accumulation=dict(enabled=True, why="FAKE")))
            atomic_json(state / "e2_feed_state.json", dict(source="FAKE", samples=n, written_wall=now, last=dict(sid=f"FAKE_{n:06d}", wall=now)))
        elif m == "hub":
            atomic_json(hub / "index.json", dict(wall=now, last_color_wall=now, writer="FAKE", frames=[dict(seq=n, stamp=now, depth_stamp=_fake_depth(root, now))],
                                                 receiver_health=dict(color=dict(hz=15.0), depth=dict(hz=15.0))))
        elif m == "observe":
            if n == 1:
                from PIL import Image
                shm.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (848, 480), (40, 40, 40)).save(shm / "frame_inference.jpg")
            if not inj.get("rich"):
                atomic_json(shm / "status.json", dict(siglip="OK", run_id="FAKE_observe", source="FAKE",
                                                      scene=dict(n_scene=n, stamp=now, regions=[dict(box=[10, 10, 50, 40], spot=1, sim=0.12, display="CANDIDATE")])))
            atomic_json(root / "runtime" / "owned" / "observe.keeper.json", dict(siglip="RUNNING", runs=[dict(pid=os.getpid())]))
        elif m in ((json.loads((root / "fake_inject.json").read_text()).get("freeze_state") or []) if (root / "fake_inject.json").exists() else []):
            pass                                 # UI test: the test owns this module's state file (fake_inject {"freeze_state": [...]})
        elif m == "e2":
            p = json.loads((state / "control" / "pause.json").read_text()) if (state / "control" / "pause.json").exists() else {}
            atomic_json(state / "e2" / "e2_state.json", dict(kind="LIVE2_E2_STATE", source="FAKE", status="PAUSED" if p.get("paused") else "IDLE",
                                                             building="kci", pass_id=0, floors=[], updated_wall=now))
        elif m == "e6":
            atomic_json(state / "e6_bridge.json", dict(kind="LIVE2_E6_BRIDGE", source="FAKE", state="RUNNING", why=None, wall=now,
                                                       focus_mode="AUTO", af_strategy="student_stepwise", n_observations=0))
        elif m == "e34":
            p = json.loads((state / "control" / "pause.json").read_text()) if (state / "control" / "pause.json").exists() else {}
            atomic_json(state / "e4_state.json", dict(state="INITIALIZING", reasons=["FAKE"], metrics={}, written_wall=now,
                                                      paused=dict(paused=True) if p.get("paused") else None))
        if inj.get("rich"):
            rich(m, n, now, t0, state, hub, shm, inj)
        time.sleep(0.2)
    print(f"FAKE {m} stopped", flush=True)


def rich(m, n, now, t0, state, hub, shm, inj):
    """UI screenshots only (fake_inject.json {"rich": true, "fixture_frame": <jpg>}): realistic-SHAPED, FAKE-labelled
    state so every V2 panel can be seen filled. Values are invented test fixtures, never results."""
    import shutil
    el = now - t0
    fx = inj.get("fixture_frame")
    if m == "hub" and fx and n % 5 == 1:
        shutil.copy(fx, hub / ".live.jpg"); os.replace(hub / ".live.jpg", hub / "live.jpg")
    if m == "observe" and fx and n % 10 == 1:
        shutil.copy(fx, shm / ".fi.jpg"); os.replace(shm / ".fi.jpg", shm / "frame_inference.jpg")
        regs = [dict(box=[506, 324, 600, 441], spot=12, sim=0.0615, display="CANDIDATE", reasons=["FAKE fixture"]),
                dict(box=[410, 0, 455, 27], spot=9, sim=0.0613, display="REJECTED_GLARE", reasons=["FAKE fixture"]),
                dict(box=[0, 0, 116, 120], spot=2, sim=0.0586, display="REJECTED_GLARE", reasons=["FAKE fixture"])]
        atomic_json(shm / "status.json", dict(siglip="OK", run_id="FAKE_observe", source="FAKE", siglip_prompts=["FAKE"],
                                              scene=dict(n_scene=n, stamp=now, regions=regs, timing=dict(scene_ms=640, siglip_infer_ms=290))))
    if m == "lio2":
        k = int(el * 10)
        trk = [[0.08 * i * math.cos(i / 40), 0.08 * i * math.sin(i / 55)] for i in range(0, k + 1, 3)]
        path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trk, trk[1:]))
        atomic_json(state / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="FAKE", written_wall=now, track_raw=trk,
                                                   odom=dict(n=k, hz_10s=9.95, last_rx_age_s=0.04, path_m=round(path, 1), frame_id="FAKE",
                                                             child_frame_id="FAKE", interarrival_ms=dict(p50=100, p95=118, max=160),
                                                             last_raw=dict(stamp=now, x=trk[-1][0], y=trk[-1][1], yaw_deg=10.0)),
                                                   accumulation=dict(enabled=True, scans=k, cells=k * 3, why="FAKE")))
    if m == "e2":
        p = json.loads((state / "control" / "pause.json").read_text()) if (state / "control" / "pause.json").exists() else {}
        done = list(range(1, min(9, int(max(0, el - 6) / 3)) + 1))
        st = "PAUSED" if p.get("paused") else ("WAITING_MOTION" if el < 6 else ("RUNNING" if len(done) < 9 else "IDLE"))
        atomic_json(state / "e2" / "e2_state.json", dict(kind="LIVE2_E2_STATE", source="FAKE", status=st, building="kci", building_name="FAKE kci",
                                                         pass_id=1 if el >= 6 else 0, floors=list(range(1, 10)), floors_total=9, floors_done=done,
                                                         wait_note="WAITING_MOTION path 0.8 m < 2.0 m in the E2 window" if el < 6 else None,
                                                         per_floor={str(f): dict(status="DONE" if f in done else "PENDING") for f in range(1, 10)},
                                                         device="cpu", procs=2, updated_wall=now))
    if m == "e34" and el > 8:
        hyp = [dict(id=f"H{i:03d}_F{fl}", floor=fl, status="ACTIVE" if i < 5 else "SUSPENDED", score=sc, gap_to_best=round(sc - 0.118, 3), n_kf=12 + i,
                    pose_m=[40 + 3 * i, 11 + i], pose_px=[1300 + 90 * i, 360 + 10 * i], heading_deg=-177 + 30 * i, e2_merit=0.2)
               for i, (fl, sc) in enumerate([(6, 0.118), (5, 0.131), (3, 0.152), (6, 0.160), (2, 0.171), (8, 0.205), (1, 0.233)])]
        atomic_json(state / "e3_state.json", dict(source="FAKE", n_alive=7, n_active=5, n_suspended=2, n_dropped=0, n_keyframes=19, kf_pending=0,
                                                  hypotheses=hyp, timing=dict(kf_ms_p50=170, kf_ms_p95=460), events=[], written_wall=now))
        atomic_json(state / "e4_state.json", dict(state="AMBIGUOUS", written_wall=now, reasons=[
            "competitor H001_F5 (OTHER_FLOOR) within 0.013 < 0.05 m of the best", "2 floors [5, 6] explain the scans within 0.05 m"],
            metrics=dict(best_id="H000_F6", best_floor=6, best_score=0.118, competitor_floor=5, gap=0.013, floors_done=9, floors_total=9),
            semantic=dict(needed=True, priority="HIGH", why="FAKE: two floors explain the scans", floors_in_question=[5, 6],
                          targets=[dict(floor=6, label="616", kind="room_number_plate", dist_m=2.4, bearing_deg=-30, observability="OBSERVABLE_NOW"),
                                   dict(floor=5, label="516", kind="room_number_plate", dist_m=2.4, bearing_deg=-30, observability="OBSERVABLE_NOW")]),
            timing_ms=dict(e4=12.0)))


if __name__ == "__main__":
    main()
