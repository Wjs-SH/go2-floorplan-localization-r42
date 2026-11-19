#!/usr/bin/env python3
"""replay_feed.py — LIVE2 STORED-INPUT test feeder (REPLAY, never live). Re-plays a recorded student dataset
(odom_link.csv + samples/ + far_points.npz, e.g. jsh testdata/S6) into a fresh LiveStore at --rate x real time, at the
live feed cadence (--sample-dt), and writes REPLAY-labelled lio_state.json / e2_feed_state.json so E2/E3/E4 and the web
run exactly as in the field. Input files are only read.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "external" / "sci_live" / "realtime"))
from live_store import LiveStore  # noqa: E402


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--sample-dt", type=float, default=1.5)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--hold-end-s", type=float, default=0.0, help="keep publishing 'stationary' status after the data ends")
    a = ap.parse_args()
    d = Path(a.data)
    rows = list(csv.DictReader(open(d / "odom_link.csv")))
    far = np.load(d / "far_points.npz")
    store = LiveStore(a.store, keep_s=900.0, reset=True)
    sess = dict(odom_session_id=f"REPLAY:{d.name}:{rows[0]['t_sec']}", started_wall=time.time(), why="REPLAY_STORED_INPUT",
                input=f"REPLAY of {d} (student a1_3_1 structure points, NOT the LIVE2 FAST-LIO proxy)", source="REPLAY")
    (Path(a.store) / "session.json").write_text(json.dumps(sess))
    state = Path(a.state); state.mkdir(parents=True, exist_ok=True)
    t0d = float(rows[0]["t_sec"]) + a.start_s
    t0w = time.time()
    last_t, n, track = None, 0, []
    stats = dict(samples=0, source="REPLAY", input=sess["input"])
    for r in rows:
        t = float(r["t_sec"])
        if t < t0d or (last_t is not None and t - last_t < a.sample_dt - 1e-6):
            continue
        while time.time() - t0w < (t - t0d) / a.rate:
            time.sleep(0.1)                       # odometry keeps flowing between samples (FAST-LIO publishes at 10 Hz)
            if track:
                atomic_json(state / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="REPLAY (stored input, not FAST-LIO)",
                                                           written_wall=time.time(), odom=dict(n=n, last_rx_age_s=0.0,
                                                           last_raw=dict(x=track[-1][0], y=track[-1][1])), track_raw=track[-1500:],
                                                           accumulation=dict(enabled=False, why="REPLAY")))
        with np.load(d / "samples" / r["sample_id"] / "data_short.npz") as z:
            xy, role = z["a1_3_1_all_candidate_xy"], z["a1_3_1_all_candidate_role"]
        fp = far[r["sample_id"]] if r["sample_id"] in far.files else np.zeros((0, 2), np.float32)
        sid = store.add(t, float(r["fastlio_x"]), float(r["fastlio_y"]), float(r["fastlio_yaw_rad"]), xy, role, fp)
        last_t = t; n += 1
        track.append((float(r["fastlio_x"]), float(r["fastlio_y"])))
        now = time.time()
        stats.update(samples=n, last=dict(sid=sid, stamp=t, wall=now, n_cells=int((role != 4).sum()), n_far=len(fp)),
                     session=dict(odom_session_id=sess["odom_session_id"]), span_s=round(store.span(), 1))
        atomic_json(state / "e2_feed_state.json", dict(stats, written_wall=now))
        atomic_json(state / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="REPLAY (stored input, not FAST-LIO)", written_wall=now,
                                                   odom=dict(n=n, hz_10s=None, last_rx_age_s=0.0, frame_id="REPLAY", child_frame_id="REPLAY",
                                                             last_raw=dict(stamp=t, x=track[-1][0], y=track[-1][1]),
                                                             interarrival_ms={}, header_to_rx_ms={}, rejected={}, session=0),
                                                   track_raw=track[-1500:], accumulation=dict(enabled=False, why="REPLAY")))
    t_end = time.time() + a.hold_end_s
    while time.time() < t_end:                    # data ended: odometry stays 'fresh but stationary' only if asked
        now = time.time()
        atomic_json(state / "lio_state.json", dict(kind="LIVE1_LIO_STATE", source="REPLAY_ENDED_HOLD", written_wall=now,
                                                   odom=dict(n=n, last_rx_age_s=0.0, last_raw=dict(x=track[-1][0], y=track[-1][1])),
                                                   track_raw=track[-1500:], accumulation=dict(enabled=False, why="REPLAY")))
        stats["last"]["wall"] = now
        atomic_json(state / "e2_feed_state.json", dict(stats, written_wall=now, note="REPLAY ended; holding last pose"))
        time.sleep(0.5)
    print(f"REPLAY done samples={n}", flush=True)


if __name__ == "__main__":
    main()
