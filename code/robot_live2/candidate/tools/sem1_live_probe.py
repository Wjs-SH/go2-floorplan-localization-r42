#!/usr/bin/env python3
"""SEM1 live probe (approved read-only shadow, visual-only when /Odometry is absent).
  --mode receiver : RosReadonlySource only (decode in the consumer), arrival/gap/out-of-order/reset/drop statistics.
  --mode tracking : + owner-thread LKTracker (scale 0.5) on every polled frame, seeded with a 3x3 grid of 64 px boxes
                    (geometry-only workload proxy; no semantics). Re-seeds lost boxes every 2 s.
No publisher, no device, no actuator. Output: JSONL samples + final summary.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "sci_sensor_bridge"))
sys.path.insert(0, str(HERE.parent / "src" / "sci_visual_observation"))
from ros_readonly import RosReadonlySource  # noqa: E402


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(len(v) * p))], 2) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["receiver", "tracking"], required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.seconds <= 60.0, "first mode run limited to 60 s"
    out = open(a.out, "w")
    src = RosReadonlySource(odom_topic="/Odometry", visual_identity_pose=False)
    out.write(json.dumps(dict(kind="START", mode=a.mode, health=src.health(), t=time.time())) + "\n")
    trk = None
    if a.mode == "tracking":
        from e5rt.tracker import LKTracker
        trk = LKTracker(track_scale=0.5, max_tracks=12)
    t0, last_s, last_seed = time.monotonic(), time.monotonic(), -1e9
    dec_ms, trk_ms, age_ms, n = [], [], [], 0
    import cv2
    while time.monotonic() - t0 < a.seconds:
        t1 = time.perf_counter()
        frames = src.poll()
        if frames:
            dec_ms.append((time.perf_counter() - t1) * 1e3 / len(frames))
        for st, seq, img in frames:
            n += 1
            rx = src.rx_mono_ns.get(seq)
            if rx:
                age_ms.append((time.monotonic_ns() - rx) / 1e6)
            if trk is not None:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                t2 = time.perf_counter()
                if time.monotonic() - last_seed > 2.0:
                    h, w = g.shape
                    k = 0
                    for yy in (0.25, 0.5, 0.75):
                        for xx in (0.25, 0.5, 0.75):
                            if trk.tracks.get(k, {}).get("status") != "LIVE":
                                trk.seed(k, [w * xx - 32, h * yy - 32, w * xx + 32, h * yy + 32], g, seq, st)
                            k += 1
                    last_seed = time.monotonic()
                trk.update(g, seq, st)
                trk_ms.append((time.perf_counter() - t2) * 1e3)
        if time.monotonic() - last_s >= 5.0:
            last_s = time.monotonic()
            out.write(json.dumps(dict(kind="SAMPLE", t=time.time(), elapsed=round(last_s - t0, 1), health=src.health(), frames=n,
                                      live_tracks=None if trk is None else sum(1 for t in trk.tracks.values() if t["status"] == "LIVE"))) + "\n")
            out.flush()
        time.sleep(0.005)
    h = src.health()
    summ = dict(kind="SUMMARY", mode=a.mode, seconds=round(time.monotonic() - t0, 1), frames_consumed=n, health=h,
                decode_ms=dict(n=len(dec_ms), p50=pct(dec_ms, .5), p95=pct(dec_ms, .95), max=pct(dec_ms, 1.0)),
                consume_age_ms=dict(n=len(age_ms), p50=pct(age_ms, .5), p95=pct(age_ms, .95), max=pct(age_ms, 1.0),
                                    note="rx (callback) -> consumer; NOT exposure latency"),
                track_ms=None if trk is None else dict(n=len(trk_ms), p50=pct(trk_ms, .5), p95=pct(trk_ms, .95), max=pct(trk_ms, 1.0),
                                                        lost=trk.stats["lost"]))
    out.write(json.dumps(summ) + "\n")
    out.close()
    src.close()
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
