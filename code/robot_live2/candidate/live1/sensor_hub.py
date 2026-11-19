#!/usr/bin/env python3
"""LIVE1 sensor hub: the ONLY LIVE1 receiver of RealSense color / aligned depth / camera_info and /Odometry
(ros_readonly allowlist, BEST_EFFORT KEEP_LAST 1, no publisher). Writes the shared frame ring + index (shared_hub.HubWriter)
and the operator live JPEG. No model, no GPU, no device. SigLIP (observe) reads this ring and can stop / pause / restart
without touching the RGB / odometry path."""
import argparse
import os
import sys
import time
from pathlib import Path

CAND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAND / "src" / "sci_sensor_bridge"))
from ros_readonly import RosReadonlySource  # noqa: E402
from shared_hub import HubWriter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True)
    ap.add_argument("--live-jpeg", required=True)
    ap.add_argument("--max-s", type=float, required=True)
    a = ap.parse_args()
    src = RosReadonlySource(odom_topic="/Odometry", visual_identity_pose=True, node_name="live1_sensor_hub")
    print(f"LIVE1_HUB subscribed={src.subscribed} refused={src.refused} hub={a.hub}", flush=True)
    w = HubWriter(a.hub, src, jpeg_path=a.live_jpeg)
    t0, last_idx = time.monotonic(), 0.0
    try:
        while time.monotonic() - t0 < a.max_s:
            n = w.step()
            if time.monotonic() - last_idx >= 0.05:
                last_idx = time.monotonic()
                w.write_index()
            if not n:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    print(f"LIVE1_HUB end written={w.counts['written']}", flush=True)
    src.close()


if __name__ == "__main__":
    main()
