#!/usr/bin/env python3
"""LIVE1 /Odometry frame-contract evidence (G1-1). No constant (45.3 / 228 deg) is assumed anywhere.

collect (needs approval A; read-only):  subscribe /Odometry (BEST_EFFORT) + /tf_static (TRANSIENT_LOCAL, the URDF static
          transforms of robot_state_publisher) for --seconds and write every sample to --out (JSONL). No publisher.
analyze (CPU, offline): from the operator's joystick motions in the same recording
    - straight FORWARD walk(s): heading of motion vs reported odom yaw  -> mounting yaw offset delta (median, spread)
    - one in-place turn with the direction the OPERATOR declares (left = counter-clockwise seen from above)
                                                                          -> handedness (mirror needed or not)
    - lever arm base<-imu from /tf_static (if published), never typed in
  and writes a CONTRACT DRAFT whose status is DRAFT_NEEDS_HUMAN_CONFIRMATION. ros_readonly refuses a contract without
  'verified_by'; that field is filled only by `live1_ctl.sh confirm-contract` after the user confirms in the chat.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live1_common as C  # noqa: E402


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_q(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def straight_segments(samples, seg_m=0.5, max_dyaw_deg=6.0, max_s=5.0):
    """Consecutive pieces of >= seg_m travelled within max_s with nearly constant yaw (no speed assumption)."""
    out = []
    i, n = 0, len(samples)
    while i < n - 1:
        j = i + 1
        while j < n - 1 and math.hypot(samples[j]["x"] - samples[i]["x"], samples[j]["y"] - samples[i]["y"]) < seg_m \
                and samples[j]["t"] - samples[i]["t"] < max_s:
            j += 1
        a, b = samples[i], samples[j]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        disp = math.hypot(dx, dy)
        yaws = [s["yaw"] for s in samples[i:j + 1]]
        dyaw = max(abs(wrap(y - yaws[0])) for y in yaws)
        if disp >= seg_m and math.degrees(dyaw) <= max_dyaw_deg:
            mean_yaw = math.atan2(sum(math.sin(y) for y in yaws), sum(math.cos(y) for y in yaws))
            out.append(dict(t0=a["t"], t1=b["t"], disp_m=round(disp, 3), bearing_deg=math.degrees(math.atan2(dy, dx)),
                            yaw_deg=math.degrees(mean_yaw), offset_deg=math.degrees(wrap(math.atan2(dy, dx) - mean_yaw))))
            i = j
        else:
            i += 1
    return out


def turn_sign(samples, t0=None, t1=None):
    s = [x for x in samples if (t0 is None or x["t"] >= t0) and (t1 is None or x["t"] <= t1)]
    if len(s) < 2:
        return None, 0.0
    tot = sum(wrap(b["yaw"] - a["yaw"]) for a, b in zip(s, s[1:]))
    return (1 if tot > 0 else -1), math.degrees(tot)


def analyze(samples, tf_static, declared_turn, turn_window=None, forward_window=None):
    od = [s for s in samples if s.get("kind") == "odom"]
    frames = sorted({(s["frame_id"], s["child_frame_id"]) for s in od})
    fw = [s for s in od if forward_window is None or forward_window[0] <= s["t"] <= forward_window[1]]
    segs = straight_segments(fw)
    sign, tot = turn_sign(od, *(turn_window or (None, None)))
    mirror = None
    if sign is not None and declared_turn in ("left", "right") and abs(tot) >= 30.0:
        expect = 1 if declared_turn == "left" else -1          # z-up right-handed body: left turn increases yaw
        mirror = sign != expect
    offs = [s["offset_deg"] for s in segs]
    if mirror and offs:
        offs = [-o for o in offs]                              # after mirroring y: bearing and yaw both change sign
    med = None
    spread = None
    if offs:
        c = math.atan2(sum(math.sin(math.radians(o)) for o in offs), sum(math.cos(math.radians(o)) for o in offs))
        med = math.degrees(c)
        spread = max(abs(math.degrees(wrap(math.radians(o) - c))) for o in offs)
    lever = None
    for t in tf_static:
        if t.get("parent") in ("base_link", "base") and "imu" in (t.get("child") or ""):
            lever = dict(parent=t["parent"], child=t["child"], xyz=t["xyz"], note="URDF static transform, from /tf_static")
    problems = []
    if len(frames) != 1:
        problems.append(f"FRAME_IDS_NOT_UNIQUE:{frames}")
    if len(segs) < 3:
        problems.append(f"TOO_FEW_STRAIGHT_SEGMENTS:{len(segs)} (walk forward >= 2 m in a straight line, several times)")
    if mirror is None:
        problems.append("HANDEDNESS_UNKNOWN (declare --turn left|right and turn >= 30 deg in place)")
    if spread is not None and spread > 10.0:
        problems.append(f"OFFSET_SPREAD_TOO_LARGE:{spread:.1f}deg")
    if lever is None:
        problems.append("LEVER_NOT_IN_TF_STATIC (base_link->imu not found; lever must come from evidence)")
    kind = None
    bt = None
    if med is not None and mirror is not None:
        if mirror:
            kind = "fastlio_mirror_lever_yaw"
            bt = dict(kind=kind, delta_deg=round(med, 2), lever_m=None if lever is None else [lever["xyz"][0], lever["xyz"][1]])
        elif abs(med) <= 3.0:
            kind = "identity"
            bt = dict(kind=kind, measured_offset_deg=round(med, 2))
        else:
            # ros_readonly 'planar_yaw' rotates the whole frame (position AND yaw), which cannot remove a mounting offset
            # between heading of motion and yaw; a yaw-only mount offset kind is not implemented -> not usable.
            problems.append(f"NON_MIRROR_MOUNT_OFFSET_{med:.1f}deg_KIND_NOT_IMPLEMENTED")
    return dict(kind="LIVE1_ODOM_FRAME_CONTRACT_DRAFT", status="DRAFT_NEEDS_HUMAN_CONFIRMATION", created=C.now_wall(),
                boot_id=C.boot_id(), frame_id=frames[0][0] if len(frames) == 1 else None,
                child_frame_id=frames[0][1] if len(frames) == 1 else None, body_transform=bt,
                measurements=dict(n_odom=len(od), straight_segments=segs, offset_median_deg=med, offset_spread_deg=spread,
                                  turn_declared=declared_turn, turn_total_deg=tot, mirror_needed=mirror, lever=lever),
                problems=problems, usable=not problems and bt is not None and (kind != "fastlio_mirror_lever_yaw" or bt["lever_m"] is not None),
                verified_by=None, evidence=None,
                note="no constant assumed; delta from straight-walk heading vs yaw, handedness from an operator-declared turn")


def collect(out, seconds):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node = rclpy.create_node("live1_frame_probe")
    f = open(out, "a")

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        f.write(json.dumps(dict(kind="odom", t=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, rx_wall=time.time(),
                                frame_id=m.header.frame_id, child_frame_id=m.child_frame_id, x=p.x, y=p.y, z=p.z,
                                q=[q.x, q.y, q.z, q.w], yaw=yaw_q(q.x, q.y, q.z, q.w))) + "\n")

    def on_tf(m):
        for t in m.transforms:
            tr, r = t.transform.translation, t.transform.rotation
            f.write(json.dumps(dict(kind="tf_static", parent=t.header.frame_id, child=t.child_frame_id, xyz=[tr.x, tr.y, tr.z],
                                    q=[r.x, r.y, r.z, r.w])) + "\n")
    node.create_subscription(Odometry, "/Odometry", on_odom,
                             QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10))
    node.create_subscription(TFMessage, "/tf_static", on_tf,
                             QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10,
                                        durability=DurabilityPolicy.TRANSIENT_LOCAL))
    t0 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    f.close()
    node.destroy_node()
    rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--seconds", type=float, default=90.0)
    z = sp.add_parser("analyze")
    z.add_argument("--samples", required=True)
    z.add_argument("--turn", choices=["left", "right", "none"], default="none", help="direction the OPERATOR turned the robot")
    z.add_argument("--turn-window", default=None, help="t0,t1 header seconds of the turn (optional)")
    z.add_argument("--forward-window", default=None, help="t0,t1 header seconds of the straight walk (optional)")
    z.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.out, a.seconds)
        return
    rows = [json.loads(line) for line in open(a.samples) if line.strip()]
    tfs = [r for r in rows if r.get("kind") == "tf_static"]
    pw = lambda s: tuple(float(v) for v in s.split(",")) if s else None  # noqa: E731
    d = analyze(sorted([r for r in rows if r.get("kind") == "odom"], key=lambda r: r["t"]), tfs, a.turn, pw(a.turn_window), pw(a.forward_window))
    d["evidence_samples"] = str(Path(a.samples).resolve())
    C.atomic_json(a.out, d)
    print(json.dumps(dict(usable=d["usable"], problems=d["problems"], body_transform=d["body_transform"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
