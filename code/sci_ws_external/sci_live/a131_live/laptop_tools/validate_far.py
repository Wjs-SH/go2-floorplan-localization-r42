#!/usr/bin/env python3
"""라이브판 far_points() 를 전달본 원거리 점(유령점 마스크 전)과 대조(2026-10-06).
기준: /data2/a131_rebuild/a131_S6_clean_v7/far_points.npz (정제 bag S6flower_tffix_clean_v7, 스크립트 extract_far_generic_deskew.py v2).
자세는 스크립트와 같이 bag 전체 /tf(odom→base_link)를 쓴다. 스캔은 sqlite 로 읽는다(스크립트와 같은 방식).
    python3 live_a131/validate_far.py [--n 300]
"""
import argparse, csv, glob, sqlite3, struct, sys, time
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
sys.path.insert(0, str(Path(__file__).parent))
from a131_live import FarPoses, far_points  # noqa

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a131", default="/data2/a131_rebuild/a131_S6_clean_v7")
    ap.add_argument("--bag", default="/data2/a131_clean_bags/S6flower_tffix_clean_v7"); ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{glob.glob(a.bag + '/*.db3')[0]}?mode=ro&immutable=1", uri=True)
    tp = {n: (i, t) for i, n, t in con.execute("SELECT id,name,type FROM topics")}
    TF = get_message(tp['/tf'][1]); ot, op, oq = [], [], []
    for (data,) in con.execute(f"SELECT data FROM messages WHERE topic_id={tp['/tf'][0]} ORDER BY timestamp"):
        for tr in deserialize_message(bytes(data), TF).transforms:
            if tr.header.frame_id == 'odom' and tr.child_frame_id == 'base_link':
                ot.append(tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9)
                op.append([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
                q = tr.transform.rotation; oq.append([q.x, q.y, q.z, q.w])
    poses = FarPoses(ot, op, oq)
    first = bytes(con.execute(f"SELECT data FROM messages WHERE topic_id={tp['/lidar_points'][0]} LIMIT 1").fetchone()[0])
    m0 = deserialize_message(first, get_message(tp['/lidar_points'][1]))
    OFF = first.find(struct.pack('<III', 26, m0.width * 26, m0.width * 26), 0, 1024) + 12
    ends, rowids = [], []
    for rid, d in con.execute(f"SELECT rowid,data FROM messages WHERE topic_id={tp['/lidar_points'][0]}"):
        b = bytes(d); _, _, nd = struct.unpack_from('<III', b, OFF - 12)
        t = np.frombuffer(b, np.uint8, count=nd, offset=OFF).reshape(-1, 26)[:, 18:26].copy().view(np.float64)
        ends.append(float(np.nanmax(t))); rowids.append(rid)
    ends = np.array(ends); rowids = np.array(rowids); o = np.argsort(ends); ends, rowids = ends[o], rowids[o]
    link = list(csv.DictReader(open(a.a131 + "/odom_link.csv")))
    ref = np.load(a.a131 + "/far_points.npz")
    idx = np.linspace(0, len(link) - 1, min(a.n, len(link))).astype(int)
    same = diff = 0; tt = []
    for i in idx:
        st = float(link[i]["t_sec"]); sid = link[i]["sample_id"]
        k = int(np.clip(np.searchsorted(ends, st), 1, len(ends) - 1)); k = k - 1 if abs(ends[k - 1] - st) < abs(ends[k] - st) else k
        b = bytes(con.execute(f"SELECT data FROM messages WHERE rowid={int(rowids[k])}").fetchone()[0])
        _, _, nd = struct.unpack_from('<III', b, OFF - 12); buf = np.frombuffer(b, np.uint8, count=nd, offset=OFF).reshape(-1, 26)
        x = buf[:, 0:4].copy().view(np.float32).ravel(); y = buf[:, 4:8].copy().view(np.float32).ravel(); z = buf[:, 8:12].copy().view(np.float32).ravel()
        ring = buf[:, 16:18].copy().view(np.uint16).ravel(); tsp = buf[:, 18:26].copy().view(np.float64).ravel()
        t0 = time.perf_counter(); f = far_points(x, y, z, ring, tsp, poses); tt.append(time.perf_counter() - t0)
        r = ref[sid] if sid in ref.files else np.zeros((0, 2), np.float32)
        if np.array_equal(f, r): same += 1
        else: diff += 1; print("다름", sid, len(f), len(r))
    print(f"표본 {len(idx)}개: 같음 {same}, 다름 {diff} | far_points() 중앙 {1000 * np.median(tt):.1f} ms")

if __name__ == "__main__":
    main()
