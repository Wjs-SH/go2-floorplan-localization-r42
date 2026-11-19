#!/usr/bin/env python3
"""fastlio_to_base() 를 전달본 녹화본 자세와 대조(2026-10-06). S6: FAST-LIO 원본(/Odometry, S6_attkleg05_full_legcloud)을
라이다 헤더 시각에 보간(build_s6_tffix_bag.py 와 같음) → fastlio_to_base(δ=녹화본 값 44.711°) → 정제 bag(S6flower_tffix_clean_v7) /tf 와 비교.
δ 를 고정값 45.3° 로 했을 때의 차이와, 거울상 교정 자세로 estimate_delta_deg 를 다시 구한 값도 낸다.
    python3 live_a131/validate_pose.py
"""
import glob, json, math, sqlite3, sys
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0, str(Path(__file__).parent))
from a131_live import fastlio_to_base, estimate_delta_deg  # noqa

def read(db, topic):
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    tp = {n: (i, t) for i, n, t in con.execute("SELECT id,name,type FROM topics")}
    M = get_message(tp[topic][1])
    return [deserialize_message(bytes(d), M) for (d,) in con.execute(f"SELECT data FROM messages WHERE topic_id={tp[topic][0]} ORDER BY timestamp")]

odm = read(glob.glob("/data2/rosbags/flower_usb/S6_attkleg05_full_legcloud/*.db3")[0], "/Odometry")
ot = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in odm])
op = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for m in odm])
oq = np.array([[m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w] for m in odm])
o = np.argsort(ot); ot, op, oq = ot[o], op[o], oq[o]; k = np.r_[True, np.diff(ot) > 0]; ot, op, oq = ot[k], op[k], oq[k]
sl = Slerp(ot, Rot.from_quat(oq))
tf = read(glob.glob("/data2/a131_clean_bags/S6flower_tffix_clean_v7/*.db3")[0], "/tf")
T, P, Q = [], [], []
for m in tf:
    for tr in m.transforms:
        if tr.header.frame_id.strip("/") == "odom" and tr.child_frame_id.strip("/") == "base_link":
            T.append(tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9)
            P.append([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
            q = tr.transform.rotation; Q.append([q.x, q.y, q.z, q.w])
T, P, Q = np.array(T), np.array(P), np.array(Q)
pi = np.c_[[np.interp(T, ot, op[:, j]) for j in range(3)]].T; qi = sl(T).as_quat()
for name, dlt in (("녹화본 δ 44.711°", 44.711220508541906), ("고정 δ 45.3°", 45.3)):
    pb, qb = fastlio_to_base(pi, qi, delta_deg=dlt)
    dp = np.linalg.norm(pb - P, axis=1); da = np.degrees((Rot.from_quat(qb).inv() * Rot.from_quat(Q)).magnitude())
    print(f"{name}: 위치 차 최대 {dp.max():.2e} m, 각도 차 최대 {da.max():.4f}° (자세 {len(T)}개)")
pb, qb = fastlio_to_base(pi, qi, delta_deg=0.0)
yaw = np.unwrap(Rot.from_quat(qb).as_euler("zyx")[:, 0])
print("거울상만 교정한 자세로 δ 다시 추정:", estimate_delta_deg(T, pb[:, 0], pb[:, 1], yaw))
