#!/usr/bin/env python3
"""처음부터 끝까지 로봇과 같은 경로로 대조(2026-10-06): 정제 녹화본 라이다 + FAST-LIO 원본(/Odometry, camera_init→imu)
→ fastlio_to_base(δ) 로 base_link 자세(FAST-LIO 주기 그대로) → make_interpolator → A131Live → 전달본(a131_S6_clean_v7, 유령점 마스크 전)과 비교.
전달본은 라이다 헤더 시각에 보간한 자세(/tf)로 만들었으므로 보간 지점이 달라 경계 칸 몇 개는 다를 수 있다.
    python3 live_a131/validate_e2e.py [--start-sec 180 --duration-sec 9] [--delta 44.711] [--fast-pca]
"""
import argparse, csv, glob, sqlite3, sys, time
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/data/kci_work/HANDOFF_1F2F3F/1_localizer_package/floorplan_localizer/vendor/a1_3_1/orchestrator")
from a131_live import A131Live, make_interpolator, fastlio_to_base  # noqa
import importlib.util

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/data2/a131_clean_bags/S6flower_tffix_clean_v7")
    ap.add_argument("--odom-bag", default="/data2/rosbags/flower_usb/S6_attkleg05_full_legcloud")
    ap.add_argument("--ref-dataset", default="/data2/a131_rebuild/a131_S6_clean_v7")
    ap.add_argument("--start-sec", type=float, default=180.0); ap.add_argument("--duration-sec", type=float, default=9.0)
    ap.add_argument("--delta", type=float, default=44.711220508541906); ap.add_argument("--fast-pca", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    db = glob.glob(a.odom_bag + "/*.db3")[0]; con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    tp = {n: (i, t) for i, n, t in con.execute("SELECT id,name,type FROM topics")}; M = get_message(tp["/Odometry"][1])
    T, P, Q = [], [], []
    for (d,) in con.execute(f"SELECT data FROM messages WHERE topic_id={tp['/Odometry'][0]} ORDER BY timestamp"):
        m = deserialize_message(bytes(d), M); T.append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)
        p, q = m.pose.pose.position, m.pose.pose.orientation; P.append([p.x, p.y, p.z]); Q.append([q.x, q.y, q.z, q.w])
    T, P, Q = np.array(T, np.int64), np.array(P), np.array(Q)
    pb, qb = fastlio_to_base(P, Q, delta_deg=a.delta)          # 로봇에서 FAST-LIO 출력마다 하는 변환
    tmp = A131Live(static_matrix=np.eye(4)); a1, seg = tmp.M["a1"], tmp.M["segment"]
    bag = Path(a.bag); storage = a1.detect_storage_id(bag); types = {i.name: i.type for i in a1.make_reader(bag, storage).get_all_topics_and_types()}
    static = a1.read_static_transform(bag_dir=bag, storage_id=storage, topic_type=types["/tf_static"], parent_frame="base_link", child_frame="hesai_lidar")
    L = A131Live(static_matrix=static, exact_pca=not a.fast_pca, workers=3)
    spec = importlib.util.spec_from_file_location("orch", "/data/kci_work/HANDOFF_1F2F3F/1_localizer_package/floorplan_localizer/vendor/a1_3_1/orchestrator/01_process_chunk_periodic.py")
    orch = importlib.util.module_from_spec(spec); spec.loader.exec_module(orch)
    _, scans, cand = seg.index_scans(bag=bag, topic="/lidar_points", timestamp_field="timestamp", start_sec=a.start_sec, duration_sec=a.duration_sec,
                                     context_before_sec=0.80, context_after_sec=0.20)
    specs = orch.periodic_target_indices(scans=scans, candidate_indices=cand, global_start_sec=0.0, chunk_start_sec=a.start_sec,
                                         chunk_duration_sec=a.duration_sec, period_sec=0.15)
    specs = [s for s in specs if len(seg.latest_indices(int(s["target_scan_index"]), 3)) == 3][::a.every]
    def interp(lo, hi, margin=2_000_000_000):                  # 로봇에서는 자세 버퍼에서 이 구간을 잘라 쓴다
        m = (T >= lo - margin) & (T <= hi + margin)
        return make_interpolator(T[m], pb[m], qb[m])
    ds = {int(r["target_point_end_ns"]): r["sample_id"] for r in csv.DictReader(open(Path(a.ref_dataset) / "all_samples.csv"))}
    needed = sorted({int(v) for s in specs for v in seg.latest_indices(int(s["target_scan_index"]), 3)})
    merge_interp = interp(min(int(scans[i]["point_start_ns"]) for i in needed), max(int(scans[i]["point_end_ns"]) for i in needed))
    done = set(); same = 0; diffs = []; tt = []
    for s in specs:
        ti = int(s["target_scan_index"]); items = []
        for i in seg.latest_indices(ti, 3):
            i = int(i)
            if i in done: continue
            tgt = int(scans[i]["point_end_ns"])
            c = a1.read_candidate_clouds(bag_dir=bag, storage_id=storage, topic="/lidar_points", topic_type=types["/lidar_points"], target_stamp_ns=tgt, search_margin_sec=0.30)
            _, arr, _, _ = a1.select_scan(c, tgt, "timestamp")
            xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64); t = np.asarray(arr["timestamp"], np.float64)
            items.append((i, xyz, arr["intensity"], arr["ring"], t, interp(int(scans[i]["point_start_ns"]), tgt), scans[i])); done.add(i)
        t0 = time.perf_counter(); L.add_scans(items); out = L.extract(seg.latest_indices(ti, 3), merge_interp); tt.append(time.perf_counter() - t0)
        ref = np.load(Path(a.ref_dataset) / "samples" / ds[out["stamp_ns"]] / "data_short.npz")
        A = {tuple(v): r for v, r in zip(map(tuple, out["xy"]), out["role"])}
        B = {tuple(v): r for v, r in zip(map(tuple, ref["a1_3_1_all_candidate_xy"]), ref["a1_3_1_all_candidate_role"])}
        nd = len(set(A) ^ set(B)) + sum(1 for k in set(A) & set(B) if A[k] != B[k]); diffs.append((nd, len(B)))
        same += nd == 0
    d = np.array([x for x, _ in diffs]); n = np.array([y for _, y in diffs])
    print(f"표본 {len(diffs)}개: 전달본과 완전히 같음 {same}개 | 다른 칸 수 중앙 {np.median(d):.0f}, 최대 {d.max()} (표본당 칸 {int(np.median(n))}개, 다른 비율 최대 {100 * (d / n).max():.2f}%)"
          f" | 표본당 처리 중앙 {1000 * np.median(tt):.0f} ms")

if __name__ == "__main__":
    main()
