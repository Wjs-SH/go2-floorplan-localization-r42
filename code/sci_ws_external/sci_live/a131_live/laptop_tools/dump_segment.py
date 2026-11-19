#!/usr/bin/env python3
"""로봇에서 ROS 없이 라이브판을 시험하도록 녹화본 일부를 배열로 저장한다(2026-10-06, 노트북에서 실행).
저장: 스캔별 x,y,z(float32), intensity, ring, timestamp + 스캔 정보, odom→base_link 자세 목록(구간 ±여유), base_link←hesai 정적 TF,
표본(target)별 3스캔 묶음과 기준 결과(all_candidate xy/role, 기존 파이프라인). 자세 보간기는 demo 에서 기존과 같은 구간으로 다시 만든다.
    python3 live_a131/dump_segment.py --max-targets 20 --out live_a131/testdata/S6_chunk0_20.npz
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/data/kci_work/HANDOFF_1F2F3F/1_localizer_package/floorplan_localizer/vendor/a1_3_1/orchestrator")
from a131_live import A131Live  # noqa
ORCH = "/data/kci_work/HANDOFF_1F2F3F/1_localizer_package/floorplan_localizer/vendor/a1_3_1/orchestrator/01_process_chunk_periodic.py"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="/data3/6flower_tffix")
    ap.add_argument("--ref", default="/data2/_a131_vec_itest/CHUNK_000_fast"); ap.add_argument("--max-targets", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); bag = Path(a.bag)
    spec = importlib.util.spec_from_file_location("orch", ORCH); orch = importlib.util.module_from_spec(spec); spec.loader.exec_module(orch)
    L = A131Live(static_matrix=np.eye(4)); a1, seg = L.M["a1"], L.M["segment"]
    storage = a1.detect_storage_id(bag); types = {i.name: i.type for i in a1.make_reader(bag, storage).get_all_topics_and_types()}
    static = a1.read_static_transform(bag_dir=bag, storage_id=storage, topic_type=types["/tf_static"], parent_frame="base_link", child_frame="hesai_lidar")
    _, scans, cand = seg.index_scans(bag=bag, topic="/lidar_points", timestamp_field="timestamp", start_sec=0.0, duration_sec=9.0,
                                     context_before_sec=0.80, context_after_sec=0.20)
    specs = orch.periodic_target_indices(scans=scans, candidate_indices=cand, global_start_sec=0.0, chunk_start_sec=0.0, chunk_duration_sec=9.0, period_sec=0.15)
    specs = [s for s in specs if len(seg.latest_indices(int(s["target_scan_index"]), 3)) == 3]
    all_needed = sorted({int(v) for s in specs for v in seg.latest_indices(int(s["target_scan_index"]), 3)})
    specs = specs[:a.max_targets]
    needed = sorted({int(v) for s in specs for v in seg.latest_indices(int(s["target_scan_index"]), 3)})
    # merge 보간기는 기존 청크(55표본 전체)와 같은 구간으로 만들어야 같은 결과가 나온다
    mn = min(int(scans[i]["point_start_ns"]) for i in all_needed); mx = max(int(scans[i]["point_end_ns"]) for i in all_needed)
    poses = a1.read_dynamic_poses(bag_dir=bag, storage_id=storage, topic_type=types["/tf"], parent_frame="odom", child_frame="base_link",
                                  start_ns=mn - 3_000_000_000, end_ns=mx + 3_000_000_000, seek_margin_ns=2_000_000_000)
    out = dict(static_base_lidar=static, merge_start_ns=np.int64(mn), merge_end_ns=np.int64(mx),
               pose_stamp_ns=np.array([p.stamp_ns for p in poses], np.int64), pose_xyz=np.array([p.translation for p in poses]),
               pose_quat_xyzw=np.array([p.quaternion_xyzw for p in poses]), scan_ids=np.array(needed, np.int64))
    for i in needed:
        tgt = int(scans[i]["point_end_ns"])
        c = a1.read_candidate_clouds(bag_dir=bag, storage_id=storage, topic="/lidar_points", topic_type=types["/lidar_points"], target_stamp_ns=tgt, search_margin_sec=0.30)
        _, arr, _, _ = a1.select_scan(c, tgt, "timestamp")
        for k in ("x", "y", "z", "intensity", "ring", "timestamp"):
            out[f"scan{i}_{k}"] = np.asarray(arr[k])
        for k in ("point_start_ns", "point_end_ns", "header_stamp_ns"):
            out[f"scan{i}_{k}"] = np.int64(scans[i][k])
    tg = []
    for s in specs:
        ti = int(s["target_scan_index"]); grp = seg.latest_indices(ti, 3); st = int(scans[ti]["point_end_ns"])
        r = np.load(Path(a.ref) / "targets" / f"TARGET_{st}" / "data_short.npz")
        out[f"target{st}_group"] = np.array(grp, np.int64)
        out[f"target{st}_ref_xy"] = r["a1_3_1_all_candidate_xy"]; out[f"target{st}_ref_role"] = r["a1_3_1_all_candidate_role"]
        tg.append(st)
    out["targets"] = np.array(tg, np.int64)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out, **out)
    print(a.out, "스캔", len(needed), "표본", len(tg), f"{Path(a.out).stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
