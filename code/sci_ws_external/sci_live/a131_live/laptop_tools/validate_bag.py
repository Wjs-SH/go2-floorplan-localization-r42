#!/usr/bin/env python3
"""라이브판(a131_live.A131Live)을 녹화본에 돌려 기존 파이프라인 결과와 비트 대조한다(2026-10-06).
기준: /data2/_a131_vec_itest/CHUNK_000_fast (S6 6flower_tffix, 0~9초, 표본 55개, 유령점 마스크 전).
스캔·자세는 기존 파이프라인과 똑같이 읽는다(A1: 스캔 시각 ±2초 자세로 보간기, merge: 청크 전체 자세로 보간기).

    python3 live_a131/validate_bag.py [--bag /data3/6flower_tffix] [--ref /data2/_a131_vec_itest/CHUNK_000_fast] [--max-targets 0]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from a131_live import A131Live  # noqa: E402

sys.path.insert(0, "/data/kci_work/HANDOFF_1F2F3F/1_localizer_package/floorplan_localizer/vendor/a1_3_1/orchestrator")
ORCH = Path("/data/kci_work/HANDOFF_1F2F3F/1_localizer_package/floorplan_localizer/vendor/a1_3_1/orchestrator/01_process_chunk_periodic.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/data3/6flower_tffix")
    ap.add_argument("--ref", default="/data2/_a131_vec_itest/CHUNK_000_fast")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--duration-sec", type=float, default=9.0)
    ap.add_argument("--max-targets", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--ref-dataset", default="", help="완성 데이터셋 폴더(all_samples.csv + samples/<sid>/data_short.npz)를 기준으로")
    ap.add_argument("--roundtrip", action="store_true", help="단계 사이 npz 를 실제 직렬화(BytesIO)로 주고받음")
    ap.add_argument("--save-dir", default="", help="표본마다 라이브판 결과(xy, role)를 npz 로 저장")
    a = ap.parse_args()
    bag = Path(a.bag)
    spec = importlib.util.spec_from_file_location("orch_chunk", ORCH)
    orch = importlib.util.module_from_spec(spec); spec.loader.exec_module(orch)

    # 자세 보간기를 만들기 위해 정적 TF 를 먼저 읽는다
    import importlib as _il  # noqa
    tmp = A131Live(static_matrix=np.eye(4))
    a1, seg = tmp.M["a1"], tmp.M["segment"]
    storage = a1.detect_storage_id(bag)
    reader = a1.make_reader(bag, storage)
    types = {i.name: i.type for i in reader.get_all_topics_and_types()}
    static = a1.read_static_transform(bag_dir=bag, storage_id=storage, topic_type=types["/tf_static"],
                                      parent_frame="base_link", child_frame="hesai_lidar")
    L = A131Live(static_matrix=static, roundtrip=a.roundtrip)

    bag_start_ns, scans, cand = seg.index_scans(bag=bag, topic="/lidar_points", timestamp_field="timestamp",
                                                start_sec=a.start_sec, duration_sec=a.duration_sec,
                                                context_before_sec=0.80, context_after_sec=0.20)
    specs = orch.periodic_target_indices(scans=scans, candidate_indices=cand, global_start_sec=0.0,
                                         chunk_start_sec=a.start_sec, chunk_duration_sec=a.duration_sec, period_sec=0.15)
    specs = [s for s in specs if len(seg.latest_indices(int(s["target_scan_index"]), 3)) == 3]
    if a.max_targets:
        specs = specs[:a.max_targets]
    needed = sorted({int(v) for s in specs for v in seg.latest_indices(int(s["target_scan_index"]), 3)})
    mn = min(int(scans[i]["point_start_ns"]) for i in needed); mx = max(int(scans[i]["point_end_ns"]) for i in needed)
    merge_poses = a1.read_dynamic_poses(bag_dir=bag, storage_id=storage, topic_type=types["/tf"], parent_frame="odom",
                                        child_frame="base_link", start_ns=mn, end_ns=mx, seek_margin_ns=2_000_000_000)
    merge_interp = a1.PoseInterpolator(merge_poses)
    print(f"표본 {len(specs)}개, 필요한 스캔 {len(needed)}개", flush=True)

    done = set()
    res = []
    t_wall = time.time()
    for s in specs:
        ti = int(s["target_scan_index"])
        for i in seg.latest_indices(ti, 3):
            i = int(i)
            if i in done:
                continue
            tgt = int(scans[i]["point_end_ns"])
            cands = a1.read_candidate_clouds(bag_dir=bag, storage_id=storage, topic="/lidar_points",
                                             topic_type=types["/lidar_points"], target_stamp_ns=tgt, search_margin_sec=0.30)
            _, arrays, _, info = a1.select_scan(cands, tgt, "timestamp")
            xyz = np.column_stack([arrays["x"], arrays["y"], arrays["z"]]).astype(np.float64)
            t = np.asarray(arrays["timestamp"], np.float64)
            tn = np.rint(t * 1e9).astype(np.int64)
            valid = np.all(np.isfinite(xyz), 1) & (np.linalg.norm(xyz, axis=1) > 1e-6) & np.isfinite(t)
            poses = a1.read_dynamic_poses(bag_dir=bag, storage_id=storage, topic_type=types["/tf"], parent_frame="odom",
                                          child_frame="base_link", start_ns=min(int(tn[valid].min()), tgt),
                                          end_ns=max(int(tn[valid].max()), tgt))
            interp = a1.PoseInterpolator(poses, boundary_tolerance_ns=750_000_000)
            L.add_scan(i, xyz, arrays["intensity"], arrays["ring"], t, interp, scan_meta=scans[i])
            done.add(i)
        out = L.extract(seg.latest_indices(ti, 3), merge_interp)
        if a.save_dir:
            Path(a.save_dir).mkdir(parents=True, exist_ok=True)
            np.savez(Path(a.save_dir) / f"live_{out['stamp_ns']}.npz", xy=out["xy"], role=out["role"])
        refp = Path(a.ref) / "targets" / f"TARGET_{out['stamp_ns']}" / "data_short.npz"
        if a.ref_dataset:
            if not hasattr(main, "_ds"):
                import csv as _csv
                main._ds = {int(r["target_point_end_ns"]): r["sample_id"] for r in _csv.DictReader(open(Path(a.ref_dataset) / "all_samples.csv"))}
            sid = main._ds.get(int(out["stamp_ns"]))
            refp = Path(a.ref_dataset) / "samples" / str(sid) / "data_short.npz" if sid else Path("/nonexistent")
        r = dict(stamp=out["stamp_ns"], n=len(out["role"]))
        if refp.is_file():
            ref = np.load(refp)
            r.update(ref_n=len(ref["a1_3_1_all_candidate_role"]),
                     xy_same=bool(np.array_equal(ref["a1_3_1_all_candidate_xy"], out["xy"])),
                     role_same=bool(np.array_equal(ref["a1_3_1_all_candidate_role"], out["role"])))
        else:
            r["ref"] = "없음"
        res.append(r)
        print(r, flush=True)
    same = [r for r in res if r.get("xy_same") and r.get("role_same")]
    print(f"\n표본 {len(res)}개 중 기준과 비트 동일 {len(same)}개 (기준 없음 {sum(1 for r in res if r.get('ref') == '없음')}), "
          f"걸린 시간 {time.time() - t_wall:.0f}초")
    for k, v in L.timing.items():
        print(f"  {k:12s} 중앙 {1000 * np.median(v):7.1f} ms  (n={len(v)})")
    if a.out:
        json.dump(dict(results=res, timing={k: list(map(float, v)) for k, v in L.timing.items()}), open(a.out, "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
