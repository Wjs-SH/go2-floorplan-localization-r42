#!/usr/bin/env python3
"""ROS 없이 라이브판을 돌려 기존 파이프라인 결과와 대조하고 속도를 잰다(로봇·노트북 공용, 2026-10-06).
dump_segment.py 가 만든 npz 하나만 있으면 된다. 실제 노드가 하는 것과 같게, 스캔이 들어올 때마다 add_scan, 표본마다 extract 를 부른다.
자세 보간기는 기존과 같은 구간(A1: 스캔 시각 ±2초, merge: 청크 전체 ±2초)으로 자세 목록에서 잘라 만든다.
    python3 demo_from_dump.py testdata/S6_chunk0_20.npz [--threads 1]
"""
import argparse, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0, help="스캔을 동시에 처리할 프로세스 수(표본 간격이 길어 표본마다 새 스캔 3개일 때)")
    ap.add_argument("--fast-pca", action="store_true", help="면 방향 PCA 빠른 판(결과가 경계 칸 몇 개 달라질 수 있음)")
    ap.add_argument("--every", type=int, default=1, help="표본을 몇 개마다 하나씩(0.15초 × every 간격). 7 이면 약 1초")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    from a131_live import A131Live, make_interpolator
    d = np.load(a.dump)
    L = A131Live(static_matrix=d["static_base_lidar"], workers=a.workers, exact_pca=not a.fast_pca)
    ps, px, pq = d["pose_stamp_ns"], d["pose_xyz"], d["pose_quat_xyzw"]
    M = 2_000_000_000
    def interp(lo, hi):
        m = (ps >= lo - M) & (ps <= hi + M)
        return make_interpolator(ps[m], px[m], pq[m])
    merge_interp = interp(int(d["merge_start_ns"]), int(d["merge_end_ns"]))
    done, same, tt = set(), 0, []
    for st in d["targets"][::a.every]:
        st = int(st); grp = [int(v) for v in d[f"target{st}_group"]]
        t0 = time.perf_counter()
        items = []
        for i in grp:
            if i in done: continue
            t = d[f"scan{i}_timestamp"].astype(np.float64)
            xyz = np.column_stack([d[f"scan{i}_x"], d[f"scan{i}_y"], d[f"scan{i}_z"]]).astype(np.float64)
            tn = np.rint(t * 1e9).astype(np.int64); end = int(d[f"scan{i}_point_end_ns"])
            v = np.all(np.isfinite(xyz), 1) & (np.linalg.norm(xyz, axis=1) > 1e-6) & np.isfinite(t)
            meta = dict(scan_index=i, point_start_ns=int(d[f"scan{i}_point_start_ns"]), point_end_ns=end, header_stamp_ns=int(d[f"scan{i}_header_stamp_ns"]))
            items.append((i, xyz, d[f"scan{i}_intensity"], d[f"scan{i}_ring"], t, interp(min(int(tn[v].min()), end), max(int(tn[v].max()), end)), meta))
            done.add(i)
        L.add_scans(items)
        out = L.extract(grp, merge_interp)
        tt.append(time.perf_counter() - t0)
        ok = np.array_equal(out["xy"], d[f"target{st}_ref_xy"]) and np.array_equal(out["role"], d[f"target{st}_ref_role"])
        same += ok
        note = ""
        if not ok:   # 다르면 칸 몇 개가 다른지(좌표 집합 차이 + 같은 칸의 역할 차이)
            rx, rr = d[f"target{st}_ref_xy"], d[f"target{st}_ref_role"]
            A = {tuple(v): r for v, r in zip(map(tuple, out["xy"]), out["role"])}; B = {tuple(v): r for v, r in zip(map(tuple, rx), rr)}
            note = f" (칸 집합 차이 {len(set(A) ^ set(B))}, 같은 칸 역할 차이 {sum(1 for k in set(A) & set(B) if A[k] != B[k])})"
        print(f"표본 {st}: 칸 {len(out['role'])}개, 기준과 {'같음' if ok else '다름'}{note}, {1000 * tt[-1]:.0f} ms", flush=True)
    import numpy as np  # noqa
    print(f"\n표본 {len(tt)}개 중 기준과 같음 {same}개 | 표본당 중앙 {1000 * np.median(tt[3:] or tt):.0f} ms (처음 3개 제외)")
    for k, v in L.timing.items():
        print(f"  {k:12s} 중앙 {1000 * np.median(v):7.1f} ms")
    print(f"numpy {np.__version__}")

if __name__ == "__main__":
    main()
