#!/usr/bin/env python3
"""C++ 커널이 파이썬 판과 같은지: 테스트 자료의 스캔마다 이웃 표(전 키)와 모멘트 10개를 비트 단위로 대조."""
import os, sys
from pathlib import Path
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
os.environ["A131_CKERNEL"] = "0"                       # 파이썬 판으로 돌리면서 입력을 가로챈다
import a131_live as AL
import a131_ckernel as CK
d = np.load(sys.argv[1]); calls = []
L = AL.A131Live(static_matrix=d["static_base_lidar"], workers=0, exact_pca=False)
V = sys.modules["a131_vec"]; py_table = V.neighbor_table; py_pca = V.support_pca
def spy_table(*a, **k):
    t = py_table(*a, **k); calls.append(("table", a, t)); return t
def spy_pca(points, table, fit_mask, xp):
    calls.append(("pca", (np.array(points), table, np.array(fit_mask)), None)); return py_pca(points, table, fit_mask, xp)
V.neighbor_table = spy_table; V.support_pca = spy_pca
ps, px, pq = d["pose_stamp_ns"], d["pose_xyz"], d["pose_quat_xyzw"]; M = 2_000_000_000
def interp(lo, hi):
    m = (ps >= lo - M) & (ps <= hi + M); return AL.make_interpolator(ps[m], px[m], pq[m])
done = set(); items = []
for st in [int(v) for v in d["targets"][::10]]:
    for i in [int(v) for v in d[f"target{st}_group"]]:
        if i in done: continue
        t = d[f"scan{i}_timestamp"].astype(np.float64); xyz = np.column_stack([d[f"scan{i}_x"], d[f"scan{i}_y"], d[f"scan{i}_z"]]).astype(np.float64)
        tn = np.rint(t * 1e9).astype(np.int64); end = int(d[f"scan{i}_point_end_ns"]); v = np.all(np.isfinite(xyz), 1) & (np.linalg.norm(xyz, axis=1) > 1e-6) & np.isfinite(t)
        meta = dict(scan_index=i, point_start_ns=int(d[f"scan{i}_point_start_ns"]), point_end_ns=end, header_stamp_ns=int(d[f"scan{i}_header_stamp_ns"]))
        items.append((i, xyz, d[f"scan{i}_intensity"], d[f"scan{i}_ring"], t, interp(min(int(tn[v].min()), end), max(int(tn[v].max()), end)), meta)); done.add(i)
L.add_scans(items)
def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
nt = npc = bad = 0
for kind, a, t in calls:
    if kind == "table":
        c = CK.neighbor_table(*a); nt += 1
        for key in ("centers", "r0", "c0", "crange", "jump", "nidx", "avail", "acc"):
            if not same(t[key], c[key]): bad += 1; print("표 다름:", key, np.asarray(t[key]).dtype, np.asarray(c[key]).dtype, np.asarray(t[key]).shape, np.asarray(c[key]).shape)
        if t["offsets"] != c["offsets"]: bad += 1; print("offsets 다름")
        neg = int((np.asarray(a[0]) < 0).sum())
    else:
        points, table, fit = a; npc += 1
        cols = np.flatnonzero(fit); acc, nidx, centers = table["acc"], table["nidx"], table["centers"]; C = points[centers[cols]]; q = len(cols)
        ref = [np.zeros(q) for _ in range(10)]
        for k in range(acc.shape[0]):
            aa = acc[k, cols]
            if not aa.any(): continue
            dlt = points[nidx[k, cols]] - C
            dx = np.where(aa, dlt[:, 0], 0.0); dy = np.where(aa, dlt[:, 1], 0.0); dz = np.where(aa, dlt[:, 2], 0.0)
            ref[0] += dx; ref[1] += dy; ref[2] += dz; ref[3] += dx * dx; ref[4] += dy * dy; ref[5] += dz * dz; ref[6] += dx * dy; ref[7] += dx * dz; ref[8] += dy * dz
            np.maximum(ref[9], dx * dx + dy * dy + dz * dz, out=ref[9])
        got = CK.moments(points, table, cols)
        for i, (r, g) in enumerate(zip(ref, got)):
            if not same(r, g): bad += 1; print("모멘트 다름:", i, int((r != g).sum()), "개", "NaN", int(np.isnan(r).sum()), int(np.isnan(g).sum()))
print(f"이웃 표 {nt}번·PCA {npc}번 대조, 다른 것 {bad}개, 격자에 음수 인덱스 {neg}개(마지막 스캔)")
