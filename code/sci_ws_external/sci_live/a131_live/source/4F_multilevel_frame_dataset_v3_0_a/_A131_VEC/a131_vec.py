#!/usr/bin/env python3
"""A1 / A1.1 구조격자(ring x column) 이웃 PCA 커널 — 점별 Python for-loop 를 배치 연산으로 대체.

원본 루프(extract_a1_single_sample.compute_structured_features,
build_a1_1_from_a1.compute_refined_features)와 같은 연산 순서를 따른다.
  지지점 순서 [중심, 오프셋 순서의 수용 이웃] / np.mean / C^T C (numpy syrk 경로) / np.linalg.eigh
  / 1-D np.dot (cblas_ddot, 연속 메모리) / np.cross / np.linalg.norm(axis=-1)
CPU(numpy)에서는 원본과 비트 동일을 목표로 하며 a131_vec_check.py 로 검증한다.
A131_XP=cupy 이면 GPU(CuPy)로 계산한다(실시간 노드용, 부동소수 마지막 자리 차이 허용).
"""
from __future__ import annotations

import os

import numpy as np


def get_xp():
    if os.environ.get("A131_XP", "numpy").lower() == "cupy":
        import cupy
        return cupy
    return np


def to_numpy(xp, a):
    return a if xp is np else xp.asnumpy(a)


def dot_rows(xp, a, b):
    """행별 3-벡터 내적. 원본의 1-D np.dot / np.linalg.norm(=cblas_ddot) 과 같은 경로."""
    a = xp.ascontiguousarray(a)
    b = xp.ascontiguousarray(b)
    return xp.matmul(a[:, None, :], b[:, :, None])[:, 0, 0]


def neighbor_table(index_grid, valid_mask, range_m, ring_radius, column_radius,
                   range_jump_base_m, range_jump_ratio, xp):
    """유효 중심점마다 오프셋 순서대로 (이웃 인덱스, 기하적 가용, 거리점프 수용) 표."""
    ring_count, column_count = index_grid.shape
    point_count = int(valid_mask.shape[0])
    grid = xp.asarray(index_grid, dtype=xp.int64)
    valid = xp.asarray(valid_mask, dtype=bool)
    rng = xp.asarray(range_m, dtype=xp.float64)

    reverse_ring = xp.full(point_count, -1, dtype=xp.int64)
    reverse_col = xp.full(point_count, -1, dtype=xp.int64)
    reverse_ring[grid] = xp.broadcast_to(
        xp.arange(ring_count, dtype=xp.int64)[:, None], grid.shape)
    reverse_col[grid] = xp.broadcast_to(
        xp.arange(column_count, dtype=xp.int64)[None, :], grid.shape)

    centers = xp.flatnonzero(valid & (reverse_ring >= 0) & (reverse_col >= 0))
    r0 = reverse_ring[centers]
    c0 = reverse_col[centers]
    crange = rng[centers]
    jump = range_jump_base_m + range_jump_ratio * crange

    offsets = [
        (dr, dc)
        for dr in range(-ring_radius, ring_radius + 1)
        for dc in range(-column_radius, column_radius + 1)
        if not (dr == 0 and dc == 0)
    ]
    k_count = len(offsets)
    m_count = int(centers.shape[0])
    nidx = xp.empty((k_count, m_count), dtype=xp.int64)
    avail = xp.empty((k_count, m_count), dtype=bool)
    acc = xp.empty((k_count, m_count), dtype=bool)
    for k, (dr, dc) in enumerate(offsets):
        rr = r0 + dr
        inside = (rr >= 0) & (rr < ring_count)
        cc = (c0 + dc) % column_count
        n = grid[xp.clip(rr, 0, ring_count - 1), cc]
        a = inside & valid[n]
        nidx[k] = n
        avail[k] = a
        # 원본: abs(range[n] - center_range) > jump_limit 이면 거부
        acc[k] = a & ~(xp.abs(rng[n] - crange) > jump)
    return {
        "grid": grid, "valid": valid, "rng": rng, "centers": centers,
        "r0": r0, "c0": c0, "crange": crange, "jump": jump,
        "offsets": offsets, "nidx": nidx, "avail": avail, "acc": acc,
    }


def ring_support(table, xp):
    """len({r} | 수용 이웃의 ring 집합)."""
    offsets, acc = table["offsets"], table["acc"]
    count = xp.ones(acc.shape[1], dtype=xp.int64)
    for dr in sorted({dr for dr, _ in offsets if dr != 0}):
        ks = [k for k, (a, _) in enumerate(offsets) if a == dr]
        count += acc[ks].any(axis=0)
    return count


def support_pca(points, table, fit_mask, xp):
    """fit_mask 인 중심마다 [중심, 수용 이웃] 지지점 PCA (safe_eigen_features / fit_surface 동일)."""
    centers, nidx, acc = table["centers"], table["nidx"], table["acc"]
    m_count = int(centers.shape[0])
    count = 1 + acc.sum(axis=0)
    rank = xp.cumsum(acc, axis=0) - 1
    eig = xp.full((m_count, 3), xp.nan)
    nrm = xp.full((m_count, 3), xp.nan)
    rad = xp.full(m_count, xp.nan)
    ok = xp.zeros(m_count, dtype=bool)
    for n in to_numpy(xp, xp.unique(count[fit_mask])).tolist():
        n = int(n)
        cols = xp.flatnonzero(fit_mask & (count == n))
        m = int(cols.shape[0])
        support = xp.empty((m, n), dtype=xp.int64)
        support[:, 0] = centers[cols]
        if n > 1:
            kk, jj = xp.nonzero(acc[:, cols])
            support[jj, 1 + rank[:, cols][kk, jj]] = nidx[:, cols][kk, jj]
        S = points[support]                                         # (m, n, 3)
        centered = S - xp.mean(S, axis=1, keepdims=True)
        cov = xp.matmul(centered.transpose(0, 2, 1), centered) / max(1, n - 1)
        w, V = xp.linalg.eigh(cov)
        w = xp.maximum(w[:, ::-1], 0.0)                             # 내림차순
        normal = xp.ascontiguousarray(V[:, :, 0])                   # 최소 고유값 벡터
        norm = xp.sqrt(dot_rows(xp, normal, normal))
        good = (norm > 1e-12) & xp.all(xp.isfinite(w), axis=1)
        normal = normal / norm[:, None]
        center = xp.ascontiguousarray(S[:, 0, :])
        flip = dot_rows(xp, normal, center) > 0.0                   # 센서 원점 쪽으로
        normal = xp.where(flip[:, None], -normal, normal)
        radius = xp.max(xp.linalg.norm(S - center[:, None, :], axis=-1), axis=1)
        eig[cols] = w
        nrm[cols] = normal
        rad[cols] = radius
        ok[cols] = good
    return count, eig, nrm, rad, ok


def structured_normals(points, table, normal, ok, xp):
    """좌우/상하 즉시 이웃 cross 로 만든 보조 normal 과 |dot(n_pca, n_structured)|."""
    grid, valid, rng = table["grid"], table["valid"], table["rng"]
    ring_count, column_count = grid.shape
    r0, c0 = table["r0"], table["c0"]
    crange, jump, centers = table["crange"], table["jump"], table["centers"]
    left = grid[r0, (c0 - 1) % column_count]
    right = grid[r0, (c0 + 1) % column_count]
    lower = grid[xp.maximum(r0 - 1, 0), c0]
    upper = grid[xp.minimum(r0 + 1, ring_count - 1), c0]
    cond = ok.copy()
    for n in (left, right, lower, upper):
        cond &= valid[n]
    for n in (left, right, lower, upper):
        cond &= xp.abs(rng[n] - crange) <= jump
    horizontal = points[right] - points[left]
    vertical = points[upper] - points[lower]
    s = xp.cross(horizontal, vertical)
    snorm = xp.sqrt(dot_rows(xp, s, s))
    cond &= snorm > 1e-12
    s = s / snorm[:, None]
    flip = dot_rows(xp, s, points[centers]) > 0.0
    s = xp.where(flip[:, None], -s, s)
    agree = xp.abs(dot_rows(xp, normal, s))
    return cond, s, agree


def a1_loop_vec(points_base, valid_mask, range_m, index_grid, ring_radius, column_radius,
                min_neighbors, range_jump_base_m, range_jump_ratio, out):
    """compute_structured_features 의 점별 for-loop 대체. out 의 원본 배열을 제자리에서 채운다."""
    xp = get_xp()
    with np.errstate(all="ignore"):
        P = xp.asarray(points_base, dtype=xp.float64)
        table = neighbor_table(index_grid, valid_mask, range_m, ring_radius, column_radius,
                               range_jump_base_m, range_jump_ratio, xp)
        fit = (1 + table["acc"].sum(axis=0)) >= min_neighbors
        count, eig, nrm, rad, ok = support_pca(P, table, fit, xp)
        rings = ring_support(table, xp)
        cond, s, agree = structured_normals(P, table, nrm, ok, xp)
        l1, l2, l3 = eig[:, 0], eig[:, 1], eig[:, 2]
        denom = xp.maximum(l1, 1e-12)
        ids = to_numpy(xp, table["centers"][ok])
        out["normals"][ids] = to_numpy(xp, nrm[ok]).astype(np.float32)
        out["eigenvalues"][ids] = to_numpy(xp, eig[ok]).astype(np.float32)
        out["linearity"][ids] = to_numpy(xp, ((l1 - l2) / denom)[ok])
        out["planarity"][ids] = to_numpy(xp, ((l2 - l3) / denom)[ok])
        out["scattering"][ids] = to_numpy(xp, (l3 / denom)[ok])
        out["verticality"][ids] = to_numpy(xp, (1.0 - xp.abs(nrm[:, 2]))[ok])
        out["support_radius"][ids] = to_numpy(xp, rad[ok])
        out["neighbor_count"][ids] = to_numpy(xp, count[ok])
        out["ring_support_count"][ids] = to_numpy(xp, rings[ok])
        sid = to_numpy(xp, table["centers"][cond])
        out["structured_normals"][sid] = to_numpy(xp, s[cond]).astype(np.float32)
        out["normal_agreement"][sid] = to_numpy(xp, agree[cond])


def a11_loop_vec(points, base_valid_mask, range_m, grid, ring_radius, column_radius,
                 fit_min_support_count, range_jump_base_m, range_jump_ratio,
                 expected_support_count, out):
    """compute_refined_features 의 점별 for-loop 대체. out 의 원본 배열을 제자리에서 채운다."""
    xp = get_xp()
    with np.errstate(all="ignore"):
        P = xp.asarray(points, dtype=xp.float64)
        table = neighbor_table(grid, base_valid_mask, range_m, ring_radius, column_radius,
                               range_jump_base_m, range_jump_ratio, xp)
        ring_count = grid.shape[0]
        accepted = table["acc"].sum(axis=0)
        available = table["avail"].sum(axis=0)
        rings = ring_support(table, xp)
        ids_all = to_numpy(xp, table["centers"])
        out["support_count"][ids_all] = to_numpy(xp, accepted + 1)
        out["support_completeness"][ids_all] = to_numpy(
            xp, (accepted + 1).astype(xp.float64) / expected_support_count)
        out["ring_support_count"][ids_all] = to_numpy(xp, rings)
        out["ring_coverage"][ids_all] = to_numpy(
            xp, rings.astype(xp.float64) / min(ring_count, 2 * ring_radius + 1))
        out["range_jump_reject_fraction"][ids_all] = to_numpy(
            xp, 1.0 - accepted.astype(xp.float64) / xp.maximum(available, 1).astype(xp.float64))

        fit = (accepted + 1) >= fit_min_support_count
        _, eig, nrm, rad, ok = support_pca(P, table, fit, xp)
        cond, s, agree = structured_normals(P, table, nrm, ok, xp)
        l1, l2, l3 = eig[:, 0], eig[:, 1], eig[:, 2]
        denom = xp.maximum(l1, 1e-12)
        rms = xp.sqrt(xp.maximum(l3, 0.0))
        tangential = xp.sqrt(xp.maximum(l1 + l2, 1e-12))
        ids = to_numpy(xp, table["centers"][ok])
        out["eigenvalues"][ids] = to_numpy(xp, eig[ok]).astype(np.float32)
        out["normals"][ids] = to_numpy(xp, nrm[ok]).astype(np.float32)
        out["linearity"][ids] = to_numpy(xp, ((l1 - l2) / denom)[ok])
        out["planarity_legacy"][ids] = to_numpy(xp, ((l2 - l3) / denom)[ok])
        out["scattering"][ids] = to_numpy(xp, (l3 / denom)[ok])
        out["verticality"][ids] = to_numpy(xp, (1.0 - xp.abs(nrm[:, 2]))[ok])
        out["support_radius_m"][ids] = to_numpy(xp, rad[ok])
        out["plane_rms_m"][ids] = to_numpy(xp, rms[ok])
        out["plane_residual_ratio"][ids] = to_numpy(xp, (rms / tangential)[ok])
        sid = to_numpy(xp, table["centers"][cond])
        out["structured_normals"][sid] = to_numpy(xp, s[cond]).astype(np.float32)
        out["normal_agreement"][sid] = to_numpy(xp, agree[cond])
