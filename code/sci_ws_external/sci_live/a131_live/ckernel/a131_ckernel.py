#!/usr/bin/env python3
"""a131_ckernel.py — C++ 커널(liba131_kernel.so)을 ctypes 로 불러 a131_vec.neighbor_table 과 면방향 PCA 누적을 대신한다(2026-11-13).
파이썬 판과 같은 연산 순서라 빠른 면방향(support_pca_fast) 결과와 끝자리까지 같아야 한다(check_ckernel.py 로 대조).
라이브러리가 없거나 못 부르면 available() 이 False → a131_live 가 파이썬 판을 그대로 쓴다.
    만들기: bash build.sh   (g++ -O3 -fPIC -shared -ffp-contract=off)
"""
import ctypes
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = None
_I64 = ctypes.POINTER(ctypes.c_int64)
_U8 = ctypes.POINTER(ctypes.c_uint8)
_F64 = ctypes.POINTER(ctypes.c_double)


def _p(a, t):
    return a.ctypes.data_as(t)


def _load():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(_HERE / "liba131_kernel.so"))
        lib.a131_centers.restype = ctypes.c_int64
        lib.a131_centers.argtypes = [_I64, ctypes.c_int64, ctypes.c_int64, _U8, ctypes.c_int64, _I64, _I64, _I64, _I64, _I64]
        lib.a131_table.restype = None
        lib.a131_table.argtypes = [_I64, ctypes.c_int64, ctypes.c_int64, _U8, _F64, ctypes.c_int64, _I64, _I64, _I64, ctypes.c_int64,
                                   _I64, ctypes.c_int64, ctypes.c_double, ctypes.c_double, _F64, _F64, _I64, _U8, _U8]
        lib.a131_moments.restype = None
        lib.a131_moments.argtypes = [_F64, ctypes.c_int64, _I64, _I64, _U8, ctypes.c_int64, ctypes.c_int64, _I64, ctypes.c_int64] + [_F64] * 10
        _LIB = lib
    return _LIB


def available():
    if os.environ.get("A131_CKERNEL", "1") == "0":
        return False
    try:
        _load()
        return True
    except OSError:
        return False


def neighbor_table(index_grid, valid_mask, range_m, ring_radius, column_radius, range_jump_base_m, range_jump_ratio, xp):
    """a131_vec.neighbor_table 과 같은 표(같은 키·같은 값). xp 는 numpy 만."""
    lib = _load()
    ring_count, column_count = index_grid.shape
    point_count = int(valid_mask.shape[0])
    grid = np.ascontiguousarray(index_grid, dtype=np.int64)
    valid = np.ascontiguousarray(valid_mask, dtype=bool)
    rng = np.ascontiguousarray(range_m, dtype=np.float64)
    v8 = valid.view(np.uint8)
    rev_r = np.empty(point_count, np.int64); rev_c = np.empty(point_count, np.int64)
    centers = np.empty(point_count, np.int64); r0 = np.empty(point_count, np.int64); c0 = np.empty(point_count, np.int64)
    m = int(lib.a131_centers(_p(grid, _I64), ring_count, column_count, _p(v8, _U8), point_count,
                             _p(rev_r, _I64), _p(rev_c, _I64), _p(centers, _I64), _p(r0, _I64), _p(c0, _I64)))
    centers = np.ascontiguousarray(centers[:m]); r0 = np.ascontiguousarray(r0[:m]); c0 = np.ascontiguousarray(c0[:m])
    offsets = [(dr, dc) for dr in range(-ring_radius, ring_radius + 1) for dc in range(-column_radius, column_radius + 1)
               if not (dr == 0 and dc == 0)]
    k = len(offsets)
    off = np.ascontiguousarray(np.asarray(offsets, dtype=np.int64).reshape(k, 2))
    crange = np.empty(m, np.float64); jump = np.empty(m, np.float64)
    nidx = np.empty((k, m), np.int64); avail = np.empty((k, m), np.uint8); acc = np.empty((k, m), np.uint8)
    lib.a131_table(_p(grid, _I64), ring_count, column_count, _p(v8, _U8), _p(rng, _F64), point_count,
                   _p(centers, _I64), _p(r0, _I64), _p(c0, _I64), m, _p(off, _I64), k,
                   float(range_jump_base_m), float(range_jump_ratio),
                   _p(crange, _F64), _p(jump, _F64), _p(nidx, _I64), _p(avail, _U8), _p(acc, _U8))
    return {"grid": grid, "valid": valid, "rng": rng, "centers": centers, "r0": r0, "c0": c0, "crange": crange, "jump": jump,
            "offsets": offsets, "nidx": nidx, "avail": avail.view(bool), "acc": acc.view(bool)}


def moments(points, table, cols):
    """support_pca_fast 의 누적 루프: (sx, sy, sz, sxx, syy, szz, sxy, sxz, syz, r2max), 각 길이 len(cols)."""
    lib = _load()
    P = np.ascontiguousarray(points, dtype=np.float64)
    centers = np.ascontiguousarray(table["centers"], dtype=np.int64)
    nidx = np.ascontiguousarray(table["nidx"], dtype=np.int64)
    acc = np.ascontiguousarray(table["acc"]).view(np.uint8)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    k, m = nidx.shape
    q = len(cols)
    out = [np.empty(q, np.float64) for _ in range(10)]
    lib.a131_moments(_p(P, _F64), P.shape[0], _p(centers, _I64), _p(nidx, _I64), _p(acc, _U8), k, m, _p(cols, _I64), q,
                     *[_p(o, _F64) for o in out])
    return out
