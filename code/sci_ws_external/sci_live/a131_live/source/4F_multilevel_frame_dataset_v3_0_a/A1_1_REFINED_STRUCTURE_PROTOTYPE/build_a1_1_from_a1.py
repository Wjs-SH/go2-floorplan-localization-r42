#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import os as _os
import sys as _sys

# 2026-09-11: 점별 for-loop 대신 배치 커널(_A131_VEC/a131_vec.py). A131_VEC=0 이면 원본 루프.
_A131_VEC = _os.environ.get("A131_VEC", "1") == "1"
if _A131_VEC:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_A131_VEC"))
    import a131_vec as _a131_vec


def build_ring_grid(ring: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ring = np.asarray(ring, dtype=np.int64)
    unique_rings = np.unique(ring)

    expected = np.arange(int(unique_rings.min()), int(unique_rings.max()) + 1)
    if not np.array_equal(unique_rings, expected):
        raise ValueError(
            f"ring 번호가 연속적이지 않습니다: {unique_rings.tolist()}"
        )

    ring_count = len(unique_rings)
    if len(ring) % ring_count != 0:
        raise ValueError(
            f"point 수가 ring 수로 나누어지지 않습니다: "
            f"N={len(ring)}, rings={ring_count}"
        )

    column_count = len(ring) // ring_count
    exact_pattern = np.tile(unique_rings, column_count)

    if np.array_equal(ring, exact_pattern):
        grid = np.arange(len(ring), dtype=np.int64).reshape(
            column_count,
            ring_count,
        ).T
        return grid, {
            "mode": "exact_repeating_ring_sequence",
            "ring_count": ring_count,
            "column_count": column_count,
        }

    per_ring = [np.flatnonzero(ring == value) for value in unique_rings]
    lengths = [len(indices) for indices in per_ring]
    if len(set(lengths)) != 1:
        raise ValueError(f"ring별 point 수가 다릅니다: {lengths}")

    return np.stack(per_ring), {
        "mode": "per_ring_original_order_fallback",
        "ring_count": ring_count,
        "column_count": lengths[0],
    }


def fit_surface(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)

    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]

    normal = vectors[:, -1]
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        return np.full(3, np.nan), np.full(3, np.nan)

    return values, normal / normal_norm


def robust_score_exp(values: np.ndarray, sigma: float) -> np.ndarray:
    result = np.exp(-0.5 * (values / max(sigma, 1e-9)) ** 2)
    result[~np.isfinite(result)] = 0.0
    return result


def compute_refined_features(
    points: np.ndarray,
    ring: np.ndarray,
    base_valid_mask: np.ndarray,
    range_m: np.ndarray,
    ring_radius: int,
    column_radius: int,
    fit_min_support_count: int,
    preferred_support_count: int,
    range_jump_base_m: float,
    range_jump_ratio: float,
    plane_residual_ratio_sigma: float,
    large_plane_rms_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    grid, grid_info = build_ring_grid(ring)
    ring_count, column_count = grid.shape
    point_count = len(points)

    reverse_ring = np.full(point_count, -1, dtype=np.int32)
    reverse_col = np.full(point_count, -1, dtype=np.int32)
    for rr in range(ring_count):
        indices = grid[rr]
        reverse_ring[indices] = rr
        reverse_col[indices] = np.arange(column_count, dtype=np.int32)

    expected_support_count = (
        (2 * ring_radius + 1) *
        (2 * column_radius + 1)
    )
    expected_neighbor_count = expected_support_count - 1

    eigenvalues = np.full((point_count, 3), np.nan, dtype=np.float32)
    normals = np.full((point_count, 3), np.nan, dtype=np.float32)
    structured_normals = np.full((point_count, 3), np.nan, dtype=np.float32)

    plane_rms_m = np.full(point_count, np.nan, dtype=np.float32)
    plane_residual_ratio = np.full(point_count, np.nan, dtype=np.float32)
    support_radius_m = np.full(point_count, np.nan, dtype=np.float32)
    support_count = np.zeros(point_count, dtype=np.uint16)
    support_completeness = np.zeros(point_count, dtype=np.float32)
    ring_support_count = np.zeros(point_count, dtype=np.uint8)
    ring_coverage = np.zeros(point_count, dtype=np.float32)
    normal_agreement = np.full(point_count, np.nan, dtype=np.float32)
    verticality = np.full(point_count, np.nan, dtype=np.float32)
    linearity = np.full(point_count, np.nan, dtype=np.float32)
    planarity_legacy = np.full(point_count, np.nan, dtype=np.float32)
    scattering = np.full(point_count, np.nan, dtype=np.float32)
    range_jump_reject_fraction = np.ones(point_count, dtype=np.float32)

    candidate_offsets = [
        (dr, dc)
        for dr in range(-ring_radius, ring_radius + 1)
        for dc in range(-column_radius, column_radius + 1)
        if not (dr == 0 and dc == 0)
    ]

    if _A131_VEC:
        _a131_vec.a11_loop_vec(
            points, base_valid_mask, range_m, grid, ring_radius, column_radius,
            fit_min_support_count, range_jump_base_m, range_jump_ratio,
            expected_support_count,
            {
                "support_count": support_count,
                "support_completeness": support_completeness,
                "ring_support_count": ring_support_count,
                "ring_coverage": ring_coverage,
                "range_jump_reject_fraction": range_jump_reject_fraction,
                "eigenvalues": eigenvalues, "normals": normals,
                "linearity": linearity, "planarity_legacy": planarity_legacy,
                "scattering": scattering, "verticality": verticality,
                "support_radius_m": support_radius_m, "plane_rms_m": plane_rms_m,
                "plane_residual_ratio": plane_residual_ratio,
                "structured_normals": structured_normals,
                "normal_agreement": normal_agreement,
            },
        )

    for idx in (() if _A131_VEC else np.flatnonzero(base_valid_mask)):
        rr = int(reverse_ring[idx])
        cc = int(reverse_col[idx])
        if rr < 0 or cc < 0:
            continue

        center = points[idx]
        center_range = float(range_m[idx])
        jump_limit = range_jump_base_m + range_jump_ratio * center_range

        accepted: list[int] = []
        accepted_rings = {rr}
        geometrically_available = 0

        for dr, dc in candidate_offsets:
            nr = rr + dr
            if nr < 0 or nr >= ring_count:
                continue

            nc = (cc + dc) % column_count
            n_idx = int(grid[nr, nc])
            if not base_valid_mask[n_idx]:
                continue

            geometrically_available += 1

            if abs(float(range_m[n_idx]) - center_range) > jump_limit:
                continue

            accepted.append(n_idx)
            accepted_rings.add(nr)

        accepted_count = len(accepted)
        support_count[idx] = accepted_count + 1
        support_completeness[idx] = (
            (accepted_count + 1) / expected_support_count
        )
        ring_support_count[idx] = len(accepted_rings)
        ring_coverage[idx] = (
            len(accepted_rings) /
            min(ring_count, 2 * ring_radius + 1)
        )
        range_jump_reject_fraction[idx] = (
            1.0 - accepted_count / max(geometrically_available, 1)
        )

        if accepted_count + 1 < fit_min_support_count:
            continue

        support_indices = np.array([idx, *accepted], dtype=np.int64)
        support_points = points[support_indices]

        values, normal = fit_surface(support_points)
        if not np.all(np.isfinite(values)):
            continue

        if np.dot(normal, center) > 0.0:
            normal = -normal

        l1, l2, l3 = map(float, values)
        denom = max(l1, 1e-12)

        eigenvalues[idx] = values.astype(np.float32)
        normals[idx] = normal.astype(np.float32)

        linearity[idx] = (l1 - l2) / denom
        planarity_legacy[idx] = (l2 - l3) / denom
        scattering[idx] = l3 / denom
        verticality[idx] = 1.0 - abs(float(normal[2]))

        distances = np.linalg.norm(support_points - center, axis=1)
        support_radius = float(np.max(distances))
        support_radius_m[idx] = support_radius

        rms = math.sqrt(max(l3, 0.0))
        plane_rms_m[idx] = rms

        tangential_scale = math.sqrt(max(l1 + l2, 1e-12))
        plane_residual_ratio[idx] = rms / tangential_scale

        # 센서 구조를 이용한 독립 normal.
        left = int(grid[rr, (cc - 1) % column_count])
        right = int(grid[rr, (cc + 1) % column_count])
        lower = int(grid[max(0, rr - 1), cc])
        upper = int(grid[min(ring_count - 1, rr + 1), cc])

        immediate = [left, right, lower, upper]
        if all(base_valid_mask[n] for n in immediate):
            if all(
                abs(float(range_m[n]) - center_range) <= jump_limit
                for n in immediate
            ):
                horizontal = points[right] - points[left]
                vertical = points[upper] - points[lower]
                structured = np.cross(horizontal, vertical)
                structured_norm = np.linalg.norm(structured)

                if structured_norm > 1e-12:
                    structured /= structured_norm
                    if np.dot(structured, center) > 0.0:
                        structured = -structured

                    structured_normals[idx] = structured.astype(np.float32)
                    normal_agreement[idx] = abs(
                        float(np.dot(normal, structured))
                    )

    feature_valid = (
        np.isfinite(verticality) &
        np.isfinite(plane_rms_m) &
        np.isfinite(plane_residual_ratio) &
        (support_count >= fit_min_support_count)
    )

    # 구성요소는 각각 따로 저장한다.
    vertical_score = np.clip(
        (verticality - 0.40) / 0.60,
        0.0,
        1.0,
    )
    # 고정 미터 RMS는 거리가 멀수록 불리하므로 원시값으로만 보존하고,
    # confidence에는 접선 방향 scale로 정규화한 residual ratio를 사용한다.
    plane_fit_score = robust_score_exp(
        plane_residual_ratio.astype(np.float64),
        plane_residual_ratio_sigma,
    ).astype(np.float32)
    agreement_score = np.where(
        np.isfinite(normal_agreement),
        np.clip(normal_agreement, 0.0, 1.0),
        0.5,
    ).astype(np.float32)

    # 전체 45개 이웃을 모두 채우도록 강요하지 않는다.
    # preferred_support_count 이상이면 support 점수는 포화한다.
    support_score = np.clip(
        support_count.astype(np.float32) /
        max(float(preferred_support_count), 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    ring_score = np.clip(
        ring_support_count.astype(np.float32) / 3.0,
        0.0,
        1.0,
    ).astype(np.float32)
    continuity_score = np.clip(
        1.0 - range_jump_reject_fraction,
        0.0,
        1.0,
    ).astype(np.float32)

    # 낮은 support는 hard delete가 아니라 soft penalty로만 사용한다.
    isolation_penalty = np.clip(
        1.0 - support_score,
        0.0,
        1.0,
    ).astype(np.float32)

    # A1.1의 진단용 결합값. 각 항은 별도 저장되며 후속 실험에서
    # 독립적으로 바꾸거나 제거할 수 있다.
    structure_confidence = (
        0.35 * vertical_score +
        0.30 * plane_fit_score +
        0.15 * agreement_score +
        0.10 * support_score +
        0.05 * ring_score +
        0.05 * continuity_score
    )
    # isolation_penalty는 별도 저장한다. 결합 confidence에서 점을
    # 영구적으로 제거하거나 이중 감점하지 않는다.
    structure_confidence[~feature_valid] = 0.0
    structure_confidence = np.clip(
        structure_confidence,
        0.0,
        1.0,
    ).astype(np.float32)

    valid_conf = structure_confidence[feature_valid]
    percentiles = {}
    percentile_masks = {}

    for percentile in [50, 60, 70, 80]:
        threshold = (
            float(np.percentile(valid_conf, percentile))
            if valid_conf.size
            else 1.0
        )
        percentiles[f"q{percentile}"] = threshold
        percentile_masks[f"confidence_q{percentile}_mask"] = (
            feature_valid &
            (structure_confidence >= threshold)
        )

    reason_masks = {
        "unfit_support_mask": (
            base_valid_mask &
            (support_count < fit_min_support_count)
        ),
        "insufficient_support_mask": (
            base_valid_mask &
            (support_count < preferred_support_count)
        ),
        "large_plane_residual_mask": (
            feature_valid &
            (plane_rms_m > large_plane_rms_m)
        ),
        "large_normalized_residual_mask": (
            feature_valid &
            (plane_residual_ratio > 2.0 * plane_residual_ratio_sigma)
        ),
        "low_verticality_mask": (
            feature_valid &
            (verticality < 0.70)
        ),
        "normal_disagreement_mask": (
            feature_valid &
            np.isfinite(normal_agreement) &
            (normal_agreement < 0.70)
        ),
        "high_range_jump_rejection_mask": (
            base_valid_mask &
            (range_jump_reject_fraction > 0.60)
        ),
    }

    arrays = {
        "a1_1_normals": normals,
        "a1_1_structured_normals": structured_normals,
        "a1_1_eigenvalues": eigenvalues,
        "a1_1_linearity": linearity,
        "a1_1_planarity_legacy": planarity_legacy,
        "a1_1_scattering": scattering,
        "a1_1_verticality": verticality,
        "a1_1_plane_rms_m": plane_rms_m,
        "a1_1_plane_residual_ratio": plane_residual_ratio,
        "a1_1_support_radius_m": support_radius_m,
        "a1_1_support_count": support_count,
        "a1_1_support_completeness": support_completeness,
        "a1_1_ring_support_count": ring_support_count,
        "a1_1_ring_coverage": ring_coverage,
        "a1_1_normal_agreement": normal_agreement,
        "a1_1_range_jump_reject_fraction": range_jump_reject_fraction,
        "a1_1_feature_valid_mask": feature_valid,
        "a1_1_vertical_score": vertical_score.astype(np.float32),
        "a1_1_plane_fit_score": plane_fit_score,
        "a1_1_agreement_score": agreement_score,
        "a1_1_support_score": support_score,
        "a1_1_ring_score": ring_score,
        "a1_1_continuity_score": continuity_score,
        "a1_1_isolation_penalty": isolation_penalty,
        "a1_1_structure_confidence": structure_confidence,
        **percentile_masks,
        **reason_masks,
    }

    info = {
        **grid_info,
        "ring_radius": ring_radius,
        "column_radius": column_radius,
        "expected_support_count": expected_support_count,
        "fit_min_support_count": fit_min_support_count,
        "preferred_support_count": preferred_support_count,
        "range_jump_base_m": range_jump_base_m,
        "range_jump_ratio": range_jump_ratio,
        "plane_residual_ratio_sigma": plane_residual_ratio_sigma,
        "large_plane_rms_m": large_plane_rms_m,
        "percentile_thresholds": percentiles,
        "confidence_note": (
            "A1.1 structure_confidence는 단일 샘플 감사용 진단값이다. "
            "최종 wall mask 또는 MCL likelihood가 아니다."
        ),
    }
    return arrays, info


def finite_statistics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    active = mask & np.isfinite(values)
    vals = np.asarray(values[active], dtype=np.float64)

    if vals.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "p10": math.nan,
            "p50": math.nan,
            "p90": math.nan,
        }

    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
    }


def save_statistics(
    output_dir: Path,
    data: dict[str, np.ndarray],
    feature_info: dict[str, Any],
) -> dict[str, Any]:
    base_valid = data["feature_range_mask"].astype(bool)
    feature_valid = data["a1_1_feature_valid_mask"].astype(bool)
    range_m = data["range_m"].astype(np.float64)

    component_keys = [
        "a1_1_verticality",
        "a1_1_plane_rms_m",
        "a1_1_plane_residual_ratio",
        "a1_1_support_radius_m",
        "a1_1_support_completeness",
        "a1_1_ring_coverage",
        "a1_1_normal_agreement",
        "a1_1_range_jump_reject_fraction",
        "a1_1_vertical_score",
        "a1_1_plane_fit_score",
        "a1_1_agreement_score",
        "a1_1_support_score",
        "a1_1_ring_score",
        "a1_1_continuity_score",
        "a1_1_isolation_penalty",
        "a1_1_structure_confidence",
    ]

    component_rows = []
    for key in component_keys:
        stats = finite_statistics(data[key], feature_valid)
        component_rows.append({"component": key, **stats})

    with (output_dir / "component_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(component_rows[0]),
        )
        writer.writeheader()
        writer.writerows(component_rows)

    bands = [
        (0.3, 3.0, "0.3-3m"),
        (3.0, 6.0, "3-6m"),
        (6.0, 9.0, "6-9m"),
        (9.0, 12.0, "9-12m"),
        (12.0, 20.0, "12-20m"),
    ]

    band_rows = []
    for low, high, label in bands:
        band = base_valid & (range_m >= low) & (range_m < high)
        valid_band = band & feature_valid

        row = {
            "range_band": label,
            "input_points": int(band.sum()),
            "feature_valid_points": int(valid_band.sum()),
            "feature_valid_rate": (
                float(valid_band.sum() / band.sum())
                if band.sum()
                else math.nan
            ),
        }

        for key in [
            "a1_1_plane_rms_m",
            "a1_1_verticality",
            "a1_1_support_completeness",
            "a1_1_normal_agreement",
            "a1_1_structure_confidence",
        ]:
            row[f"{key}_p50"] = finite_statistics(
                data[key],
                valid_band,
            )["p50"]

        for percentile in [50, 60, 70, 80]:
            mask = data[f"confidence_q{percentile}_mask"].astype(bool)
            row[f"q{percentile}_retained_rate"] = (
                float((valid_band & mask).sum() / valid_band.sum())
                if valid_band.sum()
                else math.nan
            )

        band_rows.append(row)

    with (output_dir / "range_band_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(band_rows[0]),
        )
        writer.writeheader()
        writer.writerows(band_rows)

    deskew_valid = data["deskew_valid_mask"].astype(bool)
    no_deskew = data["base_xyz_no_deskew"].astype(np.float64)
    deskewed = data["base_xyz_deskewed"].astype(np.float64)
    displacement = np.linalg.norm(deskewed - no_deskew, axis=1)
    displacement_stats = finite_statistics(displacement, deskew_valid)

    summary = {
        "input_point_count": int(len(range_m)),
        "feature_range_point_count": int(base_valid.sum()),
        "feature_valid_count": int(feature_valid.sum()),
        "feature_valid_rate": float(
            feature_valid.sum() / max(base_valid.sum(), 1)
        ),
        "deskew_displacement_m": displacement_stats,
        "percentile_thresholds": feature_info["percentile_thresholds"],
        "mask_counts": {
            key: int(data[key].astype(bool).sum())
            for key in [
                "confidence_q50_mask",
                "confidence_q60_mask",
                "confidence_q70_mask",
                "confidence_q80_mask",
                "unfit_support_mask",
                "insufficient_support_mask",
                "large_plane_residual_mask",
                "large_normalized_residual_mask",
                "low_verticality_mask",
                "normal_disagreement_mask",
                "high_range_jump_rejection_mask",
            ]
        },
    }

    (output_dir / "a1_1_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def save_pngs(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    points = data["base_xyz_deskewed"].astype(np.float64)
    base_valid = data["feature_range_mask"].astype(bool)

    def scalar_plot(
        filename: str,
        key: str,
        title: str,
        colorbar_label: str,
    ) -> None:
        values = np.asarray(data[key], dtype=np.float64)
        mask = base_valid & np.isfinite(values)

        fig, ax = plt.subplots(figsize=(11, 7))
        plot = ax.scatter(
            points[mask, 0],
            points[mask, 1],
            c=values[mask],
            s=3,
            alpha=0.8,
        )
        fig.colorbar(plot, ax=ax, label=colorbar_label)
        ax.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    scalar_plot(
        "01_structure_confidence.png",
        "a1_1_structure_confidence",
        "A1.1 diagnostic structure confidence",
        "confidence",
    )
    scalar_plot(
        "02_plane_rms.png",
        "a1_1_plane_rms_m",
        "A1.1 local plane-fit RMS residual",
        "RMS [m]",
    )
    scalar_plot(
        "03_verticality.png",
        "a1_1_verticality",
        "A1.1 verticality",
        "verticality",
    )
    scalar_plot(
        "04_support_completeness.png",
        "a1_1_support_completeness",
        "A1.1 neighborhood support completeness",
        "support ratio",
    )
    scalar_plot(
        "05_normal_agreement.png",
        "a1_1_normal_agreement",
        "A1.1 PCA/structured normal agreement",
        "|dot|",
    )
    scalar_plot(
        "06_isolation_penalty.png",
        "a1_1_isolation_penalty",
        "A1.1 soft isolation penalty",
        "penalty",
    )

    for percentile in [50, 60, 70, 80]:
        mask = data[f"confidence_q{percentile}_mask"].astype(bool)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.scatter(
            points[base_valid & ~mask, 0],
            points[base_valid & ~mask, 1],
            s=2,
            alpha=0.12,
            label="other valid input",
        )
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=4,
            alpha=0.85,
            label=f"confidence q{percentile}",
        )
        ax.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"A1.1 diagnostic confidence q{percentile} subset"
        )
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"07_confidence_q{percentile}.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ring-radius", type=int, default=2)
    parser.add_argument("--column-radius", type=int, default=4)
    parser.add_argument("--fit-min-support-count", type=int, default=4)
    parser.add_argument("--preferred-support-count", type=int, default=10)
    parser.add_argument("--range-jump-base-m", type=float, default=0.20)
    parser.add_argument("--range-jump-ratio", type=float, default=0.04)
    parser.add_argument(
        "--plane-residual-ratio-sigma",
        type=float,
        default=0.08,
    )
    parser.add_argument("--large-plane-rms-m", type=float, default=0.06)
    args = parser.parse_args()

    input_npz = args.input_npz.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)
    if output_dir.exists():
        raise FileExistsError(
            f"출력 폴더가 이미 존재합니다. 덮어쓰지 않습니다: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    with np.load(input_npz, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}

    required = {
        "base_xyz_deskewed",
        "base_xyz_no_deskew",
        "ring",
        "range_m",
        "feature_range_mask",
        "deskew_valid_mask",
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"A1 NPZ 필드가 없습니다: {missing}")

    refined, feature_info = compute_refined_features(
        points=data["base_xyz_deskewed"].astype(np.float64),
        ring=data["ring"],
        base_valid_mask=data["feature_range_mask"].astype(bool),
        range_m=data["range_m"].astype(np.float64),
        ring_radius=args.ring_radius,
        column_radius=args.column_radius,
        fit_min_support_count=args.fit_min_support_count,
        preferred_support_count=args.preferred_support_count,
        range_jump_base_m=args.range_jump_base_m,
        range_jump_ratio=args.range_jump_ratio,
        plane_residual_ratio_sigma=args.plane_residual_ratio_sigma,
        large_plane_rms_m=args.large_plane_rms_m,
    )
    data.update(refined)

    np.savez_compressed(
        output_dir / "a1_1_refined_sample.npz",
        **data,
    )

    summary = save_statistics(output_dir, data, feature_info)
    save_pngs(output_dir, data)

    metadata = {
        "schema_version": "a1_1_refined_structure_v0_1",
        "source_npz": str(input_npz),
        "feature_config": {
            "ring_radius": args.ring_radius,
            "column_radius": args.column_radius,
            "fit_min_support_count": args.fit_min_support_count,
            "preferred_support_count": args.preferred_support_count,
            "range_jump_base_m": args.range_jump_base_m,
            "range_jump_ratio": args.range_jump_ratio,
            "plane_residual_ratio_sigma": args.plane_residual_ratio_sigma,
            "large_plane_rms_m": args.large_plane_rms_m,
        },
        "feature_info": feature_info,
        "statistics": summary,
        "notes": [
            "A1 원본 배열과 특징은 모두 보존했다.",
            "isolated는 삭제하지 않고 soft penalty로만 사용했다.",
            "legacy planarity는 감사용으로 보존했다.",
            "plane RMS residual을 별도 평면 적합 특징으로 추가했다.",
            "q50/q60/q70/q80은 진단용 percentile subset이다.",
            "MCL score와 wall mask는 변경하지 않았다.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "===== A1.1 Refined Structure Prototype =====",
            f"source: {input_npz}",
            f"feature range points: {summary['feature_range_point_count']}",
            f"feature valid: {summary['feature_valid_count']}",
            (
                "deskew displacement p50: "
                f"{summary['deskew_displacement_m']['p50']:.6f} m"
            ),
            (
                "deskew displacement p90: "
                f"{summary['deskew_displacement_m']['p90']:.6f} m"
            ),
            "",
            "No existing dataset/result was overwritten.",
            "No point was permanently deleted.",
            "No MCL score was changed.",
            "No final wall mask was fixed.",
            f"output: {output_dir}",
        ]
    )
    (output_dir / "run_report.txt").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
