#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CHANNEL_NAMES = {
    0: "surface",
    1: "edge_return",
    2: "uncertain",
    3: "low_structure_support",
    -1: "invalid",
}


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
    exact = np.tile(unique_rings, column_count)

    if np.array_equal(ring, exact):
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


def ungrid(grid_values: np.ndarray, index_grid: np.ndarray) -> np.ndarray:
    result = np.zeros(index_grid.size, dtype=np.float32)
    result[index_grid] = grid_values
    return result


def compute_normal_change(
    normal_grid: np.ndarray,
    valid_grid: np.ndarray,
) -> np.ndarray:
    dot_maps = []

    for ring_shift, column_shift in [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    ]:
        shifted_normals = np.roll(
            normal_grid,
            shift=(ring_shift, column_shift),
            axis=(0, 1),
        )
        shifted_valid = np.roll(
            valid_grid,
            shift=(ring_shift, column_shift),
            axis=(0, 1),
        )

        # ring 축은 순환하지 않는다.
        if ring_shift == -1:
            shifted_valid[-1, :] = False
        elif ring_shift == 1:
            shifted_valid[0, :] = False

        active = (
            valid_grid
            & shifted_valid
            & np.all(np.isfinite(normal_grid), axis=2)
            & np.all(np.isfinite(shifted_normals), axis=2)
        )

        dot_map = np.full(valid_grid.shape, np.nan, dtype=np.float32)
        dot_map[active] = np.abs(
            np.sum(
                normal_grid[active] * shifted_normals[active],
                axis=1,
            )
        )
        dot_maps.append(dot_map)

    stack = np.stack(dot_maps, axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_dot = np.nanmedian(stack, axis=0)

    normal_change = 1.0 - np.nan_to_num(
        median_dot,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    normal_change[~valid_grid] = 0.0
    return np.clip(normal_change, 0.0, 1.0).astype(np.float32)


def smooth_columns(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32)

    accum = np.zeros_like(values, dtype=np.float64)
    count = 0
    for shift in range(-radius, radius + 1):
        accum += np.roll(values, shift)
        count += 1
    return (accum / max(count, 1)).astype(np.float32)


def compute_ring_edge_features(
    range_m: np.ndarray,
    normals: np.ndarray,
    valid_mask: np.ndarray,
    ring: np.ndarray,
    jump_base_m: float,
    jump_ratio: float,
    consensus_column_radius: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    index_grid, grid_info = build_ring_grid(ring)
    ring_count, column_count = index_grid.shape

    range_grid = range_m[index_grid].astype(np.float64)
    valid_grid = valid_mask[index_grid].astype(bool)
    normal_grid = normals[index_grid].astype(np.float64)

    previous_range = np.roll(range_grid, 1, axis=1)
    next_range = np.roll(range_grid, -1, axis=1)
    previous_valid = np.roll(valid_grid, 1, axis=1)
    next_valid = np.roll(valid_grid, -1, axis=1)

    previous_difference = np.where(
        previous_valid,
        np.abs(range_grid - previous_range),
        0.0,
    )
    next_difference = np.where(
        next_valid,
        np.abs(range_grid - next_range),
        0.0,
    )

    horizontal_range_difference = np.maximum(
        previous_difference,
        next_difference,
    )

    jump_scale = jump_base_m + jump_ratio * range_grid
    range_jump_score_grid = 1.0 - np.exp(
        -horizontal_range_difference / np.maximum(jump_scale, 1e-6)
    )
    range_jump_score_grid[~valid_grid] = 0.0

    edge_vote_grid = (
        range_jump_score_grid >= 0.55
    ) & valid_grid
    per_column_consensus = edge_vote_grid.mean(axis=0)
    smoothed_consensus = smooth_columns(
        per_column_consensus,
        radius=consensus_column_radius,
    )
    consensus_grid = np.tile(
        smoothed_consensus.reshape(1, -1),
        (ring_count, 1),
    )

    normal_change_grid = compute_normal_change(
        normal_grid=normal_grid,
        valid_grid=valid_grid,
    )

    arrays = {
        "a1_2_horizontal_range_difference_m": ungrid(
            horizontal_range_difference.astype(np.float32),
            index_grid,
        ),
        "a1_2_range_jump_score": ungrid(
            range_jump_score_grid.astype(np.float32),
            index_grid,
        ),
        "a1_2_cross_ring_edge_consensus": ungrid(
            consensus_grid.astype(np.float32),
            index_grid,
        ),
        "a1_2_normal_change_score": ungrid(
            normal_change_grid,
            index_grid,
        ),
    }
    info = {
        **grid_info,
        "jump_base_m": jump_base_m,
        "jump_ratio": jump_ratio,
        "consensus_column_radius": consensus_column_radius,
        "edge_vote_threshold": 0.55,
    }
    return arrays, info


def compute_channels(
    data: dict[str, np.ndarray],
    edge_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    feature_range = data["feature_range_mask"].astype(bool)
    feature_valid = data["a1_1_feature_valid_mask"].astype(bool)

    verticality = np.nan_to_num(
        data["a1_1_verticality"].astype(np.float64),
        nan=0.0,
    )
    plane_fit = np.nan_to_num(
        data["a1_1_plane_fit_score"].astype(np.float64),
        nan=0.0,
    )
    agreement = np.nan_to_num(
        data["a1_1_agreement_score"].astype(np.float64),
        nan=0.5,
    )
    support = np.nan_to_num(
        data["a1_1_support_score"].astype(np.float64),
        nan=0.0,
    )
    ring_support = np.nan_to_num(
        data["a1_1_ring_score"].astype(np.float64),
        nan=0.0,
    )
    continuity = np.nan_to_num(
        data["a1_1_continuity_score"].astype(np.float64),
        nan=0.0,
    )
    isolation_penalty = np.nan_to_num(
        data["a1_1_isolation_penalty"].astype(np.float64),
        nan=1.0,
    )
    range_reject = np.nan_to_num(
        data["a1_1_range_jump_reject_fraction"].astype(np.float64),
        nan=1.0,
    )
    range_m = data["range_m"].astype(np.float64)

    range_jump = edge_features["a1_2_range_jump_score"].astype(np.float64)
    cross_ring_consensus = edge_features[
        "a1_2_cross_ring_edge_consensus"
    ].astype(np.float64)
    normal_change = edge_features[
        "a1_2_normal_change_score"
    ].astype(np.float64)

    vertical_score = np.clip(
        (verticality - 0.40) / 0.60,
        0.0,
        1.0,
    )

    edge_consensus_score = np.clip(
        cross_ring_consensus * 3.0,
        0.0,
        1.0,
    )
    edge_evidence = np.maximum.reduce(
        [
            range_jump,
            normal_change,
            edge_consensus_score,
            range_reject,
        ]
    )

    surface_core = (
        0.45 * plane_fit
        + 0.20 * agreement
        + 0.15 * support
        + 0.10 * ring_support
        + 0.10 * continuity
    )
    surface_score = (
        vertical_score
        * surface_core
        * (1.0 - 0.35 * edge_evidence)
    )

    edge_return_score = (
        vertical_score
        * (
            0.45 * range_jump
            + 0.30 * normal_change
            + 0.15 * edge_consensus_score
            + 0.10 * range_reject
        )
        * (
            0.35
            + 0.65 * np.maximum(plane_fit, 0.40)
        )
    )

    far_range_evidence = np.clip(
        (range_m - 6.0) / 10.0,
        0.0,
        1.0,
    )
    uncertainty_evidence = np.maximum.reduce(
        [
            isolation_penalty,
            1.0 - support,
            range_reject,
            normal_change,
            0.60 * far_range_evidence,
        ]
    )
    strongest_structure = np.maximum(
        surface_score,
        edge_return_score,
    )
    uncertain_score = (
        vertical_score
        * uncertainty_evidence
        * (1.0 - 0.55 * strongest_structure)
    )

    low_structure_support_score = (
        0.55 * (1.0 - vertical_score)
        + 0.15 * (1.0 - plane_fit)
        + 0.15 * isolation_penalty
        + 0.10 * (1.0 - support)
        + 0.05 * (1.0 - ring_support)
    )

    surface_score = np.clip(surface_score, 0.0, 1.0)
    edge_return_score = np.clip(edge_return_score, 0.0, 1.0)
    uncertain_score = np.clip(uncertain_score, 0.0, 1.0)
    low_structure_support_score = np.clip(
        low_structure_support_score,
        0.0,
        1.0,
    )

    for score in [
        surface_score,
        edge_return_score,
        uncertain_score,
        low_structure_support_score,
    ]:
        score[~feature_range] = 0.0

    # 진단용 dominant channel. 정답 라벨이 아니다.
    dominant_channel_id = np.full(
        len(surface_score),
        -1,
        dtype=np.int8,
    )
    eligible = feature_range & feature_valid

    # surface를 기본으로 두되, 강한 edge는 우선한다.
    dominant_channel_id[
        eligible & (surface_score >= 0.55)
    ] = 0
    dominant_channel_id[
        eligible & (edge_return_score >= 0.35)
    ] = 1

    uncertain_mask = (
        eligible
        & (dominant_channel_id < 0)
        & (vertical_score >= 0.50)
        & (uncertainty_evidence >= 0.35)
    )
    dominant_channel_id[uncertain_mask] = 2

    dominant_channel_id[
        eligible & (dominant_channel_id < 0)
    ] = 3

    # feature fit이 불가능했지만 range 안에 있는 점은 uncertain으로 남긴다.
    dominant_channel_id[
        feature_range & ~feature_valid
    ] = 2

    return {
        "a1_2_vertical_score": vertical_score.astype(np.float32),
        "a1_2_edge_evidence": edge_evidence.astype(np.float32),
        "a1_2_surface_score": surface_score.astype(np.float32),
        "a1_2_edge_return_score": edge_return_score.astype(np.float32),
        "a1_2_uncertain_score": uncertain_score.astype(np.float32),
        "a1_2_low_structure_support_score":
            low_structure_support_score.astype(np.float32),
        "a1_2_dominant_channel_id": dominant_channel_id,
        "a1_2_surface_candidate_mask": (
            eligible & (surface_score >= 0.55)
        ),
        "a1_2_edge_return_candidate_mask": (
            eligible & (edge_return_score >= 0.35)
        ),
        "a1_2_uncertain_candidate_mask": (
            feature_range
            & (
                (dominant_channel_id == 2)
                | (uncertain_score >= 0.35)
            )
        ),
        "a1_2_low_structure_support_mask": (
            feature_range & (dominant_channel_id == 3)
        ),
    }


def round_robin_select(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    angle: np.ndarray,
    range_m: np.ndarray,
    quota: int,
    angle_bin_count: int,
    range_edges: np.ndarray,
) -> np.ndarray:
    if quota <= 0 or candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    angle_normalized = (angle[candidate_indices] + math.pi) % (
        2.0 * math.pi
    )
    angle_bins = np.floor(
        angle_normalized / (2.0 * math.pi) * angle_bin_count
    ).astype(np.int32)
    angle_bins = np.clip(
        angle_bins,
        0,
        angle_bin_count - 1,
    )

    range_bins = np.digitize(
        range_m[candidate_indices],
        range_edges,
        right=False,
    ) - 1
    range_bins = np.clip(
        range_bins,
        0,
        len(range_edges) - 2,
    )

    groups: dict[tuple[int, int], list[int]] = {}
    for local_idx, point_idx in enumerate(candidate_indices):
        key = (
            int(angle_bins[local_idx]),
            int(range_bins[local_idx]),
        )
        groups.setdefault(key, []).append(int(point_idx))

    for key, values in groups.items():
        values.sort(
            key=lambda idx: (
                -float(scores[idx]),
                idx,
            )
        )

    group_keys = sorted(groups)
    selected: list[int] = []
    cursor = {key: 0 for key in group_keys}

    while len(selected) < quota:
        progress = False
        for key in group_keys:
            position = cursor[key]
            values = groups[key]
            if position >= len(values):
                continue

            selected.append(values[position])
            cursor[key] = position + 1
            progress = True

            if len(selected) >= quota:
                break

        if not progress:
            break

    return np.asarray(selected, dtype=np.int64)


def build_balanced_preview(
    data: dict[str, np.ndarray],
    total_budget: int,
    surface_fraction: float,
    edge_fraction: float,
    uncertain_fraction: float,
    angle_bin_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    points = data["base_xyz_deskewed"].astype(np.float64)
    range_m = data["range_m"].astype(np.float64)
    angle = np.arctan2(points[:, 1], points[:, 0])

    surface_mask = data[
        "a1_2_surface_candidate_mask"
    ].astype(bool)
    edge_mask = data[
        "a1_2_edge_return_candidate_mask"
    ].astype(bool)
    uncertain_mask = data[
        "a1_2_uncertain_candidate_mask"
    ].astype(bool)

    # edge 중복을 surface에서 제거한다.
    surface_mask &= ~edge_mask
    uncertain_mask &= ~(edge_mask | surface_mask)

    quotas = {
        "surface": int(round(total_budget * surface_fraction)),
        "edge_return": int(round(total_budget * edge_fraction)),
        "uncertain": int(round(total_budget * uncertain_fraction)),
    }
    quotas["surface"] += total_budget - sum(quotas.values())

    range_edges = np.asarray(
        [0.3, 3.0, 6.0, 9.0, 12.0, 20.0001],
        dtype=np.float64,
    )

    selected_surface = round_robin_select(
        candidate_indices=np.flatnonzero(surface_mask),
        scores=data["a1_2_surface_score"],
        angle=angle,
        range_m=range_m,
        quota=quotas["surface"],
        angle_bin_count=angle_bin_count,
        range_edges=range_edges,
    )
    selected_edge = round_robin_select(
        candidate_indices=np.flatnonzero(edge_mask),
        scores=data["a1_2_edge_return_score"],
        angle=angle,
        range_m=range_m,
        quota=quotas["edge_return"],
        angle_bin_count=angle_bin_count,
        range_edges=range_edges,
    )
    selected_uncertain = round_robin_select(
        candidate_indices=np.flatnonzero(uncertain_mask),
        scores=data["a1_2_uncertain_score"],
        angle=angle,
        range_m=range_m,
        quota=quotas["uncertain"],
        angle_bin_count=angle_bin_count,
        range_edges=range_edges,
    )

    selected_indices = np.concatenate(
        [
            selected_surface,
            selected_edge,
            selected_uncertain,
        ]
    )
    selected_role = np.concatenate(
        [
            np.zeros(len(selected_surface), dtype=np.int8),
            np.ones(len(selected_edge), dtype=np.int8),
            np.full(len(selected_uncertain), 2, dtype=np.int8),
        ]
    )

    order = np.argsort(selected_indices)
    selected_indices = selected_indices[order]
    selected_role = selected_role[order]

    preview_mask = np.zeros(len(points), dtype=bool)
    preview_role = np.full(len(points), -1, dtype=np.int8)
    preview_mask[selected_indices] = True
    preview_role[selected_indices] = selected_role

    info = {
        "requested_budget": total_budget,
        "actual_selected_count": int(len(selected_indices)),
        "requested_quotas": quotas,
        "actual_counts": {
            "surface": int(len(selected_surface)),
            "edge_return": int(len(selected_edge)),
            "uncertain": int(len(selected_uncertain)),
        },
        "angle_bin_count": angle_bin_count,
        "range_edges_m": range_edges.tolist(),
        "important_note": (
            "balanced preview는 구조 다양성 확인용이며 "
            "아직 V1.7 MCL 입력에 사용하지 않는다."
        ),
    }
    return {
        "a1_2_balanced_preview_mask": preview_mask,
        "a1_2_balanced_preview_role": preview_role,
        "a1_2_balanced_preview_indices": selected_indices,
    }, info


def finite_stats(
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    active = mask & np.isfinite(values)
    selected = np.asarray(values[active], dtype=np.float64)

    if selected.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "p10": math.nan,
            "p50": math.nan,
            "p90": math.nan,
        }

    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "p10": float(np.percentile(selected, 10)),
        "p50": float(np.percentile(selected, 50)),
        "p90": float(np.percentile(selected, 90)),
    }


def save_csvs(
    output_dir: Path,
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    feature_range = data["feature_range_mask"].astype(bool)
    channel_id = data["a1_2_dominant_channel_id"].astype(np.int8)
    range_m = data["range_m"].astype(np.float64)

    channel_rows = []
    for channel_value in [0, 1, 2, 3]:
        mask = feature_range & (channel_id == channel_value)
        score_key = {
            0: "a1_2_surface_score",
            1: "a1_2_edge_return_score",
            2: "a1_2_uncertain_score",
            3: "a1_2_low_structure_support_score",
        }[channel_value]
        channel_rows.append(
            {
                "channel_id": channel_value,
                "channel_name": CHANNEL_NAMES[channel_value],
                "point_count": int(mask.sum()),
                "fraction_of_feature_range": float(
                    mask.sum() / max(feature_range.sum(), 1)
                ),
                **{
                    f"score_{key}": value
                    for key, value in finite_stats(
                        data[score_key],
                        mask,
                    ).items()
                    if key != "count"
                },
            }
        )

    with (output_dir / "channel_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(channel_rows[0]),
        )
        writer.writeheader()
        writer.writerows(channel_rows)

    bands = [
        (0.3, 3.0, "0.3-3m"),
        (3.0, 6.0, "3-6m"),
        (6.0, 9.0, "6-9m"),
        (9.0, 12.0, "9-12m"),
        (12.0, 20.0001, "12-20m"),
    ]
    range_rows = []

    for low, high, label in bands:
        band = (
            feature_range
            & (range_m >= low)
            & (range_m < high)
        )
        row = {
            "range_band": label,
            "point_count": int(band.sum()),
        }
        for channel_value in [0, 1, 2, 3]:
            channel_mask = band & (channel_id == channel_value)
            row[f"{CHANNEL_NAMES[channel_value]}_count"] = int(
                channel_mask.sum()
            )
            row[f"{CHANNEL_NAMES[channel_value]}_fraction"] = float(
                channel_mask.sum() / max(band.sum(), 1)
            )
        range_rows.append(row)

    with (output_dir / "range_band_channel_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(range_rows[0]),
        )
        writer.writeheader()
        writer.writerows(range_rows)

    preview_indices = data[
        "a1_2_balanced_preview_indices"
    ].astype(np.int64)
    preview_role = data[
        "a1_2_balanced_preview_role"
    ].astype(np.int8)
    points = data["base_xyz_deskewed"].astype(np.float64)

    preview_rows = []
    for index in preview_indices:
        role = int(preview_role[index])
        preview_rows.append(
            {
                "point_index": int(index),
                "role_id": role,
                "role_name": CHANNEL_NAMES[role],
                "x_m": float(points[index, 0]),
                "y_m": float(points[index, 1]),
                "z_m": float(points[index, 2]),
                "range_m": float(range_m[index]),
                "surface_score": float(
                    data["a1_2_surface_score"][index]
                ),
                "edge_return_score": float(
                    data["a1_2_edge_return_score"][index]
                ),
                "uncertain_score": float(
                    data["a1_2_uncertain_score"][index]
                ),
            }
        )

    with (output_dir / "balanced_sample_preview.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(preview_rows[0]),
        )
        writer.writeheader()
        writer.writerows(preview_rows)

    return {
        "feature_range_count": int(feature_range.sum()),
        "channel_counts": {
            CHANNEL_NAMES[channel_value]: int(
                np.sum(
                    feature_range
                    & (channel_id == channel_value)
                )
            )
            for channel_value in [0, 1, 2, 3]
        },
    }


def save_plots(
    output_dir: Path,
    data: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    points = data["base_xyz_deskewed"].astype(np.float64)
    feature_range = data["feature_range_mask"].astype(bool)

    def scalar_plot(
        filename: str,
        key: str,
        title: str,
        label: str,
    ) -> None:
        values = data[key].astype(np.float64)
        mask = feature_range & np.isfinite(values)

        fig, axis = plt.subplots(figsize=(11, 7))
        plot = axis.scatter(
            points[mask, 0],
            points[mask, 1],
            c=values[mask],
            s=3,
            alpha=0.82,
        )
        fig.colorbar(plot, ax=axis, label=label)
        axis.scatter(
            [0.0],
            [0.0],
            marker="x",
            s=90,
            linewidths=2,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    scalar_plot(
        "01_surface_score.png",
        "a1_2_surface_score",
        "A1.2 smooth vertical-surface score",
        "surface score",
    )
    scalar_plot(
        "02_edge_return_score.png",
        "a1_2_edge_return_score",
        "A1.2 edge / wall-return score",
        "edge-return score",
    )
    scalar_plot(
        "03_uncertain_score.png",
        "a1_2_uncertain_score",
        "A1.2 uncertain vertical-structure score",
        "uncertain score",
    )
    scalar_plot(
        "04_low_structure_support_score.png",
        "a1_2_low_structure_support_score",
        "A1.2 low-structure-support score",
        "low-support score",
    )
    scalar_plot(
        "06_range_jump_score.png",
        "a1_2_range_jump_score",
        "A1.2 within-ring range-jump score",
        "range-jump score",
    )
    scalar_plot(
        "07_normal_change_score.png",
        "a1_2_normal_change_score",
        "A1.2 local normal-change score",
        "normal-change score",
    )
    scalar_plot(
        "08_cross_ring_edge_consensus.png",
        "a1_2_cross_ring_edge_consensus",
        "A1.2 cross-ring edge consensus",
        "cross-ring consensus",
    )

    dominant = data["a1_2_dominant_channel_id"].astype(np.int8)
    active = feature_range & (dominant >= 0)
    channel_colors = ListedColormap(
        [
            "#1f77b4",
            "#ff7f0e",
            "#9467bd",
            "#7f7f7f",
        ]
    )

    fig, axis = plt.subplots(figsize=(11, 7))
    plot = axis.scatter(
        points[active, 0],
        points[active, 1],
        c=dominant[active],
        cmap=channel_colors,
        vmin=-0.5,
        vmax=3.5,
        s=3,
        alpha=0.85,
    )
    colorbar = fig.colorbar(
        plot,
        ax=axis,
        ticks=[0, 1, 2, 3],
    )
    colorbar.ax.set_yticklabels(
        [
            "surface",
            "edge/return",
            "uncertain",
            "low support",
        ]
    )
    axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        "A1.2 dominant diagnostic channel\n"
        "Not semantic ground truth"
    )
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "05_dominant_channel.png", dpi=180)
    plt.close(fig)

    preview_mask = data[
        "a1_2_balanced_preview_mask"
    ].astype(bool)
    preview_role = data[
        "a1_2_balanced_preview_role"
    ].astype(np.int8)
    preview_active = preview_mask & (preview_role >= 0)

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.scatter(
        points[feature_range, 0],
        points[feature_range, 1],
        s=2,
        alpha=0.08,
        label="all feature-range points",
    )
    plot = axis.scatter(
        points[preview_active, 0],
        points[preview_active, 1],
        c=preview_role[preview_active],
        cmap=ListedColormap(
            [
                "#1f77b4",
                "#ff7f0e",
                "#9467bd",
            ]
        ),
        vmin=-0.5,
        vmax=2.5,
        s=8,
        alpha=0.95,
    )
    colorbar = fig.colorbar(
        plot,
        ax=axis,
        ticks=[0, 1, 2],
    )
    colorbar.ax.set_yticklabels(
        [
            "surface",
            "edge/return",
            "uncertain",
        ]
    )
    axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        "A1.2 feature-balanced 1,024-point preview\n"
        "Not connected to MCL"
    )
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        output_dir / "09_balanced_sample_preview.png",
        dpi=180,
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A1.1 특징을 smooth surface, edge/return, uncertain, "
            "low-structure-support 채널로 분리합니다."
        )
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jump-base-m", type=float, default=0.12)
    parser.add_argument("--jump-ratio", type=float, default=0.035)
    parser.add_argument(
        "--consensus-column-radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--preview-budget",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--preview-surface-fraction",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--preview-edge-fraction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--preview-uncertain-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--preview-angle-bin-count",
        type=int,
        default=36,
    )
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
        data = {
            key: np.asarray(source[key])
            for key in source.files
        }

    required = {
        "base_xyz_deskewed",
        "ring",
        "range_m",
        "feature_range_mask",
        "a1_1_normals",
        "a1_1_feature_valid_mask",
        "a1_1_verticality",
        "a1_1_plane_fit_score",
        "a1_1_agreement_score",
        "a1_1_support_score",
        "a1_1_ring_score",
        "a1_1_continuity_score",
        "a1_1_isolation_penalty",
        "a1_1_range_jump_reject_fraction",
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"A1.1 NPZ 필드가 없습니다: {missing}")

    edge_features, edge_info = compute_ring_edge_features(
        range_m=data["range_m"].astype(np.float64),
        normals=data["a1_1_normals"].astype(np.float64),
        valid_mask=data["a1_1_feature_valid_mask"].astype(bool),
        ring=data["ring"],
        jump_base_m=args.jump_base_m,
        jump_ratio=args.jump_ratio,
        consensus_column_radius=args.consensus_column_radius,
    )
    data.update(edge_features)

    channels = compute_channels(
        data=data,
        edge_features=edge_features,
    )
    data.update(channels)

    preview, preview_info = build_balanced_preview(
        data=data,
        total_budget=args.preview_budget,
        surface_fraction=args.preview_surface_fraction,
        edge_fraction=args.preview_edge_fraction,
        uncertain_fraction=args.preview_uncertain_fraction,
        angle_bin_count=args.preview_angle_bin_count,
    )
    data.update(preview)

    np.savez_compressed(
        output_dir / "a1_2_geometric_channels.npz",
        **data,
    )

    statistics = save_csvs(
        output_dir=output_dir,
        data=data,
    )
    save_plots(
        output_dir=output_dir,
        data=data,
    )

    summary = {
        "schema_version": "a1_2_geometric_channels_v0_1",
        "source_npz": str(input_npz),
        "edge_feature_config": edge_info,
        "preview_config": preview_info,
        "statistics": statistics,
        "channel_definition": {
            "surface": (
                "매끄럽고 연속적인 수직 평면 가능성"
            ),
            "edge_return": (
                "벽 끝, 코너, 깊이 불연속, 짧은 직각 리턴 가능성"
            ),
            "uncertain": (
                "수직 구조 가능성은 있으나 단일 scan 지원이 약한 점"
            ),
            "low_structure_support": (
                "수직 평면·경계로서의 구조적 지원이 낮은 점"
            ),
        },
        "important_notes": [
            "각 채널은 의미론적 정답 라벨이 아니다.",
            "문 또는 portal을 직접 판정하지 않는다.",
            "포인트를 영구 삭제하지 않는다.",
            "balanced preview는 V1.7 MCL에 연결하지 않는다.",
            "8-keyframe temporal support 전에는 clutter를 확정하지 않는다.",
            "기존 endpoint likelihood를 변경하지 않는다.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = "\n".join(
        [
            "===== A1.2 Geometric Channel Prototype =====",
            f"source: {input_npz}",
            (
                "feature range points: "
                f"{statistics['feature_range_count']}"
            ),
            (
                "surface: "
                f"{statistics['channel_counts']['surface']}"
            ),
            (
                "edge_return: "
                f"{statistics['channel_counts']['edge_return']}"
            ),
            (
                "uncertain: "
                f"{statistics['channel_counts']['uncertain']}"
            ),
            (
                "low_structure_support: "
                f"{statistics['channel_counts']['low_structure_support']}"
            ),
            (
                "balanced preview selected: "
                f"{preview_info['actual_selected_count']}"
            ),
            "",
            "No existing dataset/result was overwritten.",
            "No point was permanently deleted.",
            "No semantic portal label was assigned.",
            "No MCL score was changed.",
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
