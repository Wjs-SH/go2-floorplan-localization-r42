#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROLE_NAMES = {
    0: "smooth_surface",
    1: "depth_jump_edge",
    2: "normal_turn_edge",
    3: "low_support_vertical",
    4: "far_uncertain",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_edges(text: str) -> np.ndarray:
    values = np.asarray(
        [float(part.strip()) for part in text.split(",") if part.strip()],
        dtype=np.float64,
    )
    if values.ndim != 1 or values.size < 3:
        raise ValueError("fine height edges는 최소 3개의 오름차순 값이어야 합니다.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("fine height edges는 엄격한 오름차순이어야 합니다.")
    return values


def grid_coordinates(
    xy: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = grid_shape
    x_min, x_max = x_range
    y_min, y_max = y_range

    finite_x = np.isfinite(xy[:, 0])
    finite_y = np.isfinite(xy[:, 1])
    safe_x = np.where(finite_x, xy[:, 0], x_min - resolution)
    safe_y = np.where(finite_y, xy[:, 1], y_min - resolution)
    column = np.floor((safe_x - x_min) / resolution).astype(np.int32)
    row_bottom = np.floor((safe_y - y_min) / resolution).astype(np.int32)
    row = (height - 1) - row_bottom

    in_bounds = (
        finite_x
        & finite_y
        & (xy[:, 0] >= x_min)
        & (xy[:, 0] < x_max)
        & (xy[:, 1] >= y_min)
        & (xy[:, 1] < y_max)
        & (column >= 0)
        & (column < width)
        & (row >= 0)
        & (row < height)
    )
    return row, column, in_bounds


def flat_count(cell_ids: np.ndarray, cell_count: int) -> np.ndarray:
    return np.bincount(cell_ids, minlength=cell_count)


def aggregate_max(
    cell_ids: np.ndarray,
    values: np.ndarray,
    cell_count: int,
    empty_value: float = 0.0,
) -> np.ndarray:
    result = np.full(cell_count, -np.inf, dtype=np.float64)
    np.maximum.at(result, cell_ids, values.astype(np.float64))
    result[~np.isfinite(result)] = empty_value
    return result


def aggregate_min(
    cell_ids: np.ndarray,
    values: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    result = np.full(cell_count, np.inf, dtype=np.float64)
    np.minimum.at(result, cell_ids, values.astype(np.float64))
    result[~np.isfinite(result)] = np.nan
    return result


def aggregate_mean(
    cell_ids: np.ndarray,
    values: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    sums = np.bincount(
        cell_ids,
        weights=np.nan_to_num(values.astype(np.float64), nan=0.0),
        minlength=cell_count,
    )
    valid = np.isfinite(values)
    counts = np.bincount(
        cell_ids[valid],
        minlength=cell_count,
    )
    result = np.full(cell_count, np.nan, dtype=np.float64)
    active = counts > 0
    result[active] = sums[active] / counts[active]
    return result


def aggregate_median(
    cell_ids: np.ndarray,
    values: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    result = np.full(cell_count, np.nan, dtype=np.float32)
    order = np.argsort(cell_ids, kind="stable")
    sorted_cells = cell_ids[order]
    sorted_values = values[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        cell = int(sorted_cells[start])
        group = sorted_values[start:end]
        group = group[np.isfinite(group)]
        if group.size:
            result[cell] = np.float32(np.median(group))
    return result


def unique_ring_count(
    cell_ids: np.ndarray,
    ring: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    ring_i = ring.astype(np.int64)
    ring_base = max(int(np.max(ring_i)) + 1, 1)
    pairs = cell_ids.astype(np.int64) * ring_base + ring_i
    unique_pairs = np.unique(pairs)
    unique_cells = unique_pairs // ring_base
    return np.bincount(unique_cells, minlength=cell_count)


def longest_consecutive_true(stack: np.ndarray) -> np.ndarray:
    current = np.zeros(stack.shape[1:], dtype=np.uint8)
    longest = np.zeros(stack.shape[1:], dtype=np.uint8)
    for layer in stack:
        current = np.where(layer, current + 1, 0).astype(np.uint8)
        longest = np.maximum(longest, current)
    return longest


def cell_centers(
    rows: np.ndarray,
    cols: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    height, _ = grid_shape
    x = x_range[0] + (cols.astype(np.float64) + 0.5) * resolution
    row_bottom = (height - 1) - rows.astype(np.float64)
    y = y_range[0] + (row_bottom + 0.5) * resolution
    return np.column_stack((x, y)).astype(np.float32)


def normalize_axis_angle(angle: np.ndarray) -> np.ndarray:
    # Wall normal/tangent is axial: theta and theta+pi are equivalent.
    return ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0


def aggregate_axial_normals(
    cell_ids: np.ndarray,
    normal_angle: np.ndarray,
    weights: np.ndarray,
    cell_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(normal_angle)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    cells = cell_ids[valid]
    theta = normal_angle[valid].astype(np.float64)
    weight = weights[valid].astype(np.float64)

    cos2_sum = np.bincount(
        cells,
        weights=weight * np.cos(2.0 * theta),
        minlength=cell_count,
    )
    sin2_sum = np.bincount(
        cells,
        weights=weight * np.sin(2.0 * theta),
        minlength=cell_count,
    )
    weight_sum = np.bincount(cells, weights=weight, minlength=cell_count)

    angle = np.full(cell_count, np.nan, dtype=np.float32)
    concentration = np.zeros(cell_count, dtype=np.float32)
    active = weight_sum > 1e-9
    angle[active] = (
        0.5 * np.arctan2(sin2_sum[active], cos2_sum[active])
    ).astype(np.float32)
    concentration[active] = (
        np.hypot(cos2_sum[active], sin2_sum[active])
        / weight_sum[active]
    ).astype(np.float32)

    unit = np.full((cell_count, 2), np.nan, dtype=np.float32)
    unit[active, 0] = np.cos(angle[active])
    unit[active, 1] = np.sin(angle[active])
    return angle, concentration, weight_sum.astype(np.float32), unit


def representative_indices(
    cell_ids: np.ndarray,
    point_indices: np.ndarray,
    score: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    result = np.full(cell_count, -1, dtype=np.int32)
    order = np.lexsort((point_indices, -score, cell_ids))
    sorted_cells = cell_ids[order]
    sorted_points = point_indices[order]
    first = np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
    result[sorted_cells[first]] = sorted_points[first].astype(np.int32)
    return result


def build_csr(
    cell_ids: np.ndarray,
    point_indices: np.ndarray,
    cell_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(cell_ids, kind="stable")
    sorted_cells = cell_ids[order]
    sorted_points = point_indices[order].astype(np.int32)
    counts = np.bincount(sorted_cells, minlength=cell_count)
    offsets = np.zeros(cell_count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
    return offsets, sorted_points


def round_robin_select_cells(
    candidate_flat_ids: np.ndarray,
    scores_flat: np.ndarray,
    cell_xy_flat: np.ndarray,
    quota: int,
    angle_bin_count: int,
    range_edges: np.ndarray,
) -> np.ndarray:
    if quota <= 0 or candidate_flat_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    xy = cell_xy_flat[candidate_flat_ids]
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    distance = np.linalg.norm(xy, axis=1)
    normalized = (angle + math.pi) % (2.0 * math.pi)
    angle_id = np.floor(
        normalized / (2.0 * math.pi) * angle_bin_count
    ).astype(np.int32)
    angle_id = np.clip(angle_id, 0, angle_bin_count - 1)
    range_id = np.digitize(distance, range_edges, right=False) - 1
    range_id = np.clip(range_id, 0, len(range_edges) - 2)

    groups: dict[tuple[int, int], list[int]] = {}
    for local_index, flat_id in enumerate(candidate_flat_ids):
        key = (int(angle_id[local_index]), int(range_id[local_index]))
        groups.setdefault(key, []).append(int(flat_id))

    for values in groups.values():
        values.sort(key=lambda flat_id: (-float(scores_flat[flat_id]), flat_id))

    selected: list[int] = []
    cursors = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < quota:
        progress = False
        for key in keys:
            cursor = cursors[key]
            values = groups[key]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            cursors[key] = cursor + 1
            progress = True
            if len(selected) >= quota:
                break
        if not progress:
            break
    return np.asarray(selected, dtype=np.int64)


def save_grid_plot(
    path: Path,
    grid: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: str,
    colorbar_label: str,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 9))
    image = axis.imshow(
        grid,
        origin="upper",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_height_layers_plot(
    path: Path,
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, grid, title in zip(
        axes,
        [low, mid, high],
        ["LOW count", "MID count", "HIGH count"],
    ):
        image = axis.imshow(
            grid,
            origin="upper",
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axis, shrink=0.8)
        axis.scatter([0.0], [0.0], marker="x", s=65, linewidths=2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(title)
    figure.suptitle("A1.3 height-layer count grids")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_normal_field_plot(
    path: Path,
    evidence: np.ndarray,
    angle: np.ndarray,
    concentration: np.ndarray,
    occupied: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    stride: int,
) -> None:
    import matplotlib.pyplot as plt

    height, width = evidence.shape
    rows, cols = np.indices((height, width))
    select = (
        occupied
        & np.isfinite(angle)
        & (concentration >= 0.25)
        & ((rows % stride) == 0)
        & ((cols % stride) == 0)
    )
    selected_rows = rows[select]
    selected_cols = cols[select]
    xy = cell_centers(
        selected_rows,
        selected_cols,
        x_range,
        y_range,
        resolution,
        (height, width),
    )
    theta = angle[select]
    u = np.cos(theta)
    v = np.sin(theta)

    figure, axis = plt.subplots(figsize=(9, 9))
    image = axis.imshow(
        evidence,
        origin="upper",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(image, ax=axis, label="structure evidence")
    if xy.size:
        axis.quiver(
            xy[:, 0],
            xy[:, 1],
            u,
            v,
            concentration[select],
            angles="xy",
            scale_units="xy",
            scale=2.5,
            width=0.003,
        )
    axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("A1.3 axial wall-normal field")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_preview_plot(
    path: Path,
    all_candidate_xy: np.ndarray,
    selected_xy: np.ndarray,
    selected_role: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure, axis = plt.subplots(figsize=(11, 8))
    if all_candidate_xy.size:
        axis.scatter(
            all_candidate_xy[:, 0],
            all_candidate_xy[:, 1],
            s=3,
            alpha=0.10,
            label="all structure candidate cells",
        )
    if selected_xy.size:
        plot = axis.scatter(
            selected_xy[:, 0],
            selected_xy[:, 1],
            c=selected_role,
            cmap=ListedColormap(
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
            ),
            vmin=-0.5,
            vmax=4.5,
            s=14,
            alpha=0.95,
        )
        colorbar = figure.colorbar(plot, ax=axis, ticks=[0, 1, 2, 3, 4])
        colorbar.ax.set_yticklabels([ROLE_NAMES[index] for index in range(5)])
    axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("A1.3 cell-balanced localization preview (up to budget)")
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_v2_overlay(
    path: Path,
    v2_wall_core: np.ndarray,
    a_candidate: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    category = np.zeros(v2_wall_core.shape, dtype=np.uint8)
    category[v2_wall_core] = 1
    category[a_candidate] = 2
    category[v2_wall_core & a_candidate] = 3

    figure, axis = plt.subplots(figsize=(9, 9))
    image = axis.imshow(
        category,
        origin="upper",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        interpolation="nearest",
        cmap=ListedColormap(["#202020", "#1f77b4", "#d62728", "#9467bd"]),
        vmin=0,
        vmax=3,
    )
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(
        ["empty", "V2.2 only", "A1.3 only", "overlap"]
    )
    axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("A1.3 candidate vs V2.2 wall_core — diagnostic, not GT")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def ensure_required(data: dict[str, np.ndarray], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(data))
    if missing:
        raise KeyError(f"A1.2.1 NPZ 필드가 없습니다: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A1.2.1 3D 기하 특징을 V2.2 호환 로컬 XY grid와 "
            "다중 채널 2D observation으로 평탄화합니다."
        )
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--input-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v2-2-data-short", type=Path)
    parser.add_argument("--v2-2-sample-json", type=Path)
    parser.add_argument("--radial-max-m", type=float, default=9.0)
    parser.add_argument(
        "--fine-height-edges",
        default="0.15,0.40,0.65,0.90,1.15,1.40,1.70",
    )
    parser.add_argument("--localization-budget", type=int, default=1024)
    parser.add_argument("--angle-bin-count", type=int, default=36)
    parser.add_argument("--normal-field-stride", type=int, default=4)
    args = parser.parse_args()

    input_npz = args.input_npz.expanduser().resolve()
    input_summary_path = args.input_summary_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)
    if not input_summary_path.is_file():
        raise FileNotFoundError(input_summary_path)
    if output_dir.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output_dir}")
    output_dir.mkdir(parents=True)

    with np.load(input_npz, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    input_summary = load_json(input_summary_path)

    required = {
        "base_xyz_deskewed",
        "ring",
        "range_m",
        "a1_2_1_raw_point_index",
        "a1_2_1_z_base_m",
        "a1_2_1_height_band_id",
        "a1_2_1_normal_azimuth_rad",
        "a1_2_1_projection_ready_mask",
        "a1_2_1_smooth_surface_score",
        "a1_2_1_depth_jump_edge_score",
        "a1_2_1_normal_turn_edge_score",
        "a1_2_1_low_support_vertical_score",
        "a1_2_1_far_uncertain_score",
        "a1_2_1_low_structure_score",
        "a1_2_1_support_density_normalized",
        "a1_2_1_reason_flags",
    }
    ensure_required(data, required)

    geometry = input_summary["grid_geometry_for_next_stage"]
    x_range = tuple(map(float, geometry["x_range_m"]))
    y_range = tuple(map(float, geometry["y_range_m"]))
    resolution = float(geometry["resolution_m"])
    grid_shape = tuple(map(int, geometry["grid_shape"]))
    height, width = grid_shape
    cell_count = height * width

    expected_width = (x_range[1] - x_range[0]) / resolution
    expected_height = (y_range[1] - y_range[0]) / resolution
    if abs(expected_width - width) > 1e-6 or abs(expected_height - height) > 1e-6:
        raise ValueError(
            "sample geometry의 range/resolution/grid_shape가 일치하지 않습니다: "
            f"expected {(expected_height, expected_width)}, got {grid_shape}"
        )

    fine_edges = parse_edges(args.fine_height_edges)
    points = data["base_xyz_deskewed"].astype(np.float64)
    xy = points[:, :2]
    z = data["a1_2_1_z_base_m"].astype(np.float64)
    projection_ready = data["a1_2_1_projection_ready_mask"].astype(bool)

    row_all, col_all, in_square = grid_coordinates(
        xy,
        x_range,
        y_range,
        resolution,
        grid_shape,
    )
    radial_distance = np.linalg.norm(xy, axis=1)
    radial_mask = radial_distance <= float(args.radial_max_m)
    projection_mask = (
        projection_ready
        & in_square
        & radial_mask
        & np.isfinite(z)
        & (z >= fine_edges[0])
        & (z <= fine_edges[-1])
    )

    source_point_indices = np.flatnonzero(projection_mask).astype(np.int32)
    rows = row_all[projection_mask]
    cols = col_all[projection_mask]
    flat_ids = rows.astype(np.int64) * width + cols.astype(np.int64)

    point_count_flat = flat_count(flat_ids, cell_count)
    occupied_flat = point_count_flat > 0

    # V2.2-compatible height count grids.
    band_id = data["a1_2_1_height_band_id"][projection_mask].astype(np.int8)
    grid_low_flat = flat_count(flat_ids[band_id == 0], cell_count)
    grid_mid_flat = flat_count(flat_ids[band_id == 1], cell_count)
    grid_high_flat = flat_count(flat_ids[band_id == 2], cell_count)

    # Fine vertical histogram.
    fine_bin_id = np.digitize(z[projection_mask], fine_edges, right=False) - 1
    fine_bin_id = np.clip(fine_bin_id, 0, len(fine_edges) - 2)
    fine_count_flat = np.zeros((len(fine_edges) - 1, cell_count), dtype=np.uint16)
    for bin_index in range(len(fine_edges) - 1):
        selected = fine_bin_id == bin_index
        fine_count_flat[bin_index] = np.clip(
            flat_count(flat_ids[selected], cell_count),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

    fine_occupied_flat = fine_count_flat > 0
    fine_occupied_count_flat = np.sum(fine_occupied_flat, axis=0).astype(np.uint8)
    fine_longest_flat = longest_consecutive_true(
        fine_occupied_flat.reshape(len(fine_edges) - 1, height, width)
    ).reshape(-1)

    vertical_persistence_flat = (
        (grid_low_flat > 0).astype(np.uint8)
        + (grid_mid_flat > 0).astype(np.uint8)
        + (grid_high_flat > 0).astype(np.uint8)
    )

    z_selected = z[projection_mask]
    range_selected = data["range_m"][projection_mask].astype(np.float64)
    z_min_flat = aggregate_min(flat_ids, z_selected, cell_count)
    z_max_negative = aggregate_min(flat_ids, -z_selected, cell_count)
    z_max_flat = -z_max_negative
    z_span_flat = z_max_flat - z_min_flat
    range_min_flat = aggregate_min(flat_ids, range_selected, cell_count)
    range_median_flat = aggregate_median(flat_ids, range_selected, cell_count)

    ring_count_flat = unique_ring_count(
        flat_ids,
        data["ring"][projection_mask],
        cell_count,
    )

    score_keys = {
        "smooth_surface": "a1_2_1_smooth_surface_score",
        "depth_jump_edge": "a1_2_1_depth_jump_edge_score",
        "normal_turn_edge": "a1_2_1_normal_turn_edge_score",
        "low_support_vertical": "a1_2_1_low_support_vertical_score",
        "far_uncertain": "a1_2_1_far_uncertain_score",
        "low_structure": "a1_2_1_low_structure_score",
        "support_density": "a1_2_1_support_density_normalized",
    }
    score_max_flat: dict[str, np.ndarray] = {}
    score_mean_flat: dict[str, np.ndarray] = {}
    for name, key in score_keys.items():
        values = data[key][projection_mask].astype(np.float64)
        score_max_flat[name] = aggregate_max(flat_ids, values, cell_count)
        score_mean_flat[name] = aggregate_mean(flat_ids, values, cell_count)

    smooth_max = score_max_flat["smooth_surface"]
    depth_max = score_max_flat["depth_jump_edge"]
    normal_turn_max = score_max_flat["normal_turn_edge"]
    low_support_max = score_max_flat["low_support_vertical"]
    far_max = score_max_flat["far_uncertain"]
    low_structure_mean = np.nan_to_num(score_mean_flat["low_structure"], nan=0.0)
    edge_max = np.maximum(depth_max, normal_turn_max)

    normal_angle_point = data["a1_2_1_normal_azimuth_rad"][projection_mask].astype(np.float64)
    normal_weight_point = np.maximum.reduce(
        [
            data["a1_2_1_smooth_surface_score"][projection_mask].astype(np.float64),
            0.65 * data["a1_2_1_depth_jump_edge_score"][projection_mask].astype(np.float64),
            0.65 * data["a1_2_1_normal_turn_edge_score"][projection_mask].astype(np.float64),
            0.40 * data["a1_2_1_low_support_vertical_score"][projection_mask].astype(np.float64),
        ]
    )
    normal_weight_point = np.where(normal_weight_point > 0.05, normal_weight_point, 0.0)
    normal_angle_flat, normal_concentration_flat, normal_weight_flat, normal_unit_flat = (
        aggregate_axial_normals(
            flat_ids,
            normal_angle_point,
            normal_weight_point,
            cell_count,
        )
    )
    tangent_angle_flat = normalize_axis_angle(
        normal_angle_flat.astype(np.float64) + math.pi / 2.0
    ).astype(np.float32)

    fine_ratio_flat = fine_occupied_count_flat.astype(np.float64) / max(
        len(fine_edges) - 1,
        1,
    )
    surface_evidence_flat = np.clip(
        smooth_max
        * (0.65 + 0.35 * np.sqrt(fine_ratio_flat))
        * (0.65 + 0.35 * normal_concentration_flat),
        0.0,
        1.0,
    )
    edge_evidence_flat = np.clip(edge_max, 0.0, 1.0)
    low_support_evidence_flat = np.clip(
        low_support_max * (0.55 + 0.45 * fine_ratio_flat),
        0.0,
        1.0,
    )
    structure_evidence_flat = np.maximum.reduce(
        [
            surface_evidence_flat,
            edge_evidence_flat,
            0.75 * low_support_evidence_flat,
        ]
    )
    uncertainty_flat = np.clip(
        np.maximum(far_max, 0.50 * low_structure_mean),
        0.0,
        1.0,
    )

    surface_candidate_flat = occupied_flat & (smooth_max >= 0.55)
    depth_candidate_flat = occupied_flat & (depth_max >= 0.35)
    normal_turn_candidate_flat = occupied_flat & (normal_turn_max >= 0.30)
    low_support_candidate_flat = occupied_flat & (low_support_max >= 0.25)
    far_candidate_flat = occupied_flat & (far_max >= 0.20)
    structure_candidate_flat = (
        surface_candidate_flat
        | depth_candidate_flat
        | normal_turn_candidate_flat
        | low_support_candidate_flat
    )
    single_scan_height_core_flat = vertical_persistence_flat >= 2

    representative_score_point = np.maximum.reduce(
        [
            data["a1_2_1_smooth_surface_score"][projection_mask].astype(np.float64),
            data["a1_2_1_depth_jump_edge_score"][projection_mask].astype(np.float64),
            data["a1_2_1_normal_turn_edge_score"][projection_mask].astype(np.float64),
            0.80 * data["a1_2_1_low_support_vertical_score"][projection_mask].astype(np.float64),
            0.30 * data["a1_2_1_far_uncertain_score"][projection_mask].astype(np.float64),
        ]
    )
    representative_flat = representative_indices(
        flat_ids,
        source_point_indices,
        representative_score_point,
        cell_count,
    )
    csr_offsets, csr_point_indices = build_csr(
        flat_ids,
        source_point_indices,
        cell_count,
    )

    all_rows, all_cols = np.indices(grid_shape)
    all_xy_flat = cell_centers(
        all_rows.reshape(-1),
        all_cols.reshape(-1),
        x_range,
        y_range,
        resolution,
        grid_shape,
    )

    role_configs = [
        (
            0,
            "smooth_surface",
            surface_candidate_flat,
            surface_evidence_flat,
            0.50,
        ),
        (1, "depth_jump_edge", depth_candidate_flat, depth_max, 0.15),
        (2, "normal_turn_edge", normal_turn_candidate_flat, normal_turn_max, 0.15),
        (
            3,
            "low_support_vertical",
            low_support_candidate_flat,
            low_support_evidence_flat,
            0.15,
        ),
        (4, "far_uncertain", far_candidate_flat, far_max, 0.05),
    ]
    range_edges = np.asarray([0.0, 3.0, 6.0, 9.0, 12.0, 20.0], dtype=np.float64)
    remaining = np.ones(cell_count, dtype=bool)
    selected_parts: list[np.ndarray] = []
    selected_role_parts: list[np.ndarray] = []
    selected_role_counts: dict[str, int] = {}
    requested_role_counts: dict[str, int] = {}
    used_total = 0
    for config_index, (role_id, name, candidate, score, fraction) in enumerate(role_configs):
        quota = int(round(args.localization_budget * fraction))
        if config_index == len(role_configs) - 1:
            quota = args.localization_budget - used_total
        requested_role_counts[name] = quota
        candidate_ids = np.flatnonzero(candidate & remaining)
        selected = round_robin_select_cells(
            candidate_ids,
            score,
            all_xy_flat,
            quota,
            args.angle_bin_count,
            range_edges,
        )
        remaining[selected] = False
        selected_parts.append(selected)
        selected_role_parts.append(np.full(len(selected), role_id, dtype=np.int8))
        selected_role_counts[name] = int(len(selected))
        used_total += len(selected)

    localization_flat_ids = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    localization_roles = (
        np.concatenate(selected_role_parts)
        if selected_role_parts
        else np.empty(0, dtype=np.int8)
    )
    localization_rows = (localization_flat_ids // width).astype(np.int32)
    localization_cols = (localization_flat_ids % width).astype(np.int32)
    localization_xy = all_xy_flat[localization_flat_ids]

    full_grids: dict[str, np.ndarray] = {
        "grid_point_count": np.clip(
            point_count_flat.reshape(grid_shape),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16),
        "grid_low": np.clip(grid_low_flat.reshape(grid_shape), 0, 65535).astype(np.uint16),
        "grid_mid": np.clip(grid_mid_flat.reshape(grid_shape), 0, 65535).astype(np.uint16),
        "grid_high": np.clip(grid_high_flat.reshape(grid_shape), 0, 65535).astype(np.uint16),
        "fine_height_count": fine_count_flat.reshape(
            len(fine_edges) - 1,
            height,
            width,
        ),
        "fine_height_occupied": fine_occupied_flat.reshape(
            len(fine_edges) - 1,
            height,
            width,
        ),
        "vertical_persistence": vertical_persistence_flat.reshape(grid_shape).astype(np.uint8),
        "fine_height_occupied_count": fine_occupied_count_flat.reshape(grid_shape),
        "fine_height_longest_consecutive": fine_longest_flat.reshape(grid_shape),
        "height_min_m": z_min_flat.reshape(grid_shape).astype(np.float32),
        "height_max_m": z_max_flat.reshape(grid_shape).astype(np.float32),
        "height_span_m": z_span_flat.reshape(grid_shape).astype(np.float32),
        "ring_count": np.clip(ring_count_flat.reshape(grid_shape), 0, 255).astype(np.uint8),
        "range_min_m": range_min_flat.reshape(grid_shape).astype(np.float32),
        "range_median_m": range_median_flat.reshape(grid_shape).astype(np.float32),
        "normal_axial_angle_rad": normal_angle_flat.reshape(grid_shape),
        "wall_tangent_axial_angle_rad": tangent_angle_flat.reshape(grid_shape),
        "normal_concentration": normal_concentration_flat.reshape(grid_shape),
        "normal_weight_sum": normal_weight_flat.reshape(grid_shape),
        "normal_xy_unit": normal_unit_flat.reshape(height, width, 2),
        "smooth_surface_max": smooth_max.reshape(grid_shape).astype(np.float32),
        "smooth_surface_mean": score_mean_flat["smooth_surface"].reshape(grid_shape).astype(np.float32),
        "depth_jump_edge_max": depth_max.reshape(grid_shape).astype(np.float32),
        "depth_jump_edge_mean": score_mean_flat["depth_jump_edge"].reshape(grid_shape).astype(np.float32),
        "normal_turn_edge_max": normal_turn_max.reshape(grid_shape).astype(np.float32),
        "normal_turn_edge_mean": score_mean_flat["normal_turn_edge"].reshape(grid_shape).astype(np.float32),
        "low_support_vertical_max": low_support_max.reshape(grid_shape).astype(np.float32),
        "low_support_vertical_mean": score_mean_flat["low_support_vertical"].reshape(grid_shape).astype(np.float32),
        "far_uncertain_max": far_max.reshape(grid_shape).astype(np.float32),
        "far_uncertain_mean": score_mean_flat["far_uncertain"].reshape(grid_shape).astype(np.float32),
        "low_structure_max": score_max_flat["low_structure"].reshape(grid_shape).astype(np.float32),
        "low_structure_mean": score_mean_flat["low_structure"].reshape(grid_shape).astype(np.float32),
        "support_density_max": score_max_flat["support_density"].reshape(grid_shape).astype(np.float32),
        "support_density_mean": score_mean_flat["support_density"].reshape(grid_shape).astype(np.float32),
        "surface_evidence": surface_evidence_flat.reshape(grid_shape).astype(np.float32),
        "edge_evidence": edge_evidence_flat.reshape(grid_shape).astype(np.float32),
        "low_support_evidence": low_support_evidence_flat.reshape(grid_shape).astype(np.float32),
        "structure_evidence": structure_evidence_flat.reshape(grid_shape).astype(np.float32),
        "uncertainty": uncertainty_flat.reshape(grid_shape).astype(np.float32),
        "surface_candidate": surface_candidate_flat.reshape(grid_shape),
        "depth_jump_candidate": depth_candidate_flat.reshape(grid_shape),
        "normal_turn_candidate": normal_turn_candidate_flat.reshape(grid_shape),
        "low_support_candidate": low_support_candidate_flat.reshape(grid_shape),
        "far_uncertain_candidate": far_candidate_flat.reshape(grid_shape),
        "structure_candidate": structure_candidate_flat.reshape(grid_shape),
        "single_scan_height_core": single_scan_height_core_flat.reshape(grid_shape),
        "representative_source_point_index": representative_flat.reshape(grid_shape),
    }

    derived_full = {
        **full_grids,
        "grid_x_range_m": np.asarray(x_range, dtype=np.float32),
        "grid_y_range_m": np.asarray(y_range, dtype=np.float32),
        "grid_resolution_m": np.asarray(resolution, dtype=np.float32),
        "grid_shape": np.asarray(grid_shape, dtype=np.int32),
        "radial_max_m": np.asarray(args.radial_max_m, dtype=np.float32),
        "fine_height_edges_m": fine_edges.astype(np.float32),
        "projection_source_point_indices": source_point_indices,
        "projection_grid_row": rows.astype(np.int32),
        "projection_grid_col": cols.astype(np.int32),
        "projection_flat_cell_id": flat_ids.astype(np.int32),
        "cell_point_offsets": csr_offsets,
        "cell_point_source_indices": csr_point_indices,
        "localization_flat_cell_id": localization_flat_ids.astype(np.int32),
        "localization_grid_row": localization_rows,
        "localization_grid_col": localization_cols,
        "localization_xy": localization_xy.astype(np.float32),
        "localization_role": localization_roles,
        "localization_structure_evidence": structure_evidence_flat[localization_flat_ids].astype(np.float32),
        "localization_surface_evidence": surface_evidence_flat[localization_flat_ids].astype(np.float32),
        "localization_edge_evidence": edge_evidence_flat[localization_flat_ids].astype(np.float32),
        "localization_low_support_evidence": low_support_evidence_flat[localization_flat_ids].astype(np.float32),
        "localization_uncertainty": uncertainty_flat[localization_flat_ids].astype(np.float32),
        "localization_normal_axial_angle_rad": normal_angle_flat[localization_flat_ids].astype(np.float32),
        "localization_normal_concentration": normal_concentration_flat[localization_flat_ids].astype(np.float32),
        "localization_vertical_persistence": vertical_persistence_flat[localization_flat_ids].astype(np.uint8),
        "localization_fine_height_occupied_count": fine_occupied_count_flat[localization_flat_ids].astype(np.uint8),
        "localization_representative_source_point_index": representative_flat[localization_flat_ids].astype(np.int32),
    }

    full_data = dict(data)
    full_data.update({f"a1_3_{key}": value for key, value in derived_full.items()})
    np.savez_compressed(output_dir / "a1_3_data.npz", **full_data)

    short_data = {
        key: value
        for key, value in derived_full.items()
        if key
        not in {
            "projection_source_point_indices",
            "projection_grid_row",
            "projection_grid_col",
            "projection_flat_cell_id",
            "cell_point_offsets",
            "cell_point_source_indices",
        }
    }
    # Compatibility aliases for future localizer experiments.
    short_data["points_xy"] = localization_xy.astype(np.float32)
    short_data["point_role"] = localization_roles
    short_data["point_weight_prototype"] = (
        structure_evidence_flat[localization_flat_ids]
        * (1.0 - 0.20 * uncertainty_flat[localization_flat_ids])
    ).astype(np.float32)
    np.savez_compressed(output_dir / "a1_3_data_short.npz", **short_data)

    occupied_cells = int(np.count_nonzero(occupied_flat))
    structure_cells = int(np.count_nonzero(structure_candidate_flat))
    grid_rows = [
        {
            "metric": "input_projection_ready_points",
            "value": int(np.count_nonzero(projection_ready)),
        },
        {"metric": "in_grid_and_radial_points", "value": int(len(source_point_indices))},
        {"metric": "occupied_cells", "value": occupied_cells},
        {"metric": "surface_candidate_cells", "value": int(np.count_nonzero(surface_candidate_flat))},
        {"metric": "depth_jump_candidate_cells", "value": int(np.count_nonzero(depth_candidate_flat))},
        {"metric": "normal_turn_candidate_cells", "value": int(np.count_nonzero(normal_turn_candidate_flat))},
        {"metric": "low_support_candidate_cells", "value": int(np.count_nonzero(low_support_candidate_flat))},
        {"metric": "far_uncertain_candidate_cells", "value": int(np.count_nonzero(far_candidate_flat))},
        {"metric": "structure_candidate_cells", "value": structure_cells},
        {"metric": "single_scan_height_core_cells", "value": int(np.count_nonzero(single_scan_height_core_flat))},
        {"metric": "localization_preview_cells", "value": int(len(localization_flat_ids))},
    ]
    with (output_dir / "grid_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(grid_rows)

    save_height_layers_plot(
        output_dir / "01_height_layer_counts.png",
        full_grids["grid_low"],
        full_grids["grid_mid"],
        full_grids["grid_high"],
        x_range,
        y_range,
    )
    save_grid_plot(
        output_dir / "02_smooth_surface_max.png",
        full_grids["smooth_surface_max"],
        x_range,
        y_range,
        "A1.3 smooth-surface max per XY cell",
        "score",
        vmin=0.0,
        vmax=1.0,
    )
    save_grid_plot(
        output_dir / "03_edge_evidence.png",
        full_grids["edge_evidence"],
        x_range,
        y_range,
        "A1.3 edge evidence per XY cell",
        "score",
        vmin=0.0,
        vmax=1.0,
    )
    save_grid_plot(
        output_dir / "04_low_support_vertical_max.png",
        full_grids["low_support_vertical_max"],
        x_range,
        y_range,
        "A1.3 low-support vertical max per XY cell",
        "score",
        vmin=0.0,
        vmax=1.0,
    )
    save_grid_plot(
        output_dir / "05_fine_height_occupied_count.png",
        full_grids["fine_height_occupied_count"],
        x_range,
        y_range,
        "A1.3 occupied fine-height bins per XY cell",
        "occupied bins",
        vmin=0,
        vmax=len(fine_edges) - 1,
    )
    save_normal_field_plot(
        output_dir / "06_normal_field.png",
        full_grids["structure_evidence"],
        full_grids["normal_axial_angle_rad"],
        full_grids["normal_concentration"],
        full_grids["structure_candidate"],
        x_range,
        y_range,
        resolution,
        max(args.normal_field_stride, 1),
    )
    save_grid_plot(
        output_dir / "07_structure_evidence.png",
        full_grids["structure_evidence"],
        x_range,
        y_range,
        "A1.3 combined structure evidence — diagnostic",
        "evidence",
        vmin=0.0,
        vmax=1.0,
    )
    save_preview_plot(
        output_dir / "08_localization_preview_budget.png",
        all_xy_flat[structure_candidate_flat],
        localization_xy,
        localization_roles,
    )

    comparison: dict[str, Any] | None = None
    if args.v2_2_data_short:
        v2_path = args.v2_2_data_short.expanduser().resolve()
        if not v2_path.is_file():
            raise FileNotFoundError(v2_path)
        with np.load(v2_path, allow_pickle=False) as v2_source:
            if "wall_core" not in v2_source.files:
                raise KeyError("V2.2 data_short.npz에 wall_core가 없습니다.")
            v2_wall_core = v2_source["wall_core"].astype(bool)
            if v2_wall_core.shape != grid_shape:
                raise ValueError(
                    f"V2.2 wall_core shape {v2_wall_core.shape} != A1.3 {grid_shape}"
                )

        a_candidate = full_grids["structure_candidate"].astype(bool)
        intersection = int(np.count_nonzero(v2_wall_core & a_candidate))
        union = int(np.count_nonzero(v2_wall_core | a_candidate))
        v2_count = int(np.count_nonzero(v2_wall_core))
        a_count = int(np.count_nonzero(a_candidate))
        comparison = {
            "v2_2_data_short": str(v2_path),
            "v2_2_sha256": sha256_file(v2_path),
            "v2_2_wall_core_cells": v2_count,
            "a1_3_structure_candidate_cells": a_count,
            "intersection_cells": intersection,
            "union_cells": union,
            "iou": float(intersection / union) if union else 1.0,
            "v2_2_coverage_by_a1_3": float(intersection / v2_count) if v2_count else 1.0,
            "a1_3_overlap_with_v2_2": float(intersection / a_count) if a_count else 1.0,
            "important_note": (
                "V2.2 wall_core는 GT가 아니다. 또한 V2.2 active long window는 여러 frame을 "
                "누적하지만 현재 A1.3은 한 raw scan이므로 이 값은 좌표/구조 감사용이다."
            ),
        }
        if args.v2_2_sample_json:
            v2_sample_path = args.v2_2_sample_json.expanduser().resolve()
            if v2_sample_path.is_file():
                v2_sample = load_json(v2_sample_path)
                comparison["v2_2_sample_json"] = str(v2_sample_path)
                comparison["v2_2_active_window"] = v2_sample.get("active_window")
                comparison["v2_2_used_frame_count"] = v2_sample.get("motion", {}).get(
                    "used_frame_count"
                )
        (output_dir / "comparison_to_v2_2.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output_dir / "comparison_to_v2_2.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "value"])
            for key, value in comparison.items():
                if isinstance(value, (str, int, float)) or value is None:
                    writer.writerow([key, value])
        save_v2_overlay(
            output_dir / "09_v2_2_wall_core_overlay.png",
            v2_wall_core,
            a_candidate,
            x_range,
            y_range,
        )

    summary = {
        "schema_version": "a1_3_flattened_multichannel_v0_1",
        "sample_id": input_summary.get("sample_id"),
        "source_npz": str(input_npz),
        "source_npz_sha256": sha256_file(input_npz),
        "source_summary_json": str(input_summary_path),
        "source_summary_sha256": sha256_file(input_summary_path),
        "grid_geometry": {
            "x_range_m": list(x_range),
            "y_range_m": list(y_range),
            "resolution_m": resolution,
            "grid_shape": list(grid_shape),
            "radial_max_m": float(args.radial_max_m),
            "row_zero_location": "positive_y_side",
            "cell_center_conversion": (
                "x=x_min+(col+0.5)*resolution; "
                "y=y_min+(grid_height-1-row+0.5)*resolution"
            ),
        },
        "height_representation": {
            "legacy_bands_m": geometry["height_bands_m"],
            "fine_height_edges_m": fine_edges.tolist(),
            "fine_height_bin_count": int(len(fine_edges) - 1),
        },
        "counts": {
            "raw_points_in_input": int(len(points)),
            "projection_ready_points_before_grid_crop": int(np.count_nonzero(projection_ready)),
            "projection_points_after_grid_and_radial_crop": int(len(source_point_indices)),
            "outside_grid_or_radial_excluded_points": int(
                np.count_nonzero(projection_ready) - len(source_point_indices)
            ),
            "occupied_cells": occupied_cells,
            "surface_candidate_cells": int(np.count_nonzero(surface_candidate_flat)),
            "depth_jump_candidate_cells": int(np.count_nonzero(depth_candidate_flat)),
            "normal_turn_candidate_cells": int(np.count_nonzero(normal_turn_candidate_flat)),
            "low_support_candidate_cells": int(np.count_nonzero(low_support_candidate_flat)),
            "far_uncertain_candidate_cells": int(np.count_nonzero(far_candidate_flat)),
            "structure_candidate_cells": structure_cells,
            "single_scan_height_core_cells": int(np.count_nonzero(single_scan_height_core_flat)),
            "localization_preview_cells": int(len(localization_flat_ids)),
        },
        "localization_preview": {
            "budget": int(args.localization_budget),
            "angle_bin_count": int(args.angle_bin_count),
            "requested_role_counts": requested_role_counts,
            "actual_role_counts": selected_role_counts,
            "note": (
                "셀 단위 진단용 샘플이며 아직 V1.7/MCL에 연결하지 않는다. "
                "고유 candidate cell 수가 budget보다 적으면 중복 없이 실제 cell 수만 저장한다."
            ),
        },
        "comparison_to_v2_2": comparison,
        "files": {
            "full": "a1_3_data.npz",
            "short": "a1_3_data_short.npz",
            "schema_manifest": "schema_manifest.json",
            "grid_statistics": "grid_statistics.csv",
        },
        "important_notes": [
            "A1.3에서 실제 3D→2D XY grid 평탄화를 수행했다.",
            "V2.2와 동일한 sample geometry의 x/y range, resolution, grid shape를 사용했다.",
            "LOW/MID/HIGH뿐 아니라 fine height histogram과 height span을 보존했다.",
            "벽 normal은 방향 부호가 아닌 axial orientation으로 셀 집계했다.",
            "structure_candidate와 point_weight_prototype은 진단용이며 최종 MCL likelihood가 아니다.",
            "temporal persistence는 한 raw scan 입력이므로 생성하지 않았다. A1.4에서 8-keyframe으로 계산한다.",
            "V2.2/V1.7 baseline과 기존 A 결과를 덮어쓰지 않았다.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        key: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for key, value in sorted(derived_full.items())
    }
    (output_dir / "schema_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "===== A1.3 Flattened Multichannel Dataset =====",
        f"source: {input_npz}",
        f"sample: {input_summary.get('sample_id')}",
        f"grid: {grid_shape[0]}x{grid_shape[1]} @ {resolution:.3f} m/cell",
        f"x range: {x_range}",
        f"y range: {y_range}",
        f"radial max: {args.radial_max_m:.2f} m",
        f"projection ready before grid crop: {int(np.count_nonzero(projection_ready))}",
        f"projected points: {len(source_point_indices)}",
        f"occupied cells: {occupied_cells}",
        f"structure candidate cells: {structure_cells}",
        f"single-scan height core cells: {int(np.count_nonzero(single_scan_height_core_flat))}",
        f"localization preview cells: {len(localization_flat_ids)}",
        "",
        "Actual 3D-to-2D flattening was performed.",
        "No temporal persistence was fabricated.",
        "No V1.7/MCL score was changed.",
        "No existing dataset/result was overwritten.",
        f"output: {output_dir}",
    ]
    (output_dir / "run_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
