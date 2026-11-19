#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ROLE_NAMES = {
    0: "smooth_surface",
    1: "depth_jump_edge",
    2: "normal_turn_edge",
    3: "low_support_vertical",
    4: "far_sparse_uncertain",
}

ROLE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#d62728",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def require_keys(data: dict[str, np.ndarray], keys: list[str]) -> None:
    missing = sorted(set(keys) - set(data))
    if missing:
        raise KeyError(f"A1.3 short NPZ 필드가 없습니다: {missing}")


def connected_components(
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = candidate.shape
    component_id = np.full((height, width), -1, dtype=np.int32)
    sizes: list[int] = []

    next_id = 0
    for start_row, start_col in zip(*np.nonzero(candidate)):
        if component_id[start_row, start_col] >= 0:
            continue

        queue: deque[tuple[int, int]] = deque()
        queue.append((int(start_row), int(start_col)))
        component_id[start_row, start_col] = next_id
        size = 0

        while queue:
            row, col = queue.popleft()
            size += 1

            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue

                    neighbor_row = row + delta_row
                    neighbor_col = col + delta_col

                    if not (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_col < width
                    ):
                        continue

                    if not candidate[neighbor_row, neighbor_col]:
                        continue

                    if component_id[neighbor_row, neighbor_col] >= 0:
                        continue

                    component_id[neighbor_row, neighbor_col] = next_id
                    queue.append((neighbor_row, neighbor_col))

        sizes.append(size)
        next_id += 1

    return component_id, np.asarray(sizes, dtype=np.int32)


def local_neighbor_count(candidate: np.ndarray) -> np.ndarray:
    height, width = candidate.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue

            destination_row = slice(
                max(0, delta_row),
                min(height, height + delta_row),
            )
            destination_col = slice(
                max(0, delta_col),
                min(width, width + delta_col),
            )
            source_row = slice(
                max(0, -delta_row),
                min(height, height - delta_row),
            )
            source_col = slice(
                max(0, -delta_col),
                min(width, width - delta_col),
            )

            result[destination_row, destination_col] += (
                candidate[source_row, source_col].astype(np.uint8)
            )

    result[~candidate] = 0
    return result


def build_isolation_penalty(
    candidate: np.ndarray,
    component_id: np.ndarray,
    component_sizes: np.ndarray,
    neighbor_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_component = component_id.reshape(-1)
    flat_candidate = candidate.reshape(-1)

    component_size_grid = np.zeros(flat_component.shape, dtype=np.int32)
    active = flat_component >= 0
    component_size_grid[active] = component_sizes[flat_component[active]]
    component_size_grid = component_size_grid.reshape(candidate.shape)

    penalty = np.zeros(candidate.shape, dtype=np.float32)
    penalty[candidate & (component_size_grid <= 1)] = 1.00
    penalty[candidate & (component_size_grid == 2)] = 0.80
    penalty[
        candidate
        & (component_size_grid >= 3)
        & (component_size_grid <= 4)
    ] = 0.55
    penalty[
        candidate
        & (component_size_grid >= 5)
        & (component_size_grid <= 8)
    ] = 0.30

    penalty = np.maximum(
        penalty,
        np.where(candidate & (neighbor_count == 0), 1.00, 0.0),
    )
    penalty = np.maximum(
        penalty,
        np.where(candidate & (neighbor_count == 1), 0.50, 0.0),
    )
    penalty[~candidate] = 0.0

    return component_size_grid, penalty.astype(np.float32)


def grid_cell_xy(data: dict[str, np.ndarray]) -> np.ndarray:
    height, width = map(int, data["grid_shape"])
    resolution = float(data["grid_resolution_m"])
    x_min, x_max = map(float, data["grid_x_range_m"])
    y_min, y_max = map(float, data["grid_y_range_m"])

    expected_width = (x_max - x_min) / resolution
    expected_height = (y_max - y_min) / resolution
    if abs(expected_width - width) > 1e-5:
        raise ValueError("grid x range와 width가 일치하지 않습니다.")
    if abs(expected_height - height) > 1e-5:
        raise ValueError("grid y range와 height가 일치하지 않습니다.")

    rows, cols = np.indices((height, width))
    x = x_min + (cols.astype(np.float64) + 0.5) * resolution
    row_from_bottom = (height - 1) - rows.astype(np.float64)
    y = y_min + (row_from_bottom + 0.5) * resolution

    return np.stack((x, y), axis=-1).astype(np.float32)


def assign_roles(
    candidate: np.ndarray,
    surface: np.ndarray,
    depth: np.ndarray,
    normal_turn: np.ndarray,
    low_support: np.ndarray,
    uncertainty: np.ndarray,
    range_median: np.ndarray,
    normal_turn_threshold: float,
    depth_threshold: float,
    low_support_threshold: float,
    far_start_m: float,
    far_uncertainty_threshold: float,
    far_low_support_threshold: float,
) -> np.ndarray:
    candidate_flat = candidate.reshape(-1)
    surface_flat = surface.reshape(-1)
    depth_flat = depth.reshape(-1)
    normal_flat = normal_turn.reshape(-1)
    low_flat = low_support.reshape(-1)
    uncertainty_flat = uncertainty.reshape(-1)
    range_flat = range_median.reshape(-1)

    role = np.full(candidate_flat.shape, -1, dtype=np.int8)

    # 희소 원거리 구조는 edge보다 먼저 보존하되,
    # 단순히 멀다는 이유만으로 이 역할을 부여하지 않는다.
    far_mask = (
        candidate_flat
        & np.isfinite(range_flat)
        & (range_flat >= far_start_m)
        & (
            (uncertainty_flat >= far_uncertainty_threshold)
            | (low_flat >= far_low_support_threshold)
        )
    )
    role[far_mask] = 4

    # 경계 정보는 smooth surface보다 우선적으로 역할을 부여한다.
    normal_mask = (
        candidate_flat
        & (role < 0)
        & (normal_flat >= normal_turn_threshold)
    )
    role[normal_mask] = 2

    depth_mask = (
        candidate_flat
        & (role < 0)
        & (depth_flat >= depth_threshold)
    )
    role[depth_mask] = 1

    low_mask = (
        candidate_flat
        & (role < 0)
        & (low_flat >= low_support_threshold)
    )
    role[low_mask] = 3

    role[candidate_flat & (role < 0)] = 0
    return role.reshape(candidate.shape)


def role_score_grids(
    data: dict[str, np.ndarray],
    isolation_penalty: np.ndarray,
) -> dict[int, np.ndarray]:
    surface = data["surface_evidence"].astype(np.float64)
    depth = data["depth_jump_edge_max"].astype(np.float64)
    normal_turn = data["normal_turn_edge_max"].astype(np.float64)
    low_support = data["low_support_evidence"].astype(np.float64)
    uncertainty = data["uncertainty"].astype(np.float64)
    concentration = data["normal_concentration"].astype(np.float64)
    vertical = data["vertical_persistence"].astype(np.float64)
    fine_height = data["fine_height_occupied_count"].astype(np.float64)

    concentration_factor = np.clip(0.65 + 0.35 * concentration, 0.65, 1.0)
    height_factor = np.clip(
        0.72
        + 0.14 * np.minimum(vertical, 2.0)
        + 0.03 * np.minimum(fine_height, 4.0),
        0.72,
        1.0,
    )
    isolation_factor = np.clip(1.0 - 0.80 * isolation_penalty, 0.15, 1.0)

    return {
        0: np.clip(
            surface * concentration_factor * height_factor * isolation_factor,
            0.0,
            1.0,
        ).astype(np.float32),
        1: np.clip(
            depth
            * (0.80 + 0.20 * concentration_factor)
            * isolation_factor,
            0.0,
            1.0,
        ).astype(np.float32),
        2: np.clip(
            normal_turn
            * (0.75 + 0.25 * concentration_factor)
            * isolation_factor,
            0.0,
            1.0,
        ).astype(np.float32),
        3: np.clip(
            low_support
            * (0.75 + 0.25 * height_factor)
            * np.clip(1.0 - 0.55 * isolation_penalty, 0.25, 1.0),
            0.0,
            1.0,
        ).astype(np.float32),
        4: np.clip(
            np.maximum(uncertainty, 0.45 * low_support)
            * np.clip(1.0 - 0.50 * isolation_penalty, 0.30, 1.0),
            0.0,
            1.0,
        ).astype(np.float32),
    }


def stratified_groups(
    flat_ids: np.ndarray,
    xy_flat: np.ndarray,
    normal_angle_flat: np.ndarray,
    angle_bin_count: int,
    normal_bin_count: int,
    range_edges: np.ndarray,
) -> dict[tuple[int, int, int], list[int]]:
    xy = xy_flat[flat_ids]
    azimuth = np.arctan2(xy[:, 1], xy[:, 0])
    distance = np.linalg.norm(xy, axis=1)

    angle_normalized = (azimuth + math.pi) % (2.0 * math.pi)
    angle_bin = np.floor(
        angle_normalized / (2.0 * math.pi) * angle_bin_count
    ).astype(np.int32)
    angle_bin = np.clip(angle_bin, 0, angle_bin_count - 1)

    range_bin = np.digitize(distance, range_edges, right=False) - 1
    range_bin = np.clip(range_bin, 0, len(range_edges) - 2)

    normal = normal_angle_flat[flat_ids].astype(np.float64)
    normal_valid = np.isfinite(normal)
    normal_normalized = (normal + math.pi / 2.0) % math.pi
    normal_bin = np.full(len(flat_ids), normal_bin_count, dtype=np.int32)
    normal_bin[normal_valid] = np.floor(
        normal_normalized[normal_valid] / math.pi * normal_bin_count
    ).astype(np.int32)
    normal_bin = np.clip(normal_bin, 0, normal_bin_count)

    groups: dict[tuple[int, int, int], list[int]] = {}
    for local_index, flat_id in enumerate(flat_ids):
        key = (
            int(angle_bin[local_index]),
            int(range_bin[local_index]),
            int(normal_bin[local_index]),
        )
        groups.setdefault(key, []).append(int(flat_id))
    return groups


def spacing_ok(
    flat_id: int,
    selected_flat_ids: list[int],
    width: int,
    minimum_spacing_cells: int,
) -> bool:
    if minimum_spacing_cells <= 0 or not selected_flat_ids:
        return True

    row = flat_id // width
    col = flat_id % width
    radius_squared = minimum_spacing_cells * minimum_spacing_cells

    for selected_id in selected_flat_ids:
        selected_row = selected_id // width
        selected_col = selected_id % width
        delta_row = row - selected_row
        delta_col = col - selected_col
        if delta_row * delta_row + delta_col * delta_col < radius_squared:
            return False

    return True


def select_role_cells(
    candidate_ids: np.ndarray,
    score_flat: np.ndarray,
    xy_flat: np.ndarray,
    normal_angle_flat: np.ndarray,
    quota: int,
    width: int,
    angle_bin_count: int,
    normal_bin_count: int,
    range_edges: np.ndarray,
    minimum_spacing_cells: int,
    globally_selected: set[int],
) -> np.ndarray:
    if quota <= 0 or candidate_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    candidate_ids = np.asarray(
        [flat_id for flat_id in candidate_ids if int(flat_id) not in globally_selected],
        dtype=np.int64,
    )
    if candidate_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    groups = stratified_groups(
        candidate_ids,
        xy_flat,
        normal_angle_flat,
        angle_bin_count,
        normal_bin_count,
        range_edges,
    )

    for values in groups.values():
        values.sort(
            key=lambda flat_id: (
                -float(score_flat[flat_id]),
                flat_id,
            )
        )

    selected: list[int] = []
    cursors = {key: 0 for key in groups}
    group_keys = sorted(groups)

    # 첫 번째 pass는 역할별 공간 간격을 강제한다.
    while len(selected) < quota:
        progress = False
        for key in group_keys:
            values = groups[key]
            cursor = cursors[key]

            while cursor < len(values):
                flat_id = values[cursor]
                cursor += 1
                if flat_id in globally_selected:
                    continue
                if not spacing_ok(
                    flat_id,
                    selected,
                    width,
                    minimum_spacing_cells,
                ):
                    continue

                selected.append(flat_id)
                globally_selected.add(flat_id)
                progress = True
                break

            cursors[key] = cursor
            if len(selected) >= quota:
                break

        if not progress:
            break

    # quota가 남으면 간격을 완화하되 stratification은 유지한다.
    if len(selected) < quota:
        for spacing in range(max(minimum_spacing_cells - 1, 0), -1, -1):
            while len(selected) < quota:
                progress = False
                for key in group_keys:
                    values = groups[key]
                    for flat_id in values:
                        if flat_id in globally_selected:
                            continue
                        if not spacing_ok(flat_id, selected, width, spacing):
                            continue

                        selected.append(flat_id)
                        globally_selected.add(flat_id)
                        progress = True
                        break

                    if len(selected) >= quota:
                        break

                if not progress:
                    break

    return np.asarray(selected, dtype=np.int64)


def select_balanced(
    data: dict[str, np.ndarray],
    role_grid: np.ndarray,
    score_grids: dict[int, np.ndarray],
    isolation_penalty: np.ndarray,
    balanced_budget: int,
    role_fractions: dict[int, float],
    angle_bin_count: int,
    normal_bin_count: int,
    surface_spacing_cells: int,
    edge_spacing_cells: int,
    low_support_spacing_cells: int,
    far_spacing_cells: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = map(int, data["grid_shape"])
    candidate = data["structure_candidate"].astype(bool)
    xy_grid = grid_cell_xy(data)
    xy_flat = xy_grid.reshape(-1, 2)
    normal_flat = data["normal_axial_angle_rad"].reshape(-1)
    role_flat = role_grid.reshape(-1)

    isolation_flat = isolation_penalty.reshape(-1)

    # Balanced subset에서만 적용하는 soft eligibility.
    # 모든 후보 cell은 all-candidate 배열에 그대로 남는다.
    isolation_limit = {
        0: 0.50,
        1: 0.80,
        2: 0.80,
        3: 0.50,
        4: 0.30,
    }

    role_candidate_count = {
        role_id: int(np.sum(role_flat == role_id))
        for role_id in ROLE_NAMES
    }
    role_eligible_count = {
        role_id: int(
            np.sum(
                (role_flat == role_id)
                & (isolation_flat <= isolation_limit[role_id])
            )
        )
        for role_id in ROLE_NAMES
    }

    quotas = {
        role_id: int(round(balanced_budget * role_fractions[role_id]))
        for role_id in ROLE_NAMES
    }
    quotas[0] += balanced_budget - sum(quotas.values())

    spacing = {
        0: surface_spacing_cells,
        1: edge_spacing_cells,
        2: edge_spacing_cells,
        3: low_support_spacing_cells,
        4: far_spacing_cells,
    }

    range_edges = np.asarray(
        [0.0, 3.0, 6.0, 9.0, 12.0, 20.0],
        dtype=np.float64,
    )

    # 희소 역할부터 먼저 뽑아 surface가 edge를 선점하지 못하게 한다.
    role_order = [4, 2, 1, 3, 0]
    globally_selected: set[int] = set()
    selected_parts: list[np.ndarray] = []
    selected_role_parts: list[np.ndarray] = []
    actual_counts = {role_id: 0 for role_id in ROLE_NAMES}

    for role_id in role_order:
        ids = np.flatnonzero(
            (role_flat == role_id)
            & (isolation_flat <= isolation_limit[role_id])
        )
        selected = select_role_cells(
            candidate_ids=ids,
            score_flat=score_grids[role_id].reshape(-1),
            xy_flat=xy_flat,
            normal_angle_flat=normal_flat,
            quota=quotas[role_id],
            width=width,
            angle_bin_count=angle_bin_count,
            normal_bin_count=normal_bin_count,
            range_edges=range_edges,
            minimum_spacing_cells=spacing[role_id],
            globally_selected=globally_selected,
        )
        selected_parts.append(selected)
        selected_role_parts.append(
            np.full(len(selected), role_id, dtype=np.int8)
        )
        actual_counts[role_id] += int(len(selected))

    used = sum(actual_counts.values())
    remaining_budget = max(0, balanced_budget - used)

    # 희소 역할의 후보가 부족해 남은 예산은 전체 미선택 후보 중
    # 종합 점수가 높은 셀로 채운다. 이때도 공간·방향 stratification을 유지한다.
    if remaining_budget > 0:
        candidate_ids = np.flatnonzero(candidate.reshape(-1))
        remaining_ids = np.asarray(
            [
                flat_id
                for flat_id in candidate_ids
                if (
                    int(flat_id) not in globally_selected
                    and (
                        isolation_flat[flat_id] <= 0.50
                        or (
                            role_flat[flat_id] in (1, 2)
                            and isolation_flat[flat_id] <= 0.80
                        )
                    )
                )
            ],
            dtype=np.int64,
        )

        if remaining_ids.size:
            composite = np.maximum.reduce(
                [
                    score_grids[0].reshape(-1),
                    score_grids[1].reshape(-1),
                    score_grids[2].reshape(-1),
                    score_grids[3].reshape(-1),
                    score_grids[4].reshape(-1),
                ]
            )

            selected_fill = select_role_cells(
                candidate_ids=remaining_ids,
                score_flat=composite,
                xy_flat=xy_flat,
                normal_angle_flat=normal_flat,
                quota=remaining_budget,
                width=width,
                angle_bin_count=angle_bin_count,
                normal_bin_count=normal_bin_count,
                range_edges=range_edges,
                minimum_spacing_cells=1,
                globally_selected=globally_selected,
            )

            fill_roles = role_flat[selected_fill].astype(np.int8)
            selected_parts.append(selected_fill)
            selected_role_parts.append(fill_roles)
            for role_id in ROLE_NAMES:
                actual_counts[role_id] += int(
                    np.sum(fill_roles == role_id)
                )

    used_after_soft_fill = sum(actual_counts.values())
    emergency_fill_count = 0

    # soft eligibility까지 적용한 뒤에도 예산이 남으면,
    # isolation 조건만 해제하고 모든 미선택 structure candidate에서 보충한다.
    # 기존 선택은 유지하며 중복 없이 정확히 balanced_budget을 맞춘다.
    emergency_budget = max(0, balanced_budget - used_after_soft_fill)
    if emergency_budget > 0:
        all_candidate_ids = np.flatnonzero(candidate.reshape(-1))
        emergency_ids = np.asarray(
            [
                flat_id
                for flat_id in all_candidate_ids
                if int(flat_id) not in globally_selected
            ],
            dtype=np.int64,
        )

        if emergency_ids.size:
            composite = np.maximum.reduce(
                [
                    score_grids[0].reshape(-1),
                    score_grids[1].reshape(-1),
                    score_grids[2].reshape(-1),
                    score_grids[3].reshape(-1),
                    score_grids[4].reshape(-1),
                ]
            )
            selected_emergency = select_role_cells(
                candidate_ids=emergency_ids,
                score_flat=composite,
                xy_flat=xy_flat,
                normal_angle_flat=normal_flat,
                quota=emergency_budget,
                width=width,
                angle_bin_count=angle_bin_count,
                normal_bin_count=normal_bin_count,
                range_edges=range_edges,
                minimum_spacing_cells=0,
                globally_selected=globally_selected,
            )
            emergency_roles = role_flat[selected_emergency].astype(np.int8)
            selected_parts.append(selected_emergency)
            selected_role_parts.append(emergency_roles)
            emergency_fill_count = int(len(selected_emergency))
            for role_id in ROLE_NAMES:
                actual_counts[role_id] += int(
                    np.sum(emergency_roles == role_id)
                )

    selected_ids = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    selected_roles = (
        np.concatenate(selected_role_parts)
        if selected_role_parts
        else np.empty(0, dtype=np.int8)
    )

    total_candidate_count = int(np.count_nonzero(candidate))
    if len(selected_ids) != balanced_budget:
        raise RuntimeError(
            "A1.3.1 balanced budget을 채우지 못했습니다: "
            f"selected={len(selected_ids)}, requested={balanced_budget}, "
            f"structure_candidate={total_candidate_count}"
        )

    # 출력은 cell ID 순서로 정렬해 재현성과 비교를 쉽게 한다.
    order = np.argsort(selected_ids)
    selected_ids = selected_ids[order]
    selected_roles = selected_roles[order]

    info = {
        "requested_budget": int(balanced_budget),
        "actual_selected_count": int(len(selected_ids)),
        "soft_fill_selected_count": int(max(0, used_after_soft_fill - used)),
        "emergency_fill_selected_count": int(emergency_fill_count),
        "emergency_fill_policy": (
            "when soft isolation eligibility leaves the subset under budget, "
            "fill from all remaining structure_candidate cells with spacing=0; "
            "no duplicate cells"
        ),
        "role_candidate_count": {
            ROLE_NAMES[role_id]: role_candidate_count[role_id]
            for role_id in ROLE_NAMES
        },
        "eligible_role_count": {
            ROLE_NAMES[role_id]: role_eligible_count[role_id]
            for role_id in ROLE_NAMES
        },
        "requested_role_quota": {
            ROLE_NAMES[role_id]: quotas[role_id]
            for role_id in ROLE_NAMES
        },
        "actual_role_count": {
            ROLE_NAMES[role_id]: actual_counts[role_id]
            for role_id in ROLE_NAMES
        },
        "selection_order": [ROLE_NAMES[role_id] for role_id in role_order],
        "isolation_limit_for_balanced_subset": {
            ROLE_NAMES[role_id]: isolation_limit[role_id]
            for role_id in ROLE_NAMES
        },
        "spacing_cells": {
            "smooth_surface": surface_spacing_cells,
            "depth_jump_edge": edge_spacing_cells,
            "normal_turn_edge": edge_spacing_cells,
            "low_support_vertical": low_support_spacing_cells,
            "far_sparse_uncertain": far_spacing_cells,
        },
        "range_edges_m": range_edges.tolist(),
        "angle_bin_count": angle_bin_count,
        "normal_bin_count": normal_bin_count,
    }
    return selected_ids, selected_roles, info


def save_role_csv(
    path: Path,
    all_role: np.ndarray,
    selected_role: np.ndarray,
    selection_info: dict[str, Any],
) -> None:
    rows = []
    for role_id, role_name in ROLE_NAMES.items():
        rows.append(
            {
                "role_id": role_id,
                "role_name": role_name,
                "all_candidate_count": int(np.sum(all_role == role_id)),
                "requested_quota": int(
                    selection_info["requested_role_quota"][role_name]
                ),
                "selected_count": int(np.sum(selected_role == role_id)),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_selection_csv(
    path: Path,
    selected_ids: np.ndarray,
    selected_roles: np.ndarray,
    data: dict[str, np.ndarray],
    score_grids: dict[int, np.ndarray],
    component_size: np.ndarray,
    neighbor_count: np.ndarray,
    isolation_penalty: np.ndarray,
) -> None:
    height, width = map(int, data["grid_shape"])
    xy = grid_cell_xy(data).reshape(-1, 2)

    component_flat = component_size.reshape(-1)
    neighbor_flat = neighbor_count.reshape(-1)
    isolation_flat = isolation_penalty.reshape(-1)
    surface = data["surface_evidence"].reshape(-1)
    depth = data["depth_jump_edge_max"].reshape(-1)
    normal_turn = data["normal_turn_edge_max"].reshape(-1)
    low_support = data["low_support_evidence"].reshape(-1)
    uncertainty = data["uncertainty"].reshape(-1)
    normal_angle = data["normal_axial_angle_rad"].reshape(-1)

    rows = []
    for flat_id, role_id in zip(selected_ids, selected_roles):
        row = int(flat_id // width)
        col = int(flat_id % width)
        role_id_int = int(role_id)

        rows.append(
            {
                "flat_cell_id": int(flat_id),
                "grid_row": row,
                "grid_col": col,
                "x_m": float(xy[flat_id, 0]),
                "y_m": float(xy[flat_id, 1]),
                "role_id": role_id_int,
                "role_name": ROLE_NAMES[role_id_int],
                "role_selection_score": float(
                    score_grids[role_id_int].reshape(-1)[flat_id]
                ),
                "surface_evidence": float(surface[flat_id]),
                "depth_jump_edge": float(depth[flat_id]),
                "normal_turn_edge": float(normal_turn[flat_id]),
                "low_support_evidence": float(low_support[flat_id]),
                "uncertainty": float(uncertainty[flat_id]),
                "normal_axial_angle_rad": float(normal_angle[flat_id]),
                "component_size": int(component_flat[flat_id]),
                "local_neighbor_count": int(neighbor_flat[flat_id]),
                "isolation_penalty": float(isolation_flat[flat_id]),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_balanced_plot(
    path: Path,
    data: dict[str, np.ndarray],
    selected_ids: np.ndarray,
    selected_roles: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    candidate = data["structure_candidate"].astype(bool).reshape(-1)
    xy = grid_cell_xy(data).reshape(-1, 2)

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.scatter(
        xy[candidate, 0],
        xy[candidate, 1],
        s=4,
        alpha=0.10,
        label="all A1.3 candidate cells",
    )

    if len(selected_ids):
        plot = axis.scatter(
            xy[selected_ids, 0],
            xy[selected_ids, 1],
            c=selected_roles,
            cmap=ListedColormap(ROLE_COLORS),
            vmin=-0.5,
            vmax=4.5,
            s=18,
            alpha=0.95,
        )
        colorbar = figure.colorbar(
            plot,
            ax=axis,
            ticks=[0, 1, 2, 3, 4],
        )
        colorbar.ax.set_yticklabels(
            [ROLE_NAMES[index] for index in range(5)]
        )

    axis.scatter([0.0], [0.0], marker="x", s=100, linewidths=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_old_new_plot(
    path: Path,
    data: dict[str, np.ndarray],
    selected_ids: np.ndarray,
    selected_roles: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    xy = grid_cell_xy(data).reshape(-1, 2)
    old_ids = data["localization_flat_cell_id"].astype(np.int64)
    old_roles = data["localization_role"].astype(np.int8)

    figure, axes = plt.subplots(1, 2, figsize=(18, 7))

    for axis, ids, roles, title in [
        (
            axes[0],
            old_ids,
            old_roles,
            "A1.3 original selector",
        ),
        (
            axes[1],
            selected_ids,
            selected_roles,
            "A1.3.1 balanced selector",
        ),
    ]:
        if len(ids):
            axis.scatter(
                xy[ids, 0],
                xy[ids, 1],
                c=roles,
                cmap=ListedColormap(ROLE_COLORS),
                vmin=-0.5,
                vmax=4.5,
                s=14,
                alpha=0.95,
            )
        axis.scatter([0.0], [0.0], marker="x", s=90, linewidths=2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(f"{title}\nselected={len(ids)}")
        axis.grid(True, alpha=0.2)

    figure.suptitle(
        "A1.3 localization selection before/after\n"
        "A1.3 grids are unchanged"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_component_plot(
    path: Path,
    data: dict[str, np.ndarray],
    component_size: np.ndarray,
    isolation_penalty: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    x_range = list(map(float, data["grid_x_range_m"]))
    y_range = list(map(float, data["grid_y_range_m"]))

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))

    image0 = axes[0].imshow(
        component_size,
        origin="upper",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        interpolation="nearest",
    )
    figure.colorbar(image0, ax=axes[0], label="8-connected component size")
    axes[0].scatter([0.0], [0.0], marker="x", s=80, linewidths=2)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("A1.3.1 candidate component size")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")

    image1 = axes[1].imshow(
        isolation_penalty,
        origin="upper",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(image1, ax=axes[1], label="soft isolation penalty")
    axes[1].scatter([0.0], [0.0], marker="x", s=80, linewidths=2)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("A1.3.1 isolation penalty")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_role_count_plot(
    path: Path,
    data: dict[str, np.ndarray],
    all_role: np.ndarray,
    selected_role: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    names = [ROLE_NAMES[index] for index in range(5)]
    all_counts = [int(np.sum(all_role == index)) for index in range(5)]
    selected_counts = [
        int(np.sum(selected_role == index))
        for index in range(5)
    ]

    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        positions - width / 2,
        all_counts,
        width,
        label="all candidate role cells",
    )
    axis.bar(
        positions + width / 2,
        selected_counts,
        width,
        label="A1.3.1 selected",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylabel("cell count")
    axis.set_title("A1.3.1 role balance")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A1.3 다중 채널 grid는 변경하지 않고, "
            "위치추정용 cell selector만 균형화합니다."
        )
    )
    parser.add_argument("--input-short-npz", type=Path, required=True)
    parser.add_argument("--input-summary-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--balanced-budget", type=int, default=384)
    parser.add_argument("--surface-fraction", type=float, default=0.50)
    parser.add_argument("--depth-fraction", type=float, default=0.20)
    parser.add_argument("--normal-fraction", type=float, default=0.10)
    parser.add_argument("--low-support-fraction", type=float, default=0.15)
    parser.add_argument("--far-fraction", type=float, default=0.05)
    parser.add_argument("--angle-bin-count", type=int, default=36)
    parser.add_argument("--normal-bin-count", type=int, default=8)
    parser.add_argument("--surface-spacing-cells", type=int, default=3)
    parser.add_argument("--edge-spacing-cells", type=int, default=2)
    parser.add_argument("--low-support-spacing-cells", type=int, default=2)
    parser.add_argument("--far-spacing-cells", type=int, default=3)
    parser.add_argument("--normal-turn-threshold", type=float, default=0.28)
    parser.add_argument("--depth-threshold", type=float, default=0.32)
    parser.add_argument("--low-support-threshold", type=float, default=0.32)
    parser.add_argument("--far-start-m", type=float, default=6.0)
    parser.add_argument("--far-uncertainty-threshold", type=float, default=0.08)
    parser.add_argument("--far-low-support-threshold", type=float, default=0.35)
    args = parser.parse_args()

    input_npz = args.input_short_npz.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)
    if output_dir.exists():
        raise FileExistsError(
            f"기존 결과를 덮어쓰지 않습니다: {output_dir}"
        )

    fractions = {
        0: args.surface_fraction,
        1: args.depth_fraction,
        2: args.normal_fraction,
        3: args.low_support_fraction,
        4: args.far_fraction,
    }
    fraction_sum = sum(fractions.values())
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(
            f"role fraction 합은 1이어야 합니다: {fraction_sum}"
        )

    output_dir.mkdir(parents=True)

    data = load_npz(input_npz)
    require_keys(
        data,
        [
            "structure_candidate",
            "surface_evidence",
            "depth_jump_edge_max",
            "normal_turn_edge_max",
            "low_support_evidence",
            "uncertainty",
            "range_median_m",
            "normal_concentration",
            "normal_axial_angle_rad",
            "vertical_persistence",
            "fine_height_occupied_count",
            "representative_source_point_index",
            "grid_x_range_m",
            "grid_y_range_m",
            "grid_resolution_m",
            "grid_shape",
            "localization_flat_cell_id",
            "localization_role",
        ],
    )

    candidate = data["structure_candidate"].astype(bool)
    component_id, component_sizes = connected_components(candidate)
    neighbor_count = local_neighbor_count(candidate)
    component_size, isolation_penalty = build_isolation_penalty(
        candidate,
        component_id,
        component_sizes,
        neighbor_count,
    )

    role_grid = assign_roles(
        candidate=candidate,
        surface=data["surface_evidence"],
        depth=data["depth_jump_edge_max"],
        normal_turn=data["normal_turn_edge_max"],
        low_support=data["low_support_evidence"],
        uncertainty=data["uncertainty"],
        range_median=data["range_median_m"],
        normal_turn_threshold=args.normal_turn_threshold,
        depth_threshold=args.depth_threshold,
        low_support_threshold=args.low_support_threshold,
        far_start_m=args.far_start_m,
        far_uncertainty_threshold=args.far_uncertainty_threshold,
        far_low_support_threshold=args.far_low_support_threshold,
    )

    score_grids = role_score_grids(data, isolation_penalty)

    selected_ids, selected_roles, selection_info = select_balanced(
        data=data,
        role_grid=role_grid,
        score_grids=score_grids,
        isolation_penalty=isolation_penalty,
        balanced_budget=args.balanced_budget,
        role_fractions=fractions,
        angle_bin_count=args.angle_bin_count,
        normal_bin_count=args.normal_bin_count,
        surface_spacing_cells=args.surface_spacing_cells,
        edge_spacing_cells=args.edge_spacing_cells,
        low_support_spacing_cells=args.low_support_spacing_cells,
        far_spacing_cells=args.far_spacing_cells,
    )

    height, width = map(int, data["grid_shape"])
    xy_grid = grid_cell_xy(data)
    xy_flat = xy_grid.reshape(-1, 2)
    rows = (selected_ids // width).astype(np.int32)
    cols = (selected_ids % width).astype(np.int32)

    selected_score = np.asarray(
        [
            score_grids[int(role)].reshape(-1)[flat_id]
            for flat_id, role in zip(selected_ids, selected_roles)
        ],
        dtype=np.float32,
    )

    # 모든 후보도 별도로 저장하여 balanced subset과 성능을 비교할 수 있게 한다.
    all_candidate_ids = np.flatnonzero(candidate.reshape(-1)).astype(np.int32)
    all_candidate_roles = role_grid.reshape(-1)[all_candidate_ids].astype(np.int8)

    output_data = dict(data)
    output_data.update(
        {
            "a1_3_1_component_id": component_id.astype(np.int32),
            "a1_3_1_component_size": component_size.astype(np.int32),
            "a1_3_1_local_neighbor_count": neighbor_count.astype(np.uint8),
            "a1_3_1_isolation_penalty": isolation_penalty.astype(np.float32),
            "a1_3_1_role_grid": role_grid.astype(np.int8),
            "a1_3_1_surface_selection_score":
                score_grids[0].astype(np.float32),
            "a1_3_1_depth_selection_score":
                score_grids[1].astype(np.float32),
            "a1_3_1_normal_selection_score":
                score_grids[2].astype(np.float32),
            "a1_3_1_low_support_selection_score":
                score_grids[3].astype(np.float32),
            "a1_3_1_far_selection_score":
                score_grids[4].astype(np.float32),
            "a1_3_1_all_candidate_flat_cell_id": all_candidate_ids,
            "a1_3_1_all_candidate_xy":
                xy_flat[all_candidate_ids].astype(np.float32),
            "a1_3_1_all_candidate_role": all_candidate_roles,
            "a1_3_1_balanced_flat_cell_id": selected_ids.astype(np.int32),
            "a1_3_1_balanced_grid_row": rows,
            "a1_3_1_balanced_grid_col": cols,
            "a1_3_1_balanced_xy": xy_flat[selected_ids].astype(np.float32),
            "a1_3_1_balanced_role": selected_roles.astype(np.int8),
            "a1_3_1_balanced_selection_score": selected_score,
            "a1_3_1_balanced_structure_evidence":
                data["structure_evidence"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_surface_evidence":
                data["surface_evidence"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_edge_evidence":
                data["edge_evidence"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_low_support_evidence":
                data["low_support_evidence"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_uncertainty":
                data["uncertainty"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_isolation_penalty":
                isolation_penalty.reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_normal_axial_angle_rad":
                data["normal_axial_angle_rad"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_normal_concentration":
                data["normal_concentration"].reshape(-1)[selected_ids].astype(np.float32),
            "a1_3_1_balanced_vertical_persistence":
                data["vertical_persistence"].reshape(-1)[selected_ids].astype(np.uint8),
            "a1_3_1_balanced_fine_height_occupied_count":
                data["fine_height_occupied_count"].reshape(-1)[selected_ids].astype(np.uint8),
            "a1_3_1_balanced_representative_source_point_index":
                data["representative_source_point_index"].reshape(-1)[selected_ids].astype(np.int32),
            # 새 위치추정 실험기의 기본 alias.
            "points_xy": xy_flat[selected_ids].astype(np.float32),
            "point_role": selected_roles.astype(np.int8),
            "point_weight_prototype": np.clip(
                selected_score
                * (
                    1.0
                    - 0.35
                    * isolation_penalty.reshape(-1)[selected_ids]
                ),
                0.05,
                1.0,
            ).astype(np.float32),
        }
    )

    np.savez_compressed(
        output_dir / "a1_3_1_data_short.npz",
        **output_data,
    )

    selection_only = {
        key: output_data[key]
        for key in output_data
        if (
            key.startswith("a1_3_1_")
            or key in {
                "points_xy",
                "point_role",
                "point_weight_prototype",
                "grid_x_range_m",
                "grid_y_range_m",
                "grid_resolution_m",
                "grid_shape",
            }
        )
    }
    np.savez_compressed(
        output_dir / "a1_3_1_selection_only.npz",
        **selection_only,
    )

    save_role_csv(
        output_dir / "role_statistics.csv",
        all_candidate_roles,
        selected_roles,
        selection_info,
    )
    save_selection_csv(
        output_dir / "balanced_selection.csv",
        selected_ids,
        selected_roles,
        data,
        score_grids,
        component_size,
        neighbor_count,
        isolation_penalty,
    )

    save_balanced_plot(
        output_dir / "01_balanced_localization_preview.png",
        data,
        selected_ids,
        selected_roles,
        (
            "A1.3.1 balanced localization cells\n"
            f"selected={len(selected_ids)}, "
            "A1.3 grids unchanged"
        ),
    )
    save_old_new_plot(
        output_dir / "02_original_vs_balanced.png",
        data,
        selected_ids,
        selected_roles,
    )
    save_component_plot(
        output_dir / "03_component_and_isolation.png",
        data,
        component_size,
        isolation_penalty,
    )
    save_role_count_plot(
        output_dir / "04_role_balance.png",
        data,
        all_candidate_roles,
        selected_roles,
    )

    input_summary = None
    if args.input_summary_json:
        input_summary_path = args.input_summary_json.expanduser().resolve()
        if input_summary_path.is_file():
            input_summary = json.loads(
                input_summary_path.read_text(encoding="utf-8")
            )

    summary = {
        "schema_version": "a1_3_1_balanced_localization_selector_v1",
        "source_a1_3_short_npz": str(input_npz),
        "source_a1_3_short_sha256": sha256_file(input_npz),
        "source_summary": input_summary,
        "important_scope": [
            "A1.3 flattened grids were not changed.",
            "Only localization cell roles, isolation penalty, and balanced selection were added.",
            "No existing A1.3 result was overwritten.",
            "No cell was permanently deleted from the all-candidate set.",
            "The balanced set is a subset for experiments; all candidates remain stored.",
            "No MCL likelihood or map score was changed.",
        ],
        "candidate_statistics": {
            "all_structure_candidate_count": int(candidate.sum()),
            "connected_component_count": int(len(component_sizes)),
            "component_sizes_descending": sorted(
                [int(value) for value in component_sizes],
                reverse=True,
            ),
            "singleton_component_count": int(
                np.sum(component_sizes == 1)
            ),
        },
        "role_assignment_thresholds": {
            "normal_turn_threshold": args.normal_turn_threshold,
            "depth_threshold": args.depth_threshold,
            "low_support_threshold": args.low_support_threshold,
            "far_start_m": args.far_start_m,
            "far_uncertainty_threshold": args.far_uncertainty_threshold,
            "far_low_support_threshold": args.far_low_support_threshold,
        },
        "selection": selection_info,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "===== A1.3.1 Balanced Localization Selector =====",
        f"source: {input_npz}",
        f"all structure candidate cells: {int(candidate.sum())}",
        f"connected components: {len(component_sizes)}",
        f"singleton components: {int(np.sum(component_sizes == 1))}",
        f"balanced requested budget: {args.balanced_budget}",
        f"balanced selected cells: {len(selected_ids)}",
        "",
        "selected role counts:",
    ]
    for role_id in ROLE_NAMES:
        report_lines.append(
            f"  {ROLE_NAMES[role_id]}: "
            f"{int(np.sum(selected_roles == role_id))}"
        )
    report_lines.extend(
        [
            "",
            "A1.3 grids were not changed.",
            "All candidate cells remain stored.",
            "Only the balanced localization subset and audit fields were added.",
            "No existing output was overwritten.",
            "No MCL score was changed.",
            f"output: {output_dir}",
        ]
    )

    report = "\n".join(report_lines)
    (output_dir / "run_report.txt").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
