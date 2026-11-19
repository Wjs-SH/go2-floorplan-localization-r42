#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BIT_INVALID_XYZ = 1 << 0
BIT_ZERO_RETURN = 1 << 1
BIT_OUTSIDE_FEATURE_RANGE = 1 << 2
BIT_SUPPORT_DEFICIT = 1 << 3
BIT_HIGH_DEPTH_JUMP = 1 << 4
BIT_HIGH_NORMAL_TURN = 1 << 5
BIT_POOR_PLANE_FIT = 1 << 6
BIT_NORMAL_DISAGREEMENT = 1 << 7
BIT_FAR_SPARSE = 1 << 8
BIT_FEATURE_UNAVAILABLE = 1 << 9


ROLE_NAMES = {
    0: 'smooth_surface',
    1: 'depth_jump_edge',
    2: 'normal_turn_edge',
    3: 'low_support_vertical',
    4: 'far_uncertain',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_scan_columns(ring: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ring_i = np.asarray(ring, dtype=np.int64)
    unique = np.unique(ring_i)
    expected = np.arange(int(unique.min()), int(unique.max()) + 1)
    if not np.array_equal(unique, expected):
        raise ValueError(f'ring 번호가 연속적이지 않습니다: {unique.tolist()}')

    ring_count = len(unique)
    if len(ring_i) % ring_count != 0:
        raise ValueError('point 수가 ring 수로 나누어지지 않습니다.')

    column_count = len(ring_i) // ring_count
    repeated = np.tile(unique, column_count)
    if np.array_equal(ring_i, repeated):
        scan_column = np.repeat(
            np.arange(column_count, dtype=np.int32),
            ring_count,
        )
        return scan_column, {
            'mode': 'exact_repeating_ring_sequence',
            'ring_count': ring_count,
            'column_count': column_count,
        }

    scan_column = np.full(len(ring_i), -1, dtype=np.int32)
    lengths = []
    for ring_value in unique:
        indices = np.flatnonzero(ring_i == ring_value)
        lengths.append(len(indices))
        scan_column[indices] = np.arange(len(indices), dtype=np.int32)
    if len(set(lengths)) != 1:
        raise ValueError(f'ring별 point 수가 다릅니다: {lengths}')
    return scan_column, {
        'mode': 'per_ring_original_order_fallback',
        'ring_count': ring_count,
        'column_count': lengths[0],
    }


def moving_nanmedian(values: np.ndarray, radius: int = 1) -> np.ndarray:
    result = values.copy().astype(np.float64)
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        selected = values[lo:hi]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            result[index] = np.median(selected)
    return result


def expected_support_by_range(
    range_m: np.ndarray,
    support_count: np.ndarray,
    stable_mask: np.ndarray,
    feature_mask: np.ndarray,
    max_support: float,
    bin_width_m: float = 1.0,
    max_range_m: float = 20.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    edges = np.arange(0.0, max_range_m + bin_width_m, bin_width_m)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected_bins = np.full(len(centers), np.nan, dtype=np.float64)
    source_counts = np.zeros(len(centers), dtype=np.int32)

    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = stable_mask & (range_m >= low) & (range_m < high)
        if np.count_nonzero(mask) < 30:
            mask = feature_mask & (range_m >= low) & (range_m < high)
        values = support_count[mask].astype(np.float64)
        source_counts[index] = int(values.size)
        if values.size:
            expected_bins[index] = float(np.percentile(values, 70))

    finite = np.flatnonzero(np.isfinite(expected_bins))
    if finite.size == 0:
        expected_bins[:] = max_support
    else:
        expected_bins = np.interp(
            np.arange(len(expected_bins)),
            finite,
            expected_bins[finite],
        )
    expected_bins = moving_nanmedian(expected_bins, radius=1)
    expected_bins = np.clip(expected_bins, 4.0, max_support)

    bin_indices = np.digitize(range_m, edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(expected_bins) - 1)
    per_point = expected_bins[bin_indices].astype(np.float32)

    info = {
        'method': (
            'range-bin 70th percentile of support among stable vertical '
            'surface points; fallback to all feature-range points; '
            '1-bin median smoothing'
        ),
        'bin_width_m': bin_width_m,
        'range_bin_edges_m': edges.tolist(),
        'expected_support_per_bin': expected_bins.tolist(),
        'source_point_count_per_bin': source_counts.tolist(),
        'max_support': max_support,
    }
    return per_point, info


def round_robin_select(
    candidate_indices: np.ndarray,
    score: np.ndarray,
    angle: np.ndarray,
    range_m: np.ndarray,
    quota: int,
    angle_bins: int,
    range_edges: np.ndarray,
) -> np.ndarray:
    if quota <= 0 or candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    normalized_angle = (angle[candidate_indices] + math.pi) % (2.0 * math.pi)
    angle_id = np.floor(normalized_angle / (2.0 * math.pi) * angle_bins).astype(np.int32)
    angle_id = np.clip(angle_id, 0, angle_bins - 1)
    range_id = np.digitize(range_m[candidate_indices], range_edges) - 1
    range_id = np.clip(range_id, 0, len(range_edges) - 2)

    groups: dict[tuple[int, int], list[int]] = {}
    for local_index, point_index in enumerate(candidate_indices):
        key = (int(angle_id[local_index]), int(range_id[local_index]))
        groups.setdefault(key, []).append(int(point_index))

    for values in groups.values():
        values.sort(key=lambda point_index: (-float(score[point_index]), point_index))

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


def save_scalar_plot(
    path: Path,
    points: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    title: str,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    active = mask & np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    figure, axis = plt.subplots(figsize=(11, 7))
    plot = axis.scatter(points[active, 0], points[active, 1], c=values[active], s=3, alpha=0.82)
    figure.colorbar(plot, ax=axis, label=label)
    axis.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('x [m]')
    axis.set_ylabel('y [m]')
    axis.set_title(title)
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-npz', type=Path, required=True)
    parser.add_argument('--sample-json', type=Path, required=True)
    parser.add_argument('--a1-summary-json', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview-budget', type=int, default=1024)
    parser.add_argument('--preview-angle-bins', type=int, default=36)
    args = parser.parse_args()

    input_npz = args.input_npz.expanduser().resolve()
    sample_json_path = args.sample_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)
    if not sample_json_path.is_file():
        raise FileNotFoundError(sample_json_path)
    if output_dir.exists():
        raise FileExistsError(f'기존 결과를 덮어쓰지 않습니다: {output_dir}')
    output_dir.mkdir(parents=True)

    with np.load(input_npz, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    sample_json = load_json(sample_json_path)
    a1_summary = None
    if args.a1_summary_json:
        a1_summary_path = args.a1_summary_json.expanduser().resolve()
        if a1_summary_path.is_file():
            a1_summary = load_json(a1_summary_path)

    required = {
        'raw_xyz_hesai', 'base_xyz_deskewed', 'ring', 'point_timestamp_ns',
        'valid_raw_mask', 'invalid_xyz_mask', 'feature_range_mask', 'range_m',
        'a1_1_normals', 'a1_1_feature_valid_mask', 'a1_1_verticality',
        'a1_1_plane_fit_score', 'a1_1_support_count', 'a1_1_ring_support_count',
        'a1_1_normal_agreement', 'a1_1_plane_residual_ratio',
        'a1_2_range_jump_score', 'a1_2_normal_change_score',
        'a1_2_cross_ring_edge_consensus',
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f'A1.2 NPZ 필드가 없습니다: {missing}')

    point_count = len(data['ring'])
    raw_point_index = np.arange(point_count, dtype=np.int32)
    scan_column, scan_info = build_scan_columns(data['ring'])

    raw_xyz = data['raw_xyz_hesai'].astype(np.float64)
    base_xyz = data['base_xyz_deskewed'].astype(np.float64)
    raw_xy_range = np.linalg.norm(raw_xyz[:, :2], axis=1)
    raw_range = np.linalg.norm(raw_xyz, axis=1)
    azimuth = np.arctan2(raw_xyz[:, 1], raw_xyz[:, 0])
    elevation = np.arctan2(raw_xyz[:, 2], np.maximum(raw_xy_range, 1e-12))

    reference_stamp_ns = int(sample_json['output_stamp_ns'])
    point_time_offset_sec = (
        data['point_timestamp_ns'].astype(np.int64) - reference_stamp_ns
    ).astype(np.float64) / 1e9

    height_bands = sample_json['geometry']['height_bands_m']
    low_min, low_max = map(float, height_bands['low'])
    mid_min, mid_max = map(float, height_bands['mid'])
    high_min, high_max = map(float, height_bands['high'])
    z_base = base_xyz[:, 2]
    low_mask = (z_base >= low_min) & (z_base < low_max)
    mid_mask = (z_base >= mid_min) & (z_base < mid_max)
    high_mask = (z_base >= high_min) & (z_base <= high_max)
    height_band_id = np.full(point_count, -1, dtype=np.int8)
    height_band_id[low_mask] = 0
    height_band_id[mid_mask] = 1
    height_band_id[high_mask] = 2

    normals = data['a1_1_normals'].astype(np.float64)
    normal_xy = normals[:, :2]
    normal_xy_norm = np.linalg.norm(normal_xy, axis=1)
    normal_xy_unit = np.full_like(normal_xy, np.nan)
    valid_normal_xy = np.isfinite(normal_xy_norm) & (normal_xy_norm > 1e-6)
    normal_xy_unit[valid_normal_xy] = normal_xy[valid_normal_xy] / normal_xy_norm[valid_normal_xy, None]
    normal_azimuth = np.full(point_count, np.nan, dtype=np.float32)
    normal_azimuth[valid_normal_xy] = np.arctan2(
        normal_xy_unit[valid_normal_xy, 1], normal_xy_unit[valid_normal_xy, 0]
    ).astype(np.float32)
    tangent_azimuth = np.full(point_count, np.nan, dtype=np.float32)
    tangent_azimuth[valid_normal_xy] = (
        normal_azimuth[valid_normal_xy] + np.float32(math.pi / 2.0)
    )
    tangent_azimuth[valid_normal_xy] = (
        (tangent_azimuth[valid_normal_xy] + math.pi) % (2.0 * math.pi) - math.pi
    )

    feature_mask = data['feature_range_mask'].astype(bool)
    feature_valid = data['a1_1_feature_valid_mask'].astype(bool)
    verticality = np.nan_to_num(data['a1_1_verticality'].astype(np.float64), nan=0.0)
    vertical_score = np.clip((verticality - 0.40) / 0.60, 0.0, 1.0)
    plane_fit = np.nan_to_num(data['a1_1_plane_fit_score'].astype(np.float64), nan=0.0)
    support_count = data['a1_1_support_count'].astype(np.float64)
    max_support = float(np.nanmax(support_count[feature_mask])) if np.any(feature_mask) else 45.0
    stable_surface_mask = feature_mask & feature_valid & (verticality >= 0.85) & (plane_fit >= 0.70)
    expected_support, support_info = expected_support_by_range(
        range_m=data['range_m'].astype(np.float64),
        support_count=support_count,
        stable_mask=stable_surface_mask,
        feature_mask=feature_mask,
        max_support=max_support,
    )
    support_density_normalized = np.clip(
        support_count / np.maximum(expected_support, 1.0), 0.0, 1.5
    ).astype(np.float32)
    support_deficit_score = np.clip(1.0 - support_density_normalized, 0.0, 1.0).astype(np.float32)

    range_jump = np.nan_to_num(data['a1_2_range_jump_score'].astype(np.float64), nan=0.0)
    normal_turn = np.nan_to_num(data['a1_2_normal_change_score'].astype(np.float64), nan=0.0)
    cross_ring = np.nan_to_num(data['a1_2_cross_ring_edge_consensus'].astype(np.float64), nan=0.0)
    agreement = np.nan_to_num(data['a1_1_normal_agreement'].astype(np.float64), nan=0.5)
    ring_support = data['a1_1_ring_support_count'].astype(np.float64)
    ring_score = np.clip(ring_support / 5.0, 0.0, 1.0)
    far_evidence = np.clip((data['range_m'].astype(np.float64) - 8.0) / 8.0, 0.0, 1.0)

    # 서로 배타적인 라벨이 아니라 동시에 보존하는 연속 채널이다.
    smooth_surface_score = np.clip(
        vertical_score
        * (0.48 * plane_fit + 0.20 * agreement + 0.18 * np.clip(support_density_normalized, 0.0, 1.0) + 0.14 * ring_score)
        * (1.0 - 0.30 * np.maximum(range_jump, normal_turn)),
        0.0,
        1.0,
    )
    depth_jump_edge_score = np.clip(
        vertical_score
        * (0.68 * range_jump + 0.20 * np.clip(cross_ring * 3.0, 0.0, 1.0) + 0.12 * support_deficit_score)
        * (0.55 + 0.45 * np.maximum(plane_fit, 0.35)),
        0.0,
        1.0,
    )
    normal_turn_edge_score = np.clip(
        vertical_score
        * (0.72 * normal_turn + 0.18 * range_jump + 0.10 * (1.0 - agreement))
        * (0.60 + 0.40 * np.maximum(plane_fit, 0.30)),
        0.0,
        1.0,
    )
    low_support_vertical_score = np.clip(
        vertical_score
        * support_deficit_score
        * (0.40 + 0.35 * plane_fit + 0.25 * np.maximum(range_jump, normal_turn)),
        0.0,
        1.0,
    )
    far_uncertain_score = np.clip(
        vertical_score
        * far_evidence
        * (0.35 + 0.40 * support_deficit_score + 0.25 * np.maximum(range_jump, normal_turn)),
        0.0,
        1.0,
    )
    strongest_structure = np.maximum.reduce([
        smooth_surface_score,
        depth_jump_edge_score,
        normal_turn_edge_score,
        low_support_vertical_score,
    ])
    low_structure_score = np.clip(
        0.65 * (1.0 - vertical_score)
        + 0.20 * (1.0 - plane_fit)
        + 0.15 * (1.0 - strongest_structure),
        0.0,
        1.0,
    )

    for array in [
        smooth_surface_score, depth_jump_edge_score, normal_turn_edge_score,
        low_support_vertical_score, far_uncertain_score, low_structure_score,
    ]:
        array[~feature_mask] = 0.0

    reason_flags = np.zeros(point_count, dtype=np.uint16)
    invalid_xyz = data['invalid_xyz_mask'].astype(bool)
    zero_return = np.linalg.norm(raw_xyz, axis=1) <= 1e-6
    reason_flags[invalid_xyz] |= BIT_INVALID_XYZ
    reason_flags[zero_return] |= BIT_ZERO_RETURN
    reason_flags[~feature_mask] |= BIT_OUTSIDE_FEATURE_RANGE
    reason_flags[feature_mask & (support_deficit_score >= 0.35)] |= BIT_SUPPORT_DEFICIT
    reason_flags[feature_mask & (range_jump >= 0.55)] |= BIT_HIGH_DEPTH_JUMP
    reason_flags[feature_mask & (normal_turn >= 0.35)] |= BIT_HIGH_NORMAL_TURN
    reason_flags[feature_mask & (data['a1_1_plane_residual_ratio'] >= 0.16)] |= BIT_POOR_PLANE_FIT
    reason_flags[feature_mask & np.isfinite(data['a1_1_normal_agreement']) & (agreement < 0.70)] |= BIT_NORMAL_DISAGREEMENT
    reason_flags[feature_mask & (far_evidence >= 0.50) & (support_deficit_score >= 0.20)] |= BIT_FAR_SPARSE
    reason_flags[feature_mask & ~feature_valid] |= BIT_FEATURE_UNAVAILABLE

    # 2D 투영 준비용 높이-구조 조합 mask. 실제 grid 투영은 A1.3에서 수행한다.
    projection_ready_mask = feature_mask & feature_valid & (height_band_id >= 0)
    vertical_surface_normal_mask = projection_ready_mask & (normal_xy_norm >= 0.70)
    horizontal_surface_normal_mask = projection_ready_mask & (normal_xy_norm < 0.35)

    angle = np.arctan2(base_xyz[:, 1], base_xyz[:, 0])
    range_edges = np.asarray([0.3, 3.0, 6.0, 9.0, 12.0, 20.0001], dtype=np.float64)
    preview_config = [
        ('smooth_surface', smooth_surface_score, smooth_surface_score >= 0.55, 0.45),
        ('depth_jump_edge', depth_jump_edge_score, depth_jump_edge_score >= 0.35, 0.15),
        ('normal_turn_edge', normal_turn_edge_score, normal_turn_edge_score >= 0.30, 0.15),
        ('low_support_vertical', low_support_vertical_score, low_support_vertical_score >= 0.25, 0.15),
        ('far_uncertain', far_uncertain_score, far_uncertain_score >= 0.20, 0.10),
    ]
    remaining = np.ones(point_count, dtype=bool)
    selected_parts = []
    role_parts = []
    requested_counts = {}
    actual_counts = {}
    used_total = 0
    for role_id, (name, score, candidate, fraction) in enumerate(preview_config):
        quota = int(round(args.preview_budget * fraction))
        if role_id == len(preview_config) - 1:
            quota = args.preview_budget - used_total
        requested_counts[name] = quota
        candidates = np.flatnonzero(candidate & projection_ready_mask & remaining)
        selected = round_robin_select(
            candidates, score, angle, data['range_m'].astype(np.float64), quota,
            args.preview_angle_bins, range_edges,
        )
        remaining[selected] = False
        selected_parts.append(selected)
        role_parts.append(np.full(len(selected), role_id, dtype=np.int8))
        actual_counts[name] = int(len(selected))
        used_total += len(selected)

    preview_indices = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    preview_roles_compact = np.concatenate(role_parts) if role_parts else np.empty(0, dtype=np.int8)
    preview_mask = np.zeros(point_count, dtype=bool)
    preview_role = np.full(point_count, -1, dtype=np.int8)
    preview_mask[preview_indices] = True
    preview_role[preview_indices] = preview_roles_compact

    enriched = {
        'a1_2_1_raw_point_index': raw_point_index,
        'a1_2_1_scan_column': scan_column,
        'a1_2_1_point_time_offset_sec': point_time_offset_sec.astype(np.float32),
        'a1_2_1_raw_range_m': raw_range.astype(np.float32),
        'a1_2_1_azimuth_rad': azimuth.astype(np.float32),
        'a1_2_1_elevation_rad': elevation.astype(np.float32),
        'a1_2_1_z_base_m': z_base.astype(np.float32),
        'a1_2_1_height_band_id': height_band_id,
        'a1_2_1_low_mask': low_mask,
        'a1_2_1_mid_mask': mid_mask,
        'a1_2_1_high_mask': high_mask,
        'a1_2_1_normal_xy': normal_xy.astype(np.float32),
        'a1_2_1_normal_xy_unit': normal_xy_unit.astype(np.float32),
        'a1_2_1_normal_xy_norm': normal_xy_norm.astype(np.float32),
        'a1_2_1_normal_azimuth_rad': normal_azimuth,
        'a1_2_1_wall_tangent_azimuth_rad': tangent_azimuth,
        'a1_2_1_vertical_surface_normal_mask': vertical_surface_normal_mask,
        'a1_2_1_horizontal_surface_normal_mask': horizontal_surface_normal_mask,
        'a1_2_1_expected_support_by_range': expected_support.astype(np.float32),
        'a1_2_1_support_density_normalized': support_density_normalized,
        'a1_2_1_support_deficit_score': support_deficit_score,
        'a1_2_1_smooth_surface_score': smooth_surface_score.astype(np.float32),
        'a1_2_1_depth_jump_edge_score': depth_jump_edge_score.astype(np.float32),
        'a1_2_1_normal_turn_edge_score': normal_turn_edge_score.astype(np.float32),
        'a1_2_1_low_support_vertical_score': low_support_vertical_score.astype(np.float32),
        'a1_2_1_far_uncertain_score': far_uncertain_score.astype(np.float32),
        'a1_2_1_low_structure_score': low_structure_score.astype(np.float32),
        'a1_2_1_reason_flags': reason_flags,
        'a1_2_1_projection_ready_mask': projection_ready_mask,
        'a1_2_1_balanced_preview_mask': preview_mask,
        'a1_2_1_balanced_preview_role': preview_role,
        'a1_2_1_balanced_preview_indices': preview_indices.astype(np.int64),
    }
    data.update(enriched)
    np.savez_compressed(output_dir / 'a1_2_1_enriched_geometry.npz', **data)

    channel_rows = []
    for name, key in [
        ('smooth_surface', 'a1_2_1_smooth_surface_score'),
        ('depth_jump_edge', 'a1_2_1_depth_jump_edge_score'),
        ('normal_turn_edge', 'a1_2_1_normal_turn_edge_score'),
        ('low_support_vertical', 'a1_2_1_low_support_vertical_score'),
        ('far_uncertain', 'a1_2_1_far_uncertain_score'),
        ('low_structure', 'a1_2_1_low_structure_score'),
    ]:
        values = data[key][feature_mask]
        channel_rows.append({
            'channel': name,
            'point_count': int(values.size),
            'mean': float(np.mean(values)),
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
            'count_ge_0p25': int(np.count_nonzero(values >= 0.25)),
            'count_ge_0p50': int(np.count_nonzero(values >= 0.50)),
        })
    with (output_dir / 'channel_statistics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(channel_rows[0]))
        writer.writeheader()
        writer.writerows(channel_rows)

    points = base_xyz
    save_scalar_plot(output_dir / '01_smooth_surface.png', points, smooth_surface_score, feature_mask, 'A1.2.1 smooth vertical surface', 'score')
    save_scalar_plot(output_dir / '02_depth_jump_edge.png', points, depth_jump_edge_score, feature_mask, 'A1.2.1 depth-jump edge', 'score')
    save_scalar_plot(output_dir / '03_normal_turn_edge.png', points, normal_turn_edge_score, feature_mask, 'A1.2.1 normal-turn edge', 'score')
    save_scalar_plot(output_dir / '04_low_support_vertical.png', points, low_support_vertical_score, feature_mask, 'A1.2.1 low-support vertical structure', 'score')
    save_scalar_plot(output_dir / '05_far_uncertain.png', points, far_uncertain_score, feature_mask, 'A1.2.1 far uncertain structure', 'score')
    save_scalar_plot(output_dir / '06_support_density_normalized.png', points, support_density_normalized, feature_mask, 'A1.2.1 range-normalized support density', 'normalized support')
    save_scalar_plot(output_dir / '07_height_band_id.png', points, height_band_id.astype(np.float32), projection_ready_mask, 'A1.2.1 V2.2-compatible height bands', '0=LOW, 1=MID, 2=HIGH')

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    active = preview_mask
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.scatter(points[projection_ready_mask, 0], points[projection_ready_mask, 1], s=2, alpha=0.06)
    plot = axis.scatter(
        points[active, 0], points[active, 1], c=preview_role[active],
        cmap=ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']),
        vmin=-0.5, vmax=4.5, s=8, alpha=0.95,
    )
    colorbar = figure.colorbar(plot, ax=axis, ticks=[0, 1, 2, 3, 4])
    colorbar.ax.set_yticklabels([ROLE_NAMES[index] for index in range(5)])
    axis.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('x [m]')
    axis.set_ylabel('y [m]')
    axis.set_title('A1.2.1 balanced feature preview — not connected to MCL')
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / '08_balanced_preview.png', dpi=180)
    plt.close(figure)

    reason_definition = {
        'bit_0': 'invalid XYZ', 'bit_1': 'zero return',
        'bit_2': 'outside feature range', 'bit_3': 'range-normalized support deficit',
        'bit_4': 'high depth jump', 'bit_5': 'high normal turn',
        'bit_6': 'poor normalized plane fit', 'bit_7': 'normal disagreement',
        'bit_8': 'far sparse observation', 'bit_9': 'feature fit unavailable',
    }
    source_scan = None
    if a1_summary:
        source_scan = a1_summary.get('source', {}).get('selected_scan')
    summary = {
        'schema_version': 'a1_2_1_enriched_geometry_v0_1',
        'source_npz': str(input_npz),
        'source_npz_sha256': sha256_file(input_npz),
        'sample_json': str(sample_json_path),
        'sample_json_sha256': sha256_file(sample_json_path),
        'sample_id': sample_json.get('sample_id'),
        'reference_stamp_ns': reference_stamp_ns,
        'source_scan': source_scan,
        'scan_structure': scan_info,
        'height_bands_m': height_bands,
        'grid_geometry_for_next_stage': sample_json['geometry'],
        'support_normalization': support_info,
        'counts': {
            'raw_points': point_count,
            'feature_range_points': int(np.count_nonzero(feature_mask)),
            'projection_ready_points': int(np.count_nonzero(projection_ready_mask)),
            'low_points': int(np.count_nonzero(projection_ready_mask & low_mask)),
            'mid_points': int(np.count_nonzero(projection_ready_mask & mid_mask)),
            'high_points': int(np.count_nonzero(projection_ready_mask & high_mask)),
            'preview_points': int(len(preview_indices)),
        },
        'preview_requested_counts': requested_counts,
        'preview_actual_counts': actual_counts,
        'reason_flag_definition': reason_definition,
        'important_notes': [
            'A1.2의 모든 배열을 보존했다.',
            'LOW/MID/HIGH는 V2.2 sample.json의 실제 높이 구간을 사용했다.',
            '구조 채널은 상호 배타적인 의미론 라벨이 아니다.',
            '실제 2D grid 투영과 셀 집계는 A1.3에서 수행한다.',
            'MCL 점수와 V1.7 baseline은 변경하지 않았다.',
            '원본 rosbag 또는 기존 결과를 덮어쓰지 않았다.',
        ],
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest = {
        key: {'shape': list(np.asarray(value).shape), 'dtype': str(np.asarray(value).dtype)}
        for key, value in sorted(enriched.items())
    }
    (output_dir / 'schema_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    report = '\n'.join([
        '===== A1.2.1 Enriched Geometry Prototype =====',
        f'source: {input_npz}',
        f'sample: {sample_json.get("sample_id")}',
        f'raw points: {point_count}',
        f'feature range: {int(np.count_nonzero(feature_mask))}',
        f'projection ready: {int(np.count_nonzero(projection_ready_mask))}',
        f'LOW/MID/HIGH: {int(np.count_nonzero(projection_ready_mask & low_mask))} / {int(np.count_nonzero(projection_ready_mask & mid_mask))} / {int(np.count_nonzero(projection_ready_mask & high_mask))}',
        f'balanced preview: {len(preview_indices)}',
        '',
        'No existing dataset/result was overwritten.',
        'No point was permanently deleted.',
        'No 2D grid projection was performed yet.',
        'No MCL score was changed.',
        f'output: {output_dir}',
    ])
    (output_dir / 'run_report.txt').write_text(report + '\n', encoding='utf-8')
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
