#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import PointCloud2, PointField
import os as _os
import sys as _sys

# 2026-09-11: 점별 for-loop 대신 배치 커널(_A131_VEC/a131_vec.py). A131_VEC=0 이면 원본 루프.
_A131_VEC = _os.environ.get("A131_VEC", "1") == "1"
if _A131_VEC:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_A131_VEC"))
    import a131_vec as _a131_vec


POINTFIELD_DTYPES = {
    PointField.INT8: ("int8", 1),
    PointField.UINT8: ("uint8", 1),
    PointField.INT16: ("int16", 2),
    PointField.UINT16: ("uint16", 2),
    PointField.INT32: ("int32", 4),
    PointField.UINT32: ("uint32", 4),
    PointField.FLOAT32: ("float32", 4),
    PointField.FLOAT64: ("float64", 8),
}


def norm_frame(name: str) -> str:
    return name.strip().lstrip("/")


def detect_storage_id(bag_dir: Path) -> str:
    if list(bag_dir.glob("*.mcap")):
        return "mcap"
    if list(bag_dir.glob("*.db3")):
        return "sqlite3"

    metadata = bag_dir / "metadata.yaml"
    if metadata.is_file():
        obj = yaml.safe_load(metadata.read_text(encoding="utf-8"))
        info = obj.get("rosbag2_bagfile_information", obj)
        value = info.get("storage_identifier")
        if value:
            return str(value)

    raise RuntimeError(f"rosbag storage id를 찾지 못했습니다: {bag_dir}")


def make_reader(bag_dir: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_dir),
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def try_seek(reader: rosbag2_py.SequentialReader, stamp_ns: int) -> bool:
    try:
        reader.seek(int(stamp_ns))
        return True
    except Exception:
        return False


def pointcloud_dtype(msg: PointCloud2) -> np.dtype:
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    endian = ">" if msg.is_bigendian else "<"

    for field in msg.fields:
        if field.datatype not in POINTFIELD_DTYPES:
            raise ValueError(
                f"지원하지 않는 PointField: {field.name}, datatype={field.datatype}"
            )
        base_name, _ = POINTFIELD_DTYPES[field.datatype]
        base_dtype = np.dtype(endian + np.dtype(base_name).str[1:])
        names.append(str(field.name))
        formats.append(
            base_dtype if int(field.count) == 1
            else (base_dtype, (int(field.count),))
        )
        offsets.append(int(field.offset))

    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(msg.point_step),
        }
    )


def decode_cloud(msg: PointCloud2) -> dict[str, np.ndarray]:
    dtype = pointcloud_dtype(msg)
    width = int(msg.width)
    height = int(msg.height)
    expected_row = int(msg.point_step) * width
    raw = memoryview(msg.data)

    if int(msg.row_step) == expected_row:
        cloud = np.frombuffer(
            raw,
            dtype=dtype,
            count=width * height,
        ).copy()
    else:
        rows = []
        for row_idx in range(height):
            start = row_idx * int(msg.row_step)
            stop = start + expected_row
            rows.append(
                np.frombuffer(
                    raw[start:stop],
                    dtype=dtype,
                    count=width,
                ).copy()
            )
        cloud = np.concatenate(rows)

    return {name: np.asarray(cloud[name]) for name in cloud.dtype.names or []}


def stamp_ns_from_header(msg: Any) -> int:
    return (
        int(msg.header.stamp.sec) * 1_000_000_000
        + int(msg.header.stamp.nanosec)
    )


def transform_to_matrix(transform: Any) -> np.ndarray:
    q = np.array(
        [
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        ],
        dtype=np.float64,
    )
    t = np.array(
        [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(q).as_matrix()
    matrix[:3, 3] = t
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def direct_transform_matrix(
    transform_stamped: Any,
    parent_frame: str,
    child_frame: str,
) -> np.ndarray | None:
    parent = norm_frame(transform_stamped.header.frame_id)
    child = norm_frame(transform_stamped.child_frame_id)
    wanted_parent = norm_frame(parent_frame)
    wanted_child = norm_frame(child_frame)

    matrix = transform_to_matrix(transform_stamped.transform)

    if parent == wanted_parent and child == wanted_child:
        return matrix
    if parent == wanted_child and child == wanted_parent:
        return invert_transform(matrix)
    return None


@dataclass
class TimedPose:
    stamp_ns: int
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


class PoseInterpolator:
    DEFAULT_BOUNDARY_TOLERANCE_NS = 750_000_000

    def __init__(
        self,
        poses: list[TimedPose],
        boundary_tolerance_ns: int = DEFAULT_BOUNDARY_TOLERANCE_NS,
    ) -> None:
        if len(poses) < 2:
            raise ValueError("동적 TF 보간에는 최소 2개 pose가 필요합니다.")
        if boundary_tolerance_ns < 0:
            raise ValueError("TF boundary tolerance는 음수일 수 없습니다.")
        self.boundary_tolerance_ns = int(boundary_tolerance_ns)

        poses = sorted(poses, key=lambda item: item.stamp_ns)

        unique: dict[int, TimedPose] = {}
        for pose in poses:
            unique[int(pose.stamp_ns)] = pose
        poses = [unique[key] for key in sorted(unique)]

        if len(poses) < 2:
            raise ValueError("중복 제거 후 동적 TF pose가 2개 미만입니다.")

        self.stamps_ns = np.array([p.stamp_ns for p in poses], dtype=np.int64)
        self.times_sec = (
            self.stamps_ns - self.stamps_ns[0]
        ).astype(np.float64) / 1e9
        self.translations = np.stack([p.translation for p in poses])
        rotations = Rotation.from_quat(
            np.stack([p.quaternion_xyzw for p in poses])
        )
        self.slerp = Slerp(self.times_sec, rotations)

    @property
    def min_stamp_ns(self) -> int:
        return int(self.stamps_ns[0])

    @property
    def max_stamp_ns(self) -> int:
        return int(self.stamps_ns[-1])

    def resolve_stamp(self, stamp_ns: int) -> tuple[int, int]:
        stamp_ns = int(stamp_ns)
        if stamp_ns < self.min_stamp_ns:
            gap_ns = self.min_stamp_ns - stamp_ns
            if gap_ns > self.boundary_tolerance_ns:
                raise ValueError(
                    "TF 보간 범위를 벗어났습니다: "
                    f"{stamp_ns}, available=[{self.min_stamp_ns}, {self.max_stamp_ns}], "
                    f"boundary_tolerance_ns={self.boundary_tolerance_ns}"
                )
            return self.min_stamp_ns, int(gap_ns)
        if stamp_ns > self.max_stamp_ns:
            gap_ns = stamp_ns - self.max_stamp_ns
            if gap_ns > self.boundary_tolerance_ns:
                raise ValueError(
                    "TF 보간 범위를 벗어났습니다: "
                    f"{stamp_ns}, available=[{self.min_stamp_ns}, {self.max_stamp_ns}], "
                    f"boundary_tolerance_ns={self.boundary_tolerance_ns}"
                )
            return self.max_stamp_ns, int(gap_ns)
        return stamp_ns, 0

    def matrix_at(self, stamp_ns: int) -> np.ndarray:
        resolved_stamp_ns, _ = self.resolve_stamp(stamp_ns)
        stamp_ns = resolved_stamp_ns

        idx = int(np.searchsorted(self.stamps_ns, stamp_ns, side="left"))
        if idx < len(self.stamps_ns) and self.stamps_ns[idx] == stamp_ns:
            translation = self.translations[idx]
            rotation = self.slerp(
                np.array([(stamp_ns - self.stamps_ns[0]) / 1e9])
            ).as_matrix()[0]
        else:
            right = min(max(idx, 1), len(self.stamps_ns) - 1)
            left = right - 1
            t0 = self.stamps_ns[left]
            t1 = self.stamps_ns[right]
            alpha = float(stamp_ns - t0) / float(t1 - t0)
            translation = (
                (1.0 - alpha) * self.translations[left]
                + alpha * self.translations[right]
            )
            rotation = self.slerp(
                np.array([(stamp_ns - self.stamps_ns[0]) / 1e9])
            ).as_matrix()[0]

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix


def read_static_transform(
    bag_dir: Path,
    storage_id: str,
    topic_type: str,
    parent_frame: str,
    child_frame: str,
) -> np.ndarray:
    reader = make_reader(bag_dir, storage_id)
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf_static"]))
    msg_type = get_message(topic_type)

    while reader.has_next():
        _, serialized, _ = reader.read_next()
        msg = deserialize_message(serialized, msg_type)
        for transform in msg.transforms:
            matrix = direct_transform_matrix(
                transform,
                parent_frame=parent_frame,
                child_frame=child_frame,
            )
            if matrix is not None:
                return matrix

    raise RuntimeError(
        "tf_static에서 직접 transform을 찾지 못했습니다: "
        f"{parent_frame} <- {child_frame}"
    )


def read_dynamic_poses(
    bag_dir: Path,
    storage_id: str,
    topic_type: str,
    parent_frame: str,
    child_frame: str,
    start_ns: int,
    end_ns: int,
    seek_margin_ns: int = 2_000_000_000,
) -> list[TimedPose]:
    reader = make_reader(bag_dir, storage_id)
    try_seek(reader, max(0, start_ns - seek_margin_ns))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf"]))
    msg_type = get_message(topic_type)

    poses: list[TimedPose] = []
    passed_end = False

    while reader.has_next():
        _, serialized, bag_stamp_ns = reader.read_next()
        if bag_stamp_ns > end_ns + seek_margin_ns and poses:
            break

        msg = deserialize_message(serialized, msg_type)
        for transform in msg.transforms:
            matrix = direct_transform_matrix(
                transform,
                parent_frame=parent_frame,
                child_frame=child_frame,
            )
            if matrix is None:
                continue

            stamp_ns = stamp_ns_from_header(transform)
            if stamp_ns < start_ns - seek_margin_ns:
                continue
            if stamp_ns > end_ns + seek_margin_ns:
                passed_end = True
                continue

            rotation = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            poses.append(
                TimedPose(
                    stamp_ns=stamp_ns,
                    translation=matrix[:3, 3].copy(),
                    quaternion_xyzw=rotation,
                )
            )

        if passed_end and poses:
            break

    return poses


def read_candidate_clouds(
    bag_dir: Path,
    storage_id: str,
    topic: str,
    topic_type: str,
    target_stamp_ns: int,
    search_margin_sec: float,
) -> list[tuple[PointCloud2, dict[str, np.ndarray], int]]:
    margin_ns = int(search_margin_sec * 1e9)
    reader = make_reader(bag_dir, storage_id)
    try_seek(reader, max(0, target_stamp_ns - margin_ns))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    msg_type = get_message(topic_type)

    candidates = []
    while reader.has_next():
        read_topic, serialized, bag_stamp_ns = reader.read_next()
        if read_topic != topic:
            continue

        msg = deserialize_message(serialized, msg_type)
        header_stamp_ns = stamp_ns_from_header(msg)

        if header_stamp_ns < target_stamp_ns - margin_ns:
            continue
        if header_stamp_ns > target_stamp_ns + margin_ns:
            break

        arrays = decode_cloud(msg)
        candidates.append((msg, arrays, int(bag_stamp_ns)))

    if not candidates:
        raise RuntimeError("target 주변 PointCloud2 후보를 읽지 못했습니다.")
    return candidates


def select_scan(
    candidates: list[tuple[PointCloud2, dict[str, np.ndarray], int]],
    target_stamp_ns: int,
    timestamp_field: str,
) -> tuple[PointCloud2, dict[str, np.ndarray], int, dict[str, Any]]:
    scored = []
    target_sec = target_stamp_ns / 1e9

    for msg, arrays, bag_stamp_ns in candidates:
        if timestamp_field not in arrays:
            raise KeyError(f"point timestamp field가 없습니다: {timestamp_field}")

        point_times = np.asarray(arrays[timestamp_field], dtype=np.float64)
        finite = np.isfinite(point_times)
        if not finite.any():
            continue

        tmin = float(np.min(point_times[finite]))
        tmax = float(np.max(point_times[finite]))
        midpoint = 0.5 * (tmin + tmax)
        contains = tmin <= target_sec <= tmax

        if contains:
            score = abs(midpoint - target_sec)
            priority = 0
        else:
            score = min(abs(tmin - target_sec), abs(tmax - target_sec))
            priority = 1

        scored.append(
            (
                priority,
                score,
                msg,
                arrays,
                bag_stamp_ns,
                {
                    "point_time_min_sec": tmin,
                    "point_time_max_sec": tmax,
                    "point_time_span_sec": tmax - tmin,
                    "contains_reference_time": contains,
                    "header_stamp_ns": stamp_ns_from_header(msg),
                    "bag_record_stamp_ns": int(bag_stamp_ns),
                },
            )
        )

    if not scored:
        raise RuntimeError("유효한 point timestamp를 가진 scan이 없습니다.")

    scored.sort(key=lambda item: (item[0], item[1]))
    _, _, msg, arrays, bag_stamp_ns, info = scored[0]
    return msg, arrays, bag_stamp_ns, info


def build_ring_grid(ring: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ring = np.asarray(ring, dtype=np.int64)
    unique_rings = np.unique(ring)

    if unique_rings.size < 2:
        raise ValueError("ring 종류가 2개 미만입니다.")

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
    repeated_pattern = np.tile(unique_rings, column_count)

    if np.array_equal(ring, repeated_pattern):
        index_grid = np.arange(len(ring), dtype=np.int64).reshape(
            column_count,
            ring_count,
        ).T
        return index_grid, {
            "mode": "exact_repeating_ring_sequence",
            "ring_count": ring_count,
            "column_count": column_count,
        }

    # Fallback: 각 ring을 원본 순서대로 나열한다.
    per_ring = [np.flatnonzero(ring == value) for value in unique_rings]
    lengths = [len(indices) for indices in per_ring]
    if len(set(lengths)) != 1:
        raise ValueError(
            "ring별 point 수가 달라 fallback grid를 만들 수 없습니다: "
            f"{lengths}"
        )

    index_grid = np.stack(per_ring, axis=0)
    return index_grid, {
        "mode": "per_ring_original_order_fallback",
        "ring_count": ring_count,
        "column_count": lengths[0],
    }


def safe_eigen_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(points) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    normal = eigenvectors[:, -1]
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return np.full(3, np.nan), np.full(3, np.nan)
    normal = normal / norm
    return eigenvalues, normal


def compute_structured_features(
    points_base: np.ndarray,
    ring: np.ndarray,
    valid_mask: np.ndarray,
    range_m: np.ndarray,
    ring_radius: int,
    column_radius: int,
    min_neighbors: int,
    range_jump_base_m: float,
    range_jump_ratio: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    index_grid, grid_info = build_ring_grid(ring)
    ring_count, column_count = index_grid.shape

    point_count = len(points_base)
    normals = np.full((point_count, 3), np.nan, dtype=np.float32)
    structured_normals = np.full((point_count, 3), np.nan, dtype=np.float32)
    eigenvalues_all = np.full((point_count, 3), np.nan, dtype=np.float32)
    planarity = np.full(point_count, np.nan, dtype=np.float32)
    linearity = np.full(point_count, np.nan, dtype=np.float32)
    scattering = np.full(point_count, np.nan, dtype=np.float32)
    verticality = np.full(point_count, np.nan, dtype=np.float32)
    support_radius = np.full(point_count, np.nan, dtype=np.float32)
    neighbor_count = np.zeros(point_count, dtype=np.uint16)
    ring_support_count = np.zeros(point_count, dtype=np.uint8)
    normal_agreement = np.full(point_count, np.nan, dtype=np.float32)

    reverse_ring = np.full(point_count, -1, dtype=np.int32)
    reverse_col = np.full(point_count, -1, dtype=np.int32)
    for r in range(ring_count):
        reverse_ring[index_grid[r]] = r
        reverse_col[index_grid[r]] = np.arange(column_count)

    offsets = [
        (dr, dc)
        for dr in range(-ring_radius, ring_radius + 1)
        for dc in range(-column_radius, column_radius + 1)
        if not (dr == 0 and dc == 0)
    ]

    if _A131_VEC:
        _a131_vec.a1_loop_vec(
            points_base, valid_mask, range_m, index_grid, ring_radius, column_radius,
            min_neighbors, range_jump_base_m, range_jump_ratio,
            {
                "normals": normals, "structured_normals": structured_normals,
                "eigenvalues": eigenvalues_all, "planarity": planarity,
                "linearity": linearity, "scattering": scattering,
                "verticality": verticality, "support_radius": support_radius,
                "neighbor_count": neighbor_count, "ring_support_count": ring_support_count,
                "normal_agreement": normal_agreement,
            },
        )

    for idx in (() if _A131_VEC else np.flatnonzero(valid_mask)):
        r = int(reverse_ring[idx])
        c = int(reverse_col[idx])
        if r < 0 or c < 0:
            continue

        center = points_base[idx]
        center_range = float(range_m[idx])
        jump_limit = range_jump_base_m + range_jump_ratio * center_range

        neighbor_indices = []
        neighbor_rings = set()

        for dr, dc in offsets:
            rr = r + dr
            if rr < 0 or rr >= ring_count:
                continue
            cc = (c + dc) % column_count
            n_idx = int(index_grid[rr, cc])
            if not valid_mask[n_idx]:
                continue
            if abs(float(range_m[n_idx]) - center_range) > jump_limit:
                continue

            neighbor_indices.append(n_idx)
            neighbor_rings.add(rr)

        if len(neighbor_indices) + 1 < min_neighbors:
            continue

        support_indices = np.array([idx, *neighbor_indices], dtype=np.int64)
        support_points = points_base[support_indices]
        eigenvalues, normal = safe_eigen_features(support_points)

        if not np.all(np.isfinite(eigenvalues)):
            continue

        # normal 부호를 센서 원점 쪽으로 통일한다.
        if np.dot(normal, center) > 0.0:
            normal = -normal

        l1, l2, l3 = map(float, eigenvalues)
        denom = max(l1, 1e-12)

        normals[idx] = normal.astype(np.float32)
        eigenvalues_all[idx] = eigenvalues.astype(np.float32)
        linearity[idx] = (l1 - l2) / denom
        planarity[idx] = (l2 - l3) / denom
        scattering[idx] = l3 / denom
        verticality[idx] = 1.0 - abs(float(normal[2]))
        support_radius[idx] = float(
            np.max(np.linalg.norm(support_points - center, axis=1))
        )
        neighbor_count[idx] = len(support_indices)
        ring_support_count[idx] = len(neighbor_rings | {r})

        # 즉시 좌우/상하 구조로 만든 보조 normal.
        left = int(index_grid[r, (c - 1) % column_count])
        right = int(index_grid[r, (c + 1) % column_count])
        lower = int(index_grid[max(0, r - 1), c])
        upper = int(index_grid[min(ring_count - 1, r + 1), c])

        candidate_ids = [left, right, lower, upper]
        if all(valid_mask[n] for n in candidate_ids):
            if all(
                abs(float(range_m[n]) - center_range) <= jump_limit
                for n in candidate_ids
            ):
                horizontal = points_base[right] - points_base[left]
                vertical = points_base[upper] - points_base[lower]
                structured = np.cross(horizontal, vertical)
                structured_norm = np.linalg.norm(structured)
                if structured_norm > 1e-12:
                    structured /= structured_norm
                    if np.dot(structured, center) > 0.0:
                        structured = -structured
                    structured_normals[idx] = structured.astype(np.float32)
                    normal_agreement[idx] = abs(float(np.dot(normal, structured)))

    feature_valid = np.isfinite(planarity) & np.isfinite(verticality)

    # 이 값은 최종 wall 확정값이 아니라 RViz/감사용 연속 prototype score다.
    vertical_score = np.clip((verticality - 0.35) / 0.65, 0.0, 1.0)
    planar_score = np.clip(planarity / 0.70, 0.0, 1.0)
    support_score = np.clip(
        neighbor_count.astype(np.float32) / max(float(min_neighbors * 2), 1.0),
        0.0,
        1.0,
    )
    ring_score = np.clip(
        ring_support_count.astype(np.float32) / 3.0,
        0.0,
        1.0,
    )
    agreement_score = np.where(
        np.isfinite(normal_agreement),
        np.clip(normal_agreement, 0.0, 1.0),
        0.5,
    )

    expected_radius = 0.10 + 0.025 * range_m
    excess = np.maximum(support_radius - expected_radius, 0.0)
    radius_score = np.exp(
        -excess / np.maximum(expected_radius, 0.05)
    )
    radius_score[~np.isfinite(radius_score)] = 0.0

    vertical_confidence = (
        0.30 * vertical_score
        + 0.25 * planar_score
        + 0.15 * support_score
        + 0.10 * ring_score
        + 0.10 * agreement_score
        + 0.10 * radius_score
    )
    vertical_confidence[~feature_valid] = 0.0

    finite_conf = vertical_confidence[feature_valid]
    q70 = float(np.quantile(finite_conf, 0.70)) if finite_conf.size else 1.0
    q85 = float(np.quantile(finite_conf, 0.85)) if finite_conf.size else 1.0

    result = {
        "normals": normals,
        "structured_normals": structured_normals,
        "eigenvalues": eigenvalues_all,
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "verticality": verticality,
        "support_radius_m": support_radius,
        "neighbor_count": neighbor_count,
        "ring_support_count": ring_support_count,
        "normal_agreement": normal_agreement,
        "feature_valid_mask": feature_valid,
        "vertical_confidence_prototype": vertical_confidence.astype(np.float32),
        "high_confidence_q70_mask": feature_valid & (vertical_confidence >= q70),
        "high_confidence_q85_mask": feature_valid & (vertical_confidence >= q85),
    }
    info = {
        **grid_info,
        "ring_radius": ring_radius,
        "column_radius": column_radius,
        "min_neighbors": min_neighbors,
        "range_jump_base_m": range_jump_base_m,
        "range_jump_ratio": range_jump_ratio,
        "q70_threshold": q70,
        "q85_threshold": q85,
        "note": (
            "vertical_confidence_prototype은 A1 시각화용 연속 점수이며 "
            "최종 wall mask나 MCL likelihood가 아닙니다."
        ),
    }
    return result, info


def save_pngs(
    output_dir: Path,
    points_base: np.ndarray,
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    features: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    def scatter_plot(
        filename: str,
        values: np.ndarray,
        title: str,
        label: str,
        mask: np.ndarray | None = None,
    ) -> None:
        active = valid_mask.copy()
        if mask is not None:
            active &= mask
        active &= np.isfinite(values)

        fig, ax = plt.subplots(figsize=(11, 7))
        plot = ax.scatter(
            points_base[active, 0],
            points_base[active, 1],
            c=values[active],
            s=3,
            alpha=0.8,
        )
        fig.colorbar(plot, ax=ax, label=label)
        ax.scatter([0.0], [0.0], marker="x", s=80, linewidths=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    scatter_plot(
        "01_range_topdown.png",
        range_m,
        "A1 base_link deskewed cloud by range",
        "range [m]",
    )
    scatter_plot(
        "02_planarity_topdown.png",
        features["planarity"],
        "A1 structured-neighborhood planarity",
        "planarity",
    )
    scatter_plot(
        "03_verticality_topdown.png",
        features["verticality"],
        "A1 verticality",
        "verticality",
    )
    scatter_plot(
        "04_vertical_confidence_topdown.png",
        features["vertical_confidence_prototype"],
        "A1 prototype vertical confidence",
        "prototype confidence",
    )
    scatter_plot(
        "05_support_radius_topdown.png",
        features["support_radius_m"],
        "A1 support radius",
        "support radius [m]",
    )
    scatter_plot(
        "06_normal_agreement_topdown.png",
        features["normal_agreement"],
        "A1 PCA vs structured normal agreement",
        "|dot(n_pca, n_structured)|",
    )

    mask = features["high_confidence_q70_mask"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        points_base[valid_mask & ~mask, 0],
        points_base[valid_mask & ~mask, 1],
        s=2,
        alpha=0.18,
        label="other valid points",
    )
    ax.scatter(
        points_base[mask, 0],
        points_base[mask, 1],
        s=4,
        alpha=0.8,
        label="top 30% prototype vertical confidence",
    )
    ax.scatter([0.0], [0.0], marker="x", s=80, linewidths=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("A1 diagnostic high-confidence vertical points")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_high_confidence_q70.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "원본 Hesai XT16 PointCloud2 한 스캔을 선택하고, "
            "실제 TF와 point timestamp를 사용해 base_link 기준 deskew 및 "
            "ring-structured 수직 구조 feature를 생성합니다."
        )
    )
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--topic", default="/lidar_points")
    parser.add_argument("--target-stamp-ns", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="hesai_lidar")
    parser.add_argument("--timestamp-field", default="timestamp")
    parser.add_argument("--ring-field", default="ring")
    parser.add_argument("--intensity-field", default="intensity")
    parser.add_argument("--search-margin-sec", type=float, default=0.30)
    parser.add_argument("--tf-boundary-tolerance-sec", type=float, default=0.75)
    parser.add_argument("--min-range-m", type=float, default=0.30)
    parser.add_argument("--feature-max-range-m", type=float, default=20.0)
    parser.add_argument("--ring-radius", type=int, default=1)
    parser.add_argument("--column-radius", type=int, default=2)
    parser.add_argument("--min-neighbors", type=int, default=6)
    parser.add_argument("--range-jump-base-m", type=float, default=0.20)
    parser.add_argument("--range-jump-ratio", type=float, default=0.04)
    args = parser.parse_args()

    bag_dir = args.bag.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not bag_dir.is_dir():
        raise FileNotFoundError(f"bag 폴더가 없습니다: {bag_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"출력 폴더가 이미 존재합니다. 덮어쓰지 않습니다: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    storage_id = detect_storage_id(bag_dir)

    topic_reader = make_reader(bag_dir, storage_id)
    topic_types = {
        info.name: info.type
        for info in topic_reader.get_all_topics_and_types()
    }

    required_topics = [args.topic, "/tf", "/tf_static"]
    missing = [topic for topic in required_topics if topic not in topic_types]
    if missing:
        raise KeyError(f"필수 topic이 없습니다: {missing}")

    candidates = read_candidate_clouds(
        bag_dir=bag_dir,
        storage_id=storage_id,
        topic=args.topic,
        topic_type=topic_types[args.topic],
        target_stamp_ns=args.target_stamp_ns,
        search_margin_sec=args.search_margin_sec,
    )
    cloud_msg, arrays, bag_stamp_ns, scan_info = select_scan(
        candidates=candidates,
        target_stamp_ns=args.target_stamp_ns,
        timestamp_field=args.timestamp_field,
    )

    required_fields = {
        "x",
        "y",
        "z",
        args.intensity_field,
        args.ring_field,
        args.timestamp_field,
    }
    missing_fields = sorted(required_fields - set(arrays))
    if missing_fields:
        raise KeyError(f"PointCloud2 필드가 없습니다: {missing_fields}")

    raw_xyz = np.column_stack(
        [
            arrays["x"],
            arrays["y"],
            arrays["z"],
        ]
    ).astype(np.float64)
    intensity = np.asarray(
        arrays[args.intensity_field],
        dtype=np.float32,
    )
    ring = np.asarray(
        arrays[args.ring_field],
        dtype=np.uint16,
    )
    point_time_sec = np.asarray(
        arrays[args.timestamp_field],
        dtype=np.float64,
    )
    point_time_ns = np.rint(point_time_sec * 1e9).astype(np.int64)

    finite_xyz = np.all(np.isfinite(raw_xyz), axis=1)
    nonzero_xyz = np.linalg.norm(raw_xyz, axis=1) > 1e-6
    finite_time = np.isfinite(point_time_sec)
    valid_raw = finite_xyz & nonzero_xyz & finite_time

    if not valid_raw.any():
        raise RuntimeError("유효한 XYZ/timestamp 포인트가 없습니다.")

    selected_time_min = int(np.min(point_time_ns[valid_raw]))
    selected_time_max = int(np.max(point_time_ns[valid_raw]))
    tf_start = min(selected_time_min, args.target_stamp_ns)
    tf_end = max(selected_time_max, args.target_stamp_ns)

    static_matrix = read_static_transform(
        bag_dir=bag_dir,
        storage_id=storage_id,
        topic_type=topic_types["/tf_static"],
        parent_frame=args.base_frame,
        child_frame=args.lidar_frame,
    )

    poses = read_dynamic_poses(
        bag_dir=bag_dir,
        storage_id=storage_id,
        topic_type=topic_types["/tf"],
        parent_frame=args.world_frame,
        child_frame=args.base_frame,
        start_ns=tf_start,
        end_ns=tf_end,
    )
    tf_boundary_tolerance_ns = int(round(args.tf_boundary_tolerance_sec * 1e9))
    interpolator = PoseInterpolator(
        poses,
        boundary_tolerance_ns=tf_boundary_tolerance_ns,
    )

    reference_resolved_ns, reference_tf_clamp_ns = interpolator.resolve_stamp(
        args.target_stamp_ns
    )
    reference_world_base = interpolator.matrix_at(reference_resolved_ns)
    reference_base_world = invert_transform(reference_world_base)

    base_no_deskew = apply_transform(raw_xyz, static_matrix)
    base_deskewed = np.full_like(base_no_deskew, np.nan)

    unique_times, inverse_indices = np.unique(
        point_time_ns[valid_raw],
        return_inverse=True,
    )
    valid_indices = np.flatnonzero(valid_raw)

    tf_clamped_point_count = 0
    tf_uncovered_point_count = 0
    tf_max_clamp_ns = int(reference_tf_clamp_ns)

    for unique_idx, stamp_ns in enumerate(unique_times):
        point_indices = valid_indices[inverse_indices == unique_idx]
        try:
            resolved_stamp_ns, clamp_ns = interpolator.resolve_stamp(int(stamp_ns))
        except ValueError:
            tf_uncovered_point_count += int(len(point_indices))
            continue
        if clamp_ns > 0:
            tf_clamped_point_count += int(len(point_indices))
            tf_max_clamp_ns = max(tf_max_clamp_ns, int(clamp_ns))

        current_world_base = interpolator.matrix_at(resolved_stamp_ns)
        lidar_to_reference_base = (
            reference_base_world
            @ current_world_base
            @ static_matrix
        )
        base_deskewed[point_indices] = apply_transform(
            raw_xyz[point_indices],
            lidar_to_reference_base,
        )

    deskew_valid = np.all(np.isfinite(base_deskewed), axis=1)
    base_range = np.linalg.norm(base_deskewed, axis=1)
    feature_range_mask = (
        deskew_valid
        & (base_range >= args.min_range_m)
        & (base_range <= args.feature_max_range_m)
    )

    features, feature_info = compute_structured_features(
        points_base=base_deskewed,
        ring=ring,
        valid_mask=feature_range_mask,
        range_m=base_range,
        ring_radius=args.ring_radius,
        column_radius=args.column_radius,
        min_neighbors=args.min_neighbors,
        range_jump_base_m=args.range_jump_base_m,
        range_jump_ratio=args.range_jump_ratio,
    )

    invalid_xyz_mask = ~finite_xyz | ~nonzero_xyz
    range_excluded_mask = (
        deskew_valid
        & (
            (base_range < args.min_range_m)
            | (base_range > args.feature_max_range_m)
        )
    )
    isolated_mask = feature_range_mask & ~features["feature_valid_mask"]

    np.savez_compressed(
        output_dir / "a1_single_sample.npz",
        raw_xyz_hesai=raw_xyz.astype(np.float32),
        base_xyz_no_deskew=base_no_deskew.astype(np.float32),
        base_xyz_deskewed=base_deskewed.astype(np.float32),
        intensity=intensity,
        ring=ring,
        point_timestamp_sec=point_time_sec,
        point_timestamp_ns=point_time_ns,
        valid_raw_mask=valid_raw,
        invalid_xyz_mask=invalid_xyz_mask,
        deskew_valid_mask=deskew_valid,
        feature_range_mask=feature_range_mask,
        range_excluded_mask=range_excluded_mask,
        isolated_mask=isolated_mask,
        range_m=base_range.astype(np.float32),
        **features,
    )

    summary = {
        "schema_version": "a1_single_sample_v0_1",
        "source": {
            "bag": str(bag_dir),
            "topic": args.topic,
            "storage_id": storage_id,
            "target_stamp_ns": int(args.target_stamp_ns),
            "selected_scan": scan_info,
            "bag_record_stamp_ns": int(bag_stamp_ns),
            "frame_id": cloud_msg.header.frame_id,
            "point_count": int(len(raw_xyz)),
            "fields": sorted(arrays),
        },
        "frames": {
            "world": args.world_frame,
            "base": args.base_frame,
            "lidar": args.lidar_frame,
        },
        "tf": {
            "dynamic_pose_count": len(poses),
            "dynamic_min_stamp_ns": interpolator.min_stamp_ns,
            "dynamic_max_stamp_ns": interpolator.max_stamp_ns,
            "boundary_tolerance_ns": interpolator.boundary_tolerance_ns,
            "reference_requested_stamp_ns": int(args.target_stamp_ns),
            "reference_resolved_stamp_ns": int(reference_resolved_ns),
            "reference_clamp_ns": int(reference_tf_clamp_ns),
            "point_clamped_count": int(tf_clamped_point_count),
            "point_uncovered_count": int(tf_uncovered_point_count),
            "maximum_clamp_ns": int(tf_max_clamp_ns),
            "boundary_policy": (
                "nearest boundary pose within tolerance; no extrapolation; "
                "timestamps beyond tolerance remain invalid"
            ),
            "static_lidar_to_base": static_matrix.tolist(),
        },
        "validity": {
            "valid_raw_count": int(valid_raw.sum()),
            "invalid_xyz_count": int(invalid_xyz_mask.sum()),
            "deskew_valid_count": int(deskew_valid.sum()),
            "feature_range_count": int(feature_range_mask.sum()),
            "range_excluded_count": int(range_excluded_mask.sum()),
            "isolated_or_feature_invalid_count": int(isolated_mask.sum()),
        },
        "feature_config": {
            "min_range_m": args.min_range_m,
            "feature_max_range_m": args.feature_max_range_m,
            "ring_radius": args.ring_radius,
            "column_radius": args.column_radius,
            "min_neighbors": args.min_neighbors,
            "range_jump_base_m": args.range_jump_base_m,
            "range_jump_ratio": args.range_jump_ratio,
        },
        "feature_structure": feature_info,
        "feature_statistics": {
            "feature_valid_count": int(features["feature_valid_mask"].sum()),
            "high_confidence_q70_count": int(
                features["high_confidence_q70_mask"].sum()
            ),
            "high_confidence_q85_count": int(
                features["high_confidence_q85_mask"].sum()
            ),
        },
        "important_note": (
            "A1의 vertical_confidence_prototype과 q70/q85 mask는 "
            "RViz 및 단일 샘플 감사를 위한 진단값이다. "
            "최종 wall mask 또는 MCL likelihood로 사용하지 않는다."
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_pngs(
        output_dir=output_dir,
        points_base=base_deskewed,
        range_m=base_range,
        valid_mask=feature_range_mask,
        features=features,
    )

    report = "\n".join(
        [
            "===== A1 Single Sample Prototype =====",
            f"bag: {bag_dir}",
            f"target stamp ns: {args.target_stamp_ns}",
            f"selected scan contains target: {scan_info['contains_reference_time']}",
            (
                "selected point time range: "
                f"{scan_info['point_time_min_sec']:.9f} ~ "
                f"{scan_info['point_time_max_sec']:.9f}"
            ),
            f"raw points: {len(raw_xyz)}",
            f"valid raw: {int(valid_raw.sum())}",
            f"deskew valid: {int(deskew_valid.sum())}",
            (
                "TF boundary clamp: "
                f"reference={reference_tf_clamp_ns / 1e9:.6f}s, "
                f"points={tf_clamped_point_count}, "
                f"uncovered={tf_uncovered_point_count}, "
                f"max={tf_max_clamp_ns / 1e9:.6f}s"
            ),
            f"feature range points: {int(feature_range_mask.sum())}",
            f"feature valid: {int(features['feature_valid_mask'].sum())}",
            "",
            "No existing dataset/result was overwritten.",
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
