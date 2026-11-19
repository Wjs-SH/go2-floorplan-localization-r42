#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROLE_NAMES = {
    0: 'smooth_surface',
    1: 'depth_jump_edge',
    2: 'normal_turn_edge',
    3: 'low_support_vertical',
    4: 'far_sparse_uncertain',
}

ROLE_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#9467bd',
    '#d62728',
]

POINTFIELD_DTYPES = {
    1: ('int8', 1),
    2: ('uint8', 1),
    3: ('int16', 2),
    4: ('uint16', 2),
    5: ('int32', 4),
    6: ('uint32', 4),
    7: ('float32', 4),
    8: ('float64', 8),
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_ros() -> dict[str, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        raise RuntimeError(
            'ROS2 Python 모듈을 불러올 수 없습니다. '
            'source /opt/ros/humble/setup.bash 후 실행하세요.'
        ) from exc
    return {
        'rosbag2_py': rosbag2_py,
        'deserialize_message': deserialize_message,
        'PointCloud2': PointCloud2,
    }


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def detect_storage_id(bag: Path) -> str:
    metadata_path = bag / 'metadata.yaml' if bag.is_dir() else bag.parent / 'metadata.yaml'
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding='utf-8'))
        info = metadata.get('rosbag2_bagfile_information', metadata)
        value = info.get('storage_identifier')
        if value:
            return str(value)
    if bag.suffix.lower() == '.mcap' or (bag.is_dir() and list(bag.glob('*.mcap'))):
        return 'mcap'
    if bag.suffix.lower() == '.db3' or (bag.is_dir() and list(bag.glob('*.db3'))):
        return 'sqlite3'
    raise RuntimeError(f'rosbag storage id를 찾지 못했습니다: {bag}')


def read_bag_start_ns(bag: Path) -> int:
    metadata_path = bag / 'metadata.yaml' if bag.is_dir() else bag.parent / 'metadata.yaml'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'metadata.yaml이 없습니다: {metadata_path}')
    metadata = yaml.safe_load(metadata_path.read_text(encoding='utf-8'))
    info = metadata.get('rosbag2_bagfile_information', metadata)
    starting_time = info.get('starting_time')
    if isinstance(starting_time, dict):
        for key in ('nanoseconds_since_epoch', 'nanoseconds'):
            if key in starting_time:
                return int(starting_time[key])
    if isinstance(starting_time, int):
        return int(starting_time)
    raise KeyError('metadata.yaml에서 bag starting_time을 찾지 못했습니다.')


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pointcloud_dtype(msg: Any) -> np.dtype:
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    endian = '>' if bool(msg.is_bigendian) else '<'
    for field in msg.fields:
        datatype = int(field.datatype)
        if datatype not in POINTFIELD_DTYPES:
            raise ValueError(
                f'지원하지 않는 PointField: {field.name}, datatype={datatype}'
            )
        base_name, _ = POINTFIELD_DTYPES[datatype]
        base_dtype = np.dtype(endian + np.dtype(base_name).str[1:])
        names.append(str(field.name))
        formats.append(
            base_dtype if int(field.count) == 1
            else (base_dtype, (int(field.count),))
        )
        offsets.append(int(field.offset))
    return np.dtype({
        'names': names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': int(msg.point_step),
    })


def decode_fields(msg: Any, requested: Iterable[str]) -> dict[str, np.ndarray]:
    dtype = pointcloud_dtype(msg)
    available = set(dtype.names or [])
    requested_set = set(requested)
    missing = sorted(requested_set - available)
    if missing:
        raise KeyError(
            f'PointCloud2 필드가 없습니다: {missing}; available={sorted(available)}'
        )
    width = int(msg.width)
    height = int(msg.height)
    expected_row = int(msg.point_step) * width
    raw = memoryview(msg.data)
    if int(msg.row_step) == expected_row:
        cloud = np.frombuffer(raw, dtype=dtype, count=width * height)
    else:
        rows = []
        for row_index in range(height):
            start = row_index * int(msg.row_step)
            stop = start + expected_row
            rows.append(np.frombuffer(raw[start:stop], dtype=dtype, count=width))
        cloud = np.concatenate(rows)
    return {name: np.asarray(cloud[name]) for name in requested_set}


def open_reader(ros: dict[str, Any], bag: Path, topic: str, seek_ns: int) -> Any:
    rosbag2_py = ros['rosbag2_py']
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=detect_storage_id(bag)),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    if hasattr(reader, 'seek'):
        reader.seek(int(seek_ns))
    return reader


def index_scans(
    bag: Path,
    topic: str,
    timestamp_field: str,
    start_sec: float,
    duration_sec: float,
    context_before_sec: float = 0.80,
    context_after_sec: float = 0.20,
) -> tuple[int, list[dict[str, Any]], np.ndarray]:
    ros = import_ros()
    bag_start_ns = read_bag_start_ns(bag)
    requested_start_ns = bag_start_ns + int(round(start_sec * 1e9))
    requested_end_ns = requested_start_ns + int(round(duration_sec * 1e9))
    read_start_ns = requested_start_ns - int(round(context_before_sec * 1e9))
    read_end_ns = requested_end_ns + int(round(context_after_sec * 1e9))

    reader = open_reader(ros, bag, topic, max(bag_start_ns, read_start_ns))
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise KeyError(f'topic이 없습니다: {topic}')
    if topic_types[topic] != 'sensor_msgs/msg/PointCloud2':
        raise TypeError(f'{topic} type은 PointCloud2가 아닙니다: {topic_types[topic]}')

    deserialize = ros['deserialize_message']
    PointCloud2 = ros['PointCloud2']
    scans: list[dict[str, Any]] = []

    while reader.has_next():
        read_topic, serialized, bag_record_ns = reader.read_next()
        if read_topic != topic:
            continue
        if int(bag_record_ns) > read_end_ns:
            break
        message = deserialize(serialized, PointCloud2)
        header_ns = stamp_to_ns(message.header.stamp) or int(bag_record_ns)
        arrays = decode_fields(message, ['x', 'y', 'z', timestamp_field])
        point_time_sec = np.asarray(arrays[timestamp_field], dtype=np.float64)
        x = np.asarray(arrays['x'], dtype=np.float64)
        y = np.asarray(arrays['y'], dtype=np.float64)
        z = np.asarray(arrays['z'], dtype=np.float64)
        valid = (
            np.isfinite(point_time_sec)
            & np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & ((x * x + y * y + z * z) > 1e-12)
        )
        if not np.any(valid):
            continue
        # 2026-09-11: 스캔 시작/끝 = 타임스탬프가 유한한 모든 점(A1 select_scan 과 같은 정의).
        #  정제 bag 에서 지운 점(xyz=NaN)이 스캔 끝에 있으면 유효점 기준 끝시각이 당겨져 A1 선택 검사가 실패했다.
        #  원본 6F bag 5030 스캔에서는 두 정의가 전부 같음(scan_time_defs.py). A131_SCANTIME_ALL=0 이면 원래 정의.
        span = np.isfinite(point_time_sec) if os.environ.get('A131_SCANTIME_ALL', '1') == '1' else valid
        point_start_ns = int(np.rint(np.min(point_time_sec[span]) * 1e9))
        point_end_ns = int(np.rint(np.max(point_time_sec[span]) * 1e9))
        if point_end_ns < read_start_ns or point_start_ns > read_end_ns:
            continue
        scans.append({
            'header_stamp_ns': int(header_ns),
            'bag_record_stamp_ns': int(bag_record_ns),
            'point_start_ns': point_start_ns,
            'point_end_ns': point_end_ns,
            'point_span_sec': float(point_end_ns - point_start_ns) / 1e9,
            'raw_point_count': int(len(point_time_sec)),
            'valid_point_count': int(np.count_nonzero(valid)),
            'frame_id': str(message.header.frame_id),
        })

    scans.sort(key=lambda row: (int(row['point_end_ns']), int(row['header_stamp_ns'])))
    for index, row in enumerate(scans):
        row['scan_index'] = int(index)
        row['header_sec_from_bag_start'] = float(int(row['header_stamp_ns']) - bag_start_ns) / 1e9
        row['point_start_sec_from_bag_start'] = float(int(row['point_start_ns']) - bag_start_ns) / 1e9
        row['point_end_sec_from_bag_start'] = float(int(row['point_end_ns']) - bag_start_ns) / 1e9

    if not scans:
        raise RuntimeError('요청 구간에서 유효한 LiDAR scan을 읽지 못했습니다.')
    point_end_array = np.asarray([row['point_end_ns'] for row in scans], dtype=np.int64)
    target_indices = np.flatnonzero(
        (point_end_array >= requested_start_ns)
        & (point_end_array <= requested_end_ns)
    ).astype(np.int64)
    if target_indices.size == 0:
        raise RuntimeError('요청 구간에 target scan이 없습니다.')
    return bag_start_ns, scans, target_indices


def latest_indices(current_index: int, count: int) -> np.ndarray:
    start = max(0, current_index - count + 1)
    return np.arange(start, current_index + 1, dtype=np.int64)


def w30_header_indices(scans: list[dict[str, Any]], current_index: int) -> np.ndarray:
    refs = np.asarray([row['header_stamp_ns'] for row in scans], dtype=np.int64)
    target = int(refs[current_index])
    lower = target - 300_000_000
    mask = (refs >= lower) & (refs <= target)
    mask &= np.arange(len(refs)) <= current_index
    return np.flatnonzero(mask)


def exact_mode_for_w30(w30: np.ndarray, s3: np.ndarray, s4: np.ndarray) -> str:
    if np.array_equal(w30, s3):
        return 'S3'
    if np.array_equal(w30, s4):
        return 'S4'
    return 'OTHER'


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f'하위 단계가 실패했습니다: return={return_code}, log={log_path}'
        )


def validate_sources(source_home: Path) -> dict[str, Path]:
    paths = {
        'a1': source_home / '4F_multilevel_frame_dataset_v3_0_a/A1_SINGLE_SAMPLE_PROTOTYPE/extract_a1_single_sample.py',
        'a1_1': source_home / '4F_multilevel_frame_dataset_v3_0_a/A1_1_REFINED_STRUCTURE_PROTOTYPE/build_a1_1_from_a1.py',
        'a1_2': source_home / '4F_multilevel_frame_dataset_v3_0_a/A1_2_GEOMETRIC_CHANNEL_PROTOTYPE/build_a1_2_from_a1_1.py',
        'a1_2_1': source_home / '4F_multilevel_frame_dataset_v3_0_a/A1_2_1_ENRICHED_GEOMETRY_PROTOTYPE/build_a1_2_1_from_a1_2.py',
        'a1_3': source_home / '4F_multilevel_frame_dataset_v3_0_a/A1_3_FLATTENED_MULTICHANNEL_DATASET/build_a1_3_from_a1_2_1.py',
        'a1_3_1': source_home / '4F_multilevel_frame_dataset_v3_0_a/A1_3_1_BALANCED_LOCALIZATION_SELECTOR/refine_a1_3_selection.py',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('필요한 소스가 없습니다:\n' + '\n'.join(missing))
    return paths


def sample_geometry() -> dict[str, Any]:
    return {
        'x_range_m': [-8.0, 8.0],
        'y_range_m': [-8.0, 8.0],
        'resolution_m': 0.08,
        'grid_shape': [200, 200],
        'height_bands_m': {
            'low': [0.15, 0.50],
            'mid': [0.50, 1.00],
            'high': [1.00, 1.70],
        },
    }


def build_scan_cache(
    scan: dict[str, Any],
    bag: Path,
    source_paths: dict[str, Path],
    wrapper_path: Path,
    cache_root: Path,
    python_executable: str,
) -> Path:
    scan_index = int(scan['scan_index'])
    cache_dir = cache_root / f'SCAN_{scan_index:06d}'
    final_npz = cache_dir / 'a1_2_1_enriched_geometry.npz'
    complete = cache_dir / 'COMPLETE.json'
    if final_npz.is_file() and complete.is_file():
        print(f'[cache] 재사용: {cache_dir.name}')
        return final_npz

    if cache_dir.exists():
        raise FileExistsError(
            f'미완성 cache가 이미 있습니다. 삭제하지 않고 중단합니다: {cache_dir}'
        )
    cache_dir.mkdir(parents=True)
    work = cache_dir / '_work'
    work.mkdir()

    reference_ns = int(scan['point_end_ns'])
    sample_json = cache_dir / 'sample.json'
    write_json(sample_json, {
        'output_stamp_ns': reference_ns,
        'geometry': sample_geometry(),
        'source_scan': scan,
        'reference_definition': 'scan point timestamp end',
    })

    a1_dir = work / 'a1'
    a11_dir = work / 'a1_1'
    a12_dir = work / 'a1_2'
    a121_dir = work / 'a1_2_1'

    commands = [
        (
            'a1',
            [
                python_executable, str(wrapper_path), '--script', str(source_paths['a1']), '--',
                '--bag', str(bag),
                '--topic', '/lidar_points',
                '--target-stamp-ns', str(reference_ns),
                '--output-dir', str(a1_dir),
            ],
        ),
        (
            'a1_1',
            [
                python_executable, str(wrapper_path), '--script', str(source_paths['a1_1']), '--',
                '--input-npz', str(a1_dir / 'a1_single_sample.npz'),
                '--output-dir', str(a11_dir),
            ],
        ),
        (
            'a1_2',
            [
                python_executable, str(wrapper_path), '--script', str(source_paths['a1_2']), '--',
                '--input-npz', str(a11_dir / 'a1_1_refined_sample.npz'),
                '--output-dir', str(a12_dir),
            ],
        ),
        (
            'a1_2_1',
            [
                python_executable, str(wrapper_path), '--script', str(source_paths['a1_2_1']), '--',
                '--input-npz', str(a12_dir / 'a1_2_geometric_channels.npz'),
                '--sample-json', str(sample_json),
                '--a1-summary-json', str(a1_dir / 'summary.json'),
                '--output-dir', str(a121_dir),
            ],
        ),
    ]

    for name, command in commands:
        print(f'[{cache_dir.name}] {name} 시작')
        run_command(command, cache_dir / f'{name}.log')

    selected_summary = load_json(a1_dir / 'summary.json')
    selected = selected_summary['source']['selected_scan']
    selected_end_ns = int(round(float(selected['point_time_max_sec']) * 1e9))
    if abs(selected_end_ns - reference_ns) > 2_000_000:
        raise RuntimeError(
            f'의도한 scan과 A1 선택 scan이 다릅니다: intended_end={reference_ns}, selected_end={selected_end_ns}'
        )

    shutil.move(str(a121_dir / 'a1_2_1_enriched_geometry.npz'), str(final_npz))
    shutil.move(str(a121_dir / 'summary.json'), str(cache_dir / 'a1_2_1_summary.json'))
    write_json(complete, {
        'schema_version': 'a_s3_s4_scan_cache_v0_1',
        'scan': scan,
        'reference_stamp_ns': reference_ns,
        'reference_definition': 'scan point timestamp end',
        'source_stage': 'A1 -> A1.1 -> A1.2 -> A1.2.1',
        'note': 'This cache is for the short S3/S4 experiment. It is not the final compact full-bag cache.',
    })
    shutil.rmtree(work)
    print(f'[{cache_dir.name}] cache 완료')
    return final_npz


def transform_vectors(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.full_like(vectors, np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(vectors), axis=1)
    result[valid] = vectors[valid] @ rotation.T
    return result


def merge_group(
    group_scan_indices: np.ndarray,
    target_scan_index: int,
    scans: list[dict[str, Any]],
    cache_root: Path,
    pose_interpolator: Any,
    output_npz: Path,
    output_summary: Path,
) -> dict[str, Any]:
    target_stamp_ns = int(scans[target_scan_index]['point_end_ns'])
    world_target = pose_interpolator.matrix_at(target_stamp_ns)
    target_world = np.linalg.inv(world_target)

    merged: dict[str, list[np.ndarray]] = {
        'base_xyz_deskewed': [],
        'ring': [],
        'range_m': [],
        'a1_2_1_raw_point_index': [],
        'a1_2_1_scan_column': [],
        'a1_2_1_point_time_offset_sec': [],
        'a1_2_1_z_base_m': [],
        'a1_2_1_height_band_id': [],
        'a1_2_1_normal_xy_unit': [],
        'a1_2_1_normal_azimuth_rad': [],
        'a1_2_1_wall_tangent_azimuth_rad': [],
        'a1_2_1_projection_ready_mask': [],
        'a1_2_1_smooth_surface_score': [],
        'a1_2_1_depth_jump_edge_score': [],
        'a1_2_1_normal_turn_edge_score': [],
        'a1_2_1_low_support_vertical_score': [],
        'a1_2_1_far_uncertain_score': [],
        'a1_2_1_low_structure_score': [],
        'a1_2_1_support_density_normalized': [],
        'a1_2_1_reason_flags': [],
        'source_scan_local_id': [],
        'source_scan_context_index': [],
        'point_timestamp_ns': [],
        'a1_1_normals_target': [],
    }

    point_offset = 0
    source_rows = []
    height_bands = sample_geometry()['height_bands_m']

    for local_id, scan_index_value in enumerate(group_scan_indices):
        scan_index = int(scan_index_value)
        scan = scans[scan_index]
        cache_npz = cache_root / f'SCAN_{scan_index:06d}' / 'a1_2_1_enriched_geometry.npz'
        if not cache_npz.is_file():
            raise FileNotFoundError(cache_npz)
        with np.load(cache_npz, allow_pickle=False) as source:
            data = {key: np.asarray(source[key]) for key in source.files}

        scan_stamp_ns = int(scan['point_end_ns'])
        world_scan = pose_interpolator.matrix_at(scan_stamp_ns)
        target_from_scan = target_world @ world_scan
        rotation = target_from_scan[:3, :3]
        translation = target_from_scan[:3, 3]

        points_scan = data['base_xyz_deskewed'].astype(np.float64)
        points_target = points_scan @ rotation.T + translation
        target_range = np.linalg.norm(points_target, axis=1)
        z = points_target[:, 2]

        normals_scan = data['a1_1_normals'].astype(np.float64)
        normals_target = transform_vectors(normals_scan, rotation)
        normal_xy = normals_target[:, :2]
        normal_xy_norm = np.linalg.norm(normal_xy, axis=1)
        normal_xy_unit = np.full_like(normal_xy, np.nan)
        valid_normal = np.isfinite(normal_xy_norm) & (normal_xy_norm > 1e-6)
        normal_xy_unit[valid_normal] = normal_xy[valid_normal] / normal_xy_norm[valid_normal, None]
        normal_angle = np.full(len(points_target), np.nan, dtype=np.float32)
        normal_angle[valid_normal] = np.arctan2(
            normal_xy_unit[valid_normal, 1],
            normal_xy_unit[valid_normal, 0],
        ).astype(np.float32)
        tangent = np.full(len(points_target), np.nan, dtype=np.float32)
        tangent[valid_normal] = (
            normal_angle[valid_normal] + np.float32(math.pi / 2.0)
        )
        tangent[valid_normal] = (
            (tangent[valid_normal] + math.pi) % (2.0 * math.pi) - math.pi
        )

        height_band_id = np.full(len(points_target), -1, dtype=np.int8)
        low_min, low_max = map(float, height_bands['low'])
        mid_min, mid_max = map(float, height_bands['mid'])
        high_min, high_max = map(float, height_bands['high'])
        height_band_id[(z >= low_min) & (z < low_max)] = 0
        height_band_id[(z >= mid_min) & (z < mid_max)] = 1
        height_band_id[(z >= high_min) & (z <= high_max)] = 2

        original_ready = data['a1_2_1_projection_ready_mask'].astype(bool)
        target_ready = (
            original_ready
            & np.all(np.isfinite(points_target), axis=1)
            & (target_range >= 0.30)
            & (target_range <= 20.0)
            & (height_band_id >= 0)
        )

        point_timestamp_ns = data['point_timestamp_ns'].astype(np.int64)
        n = len(points_target)
        merged['base_xyz_deskewed'].append(points_target.astype(np.float32))
        merged['ring'].append(data['ring'].astype(np.uint16))
        merged['range_m'].append(target_range.astype(np.float32))
        merged['a1_2_1_raw_point_index'].append(np.arange(point_offset, point_offset + n, dtype=np.int32))
        merged['a1_2_1_scan_column'].append(data['a1_2_1_scan_column'].astype(np.int32))
        merged['a1_2_1_point_time_offset_sec'].append(
            ((point_timestamp_ns - target_stamp_ns).astype(np.float64) / 1e9).astype(np.float32)
        )
        merged['a1_2_1_z_base_m'].append(z.astype(np.float32))
        merged['a1_2_1_height_band_id'].append(height_band_id)
        merged['a1_2_1_normal_xy_unit'].append(normal_xy_unit.astype(np.float32))
        merged['a1_2_1_normal_azimuth_rad'].append(normal_angle)
        merged['a1_2_1_wall_tangent_azimuth_rad'].append(tangent)
        merged['a1_2_1_projection_ready_mask'].append(target_ready)
        for key in (
            'a1_2_1_smooth_surface_score',
            'a1_2_1_depth_jump_edge_score',
            'a1_2_1_normal_turn_edge_score',
            'a1_2_1_low_support_vertical_score',
            'a1_2_1_far_uncertain_score',
            'a1_2_1_low_structure_score',
            'a1_2_1_support_density_normalized',
        ):
            merged[key].append(data[key].astype(np.float32))
        merged['a1_2_1_reason_flags'].append(data['a1_2_1_reason_flags'].astype(np.uint16))
        merged['source_scan_local_id'].append(np.full(n, local_id, dtype=np.uint8))
        merged['source_scan_context_index'].append(np.full(n, scan_index, dtype=np.int32))
        merged['point_timestamp_ns'].append(point_timestamp_ns)
        merged['a1_1_normals_target'].append(normals_target.astype(np.float32))

        source_rows.append({
            'local_scan_id': local_id,
            'context_scan_index': scan_index,
            'point_end_ns': scan_stamp_ns,
            'point_end_sec_from_bag_start': float(scan['point_end_sec_from_bag_start']),
            'target_minus_scan_end_sec': float(target_stamp_ns - scan_stamp_ns) / 1e9,
            'point_count': n,
            'projection_ready_count_after_transform': int(np.count_nonzero(target_ready)),
            'target_from_scan_base': target_from_scan.tolist(),
        })
        point_offset += n

    combined = {
        key: np.concatenate(parts, axis=0)
        for key, parts in merged.items()
    }
    np.savez_compressed(output_npz, **combined)
    summary = {
        'schema_version': 'a_s3_s4_merged_geometry_v0_1',
        'target_scan_index': int(target_scan_index),
        'target_stamp_ns': target_stamp_ns,
        'reference_definition': 'target scan point timestamp end, base_link',
        'group_scan_indices': [int(value) for value in group_scan_indices],
        'group_scan_count': int(len(group_scan_indices)),
        'source_scans': source_rows,
        'grid_geometry_for_next_stage': sample_geometry(),
        'point_count': int(len(combined['ring'])),
        'projection_ready_count': int(np.count_nonzero(combined['a1_2_1_projection_ready_mask'])),
        'important_notes': [
            'A1-A1.2.1 features were calculated independently inside each native raw scan.',
            'Only after per-scan feature extraction were points and normals rigidly transformed to the target scan-end base_link frame.',
            'Scores were not recalculated from a mixed ring grid.',
            'range_m, z, height band, normal azimuth, and projection-ready validity were updated in the target frame.',
        ],
    }
    write_json(output_summary, summary)
    return summary


def grid_coordinates(
    xy: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = grid_shape
    finite_x = np.isfinite(xy[:, 0])
    finite_y = np.isfinite(xy[:, 1])
    safe_x = np.where(finite_x, xy[:, 0], x_range[0] - resolution)
    safe_y = np.where(finite_y, xy[:, 1], y_range[0] - resolution)
    col = np.floor((safe_x - x_range[0]) / resolution).astype(np.int64)
    row_from_bottom = np.floor((safe_y - y_range[0]) / resolution).astype(np.int64)
    row = (height - 1) - row_from_bottom
    valid = (
        finite_x & finite_y
        & (xy[:, 0] >= x_range[0]) & (xy[:, 0] < x_range[1])
        & (xy[:, 1] >= y_range[0]) & (xy[:, 1] < y_range[1])
        & (col >= 0) & (col < width)
        & (row >= 0) & (row < height)
    )
    return row, col, valid


def add_scan_support_to_a1_3_short(
    short_npz: Path,
    merged_npz: Path,
    group_scan_count: int,
) -> dict[str, Any]:
    with np.load(short_npz, allow_pickle=False) as source:
        short = {key: np.asarray(source[key]) for key in source.files}
    with np.load(merged_npz, allow_pickle=False) as source:
        merged = {key: np.asarray(source[key]) for key in source.files}

    x_range = tuple(map(float, short['grid_x_range_m']))
    y_range = tuple(map(float, short['grid_y_range_m']))
    resolution = float(short['grid_resolution_m'])
    grid_shape = tuple(map(int, short['grid_shape']))
    height, width = grid_shape
    cell_count = height * width

    points = merged['base_xyz_deskewed'].astype(np.float64)
    z = merged['a1_2_1_z_base_m'].astype(np.float64)
    ready = merged['a1_2_1_projection_ready_mask'].astype(bool)
    rows, cols, in_grid = grid_coordinates(
        points[:, :2], x_range, y_range, resolution, grid_shape
    )
    radial = np.linalg.norm(points[:, :2], axis=1) <= float(short['radial_max_m'])
    fine_edges = short['fine_height_edges_m'].astype(np.float64)
    mask = (
        ready & in_grid & radial & np.isfinite(z)
        & (z >= fine_edges[0]) & (z <= fine_edges[-1])
    )
    flat = rows[mask].astype(np.int64) * width + cols[mask].astype(np.int64)
    scan_ids = merged['source_scan_local_id'][mask].astype(np.int64)

    support = np.zeros(cell_count, dtype=np.uint8)
    if flat.size:
        pairs = np.unique(np.column_stack([flat, scan_ids]), axis=0)
        support = np.bincount(pairs[:, 0], minlength=cell_count).astype(np.uint8)
    support_grid = support.reshape(grid_shape)
    ratio_grid = support_grid.astype(np.float32) / max(group_scan_count, 1)

    short['scan_support_count'] = support_grid
    short['scan_support_ratio'] = ratio_grid
    short['source_scan_count'] = np.asarray(group_scan_count, dtype=np.uint8)
    short['scan_support_ge_2'] = support_grid >= 2
    short['scan_support_all'] = support_grid >= group_scan_count
    np.savez_compressed(short_npz, **short)

    occupied = short['structure_candidate'].astype(bool)
    selected_ids = short['localization_flat_cell_id'].astype(np.int64)
    return {
        'group_scan_count': int(group_scan_count),
        'cells_observed_by_any_scan': int(np.count_nonzero(support_grid)),
        'structure_candidate_cells': int(np.count_nonzero(occupied)),
        'candidate_support_ge_2': int(np.count_nonzero(occupied & (support_grid >= 2))),
        'candidate_support_all': int(np.count_nonzero(occupied & (support_grid >= group_scan_count))),
        'legacy_a1_3_selected_count': int(len(selected_ids)),
    }


def run_flatten_and_selector(
    merged_npz: Path,
    merged_summary: Path,
    mode_dir: Path,
    source_paths: dict[str, Path],
    wrapper_path: Path,
    python_executable: str,
    group_scan_count: int,
) -> dict[str, Any]:
    a13_dir = mode_dir / '_a1_3'
    a131_dir = mode_dir / '_a1_3_1'
    run_command([
        python_executable, str(wrapper_path), '--script', str(source_paths['a1_3']), '--',
        '--input-npz', str(merged_npz),
        '--input-summary-json', str(merged_summary),
        '--output-dir', str(a13_dir),
        '--localization-budget', '1024',
    ], mode_dir / 'a1_3.log')

    support_summary = add_scan_support_to_a1_3_short(
        a13_dir / 'a1_3_data_short.npz', merged_npz, group_scan_count
    )

    run_command([
        python_executable, str(wrapper_path), '--script', str(source_paths['a1_3_1']), '--',
        '--input-short-npz', str(a13_dir / 'a1_3_data_short.npz'),
        '--input-summary-json', str(a13_dir / 'summary.json'),
        '--output-dir', str(a131_dir),
        '--balanced-budget', '384',
    ], mode_dir / 'a1_3_1.log')

    final_npz = mode_dir / 'data.npz'
    shutil.move(str(a131_dir / 'a1_3_1_data_short.npz'), str(final_npz))
    shutil.move(str(a131_dir / 'summary.json'), str(mode_dir / 'selector_summary.json'))
    shutil.move(str(a13_dir / 'summary.json'), str(mode_dir / 'flatten_summary.json'))
    write_json(mode_dir / 'scan_support_summary.json', support_summary)

    # 큰 point-level 임시 파일은 실험 결과에 남기지 않는다.
    shutil.rmtree(a13_dir)
    shutil.rmtree(a131_dir)
    if merged_npz.exists():
        merged_npz.unlink()
    return support_summary


def cell_xy(data: dict[str, np.ndarray]) -> np.ndarray:
    height, width = map(int, data['grid_shape'])
    resolution = float(data['grid_resolution_m'])
    x_min, _ = map(float, data['grid_x_range_m'])
    y_min, _ = map(float, data['grid_y_range_m'])
    rows, cols = np.indices((height, width))
    x = x_min + (cols.astype(np.float64) + 0.5) * resolution
    row_from_bottom = (height - 1) - rows.astype(np.float64)
    y = y_min + (row_from_bottom + 0.5) * resolution
    return np.stack([x, y], axis=-1).astype(np.float32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def mode_statistics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    candidate = data['structure_candidate'].astype(bool)
    selected_roles = data['a1_3_1_balanced_role'].astype(np.int8)
    selected_ids = data['a1_3_1_balanced_flat_cell_id'].astype(np.int64)
    support = data['scan_support_count'].reshape(-1)
    role_counts = {
        ROLE_NAMES[role_id]: int(np.count_nonzero(selected_roles == role_id))
        for role_id in ROLE_NAMES
    }
    return {
        'structure_candidate_cells': int(np.count_nonzero(candidate)),
        'balanced_selected_cells': int(len(selected_ids)),
        'role_counts': role_counts,
        'selected_support_mean': float(np.mean(support[selected_ids])) if len(selected_ids) else 0.0,
        'selected_support_ge_2': int(np.count_nonzero(support[selected_ids] >= 2)),
        'candidate_support_ge_2': int(np.count_nonzero(candidate & (data['scan_support_count'] >= 2))),
        'structure_evidence_mean_on_candidates': float(
            np.mean(data['structure_evidence'][candidate])
        ) if np.any(candidate) else 0.0,
        'normal_concentration_mean_on_candidates': float(
            np.mean(data['normal_concentration'][candidate])
        ) if np.any(candidate) else 0.0,
    }


def save_mode_visualizations(mode_dir: Path, mode_name: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    data = load_npz(mode_dir / 'data.npz')
    x_range = list(map(float, data['grid_x_range_m']))
    y_range = list(map(float, data['grid_y_range_m']))
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    xy = cell_xy(data).reshape(-1, 2)
    candidate = data['structure_candidate'].astype(bool).reshape(-1)
    selected = data['a1_3_1_balanced_flat_cell_id'].astype(np.int64)
    roles = data['a1_3_1_balanced_role'].astype(np.int8)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        data['structure_evidence'], origin='upper', extent=extent,
        interpolation='nearest', vmin=0.0, vmax=1.0,
    )
    fig.colorbar(image, ax=ax, label='A1.3 structure evidence')
    ax.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{mode_name}: localization structure raster')
    fig.tight_layout()
    fig.savefig(mode_dir / '01_structure_evidence.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xy[candidate, 0], xy[candidate, 1], s=5, alpha=0.12, label='all candidates')
    if len(selected):
        plot = ax.scatter(
            xy[selected, 0], xy[selected, 1], c=roles,
            cmap=ListedColormap(ROLE_COLORS), vmin=-0.5, vmax=4.5,
            s=20, alpha=0.95,
        )
        colorbar = fig.colorbar(plot, ax=ax, ticks=[0, 1, 2, 3, 4])
        colorbar.ax.set_yticklabels([ROLE_NAMES[i] for i in range(5)])
    ax.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{mode_name}: actual balanced localization points')
    fig.tight_layout()
    fig.savefig(mode_dir / '02_selected_points.png', dpi=180)
    plt.close(fig)

    normal = data['normal_xy_unit']
    concentration = data['normal_concentration']
    structure = data['structure_evidence']
    coordinates = cell_xy(data)
    mask = (
        data['structure_candidate'].astype(bool)
        & np.all(np.isfinite(normal), axis=2)
        & (concentration >= 0.30)
    )
    stride_mask = np.zeros(mask.shape, dtype=bool)
    stride_mask[::4, ::4] = True
    mask &= stride_mask
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        structure, origin='upper', extent=extent, interpolation='nearest',
        vmin=0.0, vmax=1.0, alpha=0.55,
    )
    ax.quiver(
        coordinates[..., 0][mask], coordinates[..., 1][mask],
        normal[..., 0][mask], normal[..., 1][mask],
        concentration[mask], angles='xy', scale_units='xy', scale=1.8,
        width=0.003,
    )
    ax.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{mode_name}: aggregated axial wall normals')
    fig.tight_layout()
    fig.savefig(mode_dir / '03_normal_field.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        data['scan_support_count'], origin='upper', extent=extent,
        interpolation='nearest', vmin=0,
        vmax=int(data['source_scan_count']),
    )
    fig.colorbar(image, ax=ax, label='distinct contributing scans per cell')
    ax.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{mode_name}: scan support count')
    fig.tight_layout()
    fig.savefig(mode_dir / '04_scan_support.png', dpi=180)
    plt.close(fig)


def save_comparison(target_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    s3 = load_npz(target_dir / 's3/data.npz')
    s4 = load_npz(target_dir / 's4/data.npz')
    mask3 = s3['structure_candidate'].astype(bool)
    mask4 = s4['structure_candidate'].astype(bool)
    common = mask3 & mask4
    only3 = mask3 & ~mask4
    only4 = mask4 & ~mask3
    union = mask3 | mask4
    jaccard = float(np.count_nonzero(common) / max(np.count_nonzero(union), 1))

    rgba = np.zeros((*mask3.shape, 4), dtype=np.float32)
    rgba[common] = [0.55, 0.20, 0.70, 1.0]
    rgba[only3] = [0.90, 0.15, 0.15, 1.0]
    rgba[only4] = [0.10, 0.35, 0.95, 1.0]
    x_range = list(map(float, s3['grid_x_range_m']))
    y_range = list(map(float, s3['grid_y_range_m']))
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgba, origin='upper', extent=extent, interpolation='nearest')
    ax.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'S3 vs S4 structure candidates, Jaccard={jaccard:.3f}')
    ax.legend(handles=[
        Patch(facecolor=(0.55, 0.20, 0.70), label='common'),
        Patch(facecolor=(0.90, 0.15, 0.15), label='S3 only'),
        Patch(facecolor=(0.10, 0.35, 0.95), label='S4 only'),
    ], loc='upper right')
    fig.tight_layout()
    fig.savefig(target_dir / '05_s3_s4_candidate_overlay.png', dpi=180)
    plt.close(fig)

    difference = s4['structure_evidence'].astype(np.float32) - s3['structure_evidence'].astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    max_abs = max(float(np.max(np.abs(difference))), 1e-4)
    image = ax.imshow(
        difference, origin='upper', extent=extent, interpolation='nearest',
        vmin=-max_abs, vmax=max_abs, cmap='coolwarm',
    )
    fig.colorbar(image, ax=ax, label='S4 evidence - S3 evidence')
    ax.scatter([0.0], [0.0], marker='x', s=90, linewidths=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('S4 minus S3 structure evidence')
    fig.tight_layout()
    fig.savefig(target_dir / '06_s4_minus_s3_evidence.png', dpi=180)
    plt.close(fig)

    comparison = {
        's3': mode_statistics(s3),
        's4': mode_statistics(s4),
        'candidate_overlap': {
            'common_cells': int(np.count_nonzero(common)),
            's3_only_cells': int(np.count_nonzero(only3)),
            's4_only_cells': int(np.count_nonzero(only4)),
            'union_cells': int(np.count_nonzero(union)),
            'jaccard': jaccard,
        },
        'evidence_difference': {
            'mean_s4_minus_s3': float(np.mean(difference[union])) if np.any(union) else 0.0,
            'max_absolute': max_abs,
        },
        'scope_note': (
            'This is a geometry/input audit only. It does not choose the final winner '
            'without the subsequent endpoint and normal-aware localization comparison.'
        ),
    }
    write_json(target_dir / 'comparison.json', comparison)
    return comparison


def contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 4) -> None:
    from PIL import Image, ImageOps, ImageDraw
    if not image_paths:
        return
    images = [Image.open(path).convert('RGB') for path in image_paths]
    thumb_w, thumb_h = 620, 500
    prepared = []
    for path, image in zip(image_paths, images):
        image.thumbnail((thumb_w, thumb_h))
        canvas = Image.new('RGB', (thumb_w, thumb_h + 36), 'white')
        x = (thumb_w - image.width) // 2
        y = (thumb_h - image.height) // 2
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_h + 8), path.parent.name, fill='black')
        prepared.append(canvas)
    rows = math.ceil(len(prepared) / columns)
    sheet = Image.new('RGB', (columns * thumb_w, rows * (thumb_h + 36)), 'white')
    for index, image in enumerate(prepared):
        sheet.paste(image, ((index % columns) * thumb_w, (index // columns) * (thumb_h + 36)))
    sheet.save(output_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            '4.5초 LiDAR 구간에서 S3/S4를 scan별 A 기하 특징 -> target-frame 합성 -> '
            'A1.3 평탄화 -> A1.3.1 위치추정 입력 순서로 비교합니다.'
        )
    )
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--source-home', type=Path, default=Path('/home/esy'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--topic', default='/lidar_points')
    parser.add_argument('--timestamp-field', default='timestamp')
    parser.add_argument('--start-sec', type=float, default=222.65)
    parser.add_argument('--duration-sec', type=float, default=4.50)
    parser.add_argument(
        '--target-step', type=int, default=6,
        help='target scan을 몇 개 간격으로 처리할지. 6이면 약 0.6초 간격 8개, 1이면 전체 약 45개.',
    )
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()
    source_home = args.source_home.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tools_dir = Path(__file__).resolve().parent
    wrapper_path = tools_dir / 'run_stage_no_plots.py'
    python_executable = sys.executable

    if not bag.exists():
        raise FileNotFoundError(bag)
    if args.target_step < 1:
        raise ValueError('--target-step은 1 이상이어야 합니다.')
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f'기존 결과를 덮어쓰지 않습니다: {output_dir}. 이어서 실행하려면 --resume를 사용하세요.'
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = validate_sources(source_home)
    bag_start_ns, scans, all_target_indices = index_scans(
        bag=bag,
        topic=args.topic,
        timestamp_field=args.timestamp_field,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )
    target_indices = all_target_indices[::args.target_step]
    scan_rows_path = output_dir / 'scan_index.csv'
    save_csv(scan_rows_path, scans)

    needed_scan_indices: set[int] = set()
    target_specs = []
    for target_order, target_value in enumerate(target_indices):
        target_index = int(target_value)
        s3 = latest_indices(target_index, 3)
        s4 = latest_indices(target_index, 4)
        if len(s3) < 3 or len(s4) < 4:
            raise RuntimeError(f'이전 context scan이 부족합니다: target={target_index}')
        needed_scan_indices.update(int(value) for value in s4)
        w30 = w30_header_indices(scans, target_index)
        target_specs.append({
            'target_order': target_order,
            'target_scan_index': target_index,
            'target_point_end_ns': int(scans[target_index]['point_end_ns']),
            'target_point_end_sec_from_bag_start': float(scans[target_index]['point_end_sec_from_bag_start']),
            's3_scan_indices': [int(value) for value in s3],
            's4_scan_indices': [int(value) for value in s4],
            'w30_header_scan_indices': [int(value) for value in w30],
            'w30_equivalent_mode': exact_mode_for_w30(w30, s3, s4),
        })

    write_json(output_dir / 'experiment_plan.json', {
        'schema_version': 'a_s3_s4_segment_geometry_experiment_v0_1',
        'bag': str(bag),
        'bag_start_ns': int(bag_start_ns),
        'segment': {'start_sec': args.start_sec, 'duration_sec': args.duration_sec},
        'all_target_scan_count': int(len(all_target_indices)),
        'processed_target_scan_count': int(len(target_indices)),
        'target_step': int(args.target_step),
        'needed_unique_scan_count': int(len(needed_scan_indices)),
        'reference_definition': 'each target LiDAR scan point timestamp end',
        'modes': {'S3': 'current + previous 2 scans', 'S4': 'current + previous 3 scans'},
        'target_specs': target_specs,
        'scope': [
            'This stage audits the actual flattened localization inputs.',
            'It does not run V1.7 endpoint MCL or the normal-aware matcher yet.',
            'It does not modify V2.2, A1.3/A1.3.1 source, or V1.7.',
        ],
    })

    cache_root = output_dir / 'scan_cache'
    cache_root.mkdir(exist_ok=True)
    print(f'===== scan cache: {len(needed_scan_indices)} unique scans =====')
    for scan_index in sorted(needed_scan_indices):
        build_scan_cache(
            scan=scans[scan_index],
            bag=bag,
            source_paths=source_paths,
            wrapper_path=wrapper_path,
            cache_root=cache_root,
            python_executable=python_executable,
        )

    # 전체 target 범위에서 필요한 odom->base TF를 한 번 읽고 재사용한다.
    a1_module = load_module(source_paths['a1'], 'a1_source_for_segment_tf')
    storage_id = a1_module.detect_storage_id(bag)
    reader = a1_module.make_reader(bag, storage_id)
    topic_types = {info.name: info.type for info in reader.get_all_topics_and_types()}
    min_stamp = min(int(scans[index]['point_start_ns']) for index in needed_scan_indices)
    max_stamp = max(int(scans[index]['point_end_ns']) for index in needed_scan_indices)
    poses = a1_module.read_dynamic_poses(
        bag_dir=bag,
        storage_id=storage_id,
        topic_type=topic_types['/tf'],
        parent_frame='odom',
        child_frame='base_link',
        start_ns=min_stamp,
        end_ns=max_stamp,
        seek_margin_ns=2_000_000_000,
    )
    pose_interpolator = a1_module.PoseInterpolator(poses)

    samples_root = output_dir / 'samples'
    samples_root.mkdir(exist_ok=True)
    manifest_rows = []
    comparison_paths = []
    selected_paths_s3 = []
    selected_paths_s4 = []

    for spec in target_specs:
        target_order = int(spec['target_order'])
        target_index = int(spec['target_scan_index'])
        target_dir = samples_root / f'TARGET_{target_order:03d}_SCAN_{target_index:06d}'
        complete_path = target_dir / 'COMPLETE.json'
        if complete_path.is_file():
            print(f'[target] 재사용: {target_dir.name}')
            comparison = load_json(target_dir / 'comparison.json')
        else:
            if target_dir.exists():
                raise FileExistsError(
                    f'미완성 target 결과가 있습니다. 삭제하지 않고 중단합니다: {target_dir}'
                )
            target_dir.mkdir()
            write_json(target_dir / 'target.json', spec)
            for mode_name, key in [('s3', 's3_scan_indices'), ('s4', 's4_scan_indices')]:
                mode_dir = target_dir / mode_name
                mode_dir.mkdir()
                merged_npz = mode_dir / '_merged_a1_2_1.npz'
                merged_summary = mode_dir / '_merged_summary.json'
                group_indices = np.asarray(spec[key], dtype=np.int64)
                merge_group(
                    group_scan_indices=group_indices,
                    target_scan_index=target_index,
                    scans=scans,
                    cache_root=cache_root,
                    pose_interpolator=pose_interpolator,
                    output_npz=merged_npz,
                    output_summary=merged_summary,
                )
                run_flatten_and_selector(
                    merged_npz=merged_npz,
                    merged_summary=merged_summary,
                    mode_dir=mode_dir,
                    source_paths=source_paths,
                    wrapper_path=wrapper_path,
                    python_executable=python_executable,
                    group_scan_count=len(group_indices),
                )
                if merged_summary.exists():
                    merged_summary.rename(mode_dir / 'merged_geometry_summary.json')
                save_mode_visualizations(mode_dir, mode_name.upper())

            comparison = save_comparison(target_dir)
            write_json(complete_path, {
                'target': spec,
                'comparison': comparison,
                'completed_unix_sec': time.time(),
            })

        overlap = comparison['candidate_overlap']
        manifest_rows.append({
            **spec,
            's3_candidate_cells': comparison['s3']['structure_candidate_cells'],
            's4_candidate_cells': comparison['s4']['structure_candidate_cells'],
            's3_selected_cells': comparison['s3']['balanced_selected_cells'],
            's4_selected_cells': comparison['s4']['balanced_selected_cells'],
            'common_candidate_cells': overlap['common_cells'],
            's3_only_candidate_cells': overlap['s3_only_cells'],
            's4_only_candidate_cells': overlap['s4_only_cells'],
            'candidate_jaccard': overlap['jaccard'],
            's3_selected_support_mean': comparison['s3']['selected_support_mean'],
            's4_selected_support_mean': comparison['s4']['selected_support_mean'],
        })
        comparison_paths.append(target_dir / '05_s3_s4_candidate_overlay.png')
        selected_paths_s3.append(target_dir / 's3/02_selected_points.png')
        selected_paths_s4.append(target_dir / 's4/02_selected_points.png')

    save_csv(output_dir / 'target_manifest.csv', manifest_rows)
    contact_sheet(comparison_paths, output_dir / 'CONTACT_SHEET_S3_S4_OVERLAY.jpg', columns=4)
    contact_sheet(selected_paths_s3, output_dir / 'CONTACT_SHEET_S3_SELECTED.jpg', columns=4)
    contact_sheet(selected_paths_s4, output_dir / 'CONTACT_SHEET_S4_SELECTED.jpg', columns=4)

    jaccards = np.asarray([row['candidate_jaccard'] for row in manifest_rows], dtype=np.float64)
    s4_only = np.asarray([row['s4_only_candidate_cells'] for row in manifest_rows], dtype=np.float64)
    s3_only = np.asarray([row['s3_only_candidate_cells'] for row in manifest_rows], dtype=np.float64)
    w30_modes = Counter(row['w30_equivalent_mode'] for row in manifest_rows)
    segment_summary = {
        'schema_version': 'a_s3_s4_segment_geometry_summary_v0_1',
        'processed_targets': int(len(manifest_rows)),
        'unique_scan_caches': int(len(needed_scan_indices)),
        'candidate_jaccard': {
            'mean': float(np.mean(jaccards)),
            'min': float(np.min(jaccards)),
            'max': float(np.max(jaccards)),
        },
        'candidate_difference': {
            's4_only_mean_cells': float(np.mean(s4_only)),
            's3_only_mean_cells': float(np.mean(s3_only)),
        },
        'w30_equivalent_mode_on_processed_targets': dict(w30_modes),
        'next_decision': (
            'Inspect the contact sheets and per-target numeric manifest. '
            'Do not choose S3/S4 finally until the same targets are compared with '
            'the unchanged endpoint likelihood and a normal-aware lightweight matcher.'
        ),
    }
    write_json(output_dir / 'segment_summary.json', segment_summary)

    report = '\n'.join([
        '===== A S3/S4 Segment Geometry Experiment =====',
        f'bag: {bag}',
        f'segment: {args.start_sec:.2f} ~ {args.start_sec + args.duration_sec:.2f} sec from bag start',
        f'all target scans in segment: {len(all_target_indices)}',
        f'processed target scans: {len(target_indices)}',
        f'target step: {args.target_step}',
        f'unique per-scan A1.2.1 caches: {len(needed_scan_indices)}',
        f'candidate Jaccard mean/min/max: {segment_summary["candidate_jaccard"]["mean"]:.4f} / {segment_summary["candidate_jaccard"]["min"]:.4f} / {segment_summary["candidate_jaccard"]["max"]:.4f}',
        f'S4-only candidate cells mean: {segment_summary["candidate_difference"]["s4_only_mean_cells"]:.2f}',
        f'S3-only candidate cells mean: {segment_summary["candidate_difference"]["s3_only_mean_cells"]:.2f}',
        f'W30 equivalence: {dict(w30_modes)}',
        '',
        'This run generated the actual S3/S4 flattened localization input images.',
        'No endpoint MCL or normal-aware likelihood was run in this stage.',
        'No V2.2/A1/V1.7 source or previous result was overwritten.',
        f'output: {output_dir}',
    ])
    (output_dir / 'run_report.txt').write_text(report + '\n', encoding='utf-8')
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
