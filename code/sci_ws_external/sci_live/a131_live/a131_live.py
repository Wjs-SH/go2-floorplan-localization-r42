#!/usr/bin/env python3
"""a1_3_1 구조점을 메모리 안에서 바로 만든다(라이브용, 2026-10-06 성환이 요청).

기존 파이프라인(오케스트레이터 V0.4 + 빠른 판 /data2/a131_src_fast)은 스캔마다 A1 → A1.1 → A1.2 → A1.2.1 을 서브프로세스로 돌리고
npz 를 디스크에 썼다 읽은 뒤, target 스캔과 직전 2스캔을 합쳐(merge) A1.3 격자 → A1.3.1 로 표본 하나를 만든다.
여기서는 계산 코드는 그대로 불러 쓰고(결과를 같게 유지), 다음만 바꾼다.
  - 서브프로세스 대신 같은 프로세스에서 각 단계 main() 을 부른다. 단계 사이 npz 는 메모리에 둔다(np.savez/np.load 가로채기).
    저장 dtype(예: float32)은 그대로 거쳐 가므로 디스크 왕복과 같은 값이 된다.
  - A1 의 rosbag 읽기 대신 스캔 배열(x,y,z,intensity,ring,timestamp)과 자세 보간기를 직접 받는다(계산식은 A1 main 그대로).
  - A1.3.1 의 384셀 선택(select_balanced)은 최종 all_candidate_xy/role 에 쓰이지 않으므로 하지 않는다.
    all_candidate 는 A1.3.1 main 과 같은 assign_roles + grid_cell_xy 로 만든다(같은 기본 임계값).
  - 스캔별 결과(A1.2.1)는 캐시해 다음 표본에서 다시 쓴다(표본마다 새 스캔 1개만 계산).
주의: 전달본의 role=4 에는 오프라인 유령점 마스크(apply_ghost_mask, 녹화 전체 필요)도 들어 있다. 라이브판에는 없다.

    from a131_live import A131Live
    L = A131Live(static_matrix=T_base_lidar)
    L.add_scan(scan_id, xyz, intensity, ring, t_sec, interp)   # 스캔이 들어올 때마다(interp: A1 PoseInterpolator)
    out = L.extract([id0, id1, id_target], merge_interp)      # → {"xy": (N,2) float32, "role": (N,) int8, "stamp_ns": int}
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# 단계 코드 위치: 환경변수 A131_SOURCE_HOME > 이 파일 옆 source/ (전달 묶음) > /data2/a131_src_fast (작업 서버)
_HERE_SOURCE = Path(__file__).resolve().parent / "source"
SOURCE_HOME = Path(os.environ.get("A131_SOURCE_HOME", str(_HERE_SOURCE) if _HERE_SOURCE.is_dir() else "/data2/a131_src_fast")) \
    / "4F_multilevel_frame_dataset_v3_0_a"
STAGES = {
    "a1": "A1_SINGLE_SAMPLE_PROTOTYPE/extract_a1_single_sample.py",
    "a1_1": "A1_1_REFINED_STRUCTURE_PROTOTYPE/build_a1_1_from_a1.py",
    "a1_2": "A1_2_GEOMETRIC_CHANNEL_PROTOTYPE/build_a1_2_from_a1_1.py",
    "a1_2_1": "A1_2_1_ENRICHED_GEOMETRY_PROTOTYPE/build_a1_2_1_from_a1_2.py",
    "a1_3": "A1_3_FLATTENED_MULTICHANNEL_DATASET/build_a1_3_from_a1_2_1.py",
    "a1_3_1": "A1_3_1_BALANCED_LOCALIZATION_SELECTOR/refine_a1_3_selection.py",
    "segment": "A_S3_S4_SEGMENT_GEOMETRY_EXPERIMENT_V0_1/01_run_s3_s4_segment_geometry.py",
}
NOPLOT = ("save_pngs", "save_plots", "save_scalar_plot", "save_grid_plot", "save_height_layers_plot", "save_normal_field_plot",
          "save_preview_plot", "save_v2_overlay", "save_balanced_plot", "save_old_new_plot", "save_component_plot",
          "save_role_count_plot")
# KCI 2026-11-13 경량화 스위치: 환경변수 A131_LEAN=0 이면 끈다(원래대로). 단계 이름 = "a131live_<STAGES 키>"
LEAN = os.environ.get("A131_LEAN", "1") != "0"
# KCI 2026-11-13 C++ 커널(ckernel/liba131_kernel.so): 빠른 면방향일 때 이웃 표와 PCA 모멘트 누적을 C++ 로.
# 라이브러리가 없거나 A131_CKERNEL=0 이면 파이썬 판 그대로. 같은 연산 순서라 파이썬 빠른 판과 끝자리까지 같다(ckernel/check_ckernel.py).
_CK = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ckernel"))
    import a131_ckernel as _ck_mod
    if _ck_mod.available():
        _CK = _ck_mod
except Exception:
    _CK = None


class _Zero(float):
    """보고 문구·요약 JSON 자리 채움: 어떤 키를 물어도 0.0(통계는 단계 끝 보고 문구에만 쓰인다)."""
    def __new__(cls):
        return super().__new__(cls, 0.0)

    def __getitem__(self, key):
        return self


LEAN_STUBS = {
    "a131live_a1_1": {"save_statistics": lambda: _Zero()},
    "a131live_a1_2": {"build_balanced_preview": lambda: ({}, _Zero()), "save_csvs": lambda: _Zero()},
    "a131live_a1_2_1": {"round_robin_select": lambda: np.empty(0, dtype=np.int64)},
}
# A1 기본값(extract_a1_single_sample.py main 의 argparse 기본값과 같음)
A1_ARGS = dict(min_range_m=0.30, feature_max_range_m=20.0, ring_radius=1, column_radius=2, min_neighbors=6,
               range_jump_base_m=0.20, range_jump_ratio=0.04, tf_boundary_tolerance_sec=0.75)
# merge_group 이 스캔 결과(A1.2.1)에서 읽는 키. 이것만 캐시에 남긴다(115개 중 14개, 프로세스 사이 전달·메모리 절약)
MERGE_KEYS = ("base_xyz_deskewed", "a1_1_normals", "a1_2_1_projection_ready_mask", "point_timestamp_ns", "ring", "a1_2_1_scan_column",
              "a1_2_1_smooth_surface_score", "a1_2_1_depth_jump_edge_score", "a1_2_1_normal_turn_edge_score",
              "a1_2_1_low_support_vertical_score", "a1_2_1_far_uncertain_score", "a1_2_1_low_structure_score",
              "a1_2_1_support_density_normalized", "a1_2_1_reason_flags")
# A1.3.1 역할 임계값(refine_a1_3_selection.py main 기본값과 같음)
ROLE_ARGS = dict(normal_turn_threshold=0.28, depth_threshold=0.32, low_support_threshold=0.32, far_start_m=6.0,
                 far_uncertainty_threshold=0.08, far_low_support_threshold=0.35)


class _D:
    """그림 라이브러리 가짜: 무엇을 부르든 자기 자신을 돌려준다(plt.subplots → fig, ax 둘로 풀림)."""

    def __getattr__(self, k):
        return self

    def __call__(self, *a, **k):
        return self

    def __getitem__(self, k):
        return self

    def __iter__(self):
        return iter((self, self))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_matplotlib():
    """단계 함수들이 안에서 import matplotlib.pyplot 하는 그림 저장을 통째로 없앤다(PNG 인코딩·tk 생성이 스캔당 0.1초 넘게 들었다).
    로봇에서 화면 라이브러리를 아예 불러오지 않게 된다."""
    import types
    for name in ("matplotlib", "matplotlib.pyplot", "matplotlib.colors", "matplotlib.cm"):
        m = types.ModuleType(name)
        m.__getattr__ = lambda k: _D()
        sys.modules[name] = m
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]


def _stub_missing_ros():
    """A1 모듈은 rosbag 읽기용으로 rosbag2_py·rclpy·sensor_msgs·yaml 을 맨 위에서 불러온다. 라이브판은 그 읽기 함수를 쓰지 않으므로
    없으면(로봇 Foxy 가상환경 등) 가짜 모듈로 채워 불러오기만 되게 한다. 있으면 건드리지 않는다."""
    import types
    for name in ("rosbag2_py", "rclpy.serialization", "rosidl_runtime_py.utilities", "sensor_msgs.msg", "yaml"):
        try:
            importlib.import_module(name)
            continue
        except Exception:
            pass
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub not in sys.modules:
                m = types.ModuleType(sub)
                m.__getattr__ = lambda k: _D()
                m.__path__ = []
                sys.modules[sub] = m
            if i > 1:
                setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], sys.modules[sub])


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for n in NOPLOT:
        if hasattr(mod, n):
            setattr(mod, n, lambda *a, **k: None)
    if LEAN:
        # KCI 2026-11-13 경량화: 오프라인 보고용 계산(통계 JSON·CSV·그림용 미리보기 고르기)을 건너뛴다.
        # 최종 출력(a1_3_1 구조점 xy·role)에는 안 쓰인다 — demo_from_dump.py 로 기준 55표본과 비트 동일 확인.
        for n, ret in LEAN_STUBS.get(name, {}).items():
            if hasattr(mod, n):
                setattr(mod, n, (lambda r: (lambda *a, **k: r()))(ret))
    return mod


class _MemNpz:
    """np.load 가 돌려주는 NpzFile 흉내(files, [], with)."""

    def __init__(self, d):
        self._d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self._d[k]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


class MemIO:
    """root 아래 경로의 np.savez*/np.load 를 메모리로 돌린다. 저장할 때 np.asarray 로 복사해 dtype 을 그대로 둔다
    (np.savez 는 배열을 그 dtype 그대로 쓰고 np.load 는 그대로 읽으므로 디스크 왕복과 값이 같다). 존재 검사용으로 빈 파일을 남긴다."""

    def __init__(self, root: Path, roundtrip: bool = False):
        self.root = str(root)
        self.roundtrip = roundtrip                # True: 저장·읽기를 실제 npz 직렬화(BytesIO)로 — 디스크와 똑같은 배열(메모리 배치 포함)
        self.store: dict[str, dict[str, np.ndarray]] = {}
        self._orig = (np.savez, np.savez_compressed, np.load)

    def _mine(self, p):
        return str(p).startswith(self.root)

    def _save(self, orig):
        def f(file, *args, **kw):
            if not self._mine(file):
                return orig(file, *args, **kw)
            d = {f"arr_{i}": np.array(a) for i, a in enumerate(args)}
            d.update({k: np.array(v) for k, v in kw.items()})
            key = str(file) if str(file).endswith(".npz") else str(file) + ".npz"
            if self.roundtrip:
                import io as _io
                buf = _io.BytesIO()
                self._orig[0](buf, **d)
                d = buf
            self.store[key] = d
            Path(key).parent.mkdir(parents=True, exist_ok=True)
            Path(key).touch()
        return f

    def _load_fn(self, file, *a, **kw):
        if self._mine(file) and str(file) in self.store:
            v = self.store[str(file)]
            if not isinstance(v, dict):
                v.seek(0)
                return self._orig[2](v, *a, **kw)
            return _MemNpz(v)
        return self._orig[2](file, *a, **kw)

    @contextlib.contextmanager
    def active(self):
        np.savez, np.savez_compressed, np.load = self._save(self._orig[0]), self._save(self._orig[1]), self._load_fn
        try:
            yield
        finally:
            np.savez, np.savez_compressed, np.load = self._orig

    def _as_dict(self, v):
        if v is None or isinstance(v, dict):
            return v
        v.seek(0)
        with self._orig[2](v) as z:
            return {k: np.asarray(z[k]) for k in z.files}

    def get(self, key):
        return self._as_dict(self.store.get(str(key)))

    def pop(self, key):
        return self._as_dict(self.store.pop(str(key), None))

    def put(self, key, d):
        if self.roundtrip:
            import io as _io
            buf = _io.BytesIO()
            self._orig[0](buf, **d)
            d = buf
        self.store[str(key)] = d
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        Path(key).touch()


class A131Live:
    def __init__(self, static_matrix: np.ndarray, workdir: str = "/dev/shm/a131_live", keep_scans: int = 6, debug: bool = False,
                 roundtrip: bool = False, fast: bool = True, workers: int = 0, exact_pca: bool = True):
        if fast:
            _stub_matplotlib()
        self.fast = fast                          # True: 최종 결과에 안 쓰이는 계산을 뺀다(그림, A1 구조 특징)
        _stub_missing_ros()
        self.M = {k: _load(SOURCE_HOME / v, f"a131live_{k}") for k, v in STAGES.items()}
        self.exact_pca = exact_pca                # False: 면 방향 PCA 를 한 번에 계산하는 빠른 판(부동소수 끝자리 차이 → 경계 칸 몇 개 달라질 수 있음)
        if not exact_pca:
            sys.modules["a131_vec"].support_pca = support_pca_fast
            if _CK is not None:
                sys.modules["a131_vec"].neighbor_table = _CK.neighbor_table
        self.T_base_lidar = np.asarray(static_matrix, np.float64)
        self.work = Path(workdir) / f"p{os.getpid()}"
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        self.io = MemIO(self.work, roundtrip=roundtrip)
        self.cache: dict[int, dict] = {}          # scan_id → {"npz": A1.2.1 결과, "meta": 스캔 정보}
        self.keep = keep_scans
        self.timing: dict[str, list[float]] = {}
        self.debug, self.stage_out = debug, {}    # debug: 스캔별 단계 결과(a1, a1_1, a1_2, a1_2_1)를 남긴다(대조용)
        self.pool = None                          # workers > 1: 스캔을 프로세스 여러 개로 동시에(각자 단계 모듈을 한 번 불러 둠)
        if workers > 1:
            import multiprocessing as mp
            self.pool = mp.get_context("fork").Pool(workers, initializer=_worker_init,
                                                     initargs=(self.T_base_lidar, str(Path(workdir) / "w"), fast, exact_pca))

    def _t(self, name, t0):
        self.timing.setdefault(name, []).append(time.perf_counter() - t0)

    def _run_main(self, mod, argv):
        old = sys.argv
        sys.argv = ["stage", *map(str, argv)]
        try:
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                rc = mod.main()
        finally:
            sys.argv = old
        if rc not in (0, None):
            raise RuntimeError(f"{mod.__name__} 반환 {rc}")

    # ---------- A1: 스캔 배열 → a1_single_sample (A1 main 의 계산식 그대로, bag 읽기만 뺌) ----------
    def _a1(self, raw_xyz, intensity, ring, point_time_sec, target_stamp_ns, interp, out_dir: Path):
        a1 = self.M["a1"]
        args = A1_ARGS
        raw_xyz = np.asarray(raw_xyz, np.float64)
        intensity = np.asarray(intensity, np.float32)
        ring = np.asarray(ring, np.uint16)
        point_time_sec = np.asarray(point_time_sec, np.float64)
        point_time_ns = np.rint(point_time_sec * 1e9).astype(np.int64)
        finite_xyz = np.all(np.isfinite(raw_xyz), axis=1)
        nonzero_xyz = np.linalg.norm(raw_xyz, axis=1) > 1e-6
        finite_time = np.isfinite(point_time_sec)
        valid_raw = finite_xyz & nonzero_xyz & finite_time
        if not valid_raw.any():
            raise RuntimeError("유효한 XYZ/timestamp 포인트가 없습니다.")
        static_matrix = self.T_base_lidar
        reference_resolved_ns, reference_tf_clamp_ns = interp.resolve_stamp(target_stamp_ns)
        reference_world_base = interp.matrix_at(reference_resolved_ns)
        reference_base_world = a1.invert_transform(reference_world_base)
        base_no_deskew = a1.apply_transform(raw_xyz, static_matrix)
        base_deskewed = np.full_like(base_no_deskew, np.nan)
        unique_times, inverse_indices = np.unique(point_time_ns[valid_raw], return_inverse=True)
        valid_indices = np.flatnonzero(valid_raw)
        # 원본은 unique 시각마다 inverse_indices == k 로 전체를 훑었다(O(U·N)). 같은 묶음을 정렬로 한 번에 나눈다(점 순서 동일).
        order = np.argsort(inverse_indices, kind="stable")
        bounds = np.searchsorted(inverse_indices[order], np.arange(len(unique_times) + 1))
        for unique_idx, stamp_ns in enumerate(unique_times):
            point_indices = valid_indices[order[bounds[unique_idx]:bounds[unique_idx + 1]]]
            try:
                resolved_stamp_ns, clamp_ns = interp.resolve_stamp(int(stamp_ns))
            except ValueError:
                continue
            current_world_base = interp.matrix_at(resolved_stamp_ns)
            lidar_to_reference_base = reference_base_world @ current_world_base @ static_matrix
            base_deskewed[point_indices] = a1.apply_transform(raw_xyz[point_indices], lidar_to_reference_base)
        deskew_valid = np.all(np.isfinite(base_deskewed), axis=1)
        base_range = np.linalg.norm(base_deskewed, axis=1)
        feature_range_mask = deskew_valid & (base_range >= args["min_range_m"]) & (base_range <= args["feature_max_range_m"])
        if self.fast:
            # A1 의 구조 특징(점별 PCA)은 하류(A1.1~A1.3.1)에서 읽지 않는다. 키 자리만 맞춰 둔다(isolated_mask 는 진단용).
            features = {"feature_valid_mask": feature_range_mask.copy()}
        else:
            features, _ = a1.compute_structured_features(
                points_base=base_deskewed, ring=ring, valid_mask=feature_range_mask, range_m=base_range,
                ring_radius=args["ring_radius"], column_radius=args["column_radius"], min_neighbors=args["min_neighbors"],
                range_jump_base_m=args["range_jump_base_m"], range_jump_ratio=args["range_jump_ratio"])
        invalid_xyz_mask = ~finite_xyz | ~nonzero_xyz
        range_excluded_mask = deskew_valid & ((base_range < args["min_range_m"]) | (base_range > args["feature_max_range_m"]))
        isolated_mask = feature_range_mask & ~features["feature_valid_mask"]
        data = dict(raw_xyz_hesai=raw_xyz.astype(np.float32), base_xyz_no_deskew=base_no_deskew.astype(np.float32),
                    base_xyz_deskewed=base_deskewed.astype(np.float32), intensity=intensity, ring=ring,
                    point_timestamp_sec=point_time_sec, point_timestamp_ns=point_time_ns, valid_raw_mask=valid_raw,
                    invalid_xyz_mask=invalid_xyz_mask, deskew_valid_mask=deskew_valid, feature_range_mask=feature_range_mask,
                    range_excluded_mask=range_excluded_mask, isolated_mask=isolated_mask, range_m=base_range.astype(np.float32),
                    **features)
        self.io.put(out_dir / "a1_single_sample.npz", {k: np.array(v) for k, v in data.items()})
        tmin, tmax = float(np.min(point_time_sec[finite_time])), float(np.max(point_time_sec[finite_time]))
        summary = {"source": {"target_stamp_ns": int(target_stamp_ns),
                              "selected_scan": {"point_time_min_sec": tmin, "point_time_max_sec": tmax,
                                                "point_time_span_sec": tmax - tmin, "contains_reference_time": True}}}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def _compute_scan(self, scan_id: int, xyz, intensity, ring, t_sec, interp, scan_meta: dict | None = None):
        """스캔 하나의 A1 → A1.1 → A1.2 → A1.2.1. (A1.2.1 결과 dict, 스캔 정보) 를 돌려준다(등록은 _register)."""
        t_sec = np.asarray(t_sec, np.float64)
        fin = np.isfinite(t_sec)
        end_ns = int(round(float(np.max(t_sec[fin])) * 1e9))
        meta = dict(scan_meta or {})
        meta.setdefault("scan_index", int(scan_id))
        meta.setdefault("point_end_ns", end_ns)
        meta.setdefault("point_start_ns", int(round(float(np.min(t_sec[fin])) * 1e9)))
        if not hasattr(self, "_t0_ns"):
            self._t0_ns = int(meta["point_end_ns"])
        # merge_group 요약(merged_summary.json)에만 쓰이는 기록값. 계산에는 쓰이지 않는다
        meta.setdefault("point_end_sec_from_bag_start", (int(meta["point_end_ns"]) - self._t0_ns) / 1e9)
        d = self.work / f"SCAN_{scan_id:06d}"
        if d.exists():
            shutil.rmtree(d)
        w = d / "_work"
        seg = self.M["segment"]
        with self.io.active():
            t0 = time.perf_counter()
            self._a1(xyz, intensity, ring, t_sec, int(meta["point_end_ns"]), interp, w / "a1")
            self._t("A1", t0)
            sample_json = d / "sample.json"
            sample_json.write_text(json.dumps({"output_stamp_ns": int(meta["point_end_ns"]), "geometry": seg.sample_geometry(),
                                               "source_scan": meta, "reference_definition": "scan point timestamp end"}))
            for name, argv in (
                ("A1.1", ("a1_1", ["--input-npz", w / "a1/a1_single_sample.npz", "--output-dir", w / "a1_1"])),
                ("A1.2", ("a1_2", ["--input-npz", w / "a1_1/a1_1_refined_sample.npz", "--output-dir", w / "a1_2"])),
                ("A1.2.1", ("a1_2_1", ["--input-npz", w / "a1_2/a1_2_geometric_channels.npz", "--sample-json", sample_json,
                                       "--a1-summary-json", w / "a1/summary.json", "--output-dir", w / "a1_2_1"])),
            ):
                t0 = time.perf_counter()
                self._run_main(self.M[argv[0]], argv[1])
                self._t(name, t0)
        if self.debug:
            self.stage_out[int(scan_id)] = {k: self.io.get(w / p) for k, p in (
                ("a1", "a1/a1_single_sample.npz"), ("a1_1", "a1_1/a1_1_refined_sample.npz"),
                ("a1_2", "a1_2/a1_2_geometric_channels.npz"), ("a1_2_1", "a1_2_1/a1_2_1_enriched_geometry.npz"))}
        final = self.io.pop(w / "a1_2_1/a1_2_1_enriched_geometry.npz")
        if self.fast:
            final = {k: final[k] for k in MERGE_KEYS}
        for k in [k for k in self.io.store if k.startswith(str(d))]:
            self.io.store.pop(k)
        shutil.rmtree(d, ignore_errors=True)
        return final, meta

    def _register(self, scan_id: int, final: dict, meta: dict):
        # merge_group 이 cache_root/SCAN_xxxxxx/a1_2_1_enriched_geometry.npz 를 읽는다
        self.io.put(self.work / f"SCAN_{scan_id:06d}" / "a1_2_1_enriched_geometry.npz", final)
        self.cache[int(scan_id)] = dict(meta=meta)
        for old in sorted(self.cache)[:-self.keep]:
            self.cache.pop(old)
            self.io.pop(self.work / f"SCAN_{old:06d}" / "a1_2_1_enriched_geometry.npz")
            shutil.rmtree(self.work / f"SCAN_{old:06d}", ignore_errors=True)

    def add_scan(self, scan_id: int, xyz, intensity, ring, t_sec, interp, scan_meta: dict | None = None):
        """스캔 하나를 계산해 캐시에 넣는다. scan_meta 에는 index_scans 와 같은 키(point_start_ns, point_end_ns …)."""
        final, meta = self._compute_scan(scan_id, xyz, intensity, ring, t_sec, interp, scan_meta)
        self._register(scan_id, final, meta)

    def add_scans(self, items):
        """여러 스캔을 한꺼번에(workers > 1 이면 프로세스 여러 개로 동시에). items = [(scan_id, xyz, intensity, ring, t_sec, interp, meta), …].
        이미 캐시에 있는 scan_id 는 건너뛴다. 표본 간격이 0.3초보다 길면 표본마다 새 스캔 3개가 필요해 이걸 쓴다."""
        todo = [it for it in items if int(it[0]) not in self.cache]
        if not todo:
            return
        if self.pool is None or len(todo) == 1:
            for it in todo:
                self.add_scan(*it)
            return
        t0 = time.perf_counter()
        res = self.pool.map(_worker_scan, todo)
        self._t("스캔 동시 처리(대기)", t0)
        for it, (final, meta, timing) in zip(todo, res):
            for k, v in timing.items():
                self.timing.setdefault(k, []).extend(v)
            self._register(int(it[0]), final, meta)

    def extract(self, scan_ids, merge_interp) -> dict:
        """scan_ids = [직전-2, 직전-1, target] (add_scan 으로 넣은 id). merge_interp: merge 에 쓸 자세 보간기."""
        seg, a13, a131 = self.M["segment"], self.M["a1_3"], self.M["a1_3_1"]
        ids = [int(v) for v in scan_ids]
        # merge_group 은 scans[i] 를 scan_index 로 찾으므로 id → 자리 맞춤 목록을 만든다
        scans = {i: self.cache[i]["meta"] for i in ids}
        scans_list = _IndexList(scans)
        tgt = ids[-1]
        md = self.work / f"TARGET_{tgt:06d}"
        if md.exists():
            shutil.rmtree(md)
        md.mkdir(parents=True)
        with self.io.active():
            t0 = time.perf_counter()
            seg.merge_group(group_scan_indices=np.array(ids), target_scan_index=tgt, scans=scans_list, cache_root=self.work,
                            pose_interpolator=merge_interp, output_npz=md / "merged_a1_2_1.npz",
                            output_summary=md / "merged_summary.json")
            self._t("merge", t0)
            if self.debug:
                self.stage_out[("merged", int(scans[tgt]["point_end_ns"]))] = self.io.get(md / "merged_a1_2_1.npz")
            t0 = time.perf_counter()
            self._run_main(a13, ["--input-npz", md / "merged_a1_2_1.npz", "--input-summary-json", md / "merged_summary.json",
                                 "--output-dir", md / "_a1_3", "--localization-budget", "1024"])
            self._t("A1.3", t0)
            t0 = time.perf_counter()
            short = self.io.get(md / "_a1_3" / "a1_3_data_short.npz")
            data = {k: np.asarray(v) for k, v in short.items()}
            cand = data["structure_candidate"].astype(bool)
            role_grid = a131.assign_roles(candidate=cand, surface=data["surface_evidence"], depth=data["depth_jump_edge_max"],
                                          normal_turn=data["normal_turn_edge_max"], low_support=data["low_support_evidence"],
                                          uncertainty=data["uncertainty"], range_median=data["range_median_m"], **ROLE_ARGS)
            xy_flat = a131.grid_cell_xy(data).reshape(-1, 2)
            ids_c = np.flatnonzero(cand.reshape(-1)).astype(np.int32)
            out = dict(xy=xy_flat[ids_c].astype(np.float32), role=role_grid.reshape(-1)[ids_c].astype(np.int8),
                       flat_cell_id=ids_c, stamp_ns=int(scans[tgt]["point_end_ns"]))
            # 2026-10-09 추가: 가구/벽 구분용 셀 속성(A1.3 이 이미 계산해 둔 값, 추가 계산 없음)
            for _k, _dst in (("height_max_m", "height_max_m"), ("height_span_m", "height_span_m"),
                             ("fine_height_occupied_count", "height_bins"),
                             ("fine_height_longest_consecutive", "height_run"),
                             ("normal_concentration", "normal_conc"),
                             ("range_median_m", "range_m")):
                if _k in data: out[_dst] = np.asarray(data[_k]).reshape(-1)[ids_c]
            self._t("A1.3.1(역할)", t0)
        for k in [k for k in self.io.store if k.startswith(str(md))]:
            self.io.store.pop(k)
        shutil.rmtree(md, ignore_errors=True)
        return out


_W = None


def _eig3_sym(a11, a22, a33, a12, a13, a23):
    """대칭 3×3 행렬 묶음의 고유값(내림차순 e1≥e2≥e3)과 최소 고유값의 단위 고유벡터를 공식으로(삼각함수 방법) 한 번에 구한다.
    고유벡터는 (A − e3 I) 의 두 행 외적 중 가장 긴 것. 퇴화(거의 등방)하면 ok=False 로 돌려 원래 eigh 로 다시 푼다."""
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    with np.errstate(all="ignore"):
        ip = np.where(p > 0, 1.0 / p, 0.0)
        c11, c22, c33, c12, c13, c23 = b11 * ip, b22 * ip, b33 * ip, a12 * ip, a13 * ip, a23 * ip
        det = c11 * (c22 * c33 - c23 * c23) - c12 * (c12 * c33 - c23 * c13) + c13 * (c12 * c23 - c22 * c13)
        phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    r0 = np.stack([a11 - e3, a12, a13], -1); r1 = np.stack([a12, a22 - e3, a23], -1); r2 = np.stack([a13, a23, a33 - e3], -1)
    cands = [np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)]
    norms = np.stack([np.einsum("qi,qi->q", c, c) for c in cands], -1)
    pick = np.argmax(norms, axis=1)
    v = np.where((pick == 0)[:, None], cands[0], np.where((pick == 1)[:, None], cands[1], cands[2]))
    nv = np.sqrt(norms.max(axis=1))
    scale = np.maximum(np.abs(e1), 1e-300)
    ok = (nv > 1e-9 * scale * scale) & (p > 1e-15)
    with np.errstate(all="ignore"):
        v = v / nv[:, None]
    return np.stack([e1, e2, e3], -1), v, ok


def support_pca_fast(points, table, fit_mask, xp):
    """a131_vec.support_pca 와 같은 값(지지점 [중심, 수용 이웃]의 PCA: 고유값 내림차순, 최소 고유벡터 법선을 센서 쪽으로, 지지 반경)을
    빠르게: 이웃 오프셋 K개를 하나씩 돌며 중심 기준 상대좌표의 1·2차 모멘트를 누적하고(큰 중간 배열 없음), 3×3 고유값은 공식으로.
    수학적으로 같고 부동소수 끝자리만 다르다. 공식이 불안정한(거의 등방) 소수만 np.linalg.eigh 로 다시 푼다. 2026-10-06."""
    centers, nidx, acc = table["centers"], table["nidx"], table["acc"]
    m_count = int(centers.shape[0])
    count = 1 + acc.sum(axis=0)
    eig = np.full((m_count, 3), np.nan); nrm = np.full((m_count, 3), np.nan); rad = np.full(m_count, np.nan)
    ok = np.zeros(m_count, dtype=bool)
    cols = np.flatnonzero(fit_mask)
    if len(cols) == 0:
        return count, eig, nrm, rad, ok
    C = points[centers[cols]]
    q_n = len(cols)
    if _CK is not None:
        sx, sy, sz, sxx, syy, szz, sxy, sxz, syz, r2max = _CK.moments(points, table, cols)
    else:
        sx = np.zeros(q_n); sy = np.zeros(q_n); sz = np.zeros(q_n)
        sxx = np.zeros(q_n); syy = np.zeros(q_n); szz = np.zeros(q_n); sxy = np.zeros(q_n); sxz = np.zeros(q_n); syz = np.zeros(q_n)
        r2max = np.zeros(q_n)
        for k in range(acc.shape[0]):
            a = acc[k, cols]
            if not a.any():
                continue
            dlt = points[nidx[k, cols]] - C
            dx = np.where(a, dlt[:, 0], 0.0); dy = np.where(a, dlt[:, 1], 0.0); dz = np.where(a, dlt[:, 2], 0.0)
            sx += dx; sy += dy; sz += dz
            sxx += dx * dx; syy += dy * dy; szz += dz * dz; sxy += dx * dy; sxz += dx * dz; syz += dy * dz
            np.maximum(r2max, dx * dx + dy * dy + dz * dz, out=r2max)
    n = count[cols].astype(np.float64)
    mx, my, mz = sx / n, sy / n, sz / n
    den = np.maximum(1.0, n - 1.0)
    c11 = (sxx - n * mx * mx) / den; c22 = (syy - n * my * my) / den; c33 = (szz - n * mz * mz) / den
    c12 = (sxy - n * mx * my) / den; c13 = (sxz - n * mx * mz) / den; c23 = (syz - n * my * mz) / den
    w, normal, good_f = _eig3_sym(c11, c22, c33, c12, c13, c23)
    bad = ~good_f
    if bad.any():                                               # 공식이 불안정한 소수는 원래 eigh 로
        cov = np.stack([np.stack([c11, c12, c13], -1), np.stack([c12, c22, c23], -1), np.stack([c13, c23, c33], -1)], -2)[bad]
        wb, Vb = np.linalg.eigh(cov)
        w[bad] = wb[:, ::-1]; normal[bad] = Vb[:, :, 0]
    w = np.maximum(w, 0.0)
    norm = np.sqrt(np.einsum("qi,qi->q", normal, normal))
    good = (norm > 1e-12) & np.all(np.isfinite(w), axis=1)
    with np.errstate(all="ignore"):
        normal = normal / norm[:, None]
    flip = np.einsum("qi,qi->q", normal, C) > 0.0
    normal = np.where(flip[:, None], -normal, normal)
    eig[cols] = w; nrm[cols] = normal; rad[cols] = np.sqrt(r2max); ok[cols] = good
    return count, eig, nrm, rad, ok


def _worker_init(static_matrix, workdir, fast, exact_pca=True):
    global _W
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    _W = A131Live(static_matrix=static_matrix, workdir=workdir, fast=fast, exact_pca=exact_pca)


def _worker_scan(item):
    _W.timing = {}
    final, meta = _W._compute_scan(*item)
    return final, meta, _W.timing


class _IndexList:
    """merge_group 이 scans[scan_index] 로 접근하므로 dict 를 목록처럼 보이게 한다."""

    def __init__(self, d):
        self.d = d

    def __getitem__(self, i):
        return self.d[int(i)]

    def __len__(self):
        return max(self.d) + 1


# ---------- 자세·원거리 점 (성환이 노드가 FAST-LIO 자세로 바로 쓰도록) ----------
def make_interpolator(stamps_ns, xyz, quat_xyzw, a1_module=None, boundary_tolerance_ns: int = 750_000_000):
    """FAST-LIO odom→base_link 자세 목록으로 A1 과 같은 보간기(PoseInterpolator)를 만든다.
    녹화본 경로와 같게 하려고 쿼터니언을 행렬로 바꿨다가 다시 쿼터니언으로 만든다(A1 read_dynamic_poses 와 같은 변환)."""
    from scipy.spatial.transform import Rotation
    a1 = a1_module or sys.modules.get("a131live_a1")
    poses = []
    for s, p, q in zip(np.asarray(stamps_ns, np.int64), np.asarray(xyz, np.float64), np.asarray(quat_xyzw, np.float64)):
        m = Rotation.from_quat(q).as_matrix()
        poses.append(a1.TimedPose(stamp_ns=int(s), translation=np.asarray(p, np.float64).copy(),
                                  quaternion_xyzw=Rotation.from_matrix(m).as_quat()))
    return a1.PoseInterpolator(poses, boundary_tolerance_ns=boundary_tolerance_ns)


class FarPoses:
    """원거리 점용 자세(초 단위 시각, 위치, 쿼터니언). extract_far_generic_deskew.py 의 pose_at 과 같은 보간."""

    def __init__(self, t_sec, xyz, quat_xyzw):
        from scipy.spatial.transform import Rotation, Slerp
        ot, op, oq = np.asarray(t_sec, np.float64), np.asarray(xyz, np.float64), np.asarray(quat_xyzw, np.float64)
        o = np.argsort(ot); ot, op, oq = ot[o], op[o], oq[o]
        keep = np.concatenate([[True], np.diff(ot) > 0]); self.ot, self.op, self.oq = ot[keep], op[keep], oq[keep]
        self.slerp = Slerp(self.ot, Rotation.from_quat(self.oq))

    def pose_at(self, ts):
        ot, op = self.ot, self.op
        tc = np.clip(ts, ot[0], ot[-1])
        return self.slerp(tc), np.c_[np.interp(tc, ot, op[:, 0]), np.interp(tc, ot, op[:, 1]), np.interp(tc, ot, op[:, 2])]


FAR = dict(keep=10, near=9.0, far_hi=18.0, cap=3000)


def far_points(x, y, z, ring, t_sec, poses: FarPoses) -> np.ndarray:
    """target 스캔 하나의 원거리 점(몸체 xy, 9~18 m). _a131_clean/extract_far_generic_deskew.py deskew_far 와 같은 계산
    (외부보정 base=(-ly+0.145, lx, lz+0.110), 링 수평면 제거, 20 시간 구간 deskew → 스캔 끝시각 기준, 최대 3000점).
    x,y,z 는 PointCloud2 의 float32 값을 그대로(함수 안에서 float64 로). 10점 미만이면 빈 배열."""
    x = np.asarray(x, np.float32).ravel().astype(np.float64); y = np.asarray(y, np.float32).ravel().astype(np.float64)
    z = np.asarray(z, np.float32).ravel().astype(np.float64); tsp = np.asarray(t_sec, np.float64).ravel()
    ring = np.asarray(ring, np.uint16).ravel()
    n = len(x); ok_grid = n % 16 == 0 and (ring.reshape(-1, 16) == np.arange(16)).all()
    rr = np.sqrt(x * x + y * y + z * z)
    if ok_grid:
        R = rr.reshape(-1, 16); up = np.full_like(R, np.nan); dn = np.full_like(R, np.nan)
        up[:, :-1] = np.abs(R[:, 1:] - R[:, :-1]); dn[:, 1:] = np.abs(R[:, :-1] - R[:, 1:])
        dmin = np.fmin(up, dn).ravel()
    g = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    pb = np.c_[-y + 0.145, x, z + 0.110]; r0 = np.hypot(pb[:, 0], pb[:, 1])
    cand = g & (r0 >= 8) & (r0 < 19) & (pb[:, 2] > 0.1) & (pb[:, 2] < 1.9)
    if ok_grid:
        cand &= dmin < 0.05 * r0 + 0.10
    pb, tsp = pb[cand], tsp[cand]
    if len(pb) < FAR["keep"]:
        return np.zeros((0, 2), np.float32)
    t_ref = tsp.max()
    order = np.argsort(tsp); pb, tsp = pb[order], tsp[order]
    edges = np.linspace(tsp[0], tsp[-1] + 1e-9, 21); bins = np.clip(np.searchsorted(edges, tsp) - 1, 0, 19)
    bc = (edges[:-1] + edges[1:]) / 2; Rb, posb = poses.pose_at(bc); Rr, posr = poses.pose_at(np.array([t_ref])); Rr = Rr[0]; posr = posr[0]
    out = np.empty_like(pb)
    for bi in range(20):
        mk = bins == bi
        if not mk.any():
            continue
        po = Rb[bi].apply(pb[mk]) + posb[bi]
        out[mk] = Rr.inv().apply(po - posr)
    r = np.hypot(out[:, 0], out[:, 1]); fm = (r >= FAR["near"]) & (r < FAR["far_hi"]) & (out[:, 2] > 0.2) & (out[:, 2] < 1.8)
    fxy = out[fm, :2].astype(np.float32)
    if len(fxy) > FAR["cap"]:
        fxy = fxy[np.linspace(0, len(fxy) - 1, FAR["cap"]).astype(int)]
    return fxy if len(fxy) >= FAR["keep"] else np.zeros((0, 2), np.float32)


# ---------- FAST-LIO 자세 → odom→base_link (전달본 a1_3_1 을 만든 녹화본과 같은 규약) ----------
LEVER_IMU_BASE = np.array([0.026, 0.0, -0.042])   # imu → base_link (base_link←imu tf_static t=(-0.026,0,0.042), R=I)
DELTA_DEG_DEFAULT = 45.3                          # 몸체(imu) x 축과 로봇 전진 방향의 차이. 녹화본 추정값 44.7~45.7°(6F·8F·S1·S6)


def fastlio_to_base(p_imu, q_imu_xyzw, delta_deg: float = DELTA_DEG_DEFAULT, mirror: bool = True):
    """FAST-LIO /Odometry(camera_init→imu) 자세 → odom→base_link 자세. 전달본을 만든 녹화본(build_s6_tffix_bag.py →
    build_a131_clean_bag_v7.py tffix 교정)과 같은 순서:
      ① 레버암: p_base = p_imu + R_imu · LEVER, R_base = R_imu
      ② 거울상(Sy): p.y → −p.y, q=(−qx, qy, −qz, qw)
      ③ 몸체 z 축 δ 회전: R = R_sy · Rz(δ), p = p_sy − R_sy·LEVER + R·LEVER
    p_imu (N,3), q (N,4) 묶음도 된다. 돌려주는 값: (p (N,3), q (N,4) xyzw)."""
    from scipy.spatial.transform import Rotation as Rot
    p = np.atleast_2d(np.asarray(p_imu, np.float64)); q = np.atleast_2d(np.asarray(q_imu_xyzw, np.float64))
    R = Rot.from_quat(q)
    pb = p + R.apply(LEVER_IMU_BASE)
    qb = R.as_quat()
    if mirror:
        pb = pb * np.array([1.0, -1.0, 1.0])
        qb = qb * np.array([-1.0, 1.0, -1.0, 1.0])
    R_sy = Rot.from_quat(qb)
    R_t = R_sy * Rot.from_euler("z", delta_deg, degrees=True)
    pt = pb - R_sy.apply(LEVER_IMU_BASE) + R_t.apply(LEVER_IMU_BASE)
    return pt, R_t.as_quat()


def estimate_delta_deg(t_sec, x, y, yaw_unwrapped):
    """직진 구간(속도 > 0.25 m/s, 0.5 s 동안 |Δyaw| < 3°) 이동 방향의 몸체 기준 각도 최빈값(도). 거울상 교정 뒤 자세로 부른다.
    build_a131_clean_bag_v7.py motion_direction_mode 와 같은 식. (δ, 직진 창 수, ±20° 안 비율)"""
    T, X, Y, W = map(lambda v: np.asarray(v, np.float64), (t_sec, x, y, yaw_unwrapped))
    tg = np.arange(T[0], T[-1], 0.1)
    xg, yg, wg = np.interp(tg, T, X), np.interp(tg, T, Y), np.interp(tg, T, W)
    k = 5
    dx, dy, dw = xg[k:] - xg[:-k], yg[k:] - yg[:-k], wg[k:] - wg[:-k]
    st = (np.hypot(dx, dy) / (k * 0.1) > 0.25) & (np.abs(dw) < np.radians(3))
    c, s = np.cos(wg[:-k]), np.sin(wg[:-k])
    ang = np.degrees(np.arctan2(-s * dx + c * dy, c * dx + s * dy))[st]
    if len(ang) == 0:
        return float("nan"), 0, 0.0
    h, e = np.histogram(ang, bins=np.arange(-180, 181, 1))
    m = e[np.argmax(h)] + 0.5
    near = ang[np.abs(((ang - m) + 180) % 360 - 180) < 20]
    return float(np.median(near)), int(st.sum()), float(len(near) / max(len(ang), 1))


def fastlio_to_odom_link(p_imu, q_imu_xyzw):
    """FAST-LIO /Odometry(camera_init→imu) → 성환이 위치추정 입력 odom_link.csv 규약(fastlio_x, fastlio_y, fastlio_yaw_rad).
    전달본 odom_link.csv 와 대조(2026-10-06, S6): x := −x, y 그대로, yaw := −yaw (위치 차 0.0001 m 이하).
    돌려주는 값: (x, y, yaw) 배열. yaw 는 쿼터니언의 zyx 오일러 z 각."""
    from scipy.spatial.transform import Rotation as Rot
    p = np.atleast_2d(np.asarray(p_imu, np.float64))
    yaw = Rot.from_quat(np.atleast_2d(np.asarray(q_imu_xyzw, np.float64))).as_euler("zyx")[:, 0]
    return -p[:, 0], p[:, 1].copy(), -yaw
