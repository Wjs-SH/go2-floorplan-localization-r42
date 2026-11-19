"""e2_feed.py — LIVE2: FAST-LIO OUTPUTS (/Odometry + /cloud_registered, same header stamp) -> the student's LiveStore format.

No raw /lidar_points subscription. Pure logic (no rclpy) so it is testable; lio_monitor2 calls on_odom / on_cloud.

Coordinate chain (all from files actually used at run time, no tuned constant):
  /cloud_registered (FAST-LIO dense_publish_en: true -> feats_undistort: every point after the blind filter, deskewed to the
  scan end, in camera_init) and /Odometry (camera_init -> IMU body, header stamp = the same scan end time)
    p_imu   = R_imu^T (p_w - t_imu)                              exact inverse of FAST-LIO's pointBodyToWorld
    p_lidar = R_ext^T (p_imu - T_ext)                            extrinsic_R / extrinsic_T of the running FAST-LIO yaml
    p_base  = R_bl p_lidar + t_bl                                student STATIC_BASE_LIDAR (rt_live_node.py, a1_3_1 frame)
  odom row  = fastlio_to_odom_link(p_imu, q_imu) = (-x, y, -yaw)  student a131_live convention of odom_link.csv
So samples are in the same frames as the student's recorded a1_3_1 data; beta / mirror are still estimated per window by
the student's fit (runtime estimate, never a fixed 228 deg) and E2 records what it found.

Information lost against the student's raw-LiDAR pipeline (recorded in every sample meta and in the report):
  - no ring / per-point time -> the A1 PCA structure classification (normals, edges, roles) and the esy cleaner (ghost /
    person / moving-object removal) cannot run. The structure points are a PROXY: 8 cm cells of the wall-height band
    (-0.15..1.6 m in base, the band rt_live_node uses for its live scan) with >= 2 returns and >= 0.15 m vertical span.
  - FAST-LIO's blind (0.5 m) and its deskew are applied (the student deskews itself).
  - role is 0 for every proxy cell (the student uses role != 4 only, so no role information is used downstream).
"""
import json
import math
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np

STATIC_BASE_LIDAR = np.array([0.0, -1.0, 0.0, 0.145, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.110, 0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
CELL_M, BAND_Z, NEAR_M, MIN_HITS, MIN_SPAN = 0.08, (-0.15, 1.6), (0.5, 9.0), 2, 0.15
FAR_KEEP, FAR_NEAR, FAR_HI, FAR_CAP, FAR_Z = 10, 9.0, 18.0, 1000, (0.2, 1.8)   # cap 1000 (student 3000, typical a1 ~470):
#   LiveStore rewrites far_points.npz on every add, FAST-LIO dense clouds would otherwise give 3000 every sample
PROXY_NOTE = "FASTLIO_CLOUD_REGISTERED_PROXY (no ring/time: no A1 structure roles, no esy cleaner)"


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def odom_link(p, q):
    """Student a131_live.fastlio_to_odom_link for one pose: x := -x, y, yaw := -yaw (zyx euler z of the IMU quaternion)."""
    x, y, z, w = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return -float(p[0]), float(p[1]), -yaw


def read_extrinsic(yaml_path):
    import yaml
    d = yaml.safe_load(Path(yaml_path).read_text())
    m = d['/**']['ros__parameters']['mapping']
    R = np.array(m['extrinsic_R'], float).reshape(3, 3)
    T = np.array(m['extrinsic_T'], float)
    return R, T


def structure_proxy(pb):
    """base-frame points (N,3) -> (cells xy (M,2) float32, far xy (K,2) float32, stats)."""
    pb = pb[np.isfinite(pb).all(1)]
    r = np.hypot(pb[:, 0], pb[:, 1])
    m = (pb[:, 2] > BAND_Z[0]) & (pb[:, 2] < BAND_Z[1]) & (r >= NEAR_M[0]) & (r < NEAR_M[1])
    q = pb[m]
    cells = np.zeros((0, 2), np.float32)
    if len(q):
        ij = np.floor(q[:, :2] / CELL_M).astype(np.int64)
        key = ij[:, 0] * 2_000_003 + ij[:, 1]
        order = np.argsort(key, kind='stable')
        key, z, ij = key[order], q[order, 2], ij[order]
        u, start, cnt = np.unique(key, return_index=True, return_counts=True)
        zmax = np.maximum.reduceat(z, start)
        zmin = np.minimum.reduceat(z, start)
        ok = (cnt >= MIN_HITS) & (zmax - zmin >= MIN_SPAN)
        cells = ((ij[start[ok]] + 0.5) * CELL_M).astype(np.float32)
    fm = (r >= FAR_NEAR) & (r < FAR_HI) & (pb[:, 2] > FAR_Z[0]) & (pb[:, 2] < FAR_Z[1])
    far = pb[fm, :2].astype(np.float32)
    if len(far) > FAR_CAP:
        far = far[np.linspace(0, len(far) - 1, FAR_CAP).astype(int)]
    if len(far) < FAR_KEEP:
        far = np.zeros((0, 2), np.float32)
    return cells, far, dict(n_in=int(len(pb)), n_band=int(m.sum()), n_cells=int(len(cells)), n_far=int(len(far)))


class E2Feed:
    """Pairs clouds with the odometry of the SAME stamp and writes one LiveStore sample every sample_dt seconds
    (target scan + the previous merge-1 scans, all expressed in the target base frame)."""

    def __init__(self, store_root, extrinsic, sample_dt=1.5, merge=3, live_store_cls=None, clock=time.time, session_info=None):
        self.root = Path(store_root)
        self.R_ext, self.T_ext = (np.asarray(v, float) for v in extrinsic)
        self.R_bl, self.t_bl = STATIC_BASE_LIDAR[:3, :3], STATIC_BASE_LIDAR[:3, 3]
        self.sample_dt, self.merge = float(sample_dt), int(merge)
        self.LS = live_store_cls
        self.clock = clock
        self.session_info = dict(session_info or {})
        self.poses = OrderedDict()               # stamp_ns -> (p (3,), q (4,))
        self.scans = deque(maxlen=self.merge)    # (stamp_ns, xyz_world float32)
        self.store = None
        self.session = None
        self.last_sample_ns = None
        self.lock = threading.Lock()
        self.stats = dict(clouds=0, odoms=0, samples=0, no_pose=0, too_few=0, sample_ms=deque(maxlen=200), last=None,
                          reset=0, gap_dropped=0)

    # ---------- session ----------
    def new_session(self, why, first_stamp_ns):
        self.session = dict(self.session_info, odom_session_id=f"{self.session_info.get('owner', 'fastlio')}:{first_stamp_ns}",
                            started_wall=self.clock(), why=why, first_stamp_ns=int(first_stamp_ns), input=PROXY_NOTE,
                            extrinsic_R=self.R_ext.tolist(), extrinsic_T=self.T_ext.tolist(),
                            static_base_lidar=STATIC_BASE_LIDAR.tolist(), sample_dt=self.sample_dt, merge=self.merge)
        self.store = self.LS(self.root, keep_s=900.0, reset=True)
        (self.root / 'session.json').write_text(json.dumps(self.session))
        self.poses.clear(); self.scans.clear(); self.last_sample_ns = None
        self.stats['reset'] += 1

    # ---------- inputs ----------
    def on_odom(self, stamp_ns, p, q, session_changed=False):
        with self.lock:
            if self.session is None or session_changed:
                self.new_session('FIRST_ODOMETRY' if self.session is None else 'ODOM_SESSION_CHANGED', stamp_ns)
            self.stats['odoms'] += 1
            self.poses[int(stamp_ns)] = (np.asarray(p, float), np.asarray(q, float))
            while len(self.poses) > 400:
                self.poses.popitem(last=False)

    def on_cloud(self, stamp_ns, xyz_world):
        """Cheap (callback thread): keep the cloud. Returns True when a sample is due; sample() does the work."""
        with self.lock:
            self.stats['clouds'] += 1
            if self.session is None:
                return False
            stamp_ns = int(stamp_ns)
            if self.scans and stamp_ns - self.scans[-1][0] > 0.35e9:     # missing scans: merge only consecutive scans
                self.scans.clear(); self.stats['gap_dropped'] += 1
            self.scans.append((stamp_ns, np.asarray(xyz_world, np.float32)))
            if len(self.scans) < self.merge:
                return False
            return self.last_sample_ns is None or stamp_ns - self.last_sample_ns >= self.sample_dt * 1e9 - 0.05e9

    def sample(self):
        """Worker thread: build and store one sample from the newest merge scans (snapshot under the lock)."""
        with self.lock:
            if len(self.scans) < self.merge or self.store is None:
                return None
            scans = list(self.scans)
            poses = [self._pose(s) for s, _ in scans]
            store = self.store
            if self.last_sample_ns is not None and scans[-1][0] - self.last_sample_ns < self.sample_dt * 1e9 - 0.05e9:
                return None
            self.last_sample_ns = scans[-1][0]
        return self._sample(scans, poses, store)

    # ---------- sample ----------
    def _pose(self, stamp_ns):
        v = self.poses.get(stamp_ns)
        if v is None:                                  # tolerate 1 ms header jitter, never interpolate further
            near = [k for k in self.poses if abs(k - stamp_ns) <= 1_000_000]
            v = self.poses[near[0]] if near else None
        return v

    def _sample(self, scans, poses, store):
        t0 = time.perf_counter()
        tgt_ns = scans[-1][0]
        if any(p is None for p in poses):
            self.stats['no_pose'] += 1
            return None
        p_t, q_t = poses[-1]
        R_t = quat_to_R(q_t)
        pw = np.vstack([xyz for _, xyz in scans]).astype(np.float64)
        p_imu = (pw - p_t) @ R_t                       # R_t^T (p_w - t)
        p_lidar = (p_imu - self.T_ext) @ self.R_ext    # R_ext^T (p_imu - T_ext)
        pb = p_lidar @ self.R_bl.T + self.t_bl
        cells, far, st = structure_proxy(pb)
        if len(cells) < 20:
            self.stats['too_few'] += 1
            return None
        x, y, yaw = odom_link(p_t, q_t)
        with self.lock:
            if store is not self.store:                # session changed while building: old data, drop it
                return None
            sid = store.add(tgt_ns / 1e9, x, y, yaw, cells, np.zeros(len(cells), np.int8), far)
        ms = 1000 * (time.perf_counter() - t0)
        self.stats['sample_ms'].append(ms)
        if sid is not None:
            self.stats['samples'] += 1
            self.stats['last'] = dict(sid=sid, stamp=tgt_ns / 1e9, wall=self.clock(), ms=round(ms, 1), **st)
        return sid

    def summary(self):
        s = dict(self.stats)
        sm = list(s.pop('sample_ms'))
        s['sample_ms_p50'] = None if not sm else round(float(np.median(sm)), 1)
        s['sample_ms_p95'] = None if not sm else round(float(np.percentile(sm, 95)), 1)
        s['session'] = None if self.session is None else {k: self.session[k] for k in ('odom_session_id', 'started_wall', 'why')}
        s['store'] = str(self.root)
        s['span_s'] = None if self.store is None else round(self.store.span(), 1)
        s['input'] = PROXY_NOTE
        return s
