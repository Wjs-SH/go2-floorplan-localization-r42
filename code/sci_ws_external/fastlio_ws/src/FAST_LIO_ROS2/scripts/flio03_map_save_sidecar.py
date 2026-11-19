#!/usr/bin/env python3
# FLIO-12 flio03_map_save_sidecar
#   flio03 front-end 를 전혀 건드리지 않고 /cloud_registered 만 구독하여 map 을 파일로 저장.
#   ★ 기본적으로 어떤 ROS 토픽도 publish 하지 않는다(publish_preview:=true 일 때만 preview 발행).
#   ★ 로봇 제어(publish/service/action) 전혀 없음. 구독 + 파일 저장만.
#   sidecar 가 죽어도 flio03 mapping 은 계속 동작(독립 프로세스).
import os, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField


def parse_xyz(msg):
    fx = {f.name: f.offset for f in msg.fields}
    n = msg.width * msg.height
    if n == 0 or 'x' not in fx:
        return None
    ps = msg.point_step
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)[:n*ps].reshape(n, ps)
    def colf(name):
        off = fx[name]
        return arr[:, off:off+4].copy().view(np.float32).ravel()
    return np.stack([colf('x'), colf('y'), colf('z')], axis=1)


class MapSaveSidecar(Node):
    def __init__(self):
        super().__init__('flio03_map_save_sidecar')
        d = self.declare_parameter
        d('input_topic', '/cloud_registered')
        d('voxel_size', 0.08)
        d('min_range', 0.8)
        d('max_range', 50.0)
        d('min_z', -3.0)
        d('max_z', 3.0)
        d('accumulate_every_n', 1)
        d('max_points', 5000000)
        d('save_path', '/home/unitree/go2_ws/maps')
        d('save_format', 'ply')            # ply | pcd (ASCII)
        d('save_on_shutdown', True)
        d('autosave_sec', 0.0)             # >0 이면 주기 autosave
        d('publish_preview', False)        # 기본 OFF (화면 보존)
        d('preview_topic', '/flio03_save_preview_map')
        d('preview_rate', 0.5)
        d('confirmed_min_hits', 2)         # confirmed_save 에서 유지할 최소 hit
        d('mode', 'raw_save')              # raw_save | confirmed_save
        d('debug_stats', True)
        g = self.get_parameter
        self.input_topic = g('input_topic').value
        self.voxel = float(g('voxel_size').value)
        self.min_range = float(g('min_range').value)
        self.max_range = float(g('max_range').value)
        self.min_z = float(g('min_z').value)
        self.max_z = float(g('max_z').value)
        self.every_n = max(1, int(g('accumulate_every_n').value))
        self.max_points = int(g('max_points').value)
        self.save_path = os.path.expanduser(str(g('save_path').value))
        self.save_format = str(g('save_format').value).lower()
        self.save_on_shutdown = bool(g('save_on_shutdown').value)
        self.autosave_sec = float(g('autosave_sec').value)
        self.publish_preview = bool(g('publish_preview').value)
        self.preview_topic = g('preview_topic').value
        preview_rate = float(g('preview_rate').value)
        self.confirmed_min_hits = int(g('confirmed_min_hits').value)
        self.mode = str(g('mode').value).lower()
        self.debug = bool(g('debug_stats').value)

        # voxel key -> [sum_x, sum_y, sum_z, hits]  (running average)
        self.acc = {}
        self.frame_id = None
        self.capped = False
        self.n_recv = 0; self.n_used = 0; self.n_nan = 0; self.n_range = 0; self.n_z = 0
        self.start = time.time(); self.last_recv = None; self.first_logged = False; self.n_save = 0

        qos_in = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        # ★ 구독만. flio03 화면/성능에 영향 없음.
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb, qos_in)

        # preview publisher 는 명시적으로 켤 때만 생성(기본 OFF → 화면 보존).
        self.pub = None
        if self.publish_preview:
            qos_out = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub = self.create_publisher(PointCloud2, self.preview_topic, qos_out)
            self.create_timer(1.0/max(0.1, preview_rate), self.publish_preview_cb)

        self.create_timer(5.0, self.log_stats)
        if self.autosave_sec > 0:
            self.create_timer(self.autosave_sec, lambda: self.save(tag='autosave'))

        self.get_logger().info("=== flio03_map_save_sidecar started (FLIO-12) — SAVE ONLY, flio03 화면 미변경 ===")
        self.get_logger().info(
            f"input_topic={self.input_topic} mode={self.mode} voxel={self.voxel} "
            f"range=[{self.min_range},{self.max_range}] z=[{self.min_z},{self.max_z}] "
            f"confirmed_min_hits={self.confirmed_min_hits}")
        self.get_logger().info(
            f"save_path={self.save_path} save_format={self.save_format} "
            f"save_on_shutdown={self.save_on_shutdown} autosave_sec={self.autosave_sec} "
            f"publish_preview={self.publish_preview}")

    def cb(self, msg):
        try:
            self.n_recv += 1
            self.last_recv = time.time()
            if (self.n_recv % self.every_n) != 0:
                return
            if self.frame_id is None:
                self.frame_id = msg.header.frame_id
            xyz = parse_xyz(msg)
            if xyz is None:
                self.get_logger().warn("cloud 에 x/y/z 없음 → skip", throttle_duration_sec=10.0)
                return
            n_raw = xyz.shape[0]
            fin = np.isfinite(xyz).all(axis=1)
            self.n_nan += int((~fin).sum())
            xyz = xyz[fin]
            if xyz.shape[0] == 0:
                return
            r = np.linalg.norm(xyz, axis=1)
            rm = (r >= self.min_range) & (r <= self.max_range)
            self.n_range += int((~rm).sum())
            xyz = xyz[rm]
            zm = (xyz[:, 2] >= self.min_z) & (xyz[:, 2] <= self.max_z)
            self.n_z += int((~zm).sum())
            xyz = xyz[zm]
            accepted = xyz.shape[0]
            if accepted:
                self.accumulate(xyz)
            self.n_used += 1
            if not self.first_logged:
                self.first_logged = True
                self.get_logger().info(
                    f"[first cloud] frame_id={msg.header.frame_id} width={msg.width} "
                    f"raw={n_raw} accepted={accepted} voxels={len(self.acc)}")
        except Exception as e:
            self.get_logger().error(f"cb 예외(무시하고 계속): {e}")

    def accumulate(self, xyz):
        if self.capped:
            return
        keys = np.floor(xyz / self.voxel).astype(np.int64)
        acc = self.acc
        for i in range(keys.shape[0]):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            v = acc.get(k)
            if v is None:
                acc[k] = [float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]), 1]
            else:
                v[0] += xyz[i, 0]; v[1] += xyz[i, 1]; v[2] += xyz[i, 2]; v[3] += 1
        if len(acc) >= self.max_points:
            self.capped = True
            self.get_logger().warn(f"max_points({self.max_points}) 도달 → 누적 중단(저장 계속)")

    def _points(self):
        # running average → 대표점. confirmed_save 는 hit>=min_hits 만.
        min_hits = self.confirmed_min_hits if self.mode == 'confirmed_save' else 1
        out = []
        for v in self.acc.values():
            h = v[3]
            if h >= min_hits:
                out.append((v[0]/h, v[1]/h, v[2]/h))
        return np.array(out, dtype=np.float32) if out else np.empty((0, 3), dtype=np.float32)

    def publish_preview_cb(self):
        if self.pub is None:
            return
        pts = self._points()
        if pts.shape[0] == 0:
            return
        h = Header(); h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.frame_id or 'camera_init'
        m = PointCloud2(); m.header = h; m.height = 1; m.width = pts.shape[0]
        m.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                    for n, o in (('x', 0), ('y', 4), ('z', 8))]
        m.is_bigendian = False; m.point_step = 12; m.row_step = 12*pts.shape[0]
        m.is_dense = True; m.data = pts.tobytes()
        self.pub.publish(m)

    def log_stats(self):
        if self.n_recv == 0 and (time.time() - self.start) > 5.0:
            self.get_logger().warn(
                f"5초 경과 {self.input_topic} 미수신 — flio03(FAST-LIO scan_pub) 실행/토픽 확인")
        self.get_logger().info(
            f"[stats] received_clouds={self.n_recv} used={self.n_used} voxel_count={len(self.acc)} "
            f"skipped_nan={self.n_nan} skipped_range={self.n_range} skipped_z={self.n_z} "
            f"elapsed_sec={round(time.time()-self.start,1)} capped={self.capped}")

    def save(self, tag=None):
        pts = self._points()
        if not self.save_on_shutdown and tag != 'autosave':
            return
        if pts.shape[0] == 0:
            self.get_logger().warn("save skip: 누적 point 0")
            return
        os.makedirs(self.save_path, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        ext = 'pcd' if self.save_format == 'pcd' else 'ply'
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.save_path, f"flio03_map_{ts}{suffix}.{ext}")
        n = pts.shape[0]
        with open(path, 'w') as f:
            if ext == 'ply':
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"comment flio03_map_save_sidecar frame={self.frame_id} voxel={self.voxel} mode={self.mode}\n")
                f.write(f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                for p in pts:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            else:
                f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
                f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
                for p in pts:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        self.n_save += 1
        self.get_logger().info(f"[SAVED] {path}  vertices={n}  (mode={self.mode})")


def main():
    rclpy.init()
    node = MapSaveSidecar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.save_on_shutdown:
                node.save()
        except Exception as e:
            node.get_logger().error(f"save 실패: {e}")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
