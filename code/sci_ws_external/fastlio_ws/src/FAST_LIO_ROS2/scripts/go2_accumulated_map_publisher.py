#!/usr/bin/env python3
# FLIO-10 go2_accumulated_map_publisher
#  /cloud_registered (FAST-LIO world/camera_init frame) 를 voxel 누적하여
#  /go2_accumulated_map (sensor_msgs/PointCloud2) 로 발행 + 종료 시 PLY/PCD 저장.
#  ★ 로봇 제어 토픽(publish/service/action) 전혀 없음. PointCloud2 발행 + 파일 저장만.
import os, sys, math, time, struct
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

try:
    import tf2_ros
    from tf2_ros import TransformException
    _HAVE_TF = True
except Exception:
    _HAVE_TF = False


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


def quat_to_R(x, y, z, w):
    n = math.sqrt(x*x+y*y+z*z+w*w) or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


class AccumulatedMap(Node):
    def __init__(self):
        super().__init__('go2_accumulated_map_publisher')
        d = self.declare_parameter
        d('input_topic', '/cloud_registered')
        d('output_topic', '/go2_accumulated_map')
        d('target_frame', '')            # "" => 입력 frame 그대로 누적(simple mode)
        d('publish_rate', 1.0)
        d('voxel_size', 0.05)
        d('min_range', 0.5)
        d('max_range', 80.0)
        d('min_z', -5.0)
        d('max_z', 5.0)
        d('max_points', 3000000)
        d('accumulate_every_n', 1)
        d('save_on_shutdown', True)
        d('save_path', os.path.expanduser('~/go2_ws/maps'))
        d('save_format', 'ply')          # ply | pcd (both ASCII)
        d('debug_stats', True)
        g = self.get_parameter
        self.input_topic = g('input_topic').value
        self.output_topic = g('output_topic').value
        # FLIO-10-R3: 빈 값/none/null 은 None 으로 정규화 → 입력 cloud frame_id 그대로 사용.
        _tf = g('target_frame').value
        if _tf is None or str(_tf).strip().lower() in ('', 'none', 'null'):
            self.target_frame = None
        else:
            self.target_frame = str(_tf).strip()
        self.voxel = float(g('voxel_size').value)
        self.min_range = float(g('min_range').value)
        self.max_range = float(g('max_range').value)
        self.min_z = float(g('min_z').value)
        self.max_z = float(g('max_z').value)
        self.max_points = int(g('max_points').value)
        self.every_n = max(1, int(g('accumulate_every_n').value))
        self.save_on_shutdown = bool(g('save_on_shutdown').value)
        self.save_path = os.path.expanduser(str(g('save_path').value))
        self.save_format = str(g('save_format').value).lower()
        self.debug = bool(g('debug_stats').value)
        rate = float(g('publish_rate').value)

        # voxel key(int i,j,k) -> [x,y,z] 대표점. numpy 구조로 관리.
        self.vox = {}                    # key tuple -> index into arrays
        self.pts = np.empty((0, 3), dtype=np.float32)
        self.frame_out = self.target_frame            # None 이면 첫 cloud frame_id 로 세팅
        self.n_recv = 0; self.n_used = 0; self.n_nan = 0; self.n_range = 0; self.capped = False
        self.n_pub = 0; self.first_cloud_logged = False
        self.start_time = time.time(); self.last_recv_time = None

        qos_in = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        qos_out = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, qos_out)
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb, qos_in)

        self.tf_buffer = None
        if _HAVE_TF and self.target_frame:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(1.0/max(0.1, rate), self.publish_map)
        self.create_timer(5.0, self.log_stats)   # 통계 + no-data 경고
        self.get_logger().info("=== go2_accumulated_map_publisher started (FLIO-10-R3) ===")
        self.get_logger().info(
            f"input_topic={self.input_topic} output_topic={self.output_topic} "
            f"target_frame={self.target_frame if self.target_frame else '(input frame_id)'}")
        self.get_logger().info(
            f"voxel_size={self.voxel} min_range={self.min_range} max_range={self.max_range} "
            f"publish_rate={rate} save_on_shutdown={self.save_on_shutdown} "
            f"save_path={self.save_path} save_format={self.save_format}")

    def maybe_transform(self, pts, msg):
        # simple mode: target_frame 없거나 입력과 동일 => 변환 없음
        src = msg.header.frame_id
        if not self.target_frame or self.target_frame == src:
            self.frame_out = src
            return pts
        if not self.tf_buffer:
            self.get_logger().warn("tf2 미가용 → 입력 frame 그대로 누적", throttle_duration_sec=10.0)
            self.frame_out = src
            return pts
        try:
            t = self.tf_buffer.lookup_transform(self.target_frame, src, rclpy.time.Time())
            tr = t.transform.translation; q = t.transform.rotation
            R = quat_to_R(q.x, q.y, q.z, q.w)
            out = (R @ pts.T.astype(np.float64)).T + np.array([tr.x, tr.y, tr.z])
            self.frame_out = self.target_frame
            return out.astype(np.float32)
        except (TransformException, Exception) as e:  # noqa
            self.get_logger().warn(f"tf transform 실패({src}->{self.target_frame}) → skip frame: {e}",
                                   throttle_duration_sec=10.0)
            return None

    def cb(self, msg):
        # FLIO-10-R3: callback 예외로 노드가 죽지 않도록 전체 방어.
        try:
            self.n_recv += 1
            self.last_recv_time = time.time()
            if (self.n_recv % self.every_n) != 0:
                return
            xyz = parse_xyz(msg)
            if xyz is None:
                self.get_logger().warn("cloud 에 x/y/z 필드 없음 → skip", throttle_duration_sec=10.0)
                return
            n_raw = xyz.shape[0]
            fin = np.isfinite(xyz).all(axis=1)
            self.n_nan += int((~fin).sum())
            xyz = xyz[fin]
            if xyz.shape[0] == 0:
                return
            xyz = self.maybe_transform(xyz, msg)
            if xyz is None:
                return
            r = np.linalg.norm(xyz, axis=1)
            m = (r >= self.min_range) & (r <= self.max_range) & \
                (xyz[:, 2] >= self.min_z) & (xyz[:, 2] <= self.max_z)
            self.n_range += int((~m).sum())
            xyz = xyz[m]
            accepted = xyz.shape[0]
            if accepted:
                self.accumulate(xyz)
            self.n_used += 1
            if not self.first_cloud_logged:
                self.first_cloud_logged = True
                self.get_logger().info(
                    f"[first cloud] frame_id={msg.header.frame_id} width={msg.width} "
                    f"height={msg.height} point_step={msg.point_step} raw={n_raw} "
                    f"accepted={accepted} voxels={len(self.vox)} out_frame={self.frame_out}")
        except Exception as e:
            self.get_logger().error(f"cb 예외(무시하고 계속): {e}")

    def accumulate(self, xyz):
        if self.capped:
            return
        keys = np.floor(xyz / self.voxel).astype(np.int64)
        new_pts = []
        for i in range(keys.shape[0]):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            if k not in self.vox:
                self.vox[k] = len(self.vox)
                new_pts.append(xyz[i])
        if new_pts:
            self.pts = np.vstack([self.pts, np.array(new_pts, dtype=np.float32)])
            if self.pts.shape[0] >= self.max_points:
                self.capped = True
                self.get_logger().warn(f"max_points({self.max_points}) 도달 → 누적 중단(발행/저장은 계속)")

    def build_msg(self):
        pts = self.pts
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_out or 'camera_init'
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * pts.shape[0]
        msg.is_dense = True
        msg.data = pts.astype(np.float32).tobytes()
        return msg

    def publish_map(self):
        if self.pts.shape[0] == 0:
            # 아직 누적 point 없음: 이유 안내(첫 cloud 미수신 또는 전부 필터됨)
            if self.n_recv == 0:
                self.get_logger().warn("아직 /cloud_registered 미수신 → publish 대기",
                                       throttle_duration_sec=5.0)
            else:
                self.get_logger().warn("누적 point 0 (전부 NaN/range/z 필터됨?) → publish skip",
                                       throttle_duration_sec=5.0)
            return
        self.pub.publish(self.build_msg())
        self.n_pub += 1

    def log_stats(self):
        # no-data 경고: 5초 이상 미수신
        if self.n_recv == 0 and (time.time() - self.start_time) > 5.0:
            self.get_logger().warn(
                f"5초 경과 /cloud_registered 미수신 — FAST-LIO(scan_pub) 실행/토픽/QoS 확인 "
                f"(input_topic={self.input_topic})")
        elif self.last_recv_time and (time.time() - self.last_recv_time) > 5.0:
            self.get_logger().warn(
                f"{round(time.time()-self.last_recv_time,1)}s 동안 신규 cloud 없음 "
                f"(recv={self.n_recv})")
        self.get_logger().info(
            f"[stats] recv={self.n_recv} used={self.n_used} pub={self.n_pub} "
            f"voxels={len(self.vox)} pts={self.pts.shape[0]} nan={self.n_nan} "
            f"range_skip={self.n_range} frame={self.frame_out} capped={self.capped}")

    def save(self):
        if not self.save_on_shutdown or self.pts.shape[0] == 0:
            self.get_logger().info("save skip (disabled or empty)")
            return
        os.makedirs(self.save_path, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        ext = 'pcd' if self.save_format == 'pcd' else 'ply'
        path = os.path.join(self.save_path, f"go2_accumulated_map_{ts}.{ext}")
        pts = self.pts
        n = pts.shape[0]
        with open(path, 'w') as f:
            if ext == 'ply':
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"comment go2_accumulated_map frame={self.frame_out} voxel={self.voxel}\n")
                f.write(f"element vertex {n}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("end_header\n")
                for p in pts:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            else:  # pcd ascii
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
                f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {n}\nDATA ascii\n")
                for p in pts:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        self.get_logger().info(f"[SAVED] {path}  ({n} points)")


def main():
    rclpy.init()
    node = AccumulatedMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
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
