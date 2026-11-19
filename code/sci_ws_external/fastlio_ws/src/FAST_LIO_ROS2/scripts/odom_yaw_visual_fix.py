#!/usr/bin/env python3
# odom_yaw_visual_fix.py (PHASE FLIO-03-R1)
# 시각화 전용: /Odometry 를 구독하여 orientation 에 yaw offset 을 적용한 /Odometry_yawfix 를 발행.
# FAST-LIO 코어/입출력에 영향 없음. 로봇 제어 topic 발행 없음(오직 nav_msgs/Odometry 시각화 relay).
import argparse, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry

def qmul(a, b):  # Hamilton product, q=(x,y,z,w)
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )

def yaw_quat(deg):
    h = math.radians(deg) * 0.5
    return (0.0, 0.0, math.sin(h), math.cos(h))

class OdomYawFix(Node):
    def __init__(self, args):
        super().__init__('odom_yaw_visual_fix')
        self.args = args
        self.q_off = yaw_quat(args.yaw_offset_deg)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(Odometry, args.output, qos)
        self.sub = self.create_subscription(Odometry, args.input, self.cb, qos)
        self.get_logger().info(
            "odom_yaw_visual_fix: %s -> %s | yaw_offset=%.1fdeg order=%s (viz-only, no control)"
            % (args.input, args.output, args.yaw_offset_deg, args.order))
        self.n = 0

    def cb(self, msg: Odometry):
        o = msg.pose.pose.orientation
        q = (o.x, o.y, o.z, o.w)
        if self.args.order == 'pre':      # 전역(map)에서 회전: 화면상 arrow 방향 회전
            qf = qmul(self.q_off, q)
        else:                              # 로컬(body) 회전
            qf = qmul(q, self.q_off)
        out = Odometry()
        out.header = msg.header                       # frame_id/stamp 유지
        out.child_frame_id = self.args.child_frame_id if self.args.child_frame_id else msg.child_frame_id
        out.pose.pose.position = msg.pose.pose.position  # position 유지
        out.pose.pose.orientation.x = qf[0]
        out.pose.pose.orientation.y = qf[1]
        out.pose.pose.orientation.z = qf[2]
        out.pose.pose.orientation.w = qf[3]
        if self.args.keep_pose_cov:
            out.pose.covariance = msg.pose.covariance
        if self.args.keep_twist:
            out.twist = msg.twist                     # twist 유지
        self.pub.publish(out)
        self.n += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/Odometry')
    ap.add_argument('--output', default='/Odometry_yawfix')
    ap.add_argument('--yaw-offset-deg', type=float, default=-90.0)
    ap.add_argument('--order', choices=['pre', 'post'], default='pre')
    ap.add_argument('--child-frame-id', default='')   # 빈 값이면 원본 유지
    ap.add_argument('--keep-twist', type=int, default=1)
    ap.add_argument('--keep-pose-cov', type=int, default=1)
    a, _ = ap.parse_known_args()
    a.keep_twist = bool(a.keep_twist); a.keep_pose_cov = bool(a.keep_pose_cov)
    rclpy.init()
    node = OdomYawFix(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
