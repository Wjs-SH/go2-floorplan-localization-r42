#!/usr/bin/env python3
# FLIO-16: 저장 PCD/PLY → PointCloud2 발행(RViz 확인용). 기본 frame=camera_init.
#   ★ 로봇 제어 없음. 저장맵 시각화 발행만. (frame 기본이 camera_init 인 것이 FLIO-15 대비 핵심 수정)
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcd_io import read_pcd, get_xyz
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

def read_any(path):
    if path.lower().endswith('.ply'):
        lines=open(path).read().splitlines(); n=0; i=0
        for i,ln in enumerate(lines):
            if ln.startswith('element vertex'): n=int(ln.split()[-1])
            if ln.strip()=='end_header': break
        return np.array([[float(x) for x in lines[i+1+j].split()[:3]] for j in range(n)],dtype=np.float32)
    cols,_=read_pcd(path); return get_xyz(cols).astype(np.float32)

class Pub(Node):
    def __init__(self,a):
        super().__init__('publish_saved_map')
        self.pts=read_any(a.input); self.pts=self.pts[np.isfinite(self.pts).all(1)].astype(np.float32)
        self.frame=a.frame; self.topic=a.topic
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,history=HistoryPolicy.KEEP_LAST,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub=self.create_publisher(PointCloud2,self.topic,qos)
        self.create_timer(1.0/max(0.05,a.rate),self.tick)
        self.get_logger().info(f"publish_saved_map: {a.input} → {self.topic} ({self.pts.shape[0]} pts, frame={self.frame}, transient_local)")
        self.get_logger().info("RViz Fixed Frame=map_viz 권장. 맵이 거꾸로 보이면 --frame camera_init 먼저 시도(FAST-LIO 저장맵 기본 좌표).")
        self.tick()
    def tick(self):
        h=Header(); h.stamp=self.get_clock().now().to_msg(); h.frame_id=self.frame
        m=PointCloud2(); m.header=h; m.height=1; m.width=self.pts.shape[0]
        m.fields=[PointField(name=n,offset=o,datatype=PointField.FLOAT32,count=1) for n,o in (('x',0),('y',4),('z',8))]
        m.is_bigendian=False; m.point_step=12; m.row_step=12*self.pts.shape[0]; m.is_dense=True
        m.data=self.pts.tobytes(); self.pub.publish(m)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--topic',default='/saved_map')
    ap.add_argument('--frame',default='camera_init')   # FLIO-16 수정: 기본 camera_init
    ap.add_argument('--rate',type=float,default=0.5)
    a=ap.parse_args()
    rclpy.init(); n=Pub(a)
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass

if __name__=='__main__': main()
