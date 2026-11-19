#!/usr/bin/env python3
"""Go2 에서 대조할 시험 데이터: 원본 스캔 + 자세 + 오프라인 정제가 지운 점(기준 마스크).
사용: make_testdata.py <원본 bag> <오프라인 정제 bag> <out.npz> [--n 40] [--skip 100]"""
import sys, glob, sqlite3, struct, numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as Rot
RAW, REF, OUT = sys.argv[1:4]
gv=lambda k,d:int(sys.argv[sys.argv.index(k)+1]) if k in sys.argv else d
N=gv('--n',40); SKIP=gv('--skip',100)
DT=np.dtype({'names':['x','y','z','i','ring','ts'],'formats':['<f4','<f4','<f4','<f4','<u2','<f8'],'offsets':[0,4,8,12,16,18],'itemsize':26})
def rd(b):
    g=glob.glob(b+'/*.db3')[0]; c=sqlite3.connect(f'file:{g}?mode=ro&immutable=1',uri=True)
    tp={n:(i,t) for i,n,t in c.execute('SELECT id,name,type FROM topics')}
    PC=get_message(tp['/lidar_points'][1])
    first=bytes(c.execute(f"SELECT data FROM messages WHERE topic_id={tp['/lidar_points'][0]} ORDER BY timestamp LIMIT 1").fetchone()[0])
    m0=deserialize_message(first,PC); N0=m0.width*26; OFF=first.find(struct.pack('<III',26,N0,N0),0,1024)+12
    out=[]
    for (d,) in c.execute(f"SELECT data FROM messages WHERE topic_id={tp['/lidar_points'][0]} ORDER BY timestamp"):
        blob=bytes(d); ps,rs,nd=struct.unpack_from('<III',blob,OFF-12)
        if not (ps==26 and rs==nd and nd%26==0): out.append(None); continue
        sec,nsec=struct.unpack_from('<iI',blob,4)
        out.append((sec+nsec*1e-9, np.frombuffer(blob,DT,count=nd//26,offset=OFF).copy()))
    return c,tp,out
c1,tp1,A=rd(RAW); _,_,Bq=rd(REF)
TFM=get_message(tp1['/tf'][1]); P=[]
for (d,) in c1.execute(f"SELECT data FROM messages WHERE topic_id={tp1['/tf'][0]} ORDER BY timestamp"):
    m=deserialize_message(bytes(d),TFM)
    for tr in m.transforms:
        if tr.child_frame_id.endswith('base_link') and tr.header.frame_id.endswith('odom'):
            q=tr.transform.rotation
            P.append((tr.header.stamp.sec+tr.header.stamp.nanosec*1e-9,tr.transform.translation.x,tr.transform.translation.y,
                      Rot.from_quat([q.x,q.y,q.z,q.w]).as_euler('zyx')[0]))
P=np.array(sorted(P))
# 이력이 필요하므로 SKIP 번째부터 N 개를 고르되, 앞 스캔도 전부 넣는다(실시간처럼 순서대로 먹인다)
end=min(len(A), SKIP+N)
keep=[i for i in range(end) if A[i] is not None]
d={'pose_t':P[:,0],'pose_x':P[:,1],'pose_y':P[:,2],'pose_yaw':np.unwrap(P[:,3]),'eval_from':SKIP}
for j,i in enumerate(keep):
    t,rec=A[i]; ref=Bq[i][1]
    d[f's{j}_t']=np.float64(t); d[f's{j}_x']=rec['x']; d[f's{j}_y']=rec['y']; d[f's{j}_z']=rec['z']
    d[f's{j}_i']=rec['i']; d[f's{j}_ring']=rec['ring']; d[f's{j}_ts']=rec['ts']
    d[f's{j}_killref']=(~np.isfinite(ref['x'].astype(np.float64)) & np.isfinite(rec['x'].astype(np.float64)))
d['n']=len(keep)
np.savez_compressed(OUT, **d); print(f'{OUT}: 스캔 {len(keep)}개(평가는 {SKIP}번째부터), {np.round(sum(v.nbytes for v in d.values() if hasattr(v,"nbytes"))/1e6,1)} MB')
