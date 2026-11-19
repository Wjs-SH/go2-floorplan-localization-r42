#!/usr/bin/env python3
"""실시간 정제판 검증: 원본 bag 을 LiveCleaner 로 정제한 결과와, 오프라인 정제 bag(기준)을 점 단위로 대조한다.
사용: validate_clean_live.py <원본 bag> <오프라인 정제 bag> [--max 0]"""
import sys, glob, sqlite3, struct, time, math, numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/data/kci_work/KCI_L0_P0_UNIFIED/live_a131_clean')
from a131_clean_live import LiveCleaner
RAW, REF = sys.argv[1:3]
MAXS = int(sys.argv[sys.argv.index('--max')+1]) if '--max' in sys.argv else 0
DT = np.dtype({'names':['x','y','z','i','ring','ts'],'formats':['<f4','<f4','<f4','<f4','<u2','<f8'],'offsets':[0,4,8,12,16,18],'itemsize':26})
def open_bag(b):
    g=glob.glob(b+'/*.db3')[0]; c=sqlite3.connect(f'file:{g}?mode=ro&immutable=1',uri=True)
    tp={n:(i,t) for i,n,t in c.execute('SELECT id,name,type FROM topics')}
    return c, tp
def scans(c, tp):
    PC=get_message(tp['/lidar_points'][1])
    first=bytes(c.execute(f"SELECT data FROM messages WHERE topic_id={tp['/lidar_points'][0]} ORDER BY timestamp LIMIT 1").fetchone()[0])
    m0=deserialize_message(first,PC); N0=m0.width*26; OFF=first.find(struct.pack('<III',26,N0,N0),0,1024)+12
    for (d,) in c.execute(f"SELECT data FROM messages WHERE topic_id={tp['/lidar_points'][0]} ORDER BY timestamp"):
        blob=bytes(d); ps,rs,nd=struct.unpack_from('<III',blob,OFF-12)
        if not (ps==26 and rs==nd and nd%26==0): continue
        sec,nsec=struct.unpack_from('<iI',blob,4)
        yield sec+nsec*1e-9, np.frombuffer(blob,DT,count=nd//26,offset=OFF)
c1,tp1=open_bag(RAW); c2,tp2=open_bag(REF)
TFM=get_message(tp1['/tf'][1]); P=[]
for (d,) in c1.execute(f"SELECT data FROM messages WHERE topic_id={tp1['/tf'][0]} ORDER BY timestamp"):
    m=deserialize_message(bytes(d),TFM)
    for tr in m.transforms:
        if tr.child_frame_id.endswith('base_link') and tr.header.frame_id.endswith('odom'):
            q=tr.transform.rotation
            P.append((tr.header.stamp.sec+tr.header.stamp.nanosec*1e-9,tr.transform.translation.x,tr.transform.translation.y,
                      Rot.from_quat([q.x,q.y,q.z,q.w]).as_euler('zyx')[0]))
P=np.array(sorted(P)); PT,PX,PY,PW=P[:,0],P[:,1],P[:,2],np.unwrap(P[:,3])
import os
C=LiveCleaner(**eval(os.environ.get('LC_KW','{}')))
ref_iter=scans(c2,tp2)
tot=same=only_live=only_off=npts=0; times=[]
for k,(t,rec) in enumerate(scans(c1,tp1)):
    tr_,rec2=next(ref_iter)
    x=rec['x'].astype(np.float64); y=rec['y'].astype(np.float64); z=rec['z'].astype(np.float64)
    pose=(float(np.interp(t,PT,PX)),float(np.interp(t,PT,PY)),float(np.interp(t,PT,PW)))
    t0=time.perf_counter()
    _,_,_,_,_,_,kill=C.clean(k,x,y,z,rec['i'],rec['ring'],t,pose=pose,return_mask=True)
    times.append(time.perf_counter()-t0)
    off_kill=~np.isfinite(rec2['x'].astype(np.float64)) & np.isfinite(x)     # 오프라인이 NaN 으로 지운 점(원래 유효했던 것)
    live_kill=kill & np.isfinite(x)
    npts+=int(np.isfinite(x).sum()); tot+=1
    same+=int((live_kill & off_kill).sum()); only_live+=int((live_kill & ~off_kill).sum()); only_off+=int((~live_kill & off_kill).sum())
    if MAXS and tot>=MAXS: break
times=np.array(times)
print(f'스캔 {tot}개, 유효점 {npts:,}')
print(f'  둘 다 지움 {same:,}  | 실시간만 지움 {only_live:,}  | 오프라인만 지움 {only_off:,}')
print(f'  일치율(지운 점 기준) {100*same/max(same+only_live+only_off,1):.2f}%  | 오프라인이 지운 점 중 재현 {100*same/max(same+only_off,1):.2f}%')
print(f'  속도: 스캔당 중앙 {1000*np.median(times):.1f} ms, p90 {1000*np.percentile(times,90):.1f} ms (이 PC 1코어)')
print('  단계별 누적:', dict(C.stats))
