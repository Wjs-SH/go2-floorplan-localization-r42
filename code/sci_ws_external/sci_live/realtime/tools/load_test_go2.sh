#!/bin/bash
# load_test_go2.sh — 센서 없이 로봇 전체 부하 시험: 노드(구조점 추출)가 반복 재생 점군을 처리하는 동안
# 위치추정 작업자(층 판별 FLOOR_PROCS 개 + 위치)를 S6 주행 재생으로 돌린다. 메모리·부하를 10초마다 남긴다.
#   bash tools/load_test_go2.sh [FLOOR_PROCS 3] [LOOPS 32]
HERE=$(cd "$(dirname "$0")/.." && pwd)
K=${1:-3}; LOOPS=${2:-32}
PY=$HERE/../venv_gpu/bin/python
OUT=/dev/shm/rt_load_k$K
rm -rf $OUT; mkdir -p $OUT
source /opt/ros/foxy/setup.bash
source "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=77 PYTHONNOUSERSITE=1
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
cd "$HERE"
( for i in $(seq 1 90); do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"; sleep 10; done ) > $OUT/load.txt 2>&1 &
LP=$!
OMP_NUM_THREADS=1 $PY rt_live_node.py --ros-args -p building:=sw -p start_workers:=false -p store_root:=$OUT/node > $OUT/node.log 2>&1 &
NP=$!
sleep 8
$PY tools/replay_dump_ros.py ../a131_live/testdata/S6_chunk0_55.npz 1.0 reliable $LOOPS > $OUT/replay_dump.log 2>&1 &
RP=$!
$PY replay_samples.py --building sw --data ../testdata/S6 --root $OUT/loc --sample-dt 1.5 --floor-procs $K > $OUT/loc.stdout 2>&1
kill -INT $NP; kill $RP $LP 2>/dev/null; wait $NP 2>/dev/null
echo "== K=$K 끝 $(date +%T)"
grep -E "준비 끝|바퀴" $OUT/loc/out/floor.log
grep -E "보정" $OUT/loc/out/pose.log | awk '{print $NF}' | tr '\n' ' '; echo
sort -t= -k2 -n $OUT/load.txt | head -1
$PY - $OUT/node/out/status.jsonl <<'PYEOF'
import json, sys
import numpy as np
S = [json.loads(l) for l in open(sys.argv[1])]
sec = [s['last_sample_sec'] for s in S if s.get('last_sample_sec')]
lag = [s['last_sample_lag_s'] for s in S if s.get('last_sample_lag_s') is not None]
print(f"구조점 표본 {S[-1]['samples']}개 · 실패 {S[-1]['sample_fail']} · 격자 재설정 {S[-1].get('grid_resets', 0)} · 빠진 스캔 {S[-1]['scans_missed']}")
print(f"표본 처리 초 중앙 {np.median(sec):.2f} · 최대 {max(sec):.2f} | 지연(마지막 자세-표본) 중앙 {np.median(lag):.2f} · 최대 {max(lag):.2f}")
PYEOF
