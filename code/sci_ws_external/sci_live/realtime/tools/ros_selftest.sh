#!/bin/bash
# ros_selftest.sh — 센서 없이 rt_live_node 의 ROS 경로를 시험한다(로봇·노트북 공용).
#   a131_live 시험 데이터를 점군·FAST-LIO 토픽으로 실제 시간 흐름(배속 SPEED)대로 내보내고, 노드가 만든 표본을 기준 결과와 대조한다.
#   다른 노드와 섞이지 않게 도메인 77 + 루프백(lo)만 쓴다(로봇 내부망 eth0 에 시험 트래픽이 나가지 않는다).
#
#   bash tools/ros_selftest.sh [배속 0.05] [파이썬]
#
# 로봇: Unitree 환경과 같은 CycloneDDS 를 쓴다(드라이버·FAST-LIO 도 그 환경에서 돈다). 노트북: 깔린 ROS 기본값.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
SPEED=${1:-0.05}
PY=${2:-python3}
OUT=/dev/shm/rt_ros_selftest
DUMP=$HERE/../a131_live/testdata/S6_chunk0_55.npz

if [ -f "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
    source "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash"
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
else
    for d in humble foxy; do [ -f /opt/ros/$d/setup.bash ] && { source /opt/ros/$d/setup.bash; break; }; done
fi
export ROS_DOMAIN_ID=77
unset ROS_LOCALHOST_ONLY
export OMP_NUM_THREADS=1

cd "$HERE"
echo "RMW=${RMW_IMPLEMENTATION:-기본} · 배속 $SPEED · $($PY -c 'import sys;print(sys.version.split()[0])')"
$PY rt_live_node.py --ros-args -p building:=sw -p sample_dt:=0.0 -p start_workers:=false -p store_root:=$OUT > $OUT.node.log 2>&1 &
NP=$!
sleep 8
$PY tools/replay_dump_ros.py "$DUMP" "$SPEED" reliable 2>&1 | tail -2
timeout 8 ros2 topic echo --once --full-length /floorplan_loc/status 2>/dev/null | head -1 || true
sleep 3
kill -INT $NP
for i in 1 2 3 4 5 6; do kill -0 $NP 2>/dev/null || break; sleep 1; done
kill -0 $NP 2>/dev/null && { echo "노드가 6초 안에 안 끝남 — 강제 종료"; kill -KILL $NP; } || echo "노드 정상 종료"
grep -E "\[(INFO|WARN|ERROR)\]" $OUT.node.log | tail -5
$PY tools/check_store_vs_dump.py $OUT/data "$DUMP" > $OUT.check.txt
grep 다름 $OUT.check.txt | head -12 || true
tail -1 $OUT.check.txt
