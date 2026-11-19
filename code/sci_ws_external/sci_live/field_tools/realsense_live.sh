#!/usr/bin/env bash
# 로봇에서: RealSense 를 선배 ROPUS 와 같은 설정으로, DDS 만 실시간 설정(큰 조각·버퍼 8 MB)으로 띄운다(2026-10-15).
#   선배 RealSense(기본 DDS)는 받는 프로그램이 둘(E5 + 조준 콘솔)이 되자 컬러가 거의 안 갔다(3분 반에 7장). Hesai 때와 같은 종류.
#   bash realsense_live.sh start    # 먼저 선배 것을 끈다: systemctl --user stop ropus-go2-sensor-realsense (stop 만. 재부팅하면 원래대로)
#   bash realsense_live.sh stop     # 끝나면 이것 + systemctl --user start ropus-go2-sensor-realsense
#   bash realsense_live.sh status
# 설정(선배 run_go2_sensor.sh realsense 와 같음): 컬러·깊이 640x480 15 fps, 깊이 정렬, 동기화, 점군 끔. 토픽 이름도 같다.
set -uo pipefail
LIVE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; L=$LIVE/logs/live_$(date +%m%d); mkdir -p $L; PIDF=$L/realsense.pid
case "${1:-status}" in
  start)
    n=$(pgrep -c -f realsense2_camera_node || true); [ "$n" = 0 ] || { echo "RealSense 가 이미 떠 있음($n) — 선배 것을 먼저 끄세요: systemctl --user stop ropus-go2-sensor-realsense"; exit 1; }
    set +u; source /opt/ros/foxy/setup.bash; source ~/unitree_ros2/setup.sh >/dev/null; set -u
    export CYCLONEDDS_URI=file://$LIVE/realtime/config/cyclonedds_live.xml ROS2_DISABLE_DAEMON=1
    nohup setsid taskset -c 6,7 ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true \
      pointcloud.enable:=false enable_sync:=true rgb_camera.profile:=640x480x15 depth_module.profile:=640x480x15 > $L/realsense.log 2>&1 < /dev/null &
    echo $! > $PIDF; sleep 6; bash "$0" status ;;
  stop)
    [ -f $PIDF ] && { P=$(cat $PIDF); kill -INT -- -$P 2>/dev/null; for i in $(seq 1 10); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -0 $P 2>/dev/null && kill -KILL -- -$P; rm -f $PIDF; echo "우리 RealSense 끔"; } || echo "우리 RealSense 안 떠 있음" ;;
  status)
    ps -eo pid,etime,pcpu,args | grep -E "[r]ealsense2_camera_node" | cut -c1-110; echo "선배 서비스: $(systemctl --user is-active ropus-go2-sensor-realsense)"; tail -2 $L/realsense.log 2>/dev/null | cut -c1-150 ;;
esac
