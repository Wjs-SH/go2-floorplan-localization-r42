#!/usr/bin/env bash
# fastlio_sci.sh start|stop|status — FAST-LIO from the SCI build (~/sci_ws/external/fastlio_ws; source byte-identical to
# ~/go2_ws/src/FAST_LIO_ROS2, ROOT_DIR inside sci_ws so its Log/ writes never touch go2_ws). Same composition as the
# handoff start_research_inputs.sh (no_rsp): imu_correct.py + static TF camera_init->map_viz + fastlio_mapping.
# Reuses the running Hesai (/lidar_points) and /utlidar/imu. Never starts sensor drivers or robot_state_publisher.
set -uo pipefail
WS=$HOME/sci_ws; L=$WS/logs/fastlio; PIDF=$WS/state/pids/fastlio.pid; mkdir -p $L $WS/state/pids
CFG=$WS/config/fastlio/fastlio_go2_esy_extrinsic.yaml
set +u; source /opt/ros/foxy/setup.bash; source ~/unitree_ros2/setup.sh >/dev/null 2>&1; source ~/go2_ws/install/setup.bash; source $WS/external/fastlio_ws/install/local_setup.bash; set -u
export CYCLONEDDS_URI="file://$WS/config/fastlio/cyclonedds_live.xml"
case "${1:-status}" in
start)
  [ -f $PIDF ] && kill -0 $(cat $PIDF) 2>/dev/null && { echo "already running ($(cat $PIDF))"; exit 0; }
  n_h=$(ps -eo comm | grep -c '^hesai_ros_drive'); [ "$n_h" -eq 1 ] || { echo "BLOCK: hesai drivers=$n_h (want 1)"; exit 4; }
  if ps -eo args | grep -E "[f]astlio_mapping|[i]mu_correct.py" >/dev/null; then echo "BLOCK: a FAST-LIO/imu_correct already runs — reuse its /Odometry"; exit 5; fi
  P=$(ros2 pkg prefix fast_lio); case "$P" in $WS/*) ;; *) echo "BLOCK: fast_lio resolves to $P (not SCI build)"; exit 6;; esac
  CMD="python3 $WS/external/fastlio_ws/src/FAST_LIO_ROS2/scripts/imu_correct.py & ros2 run tf2_ros static_transform_publisher 0 0 0 1 0 0 0 camera_init map_viz --ros-args -r __node:=camera_init_to_map_viz & exec ros2 run fast_lio fastlio_mapping --ros-args -r __node:=laser_mapping --params-file $CFG"
  cd $L && nohup setsid bash -c "$CMD" > $L/fastlio_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
  echo $! > $PIDF; echo "started pgid $(cat $PIDF) prefix=$P cfg=$CFG" ;;
stop)
  # resolve the process group of the SCI fastlio_mapping (launcher PID != setsid group id); only SCI-built binaries
  g=$(ps -eo pgid,args | awk -v ws="$WS/external/fastlio_ws" 'index($0, ws"/install/fast_lio/lib/fast_lio/fastlio_mapping"){print $1; exit}')
  [ -n "$g" ] || { echo "SCI FAST-LIO not running"; rm -f $PIDF; exit 0; }
  kill -INT -- -$g; for i in 1 2 3 4 5; do sleep 1; ps -eo pgid | grep -qw "$g" || break; done
  ps -eo pgid | grep -qw "$g" && kill -KILL -- -$g; echo "stopped pgid $g"; rm -f $PIDF ;;
status)
  ps -eo pid,pgid,etime,pcpu,rss,args | grep -E "[f]astlio_mapping|[i]mu_correct|[c]amera_init_to_map_viz" | cut -c1-140 ;;
esac
