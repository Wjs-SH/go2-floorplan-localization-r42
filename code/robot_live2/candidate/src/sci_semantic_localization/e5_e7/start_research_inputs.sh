#!/usr/bin/env bash
# start_research_inputs.sh — reuse-aware research startup (promoted 2026-10-12 from the audit proposal after review).
#
# REUSES the always-on raw sensors (ropus-go2-sensor-hesai / -realsense / -robot-model, robot-native DDS) and starts
# ONLY FAST-LIO (go2ws, selected — see reports/e5_e7_live/raw/20261012_fastlio_selection.md). Never starts Hesai,
# RealSense or robot_state_publisher. Canonical thesis code is not modified.
#
# Same pid/env files and log folder as ~/jsh_onboard_robot_20261001/realtime/tools/sensors_go2.sh
# ($L/fastlio.pid, fastlio.env, fastlio.cfg, fastlio.launch.sh) so `sensors_go2.sh stop_fastlio|status` keep working.
# hesai.pid is never written; the script refuses to run if one exists (stale pid safety for `sensors_go2.sh stop`).
#
# Composition (default no_rsp): the three non-RSP components of mapping_go2_flio01.launch.py, same executables, same
# config, same node names: imu_correct.py (/utlidar/imu -> /utlidar/imu_corrected), fast_lio fastlio_mapping
# (laser_mapping, --params-file <cfg>, cfg is /**:ros__parameters), static TF camera_init->map_viz (args from launch:35-36).
# FLIO_COMPOSE=canonical -> the unmodified launch file (adds a 2nd robot_state_publisher).
#
# Usage: start_research_inputs.sh [--dry-run (default) | --execute]
set -uo pipefail
MODE=dry-run
for a in "$@"; do
  case "$a" in
    --execute) MODE=execute ;;
    --dry-run) MODE=dry-run ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done
HERE="$HOME/jsh_onboard_robot_20261001/realtime"
L="$HERE/../logs/live_$(date +%m%d)"
CFG="${FASTLIO_CONFIG:-$HERE/config/fastlio_go2_esy_extrinsic.yaml}"
say(){ echo "[$MODE] $*"; }

set +u
source ~/unitree_ros2/setup.sh > /dev/null
source ~/go2_ws/install/setup.bash
set -u
export CYCLONEDDS_URI="file://$HERE/config/cyclonedds_live.xml"
pubs(){ timeout 8 ros2 topic info "$1" 2>/dev/null | awk -F': ' '/Publisher count/{print $2}' | tr -d ' '; }

[ -f "$L/hesai.pid" ] && { say "BLOCK: $L/hesai.pid exists (stale pid from sensors_go2.sh). Inspect it first; not starting."; exit 6; }
n_h=$(ps -eo comm | grep -c '^hesai_ros_drive'); n_r=$(ps -eo comm | grep -c '^realsense2_came'); n_s=$(ps -eo comm | grep -c '^robot_state_pub')
say "processes: hesai=$n_h realsense=$n_r robot_state_publisher=$n_s (want 1/1/1)"
[ "$n_h" -ne 1 ] && { say "BLOCK: expected exactly 1 hesai driver"; exit 4; }
n=$(pubs /lidar_points);  [ "${n:-0}" -ge 1 ] || { say "BLOCK: /lidar_points has no publisher"; exit 3; }
say "REUSE /lidar_points publishers=$n"
n=$(pubs /utlidar/imu);   [ "${n:-0}" -ge 1 ] || { say "BLOCK: /utlidar/imu has no publisher"; exit 3; }
say "REUSE /utlidar/imu publishers=$n"
n=$(pubs /Odometry)
if [ "${n:-0}" -ge 1 ]; then say "REUSE /Odometry publishers=$n — FAST-LIO already running, nothing started"; exit 0; fi
if pgrep -x fastlio_mapping >/dev/null || pgrep -f 'FAST_LIO_ROS2/scripts/imu_correct.py' >/dev/null; then
  say "BLOCK: fastlio_mapping/imu_correct process exists without /Odometry — inspect, not starting a duplicate"; exit 5
fi

COMPOSE="${FLIO_COMPOSE:-no_rsp}"
if [ "$COMPOSE" = canonical ]; then
  CMD="ros2 launch fast_lio mapping_go2_flio01.launch.py config:=$CFG"
else
  CMD="python3 $HOME/go2_ws/src/FAST_LIO_ROS2/scripts/imu_correct.py & ros2 run tf2_ros static_transform_publisher 0 0 0 1 0 0 0 camera_init map_viz --ros-args -r __node:=camera_init_to_map_viz & exec ros2 run fast_lio fastlio_mapping --ros-args -r __node:=laser_mapping --params-file $CFG"
fi
say "FAST-LIO impl=go2ws compose=$COMPOSE cfg=$CFG"
say "DDS CYCLONEDDS_URI=$CYCLONEDDS_URI ; log dir $L"
say "command: setsid bash -c '$CMD'"
[ "$MODE" = execute ] || { say "dry-run only. Re-run with --execute."; exit 0; }

mkdir -p "$L"
echo "FAST-LIO 설정: $CFG" > "$L/fastlio.cfg"
{ echo "impl=go2ws"; echo "compose=$COMPOSE"; echo "started=$(date +%s.%N)"; echo "fast_lio_prefix=$(ros2 pkg prefix fast_lio)"; } > "$L/fastlio.env"
echo "export FASTLIO_IMPL=go2ws FASTLIO_CONFIG=$CFG" > "$L/fastlio.launch.sh"
nohup setsid bash -c "$CMD" > "$L/fastlio.log" 2>&1 < /dev/null &
echo $! > "$L/fastlio.pid"
say "started process group $(cat "$L/fastlio.pid"); stop with: bash $HERE/tools/sensors_go2.sh stop_fastlio"
