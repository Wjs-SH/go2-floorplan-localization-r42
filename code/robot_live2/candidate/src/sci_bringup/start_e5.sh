#!/usr/bin/env bash
# Start the SCI E5 stack: UI (8095) + E5 node + async reader. Never starts sensor drivers or moves the robot.
#   start_e5.sh live   [--odom-topic /Odometry] [extra e5_node args]
#   start_e5.sh replay --dataset ~/sci_ws/datasets/bags/S-6F --realtime --assume-stopped
# Pan/tilt actuation additionally needs: SCI_PANTILT_APPROVED=1 and --pantilt-execute (operator approval).
set -euo pipefail
set +u; source "$(dirname "$0")/sci_env.sh"; set -u
MODE=${1:-live}; shift || true
RUN=${SCI_RUN:-e5_${MODE}_$(date +%Y%m%d_%H%M%S)}
cd $SCI_WS
if [ "$MODE" = live ]; then
  n=$(pgrep -fc realsense2_camera_node || true); h=$(pgrep -fc hesai_ros_driver_node || true)
  echo "[check] realsense=$n hesai=$h (reused, never restarted)"; [ "$n" -ge 1 ] || { echo "RealSense driver not running -> abort"; exit 2; }
fi
pgrep -f "sci_monitor/ui_server.py" >/dev/null || { nohup $SCI_PY src/sci_monitor/ui_server.py --port ${SCI_UI_PORT:-8095} > logs/ui_server.log 2>&1 & echo $! > state/pids/ui.pid; }
SRC=$([ "$MODE" = live ] && echo ros || echo replay)
nohup $SCI_PY src/sci_visual_observation/e5_node.py --source $SRC --run-name $RUN "$@" > logs/$RUN.node.log 2>&1 &
echo $! > state/pids/e5_node.pid
sleep 2; mkdir -p experiments/$RUN/jobs/pending
nohup nice -n 5 $SCI_PY src/sci_text_reading/reader.py --run experiments/$RUN --vlm ${SCI_VLM_MODE:-on_demand} ${SCI_READER_ARGS:-} > logs/$RUN.reader.log 2>&1 &
echo $! > state/pids/reader.pid
echo "run=$RUN  UI: http://$(hostname -I | awk '{print $1}'):${SCI_UI_PORT:-8095}/   logs: $SCI_WS/logs/$RUN.*.log"
