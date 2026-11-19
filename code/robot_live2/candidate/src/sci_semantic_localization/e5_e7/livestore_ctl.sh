#!/usr/bin/env bash
# livestore_ctl.sh start|stop|status [--trial <dir>] [--execute]
# Lean E7 input producer: canonical rt_live_node with -p start_workers:=false -p viewer_port:=0 (rt_live_node.py:130,138)
# -> writes the floor_id driving folder /dev/shm/rt_live/data (LiveStore) from /lidar_points + FAST-LIO /Odometry.
# No floor-ID/pose workers (saves ~7 GB vs the full node), no :8765 viewer (no ROPUS port collision). Code unmodified.
# Same environment as the field guide §3 (unitree_ros2, cyclonedds_live.xml, PYTHONNOUSERSITE=1, venv_gpu, taskset 2-7).
set -uo pipefail
CMD_="${1:-status}"; shift || true
TRIAL=""; MODE=dry-run
while [ $# -gt 0 ]; do case "$1" in --trial) TRIAL="$2"; shift 2;; --execute) MODE=execute; shift;; *) echo "bad arg $1"; exit 2;; esac; done
RT="$HOME/jsh_onboard_robot_20261001/realtime"
PIDF="${TRIAL:-/tmp}/livestore.pid"
say(){ echo "[$MODE] $*"; }
case "$CMD_" in
  status)
    ps -eo pid,etime,pcpu,rss,args | grep -E "[r]t_live_node|[l]oc_worker" || echo "rt_live_node not running"
    ls -la /dev/shm/rt_live/data 2>/dev/null | head -5; tail -n 2 /dev/shm/rt_live/data/odom_link.csv 2>/dev/null ;;
  start)
    [ -n "$TRIAL" ] || { echo "--trial required"; exit 2; }
    ps -eo args | grep -q '[r]t_live_node' && { say "BLOCK: rt_live_node already running"; exit 3; }
    set +u; source ~/unitree_ros2/setup.sh >/dev/null; source ~/go2_ws/install/setup.bash; set -u
    n=$(timeout 8 ros2 topic info /Odometry 2>/dev/null | awk -F': ' '/Publisher count/{print $2}' | tr -d ' ')
    [ "${n:-0}" -ge 1 ] || { say "BLOCK: /Odometry (FAST-LIO) not published — start FAST-LIO first"; exit 3; }
    CMD="cd $RT && export CYCLONEDDS_URI=file://$RT/config/cyclonedds_live.xml PYTHONNOUSERSITE=1 && exec taskset -c 2-7 nice -n 10 ../venv_gpu/bin/python rt_live_node.py --ros-args -p building:=kci -p store_root:=/dev/shm/rt_live -p delta_deg:=45.3 -p mirror:=true -p cloud_reliability:=best_effort -p start_workers:=false -p viewer_port:=0"
    say "command: $CMD"
    [ "$MODE" = execute ] || exit 0
    mkdir -p "$TRIAL"
    nohup setsid bash -c "$CMD" > "$TRIAL/livestore.log" 2>&1 < /dev/null &
    echo $! > "$PIDF"; say "pid $(cat "$PIDF")" ;;
  stop)
    [ -f "$PIDF" ] || { echo "no pid file $PIDF"; exit 0; }
    P=$(cat "$PIDF"); ps -o args= -p "$P" | grep -q rt_live_node || { echo "pid $P is not rt_live_node"; exit 4; }
    kill -INT "$P"; for i in $(seq 1 10); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
    kill -0 "$P" 2>/dev/null && { echo "still alive, inspect"; exit 5; }
    rm -f "$PIDF"; echo stopped
    [ -n "$TRIAL" ] && cp -r /dev/shm/rt_live/data "$TRIAL/livestore_data" 2>/dev/null; true ;;
esac
