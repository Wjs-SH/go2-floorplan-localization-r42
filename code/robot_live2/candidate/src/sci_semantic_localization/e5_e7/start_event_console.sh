#!/usr/bin/env bash
# start_event_console.sh — start the E5 event pan/tilt console (kci bundle version) for one trial.
#
# PHYSICAL EFFECT: opening the Nano resets it and aim_console.main() drives pan/tilt to HOME (calibration zero),
# and may write HOME to the Nano EEPROM (aim_console.py:921-925, 77-100). Requires the operator's confirmation.
# No Go2 base command is involved.
#
# Usage: start_event_console.sh --trial <dir> [--odom-topic /Odometry|/utlidar/robot_odom] [--execute]
set -uo pipefail
TRIAL=""; ODOM="/utlidar/robot_odom"; MODE=dry-run
while [ $# -gt 0 ]; do
  case "$1" in
    --trial) TRIAL="$2"; shift 2 ;;
    --odom-topic) ODOM="$2"; shift 2 ;;
    --execute) MODE=execute; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[ -n "$TRIAL" ] || { echo "--trial required"; exit 2; }
CODE="$HOME/kci_event_handoff_20261009/code"
CALIB="$HOME/pantilt_calib/configs/pantilt_calib_multi_20260928.json"
say(){ echo "[$MODE] $*"; }
fail=0
ss -lnt | grep -q ':8088 ' && { say "BLOCK: :8088 already in use"; fail=1; }
ps -eo args | grep -E '[a]im_console.py|[c]alib_console.py' && { say "BLOCK: a pan/tilt console is already running"; fail=1; }
grep -l Arducam /sys/class/video4linux/video*/name >/dev/null 2>&1 || { say "BLOCK: Arducam not found"; fail=1; }
ls /dev/ttyUSB* >/dev/null 2>&1 || { say "BLOCK: no /dev/ttyUSB* (Nano)"; fail=1; }
[ -f "$CALIB" ] || { say "BLOCK: calibration missing $CALIB"; fail=1; }
[ -f "$CODE/aim_console.py" ] || { say "BLOCK: event console missing"; fail=1; }
set +u; source /opt/ros/foxy/setup.bash; export ROS2_DISABLE_DAEMON=1; source ~/unitree_ros2/setup.sh >/dev/null; set -u
pubs(){ timeout 8 ros2 topic info "$1" 2>/dev/null | awk -F': ' '/Publisher count/{print $2}' | tr -d ' '; }
for t in /camera/color/image_raw /camera/aligned_depth_to_color/image_raw "$ODOM"; do
  n=$(pubs "$t"); say "topic $t publishers=${n:-0}"; [ "${n:-0}" -ge 1 ] || { say "BLOCK: $t not published"; fail=1; }
done
[ "$fail" = 0 ] || exit 3
say "calibration $CALIB sha256 $(sha256sum "$CALIB" | cut -c1-16)"
CMD="cd $CODE && exec python3 -u aim_console.py --use-depth --config $CALIB --odom-topic $ODOM"
say "command: $CMD"
[ "$MODE" = execute ] || { say "dry-run only (servos would move to HOME on --execute)"; exit 0; }
mkdir -p "$TRIAL/console"
date -Is > "$TRIAL/console/started_at.txt"
nohup setsid bash -c "$CMD" > "$TRIAL/console/console.log" 2>&1 < /dev/null &
echo $! > "$TRIAL/console/console.pid"
say "console pid $(cat "$TRIAL/console/console.pid"); log $TRIAL/console/console.log"
