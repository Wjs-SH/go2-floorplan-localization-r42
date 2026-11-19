#!/bin/bash
# monitor_alt.sh RUN_DIR — read-only once-per-second GO/NO-GO line (files only, no ROS subscriptions). Never commands the robot.
# Provisional development gates (NOT validated safety thresholds): IMU health READY, |height| <= 0.5 m warn / 1.0 m block,
# speed state <= 2 m/s warn / 3 m/s block, any 3D jump -> block, scan gaps -> warn.
source "$(dirname "$0")/common.sh"
run="${1:?RUN_DIR}"
watch="$HOME/sci_ws/reports/fastlio_preflight_20261028_192027/scripts/mat_pre_watch.py"
while true; do
  h=$(tail -1 "$run/bridge_health.jsonl" 2>/dev/null | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(d["health"],"hz=%d"%d["publish_hz"],"gaps=%d"%d["gaps"],"resets=%d"%d["model_resets"],"res95=%.2fms"%d["residual_ms_p95"])
except Exception: print("NO_HEALTH")')
  m=$(python3 -I "$watch" "$run/fastlio_Log/mat_pre.txt" 2>/dev/null | tail -1 | python3 -c 'import sys,ast
try:
  d=ast.literal_eval(sys.stdin.read()); print("h=[%s] vmax=%.2f gaps=%d jumps=%d"%(d["height_range"],d["speed_max"],d["scan_gaps"],d["jumps_3d"]))
except Exception: print("NO_FASTLIO_STATE")')
  verdict=WALK_BLOCKED
  case "$h" in READY*) case "$m" in *"jumps=0"*) verdict=STATIC_OK_NOT_A_WALK_PERMISSION;; esac;; esac
  echo "$(date +%T) $verdict | imu: $h | fastlio: $m"
  ls "$run"/owned/*.pid >/dev/null 2>&1 || break
  alive=0; for f in "$run"/owned/{bridge,fastlio_alt}.pid; do [ -e "$f" ] && read -r p s _ < "$f" && [ "$(proc_start_time "$p")" = "$s" ] && alive=1; done
  [ $alive = 1 ] || { echo "$(date +%T) bridge/fastlio stopped -> monitor exits"; break; }
  sleep 1
done
