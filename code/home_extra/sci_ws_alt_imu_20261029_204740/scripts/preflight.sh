#!/bin/bash
# preflight.sh --live|--offline [RUN_DIR] — read-only checks; writes only RUN_DIR/preflight.txt (if given). Exit 0 = PASS.
source "$(dirname "$0")/common.sh"
mode="${1:---offline}"; run="${2:-}"
out() { echo "$*"; if [ -n "$run" ]; then echo "$*" >> "$run/preflight.txt"; fi; return 0; }
fail=0
chk() { if [ "$2" = ok ]; then out "PASS  $1 $3"; else out "FAIL  $1 $3"; fail=1; fi; return 0; }
for p in $PROTECTED_PROCS; do n=$(pgrep -xc "${p:0:15}" || true); chk "protected:$p" "$([ "$n" = 1 ] && echo ok)" "count=$n"; done
n=$(pgrep -xc fastlio_mapping || true); chk "no_other_fastlio" "$([ "$n" = 0 ] && echo ok)" "running=$n (operational or alt FAST-LIO must not run)"
n=$(pgrep -xc bridge_node || true); chk "no_other_bridge" "$([ "$n" = 0 ] && echo ok)" "running=$n"
n=$(pgrep -f "python3 .*imu_correct\.py" | while read p; do [ "$(tr "\0" " " < /proc/$p/cmdline | cut -c1-7)" = python3 ] && echo $p; done | wc -l); chk "no_imu_correct" "$([ "$n" = 0 ] && echo ok)" "running=$n"
t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1); t=$((t/1000))
chk "temp<72C" "$([ "$t" -lt 72 ] && echo ok)" "max=${t}C"
m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); chk "mem>=4300MB" "$([ "$m" -ge 4300 ] && echo ok)" "avail=${m}MB"
d=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9); chk "disk>=5GB" "$([ "$d" -ge 5 ] && echo ok)" "free=${d}GB"
l=$(cut -d' ' -f1 /proc/loadavg); out "INFO  loadavg1=$l"
if bad=$(protected_hash_check); then chk "protected_hashes" ok "unchanged"; else chk "protected_hashes" no "$bad"; fi
if [ -x "$ALT_WS/install/sci_alt_imu_bridge/lib/sci_alt_imu_bridge/bridge_node" ]; then chk bridge_built ok ""; else chk bridge_built no ""; fi
if [ "$mode" = --live ]; then
  tmp=$(mktemp -d); ros_env live "$tmp"
  info() { timeout 20 ros2 topic info "$1" 2>/dev/null | grep -E "Publisher count" | grep -o "[0-9]*"; }
  chk "/lowstate publisher" "$([ "$(info /lowstate)" = 1 ] && echo ok)" "pubs=$(info /lowstate)"
  chk "/lidar_points publisher" "$([ "$(info /lidar_points)" = 1 ] && echo ok)" "pubs=$(info /lidar_points)"
  for tp in /sci_alt_imu/imu /sci_alt_imu/Odometry /sci_alt_imu/tf /Odometry; do
    n=$(info $tp); chk "no_publisher:$tp" "$([ -z "$n" ] || [ "$n" = 0 ] && echo ok)" "pubs=${n:-0}"; done
  rm -rf -- "${tmp:?}"
fi
[ $fail = 0 ] && out "PREFLIGHT PASS ($mode)" || out "PREFLIGHT FAIL ($mode)"
exit $fail
