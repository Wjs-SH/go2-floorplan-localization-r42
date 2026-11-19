# common.sh — shared helpers for the alt-IMU FAST-LIO path. Source only; never modifies anything outside $ALT_WS / run dirs.
set -u
ALT_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALT_REPORT_DIR="${ALT_REPORT_DIR:-$HOME/sci_ws/reports/alt_imu_one_shot_20261029_204740}"
FASTLIO_WS="$HOME/sci_ws/external/fastlio_ws"
FASTLIO_SRC="$FASTLIO_WS/src/FAST_LIO_ROS2"
OPS_FASTLIO_CFG="$HOME/sci_ws/config/fastlio/fastlio_go2_esy_extrinsic.yaml"
LIVE_DDS="$HOME/sci_ws/config/fastlio/cyclonedds_live.xml"
OFFLINE_DDS="$ALT_WS/config/cyclonedds_offline_loopback.xml"
PROTECTED_PROCS="hesai_ros_driver_node realsense2_camera_node robot_state_publisher go2_control_runner"
ALT_TOPICS_REMAP=(-r /Odometry:=/sci_alt_imu/Odometry -r /cloud_registered:=/sci_alt_imu/cloud_registered
                  -r /cloud_registered_body:=/sci_alt_imu/cloud_registered_body -r /cloud_effected:=/sci_alt_imu/cloud_effected
                  -r /Laser_map:=/sci_alt_imu/Laser_map -r /path:=/sci_alt_imu/path -r /tf:=/sci_alt_imu/tf)

say() { echo "[alt-imu] $*"; }
die() { echo "[alt-imu] REFUSED: $*" >&2; exit 3; }

ros_env() {   # $1 = live|offline ; $2 = run dir (ROS_LOG_DIR inside it)
  set +u
  source /opt/ros/foxy/setup.bash >/dev/null 2>&1
  [ -f "$HOME/unitree_ros2/setup.sh" ] && source "$HOME/unitree_ros2/setup.sh" >/dev/null 2>&1
  source "$HOME/go2_ws/install/setup.bash" >/dev/null 2>&1
  source "$FASTLIO_WS/install/local_setup.bash" >/dev/null 2>&1
  source "$ALT_WS/install/local_setup.bash" >/dev/null 2>&1
  set -u
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0 PYTHONDONTWRITEBYTECODE=1
  mkdir -p "$2/roslog"; export ROS_LOG_DIR="$2/roslog"
  if [ "$1" = live ]; then export ROS_DOMAIN_ID=0 CYCLONEDDS_URI="file://$LIVE_DDS"
  else export ROS_DOMAIN_ID=42 CYCLONEDDS_URI="file://$OFFLINE_DDS"; fi
}

new_run_dir() {   # $1 = prefix
  local d="$ALT_WS/runs/$1_$(date +%Y%m%d_%H%M%S)"
  [ -e "$d" ] && die "run dir exists: $d"
  mkdir -p "$d/owned" "$d/fastlio_Log"; echo "$d"
}

proc_start_time() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null; }

# start a command in its own session (process group) and record pid + starttime (ownership proof for stop)
owned_start() {   # $1 run dir, $2 name, $3 log file, rest = command
  local run="$1" name="$2" log="$3"; shift 3
  setsid "$@" >"$log" 2>&1 < /dev/null &
  local pid=$!
  sleep 0.3
  echo "$pid $(proc_start_time $pid) $(date +%s)" > "$run/owned/$name.pid"
  say "started $name pid $pid (pgid $pid) log $log"
}

owned_stop() {    # $1 run dir ; stops ONLY processes recorded in $1/owned with matching starttime
  local run="$1" f pid st now
  for f in "$run"/owned/*.pid; do
    [ -e "$f" ] || continue
    read -r pid st _ < "$f"
    now=$(proc_start_time "$pid")
    if [ -z "$now" ]; then say "$(basename "$f" .pid): already exited"; continue; fi
    if [ "$now" != "$st" ]; then say "$(basename "$f" .pid): pid $pid reused by another process — NOT signalled"; continue; fi
    kill -INT -- "-$pid" 2>/dev/null
    for _ in $(seq 1 20); do [ -z "$(proc_start_time "$pid")" ] && break; sleep 0.5; done
    if [ -n "$(proc_start_time "$pid")" ]; then kill -TERM -- "-$pid" 2>/dev/null; sleep 3; fi
    if [ -n "$(proc_start_time "$pid")" ]; then kill -KILL -- "-$pid" 2>/dev/null; fi
    say "$(basename "$f" .pid): stopped"
  done
}

protected_hash_check() {   # verifies the files this path could conceivably affect; prints mismatches; returns 1 on any
  local bad=0 f want got
  while read -r want f; do
    got=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then echo "HASH_CHANGED $f"; bad=1; fi
  done < <(grep -E "fastlio_ws/(src|install)/|sci_ws/config/|livefield2_20261018_151606/candidate/live[12]/(ctl|console|lio_monitor2|e2_feed|live1_common)\.py|\.bashrc|systemd/user/" \
           "$ALT_REPORT_DIR/01_ORIGINAL_SNAPSHOT_MANIFEST.sha256" | grep -v '^#')
  return $bad
}

count_procs() { pgrep -fc "$1" 2>/dev/null || echo 0; }

# FAST-LIO (unchanged binary) with: rendered alt config, own node name, all outputs remapped under /sci_alt_imu,
# Log/ and PCD/ bind-mounted to the run dir inside a private mount namespace (the original Log/mat_pre.txt is never touched).
fastlio_alt_start() {   # $1 run dir, $2 rendered cfg
  local run="$1" cfg="$2"
  mkdir -p "$run/fastlio_Log" "$run/fastlio_PCD"
  local remaps="${ALT_TOPICS_REMAP[*]}"
  local inner="mount --bind '$run/fastlio_Log' '$FASTLIO_SRC/Log' && "
  [ -d "$FASTLIO_SRC/PCD" ] && inner+="mount --bind '$run/fastlio_PCD' '$FASTLIO_SRC/PCD' && "
  inner+="cd '$run' && exec ros2 run fast_lio fastlio_mapping --ros-args -r __node:=sci_alt_laser_mapping --params-file '$cfg' $remaps"
  owned_start "$run" fastlio_alt "$run/fastlio_alt.out" unshare -Urm bash -c "$inner"
}

bridge_start() {   # $1 run dir
  owned_start "$1" bridge "$1/bridge.out" ros2 launch sci_alt_imu_bridge alt_imu_bridge.launch.py log_dir:="$1"
}
