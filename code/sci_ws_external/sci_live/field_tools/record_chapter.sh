#!/usr/bin/env bash
# 필드 챕터 녹화(2026-10-14). 로봇에서:
#   bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh test               # 30초 시험 녹화 → 주기·크기 확인 후 지움
#   bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh start <이름> [--camera]
#   bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh stop
#   bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh status
# 저장: ${BAG_ROOT:-<sci_live>/bags}/<월일_시분초>_<이름>/ (rosbag2) + 같은 이름 .log
# 2026-10-15: ~/sci_ws/external/sci_live/field_tools/ 에 둔다. DDS 설정·저장 위치를 원본 폴더(~/jsh_onboard_robot_20261001) 대신 sci_live 안에서 찾는다.
# 웹 [시작]/[멈춤]이 이 스크립트를 부를 수 있다(노드를 RT_RECORD_CHAPTER=<이 파일> 로 켰을 때).
# 기본 토픽 = 정답 경로 만들기와 FAST-LIO 다시 돌리기에 필요한 것 + 그때 실시간이 쓴 입력·출력(약 16 MB/s).
# --camera 면 RealSense 색·정렬 깊이도(약 +23 MB/s, 이벤트를 녹화본으로 다시 돌릴 때만).
# 실시간 노드와 같은 DDS 설정(cyclonedds_live.xml)으로 띄운다 — 기본 DDS 로 구독하면 라이다가 늦어진다(10-13 겪음).
# 센서·노드·FAST-LIO 는 건드리지 않는다. 녹화기만 켜고 끈다.
set -euo pipefail
LIVE="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"     # sci_live
ROOT="${BAG_ROOT:-$LIVE/bags}"
PIDF=/tmp/record_chapter.pid; NAMEF=/tmp/record_chapter.name
DDS="$LIVE/realtime/config/cyclonedds_live.xml"
[ -f "$DDS" ] || { echo "DDS 설정 없음: $DDS"; exit 1; }
MIN_FREE_GB="${MIN_FREE_GB:-10}"
TOPICS=(/lidar_points /utlidar/imu /utlidar/robot_pose /tf /tf_static /Odometry /utlidar/imu_ref /leg_odom)
CAM=(/camera/color/image_raw /camera/color/camera_info /camera/aligned_depth_to_color/image_raw /camera/aligned_depth_to_color/camera_info)

ros_env() {
  set +u
  [ -f /opt/ros/foxy/setup.bash ] && source /opt/ros/foxy/setup.bash
  source "$HOME/unitree_ros2/setup.sh"
  set -u
  export ROS2_DISABLE_DAEMON=1 CYCLONEDDS_URI="file://$DDS"
}

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

free_gb() { mkdir -p "$ROOT"; df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9; }

start() {
  local name="${1:?이름}"; shift || true
  local topics=("${TOPICS[@]}")
  [ "${1:-}" = "--camera" ] && topics+=("${CAM[@]}")
  running && { echo "이미 녹화 중: $(cat "$NAMEF")"; exit 1; }
  mkdir -p "$ROOT"
  local fg; fg=$(free_gb)
  [ "$fg" -ge "$MIN_FREE_GB" ] || { echo "남은 용량 ${fg} GB < ${MIN_FREE_GB} GB — 녹화 안 함(BAG_ROOT 를 USB 로 바꾸거나 공간 확보)"; exit 1; }
  local d="$ROOT/$(date +%m%d_%H%M%S)_${name}"
  ros_env
  # 2026-10-15: 스크립트 안 백그라운드(&)는 SIGINT 를 무시한 채 뜬다(bash 규칙) → stop 의 INT 가 안 먹고 10초 뒤 강제 종료가 됐다(PC 시험).
  #   녹화기를 띄우기 직전에 SIGINT 를 기본값으로 되돌려(파이썬 한 줄, exec) INT 로 정상 종료(로스백 마무리)되게 한다.
  setsid python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
    nice -n 5 ros2 bag record -o "$d" "${topics[@]}" > "$d.log" 2>&1 < /dev/null &
  echo $! > "$PIDF"; echo "$d" > "$NAMEF"
  sleep 2
  running || { echo "녹화기가 바로 꺼짐 — $d.log:"; tail -5 "$d.log"; exit 1; }
  echo "녹화 시작: $d  (토픽 ${#topics[@]}개, 남은 용량 ${fg} GB)"
}

stop() {
  running || { echo "녹화 중 아님"; rm -f "$PIDF"; return 0; }
  local p d; p=$(cat "$PIDF"); d=$(cat "$NAMEF")
  kill -INT -- -"$p" 2>/dev/null || kill -INT "$p"
  for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$p" 2>/dev/null && { echo "녹화 끝 실패: 10초 안에 안 꺼져 강제 종료(로스백 끝이 안 닫혔을 수 있음)"; kill -TERM -- -"$p" 2>/dev/null || true; }
  rm -f "$PIDF"
  echo "녹화 끝: $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
  summary "$d"
}

summary() {
  local d="$1"
  ros_env
  ros2 bag info "$d" 2>/dev/null | python3 -c '
import re, sys
s = sys.stdin.read()
m = re.search(r"Duration:\s*([\d.]+)s", s); dur = float(m.group(1)) if m else 0.0
print(f"  길이 {dur:.1f} s")
want = {"/lidar_points": 9.0, "/utlidar/imu": 100.0, "/Odometry": 9.0}
got = re.findall(r"Topic:\s*(\S+)\s*\|\s*Type:\s*\S+\s*\|\s*Count:\s*(\d+)", s)
for name, cnt in got:
    hz = int(cnt) / dur if dur else 0.0
    flag = "" if name not in want else ("  OK" if hz >= want[name] else f"  ※ 기대 {want[name]:.0f} Hz 이상")
    print(f"  {name:45s} {int(cnt):7d}  {hz:6.1f} Hz{flag}")
seen = {n for n, c in got if int(c) > 0}
for name in want:
    if name not in seen:
        print(f"  {name:45s}       0             ※ 없음 — 그 토픽이 안 나오고 있거나 QoS 가 안 맞음")
'
}

case "${1:-}" in
  start) shift; start "$@" ;;
  stop) stop ;;
  status)
    if running; then d=$(cat "$NAMEF"); echo "녹화 중: $d  $(du -sh "$d" 2>/dev/null | cut -f1)  남은 용량 $(free_gb) GB"; else echo "녹화 중 아님 (남은 용량 $(free_gb) GB, 저장 위치 $ROOT)"; fi ;;
  test)
    start test30
    sleep 30
    d=$(cat "$NAMEF"); stop
    echo "시험 녹화 크기: $(du -sm "$d" | cut -f1) MB / 30 s"
    rm -rf "$d" "$d.log"; echo "시험 녹화 지움" ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac
