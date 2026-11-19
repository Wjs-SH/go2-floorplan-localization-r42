#!/bin/bash
# sensors_go2.sh — Hesai 드라이버·FAST-LIO(~/go2_ws, 남의 폴더 — 고치지 않고 띄우기만)를 실시간 DDS 설정으로 띄우고 끈다.
#   bash tools/sensors_go2.sh start [기본DDS]   # 기본DDS 를 주면 Unitree 기본 설정(eth0)으로 — 비교용
#   bash tools/sensors_go2.sh stop
#   bash tools/sensors_go2.sh status
#   bash tools/sensors_go2.sh stop_fastlio | start_fastlio   # 웹 [초기화] 가 부른다
#
# FASTLIO_IMPL=esy 로 띄우면(10-09) go2_ws 대신 esy 수정 FAST-LIO(../fastlio_ws, 전달본 + QoS·Foxy 빌드 두 줄)를
#   esy 레시피(config/fastlio_esy_recipe_live.yaml)로 띄운다. 같이 뜨는 것:
#   tools/esy_inputs_bridge.py — /utlidar/imu → /utlidar/imu_ref(LiDAR 시계 + 자이로 바이어스), /utlidar/robot_pose → /leg_odom.
#   처음 약 12초는 IMU 보정이라 로봇을 세워 둘 것. 보정이 끝나야 FAST-LIO 를 띄운다.
#   레시피 1단계 환경변수는 아래 기본값. 밖에서 주면 그 값이 이긴다(예: FASTLIO_LEG_K=0 으로 다리 속력 끄기).
#   2단계: FASTLIO_YAWK=0.05 FASTLIO_MANH_WARM=150 을 밖에서 주면 그대로 넘어간다.
#   FAST-LIO 는 0·1번 코어에 묶는다(taskset). nice −5 는 root 가 아니라 못 한다.
#   BRIDGE_ARGS="--shift 0.0" 처럼 주면 IMU 시각차를 esy 방식 대신 그 값으로 고정한다(tools/imu_lidar_lag.py 로 잰 값 시험용).
HERE=$(cd "$(dirname "$0")/.." && pwd)
L=$HERE/../logs/live_$(date +%m%d); mkdir -p $L
source ~/unitree_ros2/setup.sh > /dev/null
source ~/go2_ws/install/setup.bash
[ "$2" = "기본DDS" ] || export CYCLONEDDS_URI=file://$HERE/config/cyclonedds_live.xml
# 웹 [초기화]는 start_fastlio 를 환경변수 없이 부른다 → start 때 고른 구현·설정·환경변수(fastlio.launch.sh)를 그대로 다시 쓴다.
#   (10-09 사고: esy 로 시작했는데 [초기화] 뒤 go2_ws FAST-LIO 로 바뀌어 실험 하나가 다른 조건으로 돌았다.)
if [ "$1" = start_fastlio ] && [ -z "$FASTLIO_IMPL" ] && [ -f $L/fastlio.launch.sh ]; then source $L/fastlio.launch.sh; fi
# 2026-10-15 sci_ws 판: 기본을 esy 수정판(이 폴더의 ../fastlio_ws)으로. go2_ws FAST-LIO 는 돌면 go2_ws 폴더(Log/)에 기록을 써서
#   선배 운영 규칙(원본 폴더 쓰기 금지)에 걸린다. 그래도 쓰려면 FASTLIO_IMPL=go2ws 를 직접 준다.
IMPL=${FASTLIO_IMPL:-esy}
if [ "$IMPL" = esy ]; then
  FWS=$(cd "$HERE/../fastlio_ws" && pwd)
  source $FWS/install/setup.bash                 # go2_ws 위에 마지막으로 — 같은 이름 fast_lio 는 이쪽이 이긴다
  CFG=${FASTLIO_CONFIG:-$HERE/config/fastlio_esy_recipe_live.yaml}
  export FASTLIO_SPK_KG=${FASTLIO_SPK_KG:-100} FASTLIO_SPK_KA=${FASTLIO_SPK_KA:-1} FASTLIO_SPK_ALPHA=${FASTLIO_SPK_ALPHA:-0.9}
  export FASTLIO_LEG_K=${FASTLIO_LEG_K:-0.5} FASTLIO_ATTK=${FASTLIO_ATTK:-0.05} FASTLIO_GNDK=${FASTLIO_GNDK:-0.05}
else
  # FAST-LIO 설정: 기본은 go2_ws 설정을 복사해 extrinsic_R 만 esy 로스백 정본으로 바꾼 것(10-08, IMU 위쪽과 2.3°).
  # 원래 go2_ws 설정으로 띄우려면 FASTLIO_CONFIG=go2_hesai_xt16_flio01_scan10.yaml
  CFG=${FASTLIO_CONFIG:-$HERE/config/fastlio_go2_esy_extrinsic.yaml}
fi

start_fastlio() {
  echo "FAST-LIO 설정: $CFG" > $L/fastlio.cfg     # 한 줄 — 노드 run_config 가 읽는다
  if [ "$IMPL" = esy ]; then
    { echo "impl=esy"; echo "started=$(date +%s.%N)"; echo "fast_lio_prefix=$(ros2 pkg prefix fast_lio)"; env | grep -E '^FASTLIO_' | grep -vE '^FASTLIO_(IMPL|CONFIG)=' | sort; } > $L/fastlio.env
    nohup setsid taskset -c 0,1 ros2 launch fast_lio mapping.launch.py config_path:=$(dirname $CFG) config_file:=$(basename $CFG) rviz:=false \
      > $L/fastlio.log 2>&1 < /dev/null & echo $! > $L/fastlio.pid
  else
    { echo "impl=go2ws"; echo "started=$(date +%s.%N)"; echo "fast_lio_prefix=$(ros2 pkg prefix fast_lio)"; } > $L/fastlio.env
    nohup setsid ros2 launch fast_lio mapping_go2_flio01.launch.py config:=$CFG > $L/fastlio.log 2>&1 < /dev/null & echo $! > $L/fastlio.pid
  fi
}

stop_one() {                                     # PID 파일로만 끈다(패턴 kill 은 원격 셸까지 죽인다)
  [ -f $L/$1.pid ] || return 0
  P=$(cat $L/$1.pid); kill -INT -- -$P 2>/dev/null
  for i in $(seq 1 10); do kill -0 $P 2>/dev/null || break; sleep 1; done
  kill -0 $P 2>/dev/null && kill -KILL -- -$P 2>/dev/null
  rm -f $L/$1.pid
}

case "$1" in
  start)
    echo "DDS: ${CYCLONEDDS_URI:0:80} · FAST-LIO: $IMPL"
    { echo "export FASTLIO_IMPL=$IMPL FASTLIO_CONFIG=$CFG"; env | grep -E '^FASTLIO_' | grep -vE '^FASTLIO_(IMPL|CONFIG)=' | sed 's/^/export /'; } > $L/fastlio.launch.sh
    nohup setsid ros2 launch hesai_ros_driver start.py > $L/hesai.log 2>&1 < /dev/null & echo $! > $L/hesai.pid
    sleep 4
    if [ "$IMPL" = esy ]; then
      rm -f $L/imu_ref_calib.json
      nohup setsid taskset -c 2-7 python3 $HERE/tools/esy_inputs_bridge.py --out $L --compare $BRIDGE_ARGS > $L/bridge.log 2>&1 < /dev/null & echo $! > $L/bridge.pid
      echo "IMU 보정 중(로봇 세워 두기)…"
      for i in $(seq 1 60); do [ -f $L/imu_ref_calib.json ] && break; sleep 1; done
      [ -f $L/imu_ref_calib.json ] || { echo "IMU 보정이 60초 안에 안 끝남 — $L/bridge.log 확인. FAST-LIO 안 띄움"; bash "$0" status; exit 1; }
      grep -E '"(shift_s|skew_std_ms|imu_correct_style_offset_s)"' $L/imu_ref_calib.json
    fi
    start_fastlio
    sleep 8; bash "$0" status ;;
  stop_fastlio)                                  # 웹 [초기화]: 드라이버·브리지는 두고 FAST-LIO 만
    stop_one fastlio ;;
  start_fastlio)
    start_fastlio ;;
  stop)
    for n in fastlio bridge hesai; do stop_one $n; done
    bash "$0" status ;;
  status)
    ps -eo pid,psr,pcpu,rss,args | grep -E "[h]esai_ros_driver_node|[f]astlio_mapping|[i]mu_correct.py|[e]sy_inputs_bridge|[r]obot_state_publisher|[s]tatic_transform_publisher.*map_viz" | cut -c1-140
    [ -f $L/fastlio.env ] && head -2 $L/fastlio.env
    echo "로그: $L" ;;
esac
