#!/usr/bin/env bash
# KCI 2026-10-17 현장: 재부팅(배터리 교체) 뒤 우리 묶음을 한 번에 띄운다. 선배 원본(~/sci_ws 원래 트리)은 안 건드림.
#   순서: 선배 ROPUS Hesai·RealSense 서비스 stop → 센서(Hesai·브리지·FAST-LIO) → 실시간 노드(웹 8765) → RealSense
#         → 조준 콘솔(8088, 단계식 초점 45 s) → E5(8095, 팬틸트 실제 조준·초점·촬영) + 판독기(OCR·E6)
#   단계마다 subshell 로 따로 띄운다(환경 섞임 방지: PYTHONNOUSERSITE 가 콘솔로 새면 'pyserial 이 없습니다'로 죽음, 10-17).
#   센서 시작 때 IMU 보정(최대 60 초) — 로봇을 세워 둘 것.
# 사용: bash ~/sci_ws/deploy/kci_r1_20261017/kci_field_up.sh          # 팬틸트 실제
#       bash ~/sci_ws/deploy/kci_r1_20261017/kci_field_up.sh --dry    # 팬틸트 안 움직임(E5 DRY_RUN)
# 끄기: bash ~/sci_ws/deploy/kci_r1_20261017/kci_field_down.sh
set -uo pipefail
D=$HOME/sci_ws/deploy/kci_r1_20261017
PANTILT="--pantilt-execute"; [ "${1:-}" = "--dry" ] && PANTILT=""
L=$D/external/sci_live/logs/live_$(date +%m%d); mkdir -p $L
alive() { [ -f "$1" ] && grep -q "$2" /proc/$(cat "$1")/cmdline 2>/dev/null; }

# 0. 이중 실행 방지(PID 파일 기준)
for chk in "$L/node.pid:rt_live_node.py" "$D/external/sci_event/out/aim_console.pid:aim_console.py" "$D/state/pids/e5_node.pid:e5_onboard.py"; do
  if alive "${chk%%:*}" "${chk##*:}"; then echo "이미 떠 있음: ${chk##*:} ($(cat ${chk%%:*})) — 먼저 kci_field_down.sh"; exit 1; fi
done

echo "== 1/5 선배 ROPUS 센서 서비스 stop → 센서(Hesai·브리지·FAST-LIO) — 로봇 세워 두기(IMU 보정)"
systemctl --user stop ropus-go2-sensor-hesai ropus-go2-sensor-realsense; sleep 2
( cd $D/external/sci_live/realtime && timeout 90 bash tools/sensors_go2.sh start 2>&1 | grep -E "shift_s|skew|started|오류|error|안 끝남" | head -5 )

echo "== 2/5 실시간 노드(웹 8765) + RealSense"
( cd $D/external/sci_live/realtime
  source ~/unitree_ros2/setup.sh >/dev/null 2>&1
  export CYCLONEDDS_URI=file://$PWD/config/cyclonedds_live.xml PYTHONNOUSERSITE=1
  rm -f /dev/shm/rt_live_trigger
  nohup setsid taskset -c 2-5 ../venv_gpu/bin/python rt_live_node.py --ros-args -p building:=kci -p store_root:=/dev/shm/rt_live \
    -p delta_deg:=45.3 -p mirror:=true -p cloud_reliability:=best_effort -p floor_trigger:=/dev/shm/rt_live_trigger \
    > $L/node.log 2>&1 < /dev/null &
  echo $! > $L/node.pid; echo "노드 $(cat $L/node.pid)"
  timeout 60 bash $D/external/sci_live/field_tools/realsense_live.sh start 2>&1 | grep -E "Up|이미|오류|error" | head -2 )

echo "== 3/5 조준 콘솔(8088)"
( cd $D/external/sci_event/code; mkdir -p ../out
  unset PYTHONNOUSERSITE CYCLONEDDS_URI
  source /opt/ros/foxy/setup.bash >/dev/null 2>&1; export ROS2_DISABLE_DAEMON=1; source ~/unitree_ros2/setup.sh >/dev/null 2>&1
  nohup taskset -c 2-5 python3 aim_console.py --config ../configs/pantilt_calib_multi_20260928.json --odom-topic /Odometry \
    --odom-base --odom-delta-deg 45.3 --use-depth --no-eeprom-sync \
    --af-profile ../configs/af_profile_kci_20261017.json --af-budget-s 45 > ../out/aim_console.log 2>&1 < /dev/null &
  echo $! > ../out/aim_console.pid; echo "콘솔 $(cat ../out/aim_console.pid)" )
ok=0; for i in $(seq 1 45); do curl -s -m 2 http://127.0.0.1:8088/state | grep -q '"r1-aim-1"' && { ok=1; break; }; sleep 1; done
if [ $ok = 1 ]; then echo "콘솔 준비(${i}s)"; else echo "콘솔 8088 이 45 초 안에 안 뜸 — tail $D/external/sci_event/out/aim_console.log"; tail -5 $D/external/sci_event/out/aim_console.log; [ -n "$PANTILT" ] && exit 2; fi

echo "== 4/5 E5 + 판독기(8095) ${PANTILT:-DRY_RUN}"
( cd ~; SCI_WS_DEPLOY=$D SCI_PANTILT_APPROVED=1 SCI_E5_CPUS=0,1 SCI_AIM_TIMEOUT_S=70 \
    timeout 60 bash $D/external/sci_event/e5_onboard/start_e5_onboard.sh $PANTILT 2>&1 | grep -vE "^화면:" )

echo "== 5/5 확인"
sleep 15
R=$(ls -t $D/logs/*.node.log 2>/dev/null | head -1); echo "E5 로그 $R"; tail -2 "$R" | cut -c1-160
for p in 8765 8088 8095; do ss -ltn | grep -qE ":$p\b" && echo "포트 $p 열림" || echo "포트 $p 아직 안 열림"; done
for f in $L/node.pid $D/external/sci_event/out/aim_console.pid $D/state/pids/e5_node.pid $D/state/pids/reader.pid; do
  [ -f $f ] && { p=$(cat $f); kill -0 $p 2>/dev/null && echo "살아 있음 $(basename $f .pid) $p" || echo "죽음 $(basename $f .pid) $p"; }
done
echo "온도: $(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 | awk '{printf "%.1f°C", $1/1000}') · 주소 $(hostname -I)"
echo "웹: http://<로봇>:8765 [시작] 누르고 걷기 · 콘솔 :8088 · E5 :8095"
