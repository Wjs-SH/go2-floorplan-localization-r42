#!/usr/bin/env bash
# KCI 2026-10-17 현장: kci_field_up.sh 로 띄운 것만 PID 파일로 끄고 선배 ROPUS 센서 서비스를 되돌린다.
#   패턴 kill(pkill -f) 안 씀 — 원격 셸까지 죽인다. 선배 원본·다른 사람 노드는 안 건드림.
# 사용: bash ~/sci_ws/deploy/kci_r1_20261017/kci_field_down.sh
set -uo pipefail
D=$HOME/sci_ws/deploy/kci_r1_20261017
L=$D/external/sci_live/logs/live_$(date +%m%d)
stop_pid() {   # $1 PID 파일, $2 cmdline 에 있어야 할 글자, $3 이름, $4 그룹째 끄기(1)
  [ -f "$1" ] || return 0; local p; p=$(cat "$1")
  if grep -q "$2" /proc/$p/cmdline 2>/dev/null; then
    if [ "${4:-0}" = 1 ]; then kill -INT -- -$p 2>/dev/null; else kill -TERM $p; fi
    for i in $(seq 1 15); do kill -0 $p 2>/dev/null || break; sleep 1; done
    kill -0 $p 2>/dev/null && { [ "${4:-0}" = 1 ] && kill -KILL -- -$p 2>/dev/null || kill -KILL $p; echo "  $3 강제 종료"; }
    echo "끔 $3 ($p)"
  fi
  rm -f "$1"
}
echo "== E5·판독기·UI"
stop_pid $D/state/pids/reader.pid  reader_onboard.py 판독기
stop_pid $D/state/pids/e5_node.pid e5_onboard.py     E5
stop_pid $D/state/pids/ui.pid      ui_server.py      "UI 8095"
echo "== 조준 콘솔"
stop_pid $D/external/sci_event/out/aim_console.pid aim_console.py "조준 콘솔"
echo "== RealSense·실시간 노드·센서"
bash $D/external/sci_live/field_tools/realsense_live.sh stop
stop_pid $L/node.pid rt_live_node.py "실시간 노드" 1
( cd $D/external/sci_live/realtime && bash tools/sensors_go2.sh stop | tail -3 )
echo "== 선배 ROPUS 센서 서비스 되돌림"
systemctl --user start ropus-go2-sensor-realsense ropus-go2-sensor-hesai; sleep 2
systemctl --user is-active ropus-go2-sensor-realsense ropus-go2-sensor-hesai | paste -sd' ' | sed 's/^/realsense·hesai: /'
echo "== 남은 우리 프로세스"
ps -eo pid,args | grep -E "[k]ci_r1_20261017|[r]t_live_node|[a]im_console.py" | cut -c1-120 || true
