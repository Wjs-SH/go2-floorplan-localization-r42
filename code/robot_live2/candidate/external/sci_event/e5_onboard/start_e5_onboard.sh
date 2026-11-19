#!/usr/bin/env bash
# 로봇에서: 선배 SCI E5(UI 8095 + e5_node + 판독기)를 우리 것을 얹은 판으로 띄운다(2026-10-15). 드라이버·FAST-LIO·로봇 이동은 안 건드림.
#   bash start_e5_onboard.sh                                          # 팬틸트 안 움직임(DRY_RUN): 찾기·멈춤 표시·RealSense 대체 사진·판독
#   SCI_PANTILT_APPROVED=1 bash start_e5_onboard.sh --pantilt-execute  # 팬틸트 실제 조준·초점·촬영(조준 콘솔 8088 을 먼저 띄울 것)
#   끄기: bash ~/sci_ws/src/sci_bringup/stop_e5.sh   (선배 끄기 스크립트 그대로 — 같은 PID 파일 이름을 쓴다)
# 오돔: FAST-LIO /Odometry 를 몸체 기준으로(SCI_ODOM_BASE=1, δ 45.3°, 실시간 노드·조준 콘솔과 같은 식).
# 화면: 노트북 브라우저 http://<로봇>:8765 (위쪽 띠: 멈추세요/세워 두세요/출발) · 자세한 것 http://<로봇>:8095 (선배 UI)
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# 로봇 터미널이 로그인 때 불러오는 Unitree ROS 환경(RMW·DDS 설정)을 먼저 — 없으면 ssh 로 띄울 때 컬러 영상이 bad_alloc 으로 안 들어온다(10-15)
set +u; source "$HOME/unitree_ros2/setup.sh" > /dev/null; source "$HOME/sci_ws/src/sci_bringup/sci_env.sh"; set -u
# 2026-10-17 (KCI): sci_env.sh 가 SCI_WS=$HOME/sci_ws 로 덮어쓴다(R1 배포안 1-3 지적). 따로 설치한 트리로 띄우려면 SCI_WS_DEPLOY=<그 트리>.
if [ -n "${SCI_WS_DEPLOY:-}" ]; then export SCI_WS="$SCI_WS_DEPLOY"; fi
# 받는 쪽 DDS: 수신 버퍼 8 MB 인 실시간 설정(sci_live). 선배 RealSense(기본 DDS, 작은 조각)를 조준 콘솔과 같이 받으면 기본 설정에서는
#   컬러가 거의 안 들어왔다(10-15: 3분 반에 7장, 깊이는 313장). SCI_DDS_XML= 로 비우면 Unitree 설정 그대로.
SCI_DDS_XML=${SCI_DDS_XML-$HOME/sci_ws/external/sci_live/realtime/config/cyclonedds_live.xml}
[ -n "$SCI_DDS_XML" ] && export CYCLONEDDS_URI=file://$SCI_DDS_XML
RUN=${SCI_RUN:-kci_e5_$(date +%Y%m%d_%H%M%S)}
cd "$SCI_WS"
n=$(pgrep -fc realsense2_camera_node || true); [ "$n" -ge 1 ] || { echo "RealSense 드라이버 없음 — 중단"; exit 2; }
if pgrep -f "bin/python[^ ]* [^ ]*(sci_visual_observation/e5_node|e5_onboard/e5_onboard)\.py" >/dev/null; then echo "E5 가 이미 돌고 있음 — 먼저 stop_e5.sh"; exit 3; fi
if [[ " $* " == *" --pantilt-execute "* ]]; then
  [ "${SCI_PANTILT_APPROVED:-0}" = 1 ] || { echo "--pantilt-execute 는 SCI_PANTILT_APPROVED=1 과 같이"; exit 4; }
  curl -s -m 2 http://127.0.0.1:8088/state | grep -q '"r1-aim-1"' || { echo "조준 콘솔(8088)이 없거나 r1-aim-1 이 아님(구형 콘솔은 PROTOCOL_MISMATCH) — 중단"; exit 5; }
fi
export SCI_ODOM_BASE=${SCI_ODOM_BASE:-1} SCI_ODOM_DELTA_DEG=${SCI_ODOM_DELTA_DEG:-45.3} SCI_AIM_TIMEOUT_S=${SCI_AIM_TIMEOUT_S:-30}
# 2026-10-16: P2 반응 시간(SCI 1.5 → 2.5 s, 10-15 실측)·멈춤 확인 속도(1 s 기울기). 비우면 SCI 그대로: SCI_P2_REACTION_S= SCI_ODOM_SPEED_WIN_S=0
export SCI_P2_REACTION_S=${SCI_P2_REACTION_S-2.5} SCI_ODOM_SPEED_WIN_S=${SCI_ODOM_SPEED_WIN_S:-1.0}
# 장면 사진 저장(1초에 한 장, 약 2.4 MB/분): 놓친 표지판·O/X 를 나중에 사진으로 확인. 끄려면 SCI_SAVE_FRAMES=0
SAVE=""; if [ "${SCI_SAVE_FRAMES:-1}" = 1 ]; then SAVE="--save-frames"; fi
pgrep -f "sci_monitor/ui_server.py" >/dev/null || { nohup $SCI_PY src/sci_monitor/ui_server.py --port ${SCI_UI_PORT:-8095} > logs/ui_server.log 2>&1 & echo $! > state/pids/ui.pid; }
# R1: SCI_WS 는 실제로 실행할 트리(candidate 배포 시 그 경로). trial 포인터(SCI_TRIAL_POINTER)는 웹이 쓰는 trial_current.json.
# 위치추정(sci_live 노드)과 같이 돌 때: E5 는 한 단계 낮은 우선순위·작업자 2개(SCI 기본 4). 밀리면 SCI 가 장면을 건너뛴다(--max-lag).
#   10-15 로봇: 기본 설정으로 같이 띄우니 위치추정 지연 0.1 → 1.1~1.3 초, 부하 5 → 7.7. nice 5·작업자 2 로도 2.1~2.9 초·부하 10.
#   SCI_E5_CPUS=0,1 이면 E5·판독기를 FAST-LIO 와 같은 0·1번 코어에 가둔다(위치추정 노드는 2-7).
nohup taskset -c ${SCI_E5_CPUS:-0-7} nice -n ${SCI_E5_NICE:-5} $SCI_PY "$HERE/e5_onboard.py" --source ros --odom-topic /Odometry --workers ${SCI_E5_WORKERS:-2} --run-name "$RUN" $SAVE "$@" > "logs/$RUN.node.log" 2>&1 &
echo $! > state/pids/e5_node.pid
sleep 2; mkdir -p "experiments/$RUN/jobs/pending"
# E6(판독 호실 → 의미지도 → 층 확률·로봇 후보 영역): 정보관 9개 층. 선배 E6 그대로(지도 ~/sci_ws/maps/kci_handoff = 우리 웹 지도와 같은 픽셀)
SCI_READER_ARGS=${SCI_READER_ARGS-"--e6-floors 1F 2F 3F 4F 5F 6F 7F 8F 9F --building kci"}
# VLM 기본값은 SCI 그대로(on_demand). 10-15 의 '600초 재시도'는 R1 판독기(OCR 우선·작업별 유한 대기)가 고쳤다 — 메모리가 모자라면 OCR 만으로 끝남.
nohup taskset -c ${SCI_E5_CPUS:-0-7} nice -n 5 $SCI_PY "$HERE/reader_onboard.py" --run "experiments/$RUN" --vlm ${SCI_VLM_MODE:-on_demand} $SCI_READER_ARGS > "logs/$RUN.reader.log" 2>&1 &
echo $! > state/pids/reader.pid
echo "run=$RUN · 오돔 몸체 기준 δ $SCI_ODOM_DELTA_DEG° · 조준 대기 ${SCI_AIM_TIMEOUT_S}s · P2 반응 ${SCI_P2_REACTION_S:-SCI}s · VLM ${SCI_VLM_MODE:-on_demand} · 장면 저장 ${SAVE:-끔} · 팬틸트 $([[ " $* " == *" --pantilt-execute "* ]] && echo 실제 || echo 'DRY_RUN')"
echo "화면: http://<로봇 와이파이 주소>:8765 (띠) · :${SCI_UI_PORT:-8095} (선배 UI) — 주소 $(hostname -I) · 로그 $SCI_WS/logs/$RUN.*.log"
