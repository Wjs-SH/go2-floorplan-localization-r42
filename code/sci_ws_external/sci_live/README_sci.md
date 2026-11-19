# sci_live — 실시간 층 판별·지도 위치 (sci_ws 판, 2026-10-15)

`~/jsh_onboard_robot_20261001` 의 실시간 묶음을 sci_ws 안으로 복사하고 그 위에 10-13~10-15 수정을 얹은 것이다. 원본 폴더는 그대로 두었고(읽기만 함), 이 폴더 밖에는 아무것도 쓰지 않는다. 위치추정·층 판별 알고리즘과 축척 탐색은 원본과 같다. 바뀐 것은 지도(정보관 9차-문), 시작 속도, 웹 화면, 녹화 도구다.

같은 sci_ws 의 `external/sci_event/` 는 조준·판독 묶음(원본 `~/kci_event_handoff_20261009`)이다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `realtime/` | ROS 노드(`rt_live_node.py`), 층·위치 작업자(`loc_worker.py`), 웹(`tools/live_viewer.py`, 8765), 드라이버·FAST-LIO 켜기(`tools/sensors_go2.sh`), 입력 다리 노드(`tools/esy_inputs_bridge.py`), DDS·FAST-LIO 설정(`config/`) |
| `floor_id/` | 위치추정 패키지와 층별 지도(정보관 = 9차-문) |
| `a131_live/`, `a131_clean_live/` | 구조점 추출·정제 실시간판 |
| `fastlio_ws/` | 수정판 FAST-LIO(걸음 진동 적응 공분산, 다리 속력 융합, 자세·높이 앵커). 여기서 빌드했고 실행 기록(`src/fast_lio/Log/`)도 여기에 쓴다 |
| `venv_gpu/` | 파이썬 환경(원본 복사, 시스템 패키지 사용) |
| `field_tools/record_chapter.sh` | 챕터 로스백 녹화 |
| `logs/` | 실행 로그·실험 보관(`live_<월일>/experiments/`), 정답 지점(`live_<월일>/checkpoints.jsonl`) |
| `bags/` | 로스백 |

원본에서 안 가져온 것: 지도 생성기(`map_gen`), 시험 데이터, 옛 로그·백업, CPU 용 venv, floor_id 결과 폴더, FAST-LIO 빌드 산출물.

## 원본 대비 바뀐 것

| 파일 | 내용 |
|---|---|
| `floor_id/maps/kci_maps_perfloor/{1..9}F/maps/` | 정보관 9차-문 지도(문 마스크를 map_gen 최신 문 위치로, 방 앞 문 아닌 틈만 막음, 벽은 기존 그대로) |
| `realtime/loc_worker.py` | 층별 지도 준비 결과를 `realtime/cache/map_prep/` 에 저장해 다음부터 불러옴(노드 켤 때 약 76초 → 몇 초, 판정 결과 동일). 지도·코드가 바뀌면 자동으로 다시 만듦 |
| `realtime/rt_live_node.py` | 웹 왼쪽 그림만 센서 방향(β 228°)대로 2.7° 더 돌림(덜 번짐). 위치추정 입력은 그대로 |
| `realtime/tools/live_viewer.py` | [정답 지점] 버튼, [시작]/[멈춤]에 녹화 묶기(아래) |
| `realtime/tools/sensors_go2.sh` | FAST-LIO 기본을 수정판(`esy`)으로. go2_ws FAST-LIO 는 돌면 go2_ws 폴더에 기록을 써서 기본에서 뺐다(`FASTLIO_IMPL=go2ws` 로 직접 고를 수는 있음) |
| `field_tools/record_chapter.sh` | DDS 설정·저장 위치를 이 폴더 안에서 찾음. 녹화기가 Ctrl+C 신호(INT)를 받게 띄움(전에는 무시돼 끌 때 10초 뒤 강제 종료됐음) |

## 켜기 (로봇은 세워 둔 채)

```bash
# 0) 선배 Hesai 드라이버 잠깐 끄기(stop 만. 로봇을 껐다 켜면 원래대로 돌아옴)
systemctl --user stop ropus-go2-sensor-hesai
ps -eo comm | grep -c '^hesai_ros_drive'        # 0 이어야 함

# 1) 드라이버(실시간 DDS) + 수정판 FAST-LIO. 처음 약 12초 IMU 보정 — 로봇 완전히 정지
cd ~/sci_ws/external/sci_live/realtime
bash tools/sensors_go2.sh start
bash tools/sensors_go2.sh status

# 2) 노드 + 웹(http://<로봇 주소>:8765). 녹화 묶기를 쓰려면 RT_RECORD_CHAPTER 를 준다
source ~/unitree_ros2/setup.sh
export CYCLONEDDS_URI=file://$PWD/config/cyclonedds_live.xml PYTHONNOUSERSITE=1
export RT_RECORD_CHAPTER=$PWD/../field_tools/record_chapter.sh      # 빼면 녹화 묶기 꺼짐(웹은 예전과 같음)
rm -f /dev/shm/rt_live_trigger
taskset -c 2-7 nice -n 10 ../venv_gpu/bin/python rt_live_node.py --ros-args \
  -p building:=kci -p store_root:=/dev/shm/rt_live \
  -p delta_deg:=45.3 -p mirror:=true -p cloud_reliability:=best_effort \
  -p floor_trigger:=/dev/shm/rt_live_trigger
```

끄기: 노드 터미널 Ctrl+C → `bash tools/sensors_go2.sh stop` → `systemctl --user start ropus-go2-sensor-hesai`(로봇 전원을 끌 거면 생략 가능).

## 웹에 더한 것

**정답 지점**: 바닥에 표시해 둔 지점에 로봇을 세우고, 칸에 이름(예: `417-A`)을 적고 누른다. 그 순간의 시각, 실시간 자세, 마지막 표본이 `logs/live_<월일>/checkpoints.jsonl` 에 한 줄 남고, 실험 중이면(결과가 나온 뒤 다음 [시작] 전까지 포함) 그 실험 보관본 `experiment.json` 의 `marks` 에도 들어간다. 로스백 없이도 "이 시각에 여기 있었다"는 점 정답이 된다.

**녹화 묶기**(노드를 `RT_RECORD_CHAPTER` 로 켰을 때만): [시작]을 누르면 `record_chapter.sh start exp_<시분초>` 로 녹화가 켜지고, [멈춤]을 누르면 5초(`RT_RECORD_TAIL`) 뒤 꺼진다. 녹화 이름은 실험 보관 폴더(`exp_<시분초>_<층>F`)와 앞부분이 같다. 화면 위 녹화 줄에 상태와 끌 때의 주기 요약(라이다·IMU·`/Odometry` Hz)이 나오고, 보관본 `experiment.json` 의 `record` 에도 남는다.
- 손으로 미리 켠 녹화가 있으면 웹 녹화는 '녹화 실패(이미 녹화 중)'로 표시되고, [멈춤]은 그 손 녹화를 끄지 않는다.
- [멈춤] 직후 바로 [시작]하면 앞 녹화가 꺼진 다음 새 녹화를 켠다.
- 걷는 중 [초기화]를 누르면 녹화도 바로 끈다.

PC 시험(10-13 4층 실험 기록을 가짜 노드로 재생, 진짜 녹화 스크립트 + 가짜 ros2, 화면 없는 크롬으로 실제 버튼 클릭): 35개 항목 모두 통과. 로봇에서 처음 쓸 때 `record_chapter.sh test` 로 진짜 녹화가 켜지고 꺼지는지 한 번 본다.

## 녹화

```bash
bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh test            # 30초 시험 → 주기·크기 확인 후 지움
bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh start c01_417앞   # 손으로
bash ~/sci_ws/external/sci_live/field_tools/record_chapter.sh stop
```

저장 위치 `bags/<월일_시분초>_<이름>/`(`BAG_ROOT` 로 USB 지정 가능). 남은 용량 10 GB 아래면 녹화하지 않는다. 기본 토픽은 정답 경로 만들기와 FAST-LIO 다시 돌리기에 필요한 것(약 16 MB/s, 3분 ≈ 3 GB).
