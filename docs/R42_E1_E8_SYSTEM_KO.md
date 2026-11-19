# Go2 평면도 기반 층·자세 위치추정 — 시스템 설명 (E1~E8)

이 문서는 저장소에 실린 운영 코드가 실험 한 건 동안 무엇을 하는지 처음부터 확정까지 설명한다. 다룰 내용은 다음과 같다.
- 어떤 프로세스가 뜨는가
- 프로세스끼리 무엇을 주고받는가
- 각 단계가 어떤 계산으로 무엇을 내는가
- 층과 자세가 어떤 조건에서 확정되는가

모든 설명에는 코드 위치(`파일:줄`)를 붙였다. 9장은 저장소의 실험 기록 `examples/X20261116_182124` 를 시간순으로 따라간다.

## 0. 개요

### 0.1 문제

로봇은 9층 건물 어딘가에서 출발한다. 출발 층과 출발 위치·방향은 주어지지 않는다. 시스템이 찾아야 하는 것은 두 가지다.
1. **층**
2. **그 층 평면도 위의 위치(x, y)와 방향 θ.** FAST-LIO 주행거리계 좌표를 평면도 좌표로 옮기는 변환으로 표현한다.

쓰는 정보는 세 가지다.
- 건물 안내도로 만든 층별 평면도(벽, 방·복도 다각형, 방 번호)
- LiDAR 관성 주행거리계 FAST-LIO
- 복도의 방 번호 명판

**운용 방식.** 운용자가 로봇을 조이스틱으로 걷게 한다. 시스템이 명판을 찾으면 소리로 알리고, 운용자는 로봇을 명판 앞에 세워 팬틸트 카메라로 찍는다.

### 0.2 단계

| 단계 | 하는 일 | 주 코드 |
|---|---|---|
| E1 | 안내도로 만든 구조지도·의미지도. 런타임은 읽기만 한다 | 지도 파일(저장소에 없음), `L2/e2_worker.py` 의 경량 지도 생성 |
| E2 | 전 층 거친 탐색. 최근 20~60 s 의 LiDAR 로 9개 층 모두에서 변환 후보를 찾는다 | `L2/e2_worker.py`, `e2_core.py`, `e2_coarse.py` |
| E3 | 추적과 순위. 걸을 때마다(0.5 m·15°) 모든 후보를 다시 채점·보정하고 순위를 매긴다 | `L2/e3_tracker.py`, `e34_node.py` |
| E4 | 구조만으로 층이 갈리는지 판단한다. 표지 탐지를 켤지 정한다 | `L2/e4_ambiguity.py` |
| E5 | 앞 카메라 + SigLIP 으로 명판을 찾고 운용자에게 정지를 요청한다 | `L2/visual_stop.py`, `SRC/sci_visual_observation`, `SRC/sci_legacy_e5` |
| E6 | 명판 앞에서 팬틸트 사진을 찍고 OCR·VLM 으로 번호를 읽는다. 층 가능도를 계산한다 | 촬영 도구 `plate_site.py`, `L2/e6_ocr.py`, `e6_semantic.py`, `SRC/sci_text_reading` |
| E7 | 명판 기준 후보를 만들고, 후보를 LiDAR 벽 점으로 정밀 정합한다. 기하와 명판을 합친 층 점수로 확정한다 | `L2/semantic_anchor.py`, `e7_proc.py`, `e7_rematch.py`, `fused_floor.py`, `e6e7_link.py` |
| E8 | 명판 두 장 이상이 한 층의 한 자리로 설명되는지 판정한다(쌍 판정). 화면 표시도 맡는다 | `L2/e8_pair.py`, `web_v2/index.html` |

### 0.3 확정에 이르는 주 경로

1. **E2·E3 의 단계.** 층마다 구조가 비슷한 이 건물에서는 몇 개 층이 거의 같은 점수로 남는다. E4 는 대부분 '모호'를 낸다.
2. **첫 명판.** 확실한 판독(라벨 확률 ≥ 0.6)이 나오면 두 가지가 일어난다.
   - 그 방 앞 복도 벽에 명판을 고정해 변환 후보를 만든다(명판 기준 후보). 이 후보는 E3 에 들어간다.
   - 판독과 맞는 가설이 E3 순위의 앞으로 간다.
3. **두 번째 명판.** 다른 방의 명판을 읽으면 쌍 판정이 돈다. 두 명판 사이의 주행거리, 방 사이 거리, 가설마다 복도 반폭 규칙, 경로의 벽 통과를 본다. 남은 후보가 한 층뿐이면 READY 다.
4. **E7 정합.** 명판 후보 최대 4개 + 경쟁 1개를 최근 90 s 벽 모양 점으로 정밀 정합한다.
5. **층 확정.** 후보마다 (기하 비용 차 ÷ 0.05) + 명판마다 min(d²/2, ln 20) 을 계산한다. 1·2위 층의 차가 ln 100 이상이면 확정이다.
6. **확정 뒤.** 현재 자세 = 확정 변환 ∘ 최신 FAST-LIO 자세다.

명판 한 장으로 확정하는 경로(8.6절)와, 구조만으로 확정하는 경로(설정상 SHADOW 또는 정지 뒤 검증)도 있다.

## 1. 하드웨어와 실행 환경

코드·설정·기록에 적힌 것만 옮긴다. 경로 약어는 다음과 같다.

| 약어 | 경로 |
|---|---|
| L2 | `code/robot_live2/candidate/live2` |
| L1 | `code/robot_live2/candidate/live1` |
| SRC | `code/robot_live2/candidate/src` |
| CFG | `code/robot_live2/candidate/config` |
| EXT | `code/sci_ws_external` |
| REC | `examples/X20261116_182124` |

**센서와 기기**

| 구성 | 내용 | 근거 |
|---|---|---|
| 로봇 | Unitree Go2. 이동은 운용자가 조이스틱으로 하고, 위치추정 시스템은 이동 명령을 보내지 않는다 | L2/console.py:1169-1170 |
| LiDAR | Hesai(`hesai_ros_driver_node`), 설정 이름 `go2_hesai_xt16`, 16선, 10 Hz, 스캔당 약 64 000 점 | EXT/sci_live/realtime/tools/sensors_go2.sh:35, REC `fastlio_*.out` |
| IMU | 두 방식이 있다. `l1` 은 LiDAR 내장 IMU `/utlidar/imu`, `body` 는 Go2 본체 IMU(`/lowstate`). REC 는 `body` | L1/ctl.py:45-52 |
| RGB-D | RealSense D435i, 640×480, 색 약 7.5 Hz · 깊이 약 11 Hz. 내부 변수 fx 606.80, fy 606.75, cx 310.49, cy 240.52. 깊이는 uint16 mm | REC `state/e5_shm/status.json`, hub index |
| 팬틸트 카메라 | Arducam IMX477(V4L, 촬영 1920×1080), Nano 제어기(FTDI 직렬), 서보 범위 pan 50–150°·tilt 115–180°, HOME 95.35/157.19 | L2/e6_aim_console_ctl.py:4-6, 33-40 |
| 연산 | Jetson Orin NX, CPU 코어 0–7. GPU 는 SigLIP(fp16)과 VLM 서버가 함께 쓴다 | L2/wallfit_cpu_fast.py:3, SRC/sci_visual_observation/siglip_worker.py:59-80 |
| 소프트웨어 | ROS 2 Foxy, CycloneDDS(`rmw_cyclonedds_cpp`), `ROS_DOMAIN_ID=0` | L1/ctl.py:67-72 |
| 모델 | SigLIP `siglip-base-patch16-224`(추가 학습 없음), Qwen3-VL-2B-Instruct Q8_0 GGUF + mmproj F16(llama.cpp), RapidOCR(PP-OCRv4) | L1/ctl.py:55, L2/vlm_ondemand.py:20-22, L2/e6_ocr.py:1-5 |

**보호 프로세스.** 로봇 기본 서비스가 띄우는 `realsense2_camera_node`, `hesai_ros_driver_node`, `go2_control_runner`, `robot_state_publisher` 는 위치추정 시스템이 시작하거나 끄지 않는다. 감독기가 살아 있는지만 본다(L1/ctl.py:66).

## 2. 프로세스 구성과 통신

### 2.1 실행 계층

1. **`live2-console.service`(systemd 사용자 서비스)**
   - 웹 콘솔 하나만 띄운다(`scripts/live2 serve`). 연구 모듈은 이 서비스가 시작하지 않는다.
   - 실패하면 10 s 뒤 다시 띄운다. 600 s 안에 5번까지다(`code/robot_live2/scripts/live2-console.service:1-21`).
2. **`live2_cli.py serve`**
   - Tailscale 을 최대 180 s 기다린다.
   - 이후 10 s 마다 콘솔을 확인하고, 죽었으면 다시 띄운다(`code/robot_live2/scripts/live2_cli.py:189-213`, 포트 8770).
3. **`console.py`(HTTP 8770)**
   - 운용자가 [실험 시작]을 누르면 모듈마다 정해진 인자로 `ctl.py start <모듈>` 을 부른다(L2/console.py:188-196, 457-467).
   - 콘솔 자신은 감독기 밑에 두지 않는다. 감독기는 끝날 때 자손을 모두 끄기 때문이다(L1/ctl.py:501-515).
4. **`ctl.py`**
   - 시작 전 검사를 한 뒤 `live1_supervisor.py -- bash -c <명령>` 을 실행한다.
   - 소유 기록은 `runtime/owned/<모듈>.json` 에 PID 와 시작 시각으로 남긴다(L1/ctl.py:249-273).
5. **`live1_supervisor.py`**
   - 자식을 새 세션으로 띄우고 0.5 s 마다 본다. 보는 항목은 온도, 남은 메모리, 보호 프로세스, 최대 실행 시간, 중지 파일이다.
   - 프로세스 묶음을 벗어난 자손도 따로 추적한다(L1/live1_supervisor.py:178-194, 280).
   - **다시 띄우지 않는다.** 자식이 끝나면 그 모듈은 끝난 것으로 기록한다(:211-213).
   - 끌 때는 SIGINT → 10 s → SIGTERM → SIGKILL 순서다(:230-279).

예전 실행기(`candidate/kci_field_up.sh`)가 쓰던 구성요소(포트 8765·8088·8095)가 떠 있으면, 충돌 목록에 걸려 `ctl` 이 시작을 거부한다(L1/live1_common.py:29-42, L1/ctl.py:179-180).

### 2.2 모듈과 시작 순서

시작 순서는 `fastlio, lio2, hub, rec, observe, e34, e6, e2` 이고, 끄는 순서는 `e6, observe, e2, e34, rec, hub, lio2, fastlio` 다(L2/console.py:76-77). `fastlio`·`lio2` 중 하나라도 실패하면 뒤 모듈은 건너뛴다(:82, 608-616). 각 모듈은 아래 건강 확인을 통과해야 다음 모듈로 넘어간다(:480-520).

| 모듈 | 하는 일 | 건강 확인 (제한 시간) | 코어·우선순위 |
|---|---|---|---|
| `fastlio` | FAST-LIO 관성 주행거리계. `body` 방식은 본체 IMU 브리지와 출력 어댑터를 함께 띄운다 | 로그에 "Initialize the map kdtree" (60 s) | — |
| `lio2` | `lio_monitor2.py`. 점군을 받아 2D 누적 지도와 E2 표본 저장소를 만든다 | `lio_state.json` 이 2 s 안에 갱신, 자세 수신 > 0 (25 s) | core 4 |
| `hub` | `sensor_hub.py`. 카메라 색·깊이·내부 변수와 자세를 공유 메모리에 놓는다 | `index.json` 2 s 안 (20 s) | core 2,3 |
| `rec` | `field_recorder.py`(8771). 현장 표시 페이지·표식 | 1.5 s 생존 (5 s) | — |
| `observe` | `observe_keeper.py` → `observe_node.py` + SigLIP. E5 표지 탐지 | SigLIP RUNNING (40 s) | core 0,1, GPU |
| `e34` | `e34_node.py`. E3 추적, E4, E6 판독 연결, E8 쌍 판정, E7 연결, 자동 상태기계, 최종 검증 | `e4_state.json` 3 s 안 (20 s) | core 5, nice 5 |
| `e6` | 자동 조준 경로일 때만(조준 콘솔 + `e6_precision_bridge.py`). 명판 촬영 방식에서는 `e6_plate` 기록만 남는다 | `e6_bridge.json` RUNNING (30 s) | core 4 |
| `e2` | `e2_worker.py --procs 4 --device cpu --refresh-s 90`. 전 층 거친 탐색 | `e2/e2_state.json` 5 s 안 (90 s) | core 0,1,6,7, nice 10 |

모듈별 정확한 명령줄은 L1/ctl.py:377-476 에 있다. `e34` 아래에서 필요할 때 생기는 프로세스는 다음과 같다.

| 프로세스 | 실행 조건 | 근거 |
|---|---|---|
| `e7_long.py` | core 0,1,6,7, nice 5 | L2/e7_long.py:189-205 |
| `e7_proc.py` | E7 정합 전용, 시간 제한 300 s. 스위치 `LIVE2_E7_PROC`, 기본 켜짐 | L2/e7_proc.py:25-27, L2/e6e7_link.py:1195 |
| RapidOCR | OCR 스레드만 core 6,7 | L2/e6_ocr.py:25, L2/e8_pair.py:630-633 |
| `llama-server` | Qwen3-VL. `127.0.0.1:8097`, `-ngl 99 -c 4096 -t 2` | L2/vlm_ondemand.py:20-24, 47-58 |

VLM 서버는 필요할 때 띄우고 90 s 동안 요청이 없으면 끈다. 시작 조건은 남은 메모리 5400 MB 이상, 온도 78 °C 이하, 자세 8.5 Hz 이상이다(L2/vlm_ondemand.py:24). SigLIP 은 80 °C 나 2500 MB 미만에서 잠시 멈췄다가 다시 돈다(L1/observe_keeper.py:1-26).

**REC 의 시작 시각**(실험 시작부터, `timeline.jsonl`):

| 모듈 | fastlio | lio2 | hub | rec | observe | e34 | e2 |
|---|---|---|---|---|---|---|---|
| 건강 확인 통과 | 17.4 s | 24.8 s (자세 10.03 Hz) | 32.4 s | 34.3 s | 40.5 s | 41.9 s | 45.8 s |

`e6_plate` 는 41.9 s, `experiment_running` 은 45.8 s, `experiment_ready_to_move` 는 57.7 s 다.

**준비 판정.**
- `ready_loop` 가 2 s 마다 `ready_gate.evaluate` 를 부르고, 한 실험에 한 번만 '이동 준비' 를 낸다(L2/console.py:841-870, L2/ready_gate.py:17-58).
- 조건은 다음과 같다.
  - 실험 세션이 있다.
  - FAST-LIO 가 8 Hz 이상이고 최신 자세가 0.5 s 안이다.
  - 색 영상과 깊이가 정상이다.
  - SigLIP 이 돈다.
  - E2·E3·E4 상태가 5 s 안에 갱신됐고 지도 층 수가 0 보다 크다.
  - 안전 감시가 동작한다.
- E6 는 판정에 넣지 않는다.
- REC 의 값: "FAST-LIO 10.0 Hz · 최신 자세 0.013 s 전", E2/E3/E4 갱신 0.90/0.18/0.18 s, 지도 층 9.

### 2.3 ROS 2 토픽과 DDS

| 토픽 | 형식 | 내보내는 쪽 → 받는 쪽 | 근거·실측 |
|---|---|---|---|
| `/lidar_points` | PointCloud2 | Hesai 드라이버 → FAST-LIO(센서 QoS) | EXT/fastlio_ws/src/FAST_LIO_ROS2/src/laserMapping.cpp:957 |
| `/utlidar/imu` → `/utlidar/imu_corrected` | Imu | LiDAR IMU → `imu_correct.py` → FAST-LIO | `l1` 방식만 |
| `/lowstate` → `/sci_alt_imu/imu` | Go2 상태 → Imu | Go2 → IMU 브리지 → FAST-LIO | `body` 방식 |
| `/Odometry` | nav_msgs/Odometry | FAST-LIO(`body` 방식은 출력 어댑터) → `lio2`, `hub` | REC 10.01 Hz, 수신 간격 중앙 99.6 ms, 발행→수신 중앙 66.1 ms, `camera_init→base_link` (`state/lio_state.json`) |
| `/cloud_registered` | PointCloud2(세계 좌표) | 위와 같음 → `lio2` 만 | L2/lio_monitor2.py:100-103. REC 약 316 s 동안 3146개 |
| `/camera/color/image_raw` 외 2개 | Image, CameraInfo | RealSense → `hub` 만 | SRC/sci_sensor_bridge/ros_readonly.py:43-46, 79 |

- **DDS 설정**: `fastlio`·`lio2` 는 `cyclonedds_live.xml` 을 쓴다. 최대 메시지 65 500 B, 조각 65 000 B, 소켓 버퍼 8 MB 이상이다(CFG/fastlio/cyclonedds_live.xml:7-17). 주석에 따르면 기본 조각 크기(1344 B)에서는 FAST-LIO 가 6.9 Hz 로 떨어졌다. `hub`·`observe` 는 로봇 기본 DDS 설정을 쓴다(L1/ctl.py:409, 434).
- **팬틸트 제어 경로**: 팬틸트는 ROS 토픽이 아니라 직렬 포트와 조준 콘솔 HTTP(`127.0.0.1:8088`)로 제어한다(L2/e6_aim_console_ctl.py:33-35).

### 2.4 파일로 주고받는 것

같은 로봇 안의 모듈끼리는 대부분 파일로 주고받는다. JSON 은 모두 임시 파일에 쓴 뒤 이름을 바꾼다(원자적 교체). 아래 `state/` 는 실행 중 `runtime/state`, 끝나면 `experiments/<실험ID>/state` 로 옮겨진다.

| 파일 | 쓰는 쪽 → 읽는 쪽 | 주기 |
|---|---|---|
| `/dev/shm/live2_store/`(`odom_link.csv`, `samples/LIVE_nnnnnn/data_short.npz`, `far_points.npz`, `session.json`) | lio2 → e2, e34(E3·E7), e7_long, e7_proc | 표본 1.5 s 마다 (3장) |
| `/dev/shm/live1_hub/`(`index.json`, `slot_XX.color/.depth`, `live.jpg`) | hub → observe, e34(E6), console | index 0.05 s, 칸 16개, 최근 자세 200개 (SRC/sci_sensor_bridge/shared_hub.py:26, 154-203) |
| `state/lio_state.json`, `e2_feed_state.json` | lio2 → console, 준비 판정, 안전 감시, e34 | 0.5 s (L2/lio_monitor2.py:118-132) |
| `state/e2/e2_state.json`, `e2_passes.jsonl`, `semantic_anchor.json`, `e2_region_*` | e2 → e34, console | 층 하나 끝날 때마다 (L2/e2_worker.py:9-12) |
| `runtime/e2_cache/light/<건물>_<층>F_<판>/*.npy` | e2 → E3(메모리 매핑) | 층·지도 판마다 한 번 (L2/e2_worker.py:233-250) |
| `state/e2/e2_request.json`, `e2_budget.json` | e34·console → e2 | 재탐색 요청(120 s 간격 제한), 동시 작업 수 0~4 (L2/e34_node.py:226-229, L2/auto_fsm.py:195-200) |
| `state/e3_state.json`, `e4_state.json`, `e4_feed.json`, `siglip_request.json`, `e6_state.json`, `fix_current.json` | e34 → console, observe | 2 Hz (L2/e34_node.py:51, 131, 166-179) |
| `state/e3_history.jsonl`, `e4_history.jsonl`, `e4_events.jsonl` | e34 | 변할 때 |
| `state/auto_state.json`, `e8_pair_state.json`, `e8_pairs.jsonl`, `e8_observations.jsonl`, `final_verify.json`, `e7_long*.json`, `vlm_state.json`, `loc_overlay.json` | e34 → console | e34 주기 |
| `runtime/e5_shm/status.json`, `frame*.jpg` | observe → console, rec, e34 | 장면마다. 이름만 shm 이고 디스크 폴더다 |
| `experiments/live1_observe_<tag>/live2_stop_requests.jsonl`, `live2_detection_pose.jsonl`, `scenes.jsonl` | observe(정지 요청) → e34(E6) | 탐지 때 (L2/e6_tracker.py:4-5, 195) |
| `state/e6_precision_results.jsonl`, `state/e6_captures/*.jpg` | 명판 촬영 도구(또는 자동 조준 경로) → e34 | 촬영 때. REC 는 `capture_tool="plate_site_case1"`, 1920×1080 |
| `state/e6_operator_requests.jsonl`, `state/control/*.json(l)` | console → e34, e2 | 운용자 지시 때 |

### 2.5 HTTP 와 운영 화면

- **8770 콘솔**(L2/console.py:1524-1976)
  - 읽기: `/api/state` 와 `/api/ctl/state` 는 화면이 1 s 마다 함께 읽는다(L2/web_v2/index.html:529-530). 영상 경로(`/img/...`)도 있다.
  - 쓰기: `/api/ctl/<동작>` 허용 목록만 받는다. start, end, pause, resume, siglip_on/off, e2_start, e2_research, start_missing, e6_operator_target, e6_manual, e6_settle, gt_save 등이다(:1750-1852).
  - 요청 형식: JSON 64 KiB 이하만 받는다. 요청 번호로 중복을 걸러내고, 제어 작업은 한 번에 하나만 받는다.
- **운영 화면**(`web_v2/index.html`)
  - 탭은 운용 / 전역 위치 후보 / 의미 관측 / 실험 평가 / 기록·진단이다.
  - [실험 시작]은 `{building, siglip, input(l1|body), e6_mode, pantilt_consent}` 를 보낸다(:646-655).
  - 시작 위치를 지도에 찍어야 시작된다(`LIVE2_REQUIRE_START_GT` 기본 1, L2/console.py:68, 1801). 이 점은 평가용 기록일 뿐 추정에는 쓰이지 않는다.
- **8791 명판 촬영 도구**
  - 운용 탭 안에 내장 창(iframe)으로 뜬다. 도구는 `postMessage({src:"plate_site", ...})` 로 상태를 알린다(index.html:263, 1636-1665).
  - 도구는 실험 시작 때 `runtime/student_start_hook` 에 적힌 스크립트가 띄운다(L2/console.py:588-598). 이 스크립트와 도구 코드는 이 저장소에 없다.
- **접근 권한**(코드 기준, L2/console.py:96-166, 1475-1514; L2/auth.py)
  - 보기: 로봇 자신과, 로봇 소유 계정의 Tailscale 기기에 허용한다.
  - 제어: 로봇 자신과, 신뢰 목록에 오른 기기만 허용한다. 같은 LAN 이라도 허용하지 않는다.
  - 제어 요청 조건: 세션 쿠키, CSRF 헤더, Host·Origin 검사를 모두 통과해야 한다.
  - 팬틸트: 실험마다 태블릿에서 동의를 받아 그 실험에만 쓴다(L2/console.py:1190-1229).

### 2.6 실험의 시작과 끝, 안전 감시

- **상태값**: `IDLE → STARTING → RUNNING | PARTIAL → (PAUSED) → STOPPING → ENDED`(L2/console.py:170, 569, 617, 646).
  - `PAUSED` 는 분석만 멈춘다. SigLIP 은 끄고, FAST-LIO·누적·hub 는 계속 돈다(:663-681).
  - 운영 중인 모듈이 예기치 않게 끝나면 2 s 감시가 그 모듈을 `ENDED_UNEXPECTEDLY` 로 적고 실험을 `PARTIAL` 로 바꾼다. 다시 띄우는 것은 운용자의 [start_missing] 뿐이다(:893-930).
- **시작 절차**(L2/console.py:556-620):
  1. 승인·사전점검을 통과해야 한다. 온도 72 °C 미만, 메모리 4300 MB 이상, 디스크 5 GB 이상, 센서 정상, FAST-LIO 중복 없음, 이전 실험 보관 확인, 시작 위치 표시.
  2. 남은 파일을 옮긴다.
  3. 시작 훅을 실행한다.
  4. 팬틸트 동의를 기록한다.
  5. 모듈을 순서대로 시작한다.
- **종료 절차**(:622-661):
  1. 끄는 순서대로 모듈을 끈다.
  2. `state/` 를 `experiments/<ID>/state` 로 옮긴다.
  3. 모듈 로그를 복사한다.
  4. 보관한다(`exp_archive.archive_run`, L2/exp_archive.py:24-152). 대상은 표본 저장소, FAST-LIO 로그, 관측 기록이고, sha256 검증 후 `MANIFEST.json`·`SHA256SUMS` 를 쓴다.
  5. 평가한다.
  - REC: 정지 14.9 s, 보관 파일 772개.
- **시간 기록**: 실험마다 `timeline.jsonl` 에 사건이 쌓인다(L2/console.py:326-333). 종류는 `module_started`, `experiment_ready_to_move`, `e6_plate`, `siglip_on/off`, `e2_research_requested`, `experiment_ended`, `archive_verified` 등이다.
- **안전 감시**(L2/safety_guard.py:5-27): 아래 중 하나가 일어나면 실험을 자동으로 끝낸다.
  - 80 °C 이상 60 s
  - 장치 스로틀링 10 s
  - lio2·hub 모두 10 s 동안 자세 없음
  - 핵심 모듈 실패
  - 센서 고장 10 s
  - 메모리 2500 MB 미만 10 s

### 2.7 구성도

```
 로봇 기본 서비스(시작·종료 안 함): Hesai LiDAR, RealSense, Go2 제어, IMU
   │ /lidar_points           │ color/depth/info         │ /lowstate (body 방식)
   ▼                         │                          ▼
 [fastlio] FAST-LIO (+본체 IMU 브리지·출력 어댑터) ── /Odometry 10 Hz, /cloud_registered
   │                                   │
   ▼                                   ▼
 [lio2] lio_monitor2 (core 4)       [hub] sensor_hub (core 2,3) ◄── 카메라 3토픽 + /Odometry
   │ /dev/shm/live2_store (표본 1.5 s)   │ /dev/shm/live1_hub (영상·깊이·자세)
   │                                   ▼
   │                         [observe] SigLIP 탐지 (core 0,1, GPU) → 정지 요청 jsonl
   ▼                                   │
 [e2] e2_worker ×4 (core 0,1,6,7) ── e2_state.json · light/*.npy
   │            ▲ e2_request / e2_budget
   ▼            │
 [e34] e34_node (core 5): E3 · E4 · E6 연결 · E8 쌍 판정 · E7 연결 · 자동 상태기계 · 최종 검증
   ├─ e7_proc.py / e7_long.py (core 0,1,6,7)
   ├─ RapidOCR (core 6,7)
   └─ llama-server :8097 (Qwen3-VL, GPU)
 명판 촬영 도구 :8791 ──► state/e6_precision_results.jsonl, e6_captures/ ──► e34
 [console] :8770 ◄── 태블릿(브라우저) ; ctl.py → live1_supervisor → 각 모듈
```

## 3. 센서에서 표본까지

1. **FAST-LIO 설정**(CFG/fastlio/fastlio_go2_esy_extrinsic.yaml:15-48)
   - 16선, 10 Hz, 근거리 제외 0.5 m, 복셀 0.3 m, 반복 3회, 외부 파라미터 추정 끔.
   - LiDAR–IMU 외부 회전·이동은 파일에 고정값으로 있다. REC `archive/live2_store/session.json` 에 같은 sha256 이 남아 있다.
2. **`body` 방식**(REC 에서 사용): 본체 IMU 를 IMU 메시지로 바꾸는 브리지 → FAST-LIO → 출력 어댑터 순서다.
   - 어댑터는 출력을 `l1` 방식과 같은 좌표 약속으로 맞춘다. 그래서 그 뒤 단계(lio2, E2~E8)는 방식과 무관하게 같다(L1/ctl.py:45-52, 212-213).
   - 브리지가 준비되지 않으면 어댑터가 출력을 막는다. 그러면 E2 에 새 표본이 들어가지 않는다(index.html:653).
   - 브리지·어댑터 코드는 이 저장소에 없다.
3. **lio2**(L2/lio_monitor2.py:39-143)
   - 점군 하나를 두 곳에 쓴다.
     - 화면용 2D 누적 지도: 0.1 m 칸.
     - E2 표본(E2Feed): 별도 스레드, 대기열 1.
   - 표본 간격은 1.5 s 다.
4. **E2Feed — 표본 한 장 만들기**(L2/e2_feed.py:5-184)
   - **좌표 변환**: 세계 좌표 점을 같은 시각의 자세(허용 1 ms)로 로봇 몸체 좌표로 되돌린다. LiDAR–몸체 고정 변환은 t = (0.145, 0, 0.110) m, z축 90° 회전이다(:31).
   - **스캔 합치기**: 연속 3 스캔을 합친다. 0.35 s 넘게 끊기면 버린다(:136, 159-161).
   - **가까운 구조점**: 8 cm 칸으로 나눈다. 높이 −0.15~1.6 m, 거리 0.5~9 m 안에서 칸당 2점 이상, 높이 폭 0.15 m 이상인 칸만 남긴다(:32, 61-77). 벽처럼 세로로 선 것만 남기는 대용 필터다.
   - **먼 점**: 9~18 m, 높이 0.2~1.8 m, 최대 1000점(:33, 78-83).
   - **점 종류(role)**: 모두 0 이다. 점별 시각과 선 번호가 없어서, 정밀 구조 분류기는 쓰지 않는다(:20, 184).
   - **저장 좌표**: `(x, y, yaw) := (−x, y, −yaw)` 로 바꿔 저장한다(:45-49). 저장소 좌표는 FAST-LIO 발행 좌표의 거울상이다.
5. **자세 보정(PoseFix)**(L2/e2_feed_posefix.py:15-104)
   - `~/kci_loadtest_20261113/e2_clean.on` 의 내용이 `pose` 면 켜진다(L2/lio_monitor2.py:56-71).
   - 최근 8 s 의 5 cm 복셀 지도에 2D ICP 로 맞춘다. 한 번에 최대 25 cm·2° 까지, 그중 60 % 만 반영한다.
   - 보정된 자세는 표본 저장소(`odom_link.csv`)에만 들어간다. 원래 FAST-LIO 자세는 `lio_state`·hub 에 그대로 있다.
   - REC 에서 보정 누적은 마지막에 61.2 cm 였다.
6. **표본 저장소 형식**(EXT/sci_live/realtime/live_store.py:21-105)
   - `odom_link.csv` 열: `sample_id, t_sec, fastlio_x, fastlio_y, fastlio_yaw_rad, delta_s_m, delta_yaw_deg`
   - `samples/LIVE_nnnnnn/data_short.npz`: `a1_3_1_all_candidate_xy`(N×2, 몸체 좌표 m), `a1_3_1_all_candidate_role`
   - `far_points.npz`: 먼 점
   - 보관은 900 s 이지만, 첫 표본과 고정(pin)된 구간은 지우지 않는다.
   - REC 는 표본 210개, 315 s 다. 표본 처리 시간은 중앙 169 ms, 95 % 310 ms 다.

## 4. E1 — 지도

이 장과 다음 장에서 X = `code/robot_live2/candidate/external/sci_live` 다. E2 가 실제로 불러오는 사본이 이것이다(L2/e2_worker.py:38-39, 117-119). `code/sci_ws_external/sci_live` 는 이와 다섯 파일이 다르다. `metric_fit_v2.py` 의 `_window_beta`, `sensor_frame.py`, `live_store.py`, `loc_worker.py`, `rt_live_node.py` 다. 그쪽 판으로는 E2 가 돌지 않는다.

### 4.1 런타임이 읽는 지도

| 읽는 곳 | 파일 | 용도 |
|---|---|---|
| E2 층별 위치추정기 | `{층}F/maps/base_map.png`(0/255 이진), `door_mask.png` (X/floor_id/config/buildings.yaml:8, X/floor_id/localizer/node_local.py:78-86) | 벽·자유공간·거리장, 문 자리 |
| E2 | `{층}F/sem/*_mask.png`(방·복도·계단·화장실·문), `{층}F/mcl/{seed_allowed, no_go, building_footprint, elevator}.png` (X/floor_id/localizer/semantic_map.py:104-135) | 후보 허용 영역, 벌점, 건물 밖 |
| E3 | `runtime/e2_cache/light/kci_{층}F_{지도판}/` (L2/e2_worker.py:233-251) | 키프레임 채점용 경량 지도 |
| E4·E6·E8·명판 기준 후보 | `~/sci_ws/maps/kci_handoff/{층}F/{층}F_semantic_map.json` (L2/e34_node.py:50) | 방 번호, 방·복도 다각형, 축척 |
| E6·E8 글자 근거 | L2/map_text_rooms.json | 층별 방 글자(이 저장소에서는 사람 이름을 지움) |

**구조지도 → 내부 층**(X/floor_id/localizer/rt_localizer.py:93-115, X/floor_id/localizer/metric_fit_v2.py:30-42)
- **가짜 벽 제거**(`wall_clean.py:16-41`): 메워진 면·빗금처럼 두께가 벽 두께 중앙값의 2.5배를 넘는 덩어리를 지운다. 남은 것이 `wall_real` 이다.
- **벽 거리장**: F = EDT(¬wall_real) + max(벽 속 깊이 − h, 0). h 는 벽 내부 깊이의 90 백분위다.
- **통행 가능 영역**: ok = 자유 ∩ 건물 안 ∩ ¬계단.
- **벌점 영역**: pen = ok ∩ (no_go ∪ 화장실).
- **관통 판정용 벽**: wall_tr = wall_real ∧ ¬문.
- **건물 밖 거리장**: out_d.

**경량 지도(light map)**: F, out_d 를 1/4 화소 단위 uint16 으로, `wall_tr`·`ok`·`wall_real` 을 bool 로 저장한다. 층·지도 판마다 한 번 만든다. E3 는 이것을 메모리 매핑으로 열고 최근접 화소로 읽는다(L2/e3_tracker.py:87-120). E2 는 같은 F 를 쌍선형 보간으로 읽으므로, 같은 자세라도 값이 조금 다르다.

### 4.2 의미지도 JSON

**최상위 키**: `map_meta`, `layers`, `spaces`, `corridors`, `text_landmarks`, `semantic_objects` 등.

**`map_meta`**
- `image_size` 는 5층에서 3694 × 1074 다.
- `resolution_m_per_px` 는 5층에서 0.0375 다.
- 화소 좌표계는 왼쪽 위 원점, x 오른쪽, y 아래다. 지도 좌표계는 왼쪽 아래 원점, y 위다.
- `scale_handshake` 는 복도 폭(LiDAR 3.3 m ↔ 88 px)으로 축척을 맞춘 기록이다.

**`spaces`**
- 방마다 다음 필드가 있다: `id`(예 `room_502`), `type`(room/stair/restroom/elevator 등), `room_number`, `name`, `polygon_px`(5층은 모두 4꼭짓점), `bbox_px`, `center_px`, `polygon_map_m`.
- `name` 에는 사람 이름이 들어 있다. 그래서 이 저장소에는 지도를 싣지 않았다.

**`corridors`**: 복도 다각형이다. 5층은 `corridor_main_5f` 하나다.

**층 제외 규칙**: 의미지도 `image_size` 가 E2 지도 크기와 다르면 그 층은 E4 에서 뺀다(L2/e4_ambiguity.py:268-292). 9개 층 모두 크기가 일치한다.

### 4.3 좌표 약속

- **저장소 좌표(odom_link)**: FAST-LIO 자세를 (−x, y, −yaw) 로 바꾼 것이다(L2/e2_feed.py:45-49). 구조점은 몸체 좌표다.
- **후보 변환**(L2/e2_core.py:175-180 = L2/e3_tracker.py:76-80)
  - 화소 위치: p_px = pos + ppm·A(φ)(xy − anchor). 여기서 A(φ) = [[cos, −sin], [−sin, −cos]] 이고, 화소 y 뒤집기가 들어 있다.
  - 지도 위 방향: φ + yaw + β.
  - 점: p_px + ppm·((pts ⊙ [1, mirror]) A(hd)ᵀ).
  - β 는 센서와 몸체 사이 요 오프셋, mirror 는 센서 y 반전이다.
- **축척(px/m)**
  - E2 는 의미지도의 해상도를 읽지 않는다. 모든 층에 축척 씨앗 {15, 19.5, 24, 28.5, 33, 38, 43} 을 쓰고, 국소 정합이 ×0.8~1.25 안에서 연속으로 맞춘다(X/floor_id/localizer/run_floor_id.py:20, L2/e2_core.py:93-94).
  - 따라서 후보의 `x_m`, `y_m` 은 맞춘 축척으로 낸 값이다.
  - `semantic_scale_v1.json`(2층 24.83, 3층 28.82, 4층 23.96, 5층 31.40 px/m, 불확실도 5 %)은 E8·명판 기준 후보·`auto_fsm` 의 사전값으로만 쓴다.

## 5. E2 전 층 거친 탐색, E3 추적과 순위

### 5.1 E2 입력

- **표본 창**: 3장의 표본 저장소(1.5 s 간격)에서 t_now − k·dt 격자에 가장 가까운 과거 표본을 고른다(X/realtime/scan_timing.py:6-31).
- **점**: 장마다 가까운 구조점 + 먼 점을 쓴다. 정밀 단계는 400점, 거친 단계는 100점을 뽑는다.
- **벽 통과 검사용 경로**: 창 안 자세를 0.2 s 간격으로 솎은 것이다.

### 5.2 탐색 공간

| 변수 | 정하는 법 |
|---|---|
| 층 | 1~9층을 모두, 고정 순서로 본다(L2/e2_worker.py:366) |
| β·mirror | 창마다 `estimate_beta` 로 추정한다(X/floor_id/localizer/sensor_frame.py:26-93). 3 s 떨어진 표본 쌍(이동 ≥ 0.3 m)의 점군을 β 0~358°(2° 간격) × mirror ±1 로 겹쳐 최근접 거리 중앙값이 가장 작은 것을 고른다. 쓸 쌍이 없으면 `CalibrationUnavailable` 이다. 한 회차의 모든 층이 같은 값을 쓴다. REC 는 88개 후보 모두 β 228°, mirror +1 |
| φ 씨앗 | 창 점들의 선 방향 히스토그램과 지도 방향 히스토그램의 순환 상관 최대 φ0 → φ0 + {0, 90, 180, 270}° (X/floor_id/localizer/rt_localizer.py:148-184) |
| 위치 씨앗 | 축척마다 격자(22 px) 신념 상위 칸 (L2/e2_core.py:85-90). 신념은 프레임별 벽 거리 점수 + 장면 사전(교차로·막다른 곳·복도) − 비자유 벌점 − 열린 공간 벌점 − 금지 구역 벌점이다 (X/floor_id/localizer/rt_localizer.py:241-294) |
| 국소 정합 | 변수 (px, py, φ, log ppm), 잔차 min(F/ppm, 5 m), soft_l1, 경계 ±33 px·±8°·×[0.8, 1.25] (L2/e2_core.py:64-96) |

| | 정밀 | 거친·되풀이 | 지역 |
|---|---|---|---|
| 창 | 60 s | 20 s | 60 s |
| 프레임 간격 | 6 s | 6 s | 6 s |
| 점/프레임 | 400 | 100 | 400 |
| 축척 씨앗 | 7 | 4 (15, 24, 33, 43) | 7 |
| (φ·축척)당 위치 | 3 | 1 | 6 |

### 5.3 E2 점수

후보마다 아래 항목을 계산한다(L2/e2_core.py:100-138, X/floor_id/localizer/metric_fit_negev.py:24-64). 모두 낮을수록 좋고, 확률이 아니다.

| 항목 | 정의 |
|---|---|
| `wall_cost_m` | 점–벽 거리 하위 70 % 평균 |
| `T2` | 0.6 m 넘게 떨어진 점의 비율 |
| `T3` | 건물 밖 0.5 m 넘는 점의 비율 |
| `X_cross_m` | 경로가 벽 위를 지난 길이 |
| `blocked` | 통행 불가 위치 비율 |
| `nogo` | 금지 구역 벌점 |
| `T4` | 프레임 위치에서 180 방향 광선을 쏴 0.3~18 m 안 첫 벽을 찾았을 때, 그 근처에 관측점이 없는 '안 보인 벽'의 비율. 상위 12개만 계산 |

- part = wall_cost + 0.5·blocked + 0.3·nogo + 1.5·T2 + 3.0·T3 + 1.0·min(X, 0.3)
- **merit = part + 0.6·T4**
- **twoway**(창 끝 표본 하나): 36방향마다 관측 거리와 지도의 첫 벽 거리를 비교한 값이다. 0 이 가장 좋다.
- **wall_pass**: 층별 상위 4개 후보만 계산한다. 창 경로가 벽을 뚫고 반대편으로 나간 횟수다.

**선별**
- 경로가 벽을 0.3 m 넘게 지난 후보는 버린다. 모두 넘으면 최소인 것만 남긴다.
- 같은 자세(1 m, 10°, 축척비 1.1 안)는 하나로 합친다.
- 층당 최대 12개를 남긴다.
- 건물 전체 순위는 wall_pass > 0 을 뒤로 보낸 뒤 twoway, 그다음 merit 순이다.

### 5.4 E2 회차와 자원

- **첫 회차**: 저장소 길이 ≥ 21.5 s 이고 창 안 경로 ≥ 2 m 일 때 시작한다. 20 s 창 거친 단계다.
- **`e2_acc.on`**: 이 파일이 있으면 정밀 단계를 하지 않는다. 콘솔이 준 `--refresh-s 90` 대신 20 s 창 되풀이를 쓴다(L2/e2_worker.py:1173, 1216-1219).
- **되풀이 간격**: rs = 20 × 4 / budget 이다. 기준은 직전 회차의 시작 시각이다.
  - `e2_budget.json` 의 `max_busy` 가 동시 작업 수를 정한다(L2/auto_fsm.py:195-200). 0 = VLM 상주 중, 1 = 쌍 E7 검증 중, 2 = 판독 중, 3 = 모호, 4 = 그 밖.
  - 현장 기록 79건에서 9개 층 되풀이 한 번은 중앙 57 s 걸렸고, 회차 시작 간격은 중앙 72 s 였다.
- **실행 환경**: 자식 프로세스 4개, CPU 전용이다(wallfit 를 CPU 판으로 바꿔 끼움, 결과는 같음).
  - 일시정지, 자세 주기 저하, 메모리 < 3500 MB, 온도 ≥ 76 °C 이면 기다린다.
  - 메모리 < 2800 MB 나 온도 ≥ 78 °C 이면 자식을 중단한다.
- **지역 탐색**(L2/e2_worker.py:783-926, L2/e2_region_ledger.py)
  - E8·E6 가 요청한 원 안에서만 정밀 맞춤을 한다. 반경은 방 크기 + (명판 뒤 걸은 거리)·px/m 이다.
  - 요청 원장은 대기 → 실행 → 완료/재시도 상태를 가진다.
- **출력**
  - `state/e2/e2_state.json`: `candidates_coarse` 상위 24, `candidates_refresh` 층·회차당 3, `per_floor`, `ranking` 등.
  - `state/e2/e2_passes.jsonl`: 회차·층 단위 사건.
- **알려진 주의점**
  - 회차 끝에서 `candidates` 를 그 회차 결과로 통째로 바꾼다(L2/e2_worker.py:743-744). 정밀 단계가 꺼진 지금은 다른 층 E2 경쟁 목록이 비게 된다.
  - 되풀이는 정지 중에도 돌아 `CalibrationUnavailable` 로 끝난다.

### 5.5 E3 흡수와 키프레임

- **주기**: `e34_node` 가 2 Hz 로 돈다(L2/e34_node.py:51). 순서는 세션 확인 → 키프레임 → 흡수 → 명판 기준 후보 흡수 → 평가 → 모든 가설 자세 전파다(L2/e3_tracker.py:615-644).
- **흡수**(:432-505)
  - E2 의 거친·되풀이·정밀 후보를 한 번씩 받는다.
  - 같은 층이고 1 m·10° 안에 기존 가설이 있으면 합친다.
  - ID 는 `H{번호}_F{층}` 이다.
  - 최대 120개다. 넘으면 보류된 나쁜 것부터 버린다.
- **키프레임**: 표본(1.5 s)마다, 직전 키프레임에서 **0.5 m 이상 움직이거나 15° 이상 돌았을 때** 만든다(:42, 371-380). REC 는 315 s 동안 41개다. **서 있으면 키프레임이 없어 순위가 바뀌지 않는다.**
- **평가 대상**: 키프레임당 40개 가설을 평가한다(:687-714). 순서는 명판 일치 가설(`LIVE2_E3_SEM_FIRST`) → 근거 부족 ACTIVE → ACTIVE 점수 순 → 나머지 순환이다.

### 5.6 E3 채점과 추적 보정

- **키프레임 점수**(:508-553)
  - frame_merit = cost + 1.5·T2 + 3.0·T3 + 1.0·min(X, 0.3) + 0.5·[통행 불가]. X 는 직전 키프레임부터의 경로가 벽 위를 지난 길이다.
  - 두 번째 평가마다 최근 30 m 경로를 현재 변환으로 올려 본다. 벽을 넘으면 X 항을 0.3 으로 채우고 hard 로 표시한다.
- **누적 점수**: S = (3·e2_merit + Σ 최근 20 frame_merit) / (3 + n) (:43, 250-252).
- **추적 보정**(:555-613)
  - 최근 3 키프레임의 비용 합을 줄이는 작은 보정을 찾는다. 위치 ±0.2 m(0.1 m 간격), 회전 ±1°(0.5° 간격), 축척 {0.99, 1, 1.01} 이다.
  - 항등보다 엄격히 나을 때만 적용한다.
  - 축척 누적은 E2 값 대비 ±5 % 로 묶는다. β·mirror 는 바꾸지 않는다.
  - 기록은 `state/e3_track.jsonl` 이다.
- **상태**(:767-787)

| 전환 | 조건 |
|---|---|
| ACTIVE → SUSPENDED | (n_kf ≥ 3 이고 S − best > 0.25 가 3번 연속 = `GAP>0.25 x3`) 또는 최근 5개 중 hard ≥ 2 (`HARD_VIOLATIONS>=2/5`) |
| SUSPENDED → ACTIVE | gap < 0.12 이고 위반 < 2 가 2번 연속 |
| DROPPED | 상한 초과나 세션 변경만 |

보류된 가설도 살아 있는 가설이다. E8 쌍 판정은 보류된 것까지 본다.

### 5.7 E3 순위

- **정렬 키**: `rank_key` = (최근 30 m 벽 통과, 근거 묶음[n_kf ≥ 3], 의미 순서, S) 다(:257-264).
- **의미 순서**: `SEMANTIC_CONSISTENT` 0, `TEXT_FLOOR_SUPPORT` 0.5, 판독 없음 1, `PLACE_MISMATCH` 2, `FLOOR_MISMATCH` 3 이다. 쌍 판정이 붙이는 `PAIR_INCONSISTENT` 는 표에 없어 1(판독 없음)과 같다.
  - 의미 판정은 확실한 판독(라벨 확률 ≥ 0.6)이 들어올 때만 바뀐다.
- **leader**: ACTIVE 중 `rank_key` 1위다. 상태 파일의 정렬 목록 1위는 보류된 가설일 수 있다.
- **출력**
  - `state/e3_state.json`: 가설마다 `global_pose{floor, px, py, heading_deg, odom_t}`, `ppm`, `phi`, `beta_deg`, `score`, `status` 등.
  - `state/e3_history.jsonl`: 상위 6개.
  - `state/e3_events.jsonl`: 흡수·합침·보류·회복 사건.
- **E7·E8 과의 연결**: E7·E8 은 같은 프로세스 안의 추적기 객체를 직접 쓴다. 후보 변환(`rec`)은 E2 기록에 추적 보정을 반영한 pos·anchor·φ·ppm 이고, β·mirror 는 E2 값 그대로다.

## 6. E4 모호성, E5 명판 탐지, E6 촬영·판독

이 장에는 경로 약어가 하나 더 있다. PS = `code/home_extra/sci_ws_student_staging_20261109/tools/plate_site`(명판 촬영 도구)이다.

### 6.1 E4 — 구조만으로 층이 갈리는가

E4 는 E2·E3 가설을 보고 상태 하나를 고른다. 아래 표의 위에서부터 처음 맞는 규칙이 적용된다(L2/e4_ambiguity.py:53-177). 점수 단위는 E3 비용(벽 불일치, m)이다.

| 상태 | 조건 |
|---|---|
| `RECOVERY_REQUIRED` | odom 세션 변경 30 s 안, odom 나이 > 2 s, E2 표본 나이 > 6 s, 전 층 실패, 탐색 뒤에도 가설 없음 중 하나 |
| `GEOMETRY_INCONSISTENT` | E3 가 30 s 안에 재탐색을 요청함 |
| `INITIALIZING` / `SEARCH_INCOMPLETE` | 세션 없음·첫 층 진행 중 / 첫 전 층 탐색 미완료 |
| `AMBIGUOUS` | 1위와 0.05 m 안에 다른 층의 구별되는 경쟁자가 있음(2 m 또는 30° 넘게 떨어진 것). 또는 0.05 m 안에 2개 층 이상 |
| `CONFIDENT_GEOMETRY` | 아래 조건을 모두 만족 |
| `DOMINANT_UNVERIFIED` | 그 밖 |

`CONFIDENT_GEOMETRY` 의 조건은 다음 여섯 가지다.
- 경쟁과의 차 ≥ 0.15 m
- 키프레임 ≥ 8
- 경로 ≥ 8 m
- 최근 비용 ≤ 0.35
- 1위 유지 ≥ 15 키프레임
- 위반 0

**E4 가 실제로 정하는 것.**
1. **SigLIP 추론의 켜고 끄기.**
   - 모호하거나 회복이 필요하면 `siglip_request.json` 이 ACTIVE 가 된다. 이유가 사라져도 15 s 유지한다(L2/e4_ambiguity.py:204, 233-252).
   - 관측 쪽은 요청이 5 s 안으로 새롭고 ACTIVE 일 때만 추론한다(L2/siglip_gate.py:14-48).
   - E8 이 판독을 기다리는 동안과 자동 상태기계가 모호할 때는 ACTIVE 로, 확정 뒤에는 꺼짐(`IDLE_OK`)으로 덮어쓴다(L2/e34_node.py:172-179).
2. **E5 영역 예산의 순서.** 예산이 모자랄 때만 E4 목표 방향 ±15° 영역을 먼저 본다(L2/e5_e4_link.py:100-191). REC 에서는 예산이 모자란 장면이 없었다.
3. **E2 재탐색 요청.** 요청 간격은 120 s 다(L2/e34_node.py:214-229).
4. **정지 요청을 막는 것은 E4 가 아니다.** 자동 상태기계의 `stop_gate` 가 E3 가설을 직접 보고 정한다(L2/auto_fsm.py:218-229).

**REC 의 E4 상태 변화.** `INITIALIZING`(41 s) → `SEARCH_INCOMPLETE`(88 s) → `AMBIGUOUS`(105 s, 3층과 8층의 차 0.038 m)이고, 끝까지 `AMBIGUOUS` 였다. `CONFIDENT_GEOMETRY` 는 한 번도 없었다.

### 6.2 E5 — 앞 카메라로 명판 찾기

**입력과 주기.** hub 가 RealSense 색 영상(640×480)과 정렬 깊이, FAST-LIO 자세를 공유 메모리에 둔다. 관측 노드는 1 s 에 한 장면을 처리한다. 85 °C 이상이거나 메모리 1500 MB 미만이면 2 s 에 한 장면으로 늦춘다(SRC/sci_visual_observation/e5_node.py:628-639). REC 는 268장면, 첫 장면은 58.2 s 다.

**영역 후보**(SRC/sci_visual_observation/frontend.py:158-204)
- 작은 후보: MSER(원본·반전), top/black-hat, 채도 대비로 뽑는다. 크기는 가로 4–50 px, 세로 6–90 px 다(SRC/sci_legacy_e5/small_proposals.py:1-31).
- 벽 영역 후보: 깊이 7 m 이하이고 바닥이 아닌 영역 중 두드러짐 상위 8개다.
- 1.2 s 창으로 추적해 누적한다.

**SigLIP 유사도**(SRC/sci_legacy_e5/region_siglip_rerank.py:47-150)
- 모델은 `siglip-base-patch16-224` 를 추가 학습 없이 쓴다. 문구 묶음은 "a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall" 이다(REC `archive/observe_run/siglip_prompts.json`).
- 영역을 100 px 이하 조각으로 나눈다. fp16 영상 특징과 미리 계산한 문구 특징의 코사인을 문구 중 최댓값으로 잡고, 다시 조각 중 최댓값을 그 영역의 유사도로 쓴다.
- 반대 문구(포스터·게시판·반사·조명·엘리베이터·층 안내)를 함께 보고, 그쪽이 더 비슷하면 거른다.

**문턱**(`legacy_p90`): 이번 실행에서 본 모든 영역 유사도의 누적 90 백분위다. 50개가 모이기 전까지는 통과가 없다(SRC/sci_legacy_e5/live_event_runner.py:531). 같은 벽 자리의 평균 유사도는 최근 3회 관측으로 낸다. 0.7 m 넘게 이동하기 전의 관측은 평균에서 뺀다(L1/observe_node.py:200-205).

**검증 조건(P2 VERIFIED)**(SRC/sci_visual_observation/timing.py:145-214, policy.py:78-158; 설정은 CFG/e5_policy.json)

| 항목 | 조건 |
|---|---|
| 지속성 | 최근 3장면 중 서로 다른 2장면 이상 |
| 유사도 | 평균 ≥ 문턱, 단일 ≥ 0.078 |
| 모양 | 폭 ≤ 0.85 m, 높이 0.6–2.8 m |
| 반대 문구 | 반대 문구가 더 높지 않음 |
| 화질 | 포화·반사 ≤ 0.3 |
| 깊이 | 깊이 ≤ 5 m, 화면 가장자리에서 6 px 이상, 경사 ≤ 80° |

**정지 요청**(L2/visual_stop.py)
- FAST-LIO 좌표 약속이 없어 E5 자체의 정지는 미뤄진다. 대신 `visual_stop` 이 운용자에게 정지를 요청한다.
- 영역을 유사도 높은 순으로 보며 아래를 모두 통과해야 요청이 나간다.
  - 깊이–색 시차 ≤ 0.5 s
  - P2 VERIFIED, 유사도 ≥ 0.09, 거리 ≤ 5 m
  - 냉각 시간: 전역 8 s, 같은 자리 30 s
  - 자동 상태기계가 구조로 충분하다고 보지 않음
  - 유사도 ≥ 0.10 또는 평균 ≥ 0.11
  - 상자 윗변 ≥ 1.30 m
  - 이미 찍은 명판에서 1.5 m 밖
  - 최단 정지(1.2 m) 뒤에도 팬틸트로 조준 가능
- 이미 서 있으면 요청하지 않는다(:117-125). 같은 명판이 1 m·60 s 안에 다시 잡혀도 요청하지 않는다(:126-134).
- 요청이 나가면 운영 화면에서 880 Hz 소리와 진동이 난다(L2/web_v2/index.html:703-712).

**검출 자세.** 영상 시각에 FAST-LIO 원래 자세를 선형 보간한다. 두 표본 간격은 0.25 s 이하여야 하고, 없으면 가장 가까운 것(0.3 s 이하)을 쓴다(L2/det_pose.py:3-23). 명판의 카메라 기준 3-D 위치는 상자 안쪽 50 % 깊이의 중앙값으로 역투영한다(SRC/sci_legacy_e5/region_memory.py:39-75).

**기록**(관측 실행 폴더): `live2_detection_pose.jsonl`, `live2_stop_requests.jsonl`, `live2_stop_suppressed.jsonl`, `scenes.jsonl`. REC 는 정지 요청 6건이었다. 억제는 점수 부족 10, 이미 정지 4, 높이 부족 3, 조준 불가 2건이다.

### 6.3 E6 — 걸어가서 팬틸트로 찍기

`runtime/plate_case1.on` 이 있고 팬틸트 동의가 있으면, 자동 촬영 경로 대신 **명판 촬영 도구**(PS/plate_site.py, 8791)가 팬틸트를 쓴다(L2/console.py:426-435). 운용자는 운영 화면의 촬영 카드에서 다음 순서로 진행한다.

1. **명판 기억**(PS/plate_site.py:278-385). 세 가지 방법이 있다.
   - E5 정지 요청(20 s 안)을 받는다.
   - 화면에서 SigLIP 상자를 누른다.
   - 검증된 영역이 두 장면 연속 같은 자리면 자동으로 기억한다.
   - 위치는 상자 중심과 깊이로 몸체 기준 점을 만든 뒤 평면 좌표로 기억한다(L2/e6_core.py:23-51). 세로로 긴 상자는 높이 1.72 m 를 상자 안으로 맞춘다(L2/e6_tracker.py:28-41).
2. **안내**(:387-436). 명판을 1.0 m 앞에 두는 지점과 팬틸트가 닿는 최단 거리 + 0.10 m 중 먼 쪽을 정지 지점으로 한다. 남은 거리 1.0 m·0.5 m, 정지 지점, 지나침에 각각 다른 소리가 난다(PS/page.html:104-109).
3. **조준**(PS/aimgeo.py:48-99).
   - 현재 FAST-LIO 자세로 몸체 기준 명판 위치를 구하고, 보정 파일(`pantilt_calib_multi_20260928` trim1)로 서보 각을 낸다. 범위는 팬 20–180°, 틸트 100–180° 다.
   - 보내기 직전에 로봇이 서 있는지 다시 확인한다. 정지 기준은 최근 1.5 s 속도 < 0.05 m/s, 회전 < 3°/s 다(L2/e6_core.py:65-83).
4. **촬영**: IMX477 1920×1080. 화살표로 미세 조정해 다시 찍을 수 있다. 선명도(라플라시안 분산)는 보여 주기만 하고 막지는 않는다.
5. **[이 사진으로 위치추정]**(:1120-1206)
   - 사진을 `state/e6_captures/plate_<실험>_P00n.jpg` 로, 기록을 `state/e6_precision_results.jsonl` 에 한 줄 쓴다.
   - 명판 위치 `target_odom_link_xy` 는 표본 저장소 좌표(자세 보정 반영)로 옮긴 값이다. 최근 저장소 줄과 그 시각의 원래 자세를 강체로 맞춘다. 줄 나이 6 s 이하, 차이 5 m·20° 이하일 때만 옮긴다(:131-193).
   - `stopped_wall` 은 그 사진의 조준을 시작한 시각이다.

### 6.4 E6 — 판독

판독은 `e6e7_link` 가 `e6_precision_results.jsonl` 을 따라 읽어 쌍 판정 관리자에 넘기면서 시작된다(L2/e6e7_link.py:229-249). 판독 스레드는 코어 6, 7 에서 돈다.

**OCR**(L2/e6_ocr.py:107-191)
- RapidOCR(PP-OCRv4, CPU) 하위 프로세스를 쓴다. 모델은 한 번만 올린다.
- 명판 영역(원해상도)과 전체 사진을 함께 읽고, 글자별 최고 신뢰도를 고른다.
- 쓸 만한 글자(3자리 이상 숫자로 시작하거나 화장실 낱말, 신뢰도 ≥ 0.85)가 없으면 영역을 35 % 넓혀 한 번 더 읽는다.

**VLM**(L2/e8_pair.py:671-740, L2/vlm_ondemand.py, SRC/sci_text_reading/vlm.py:27-100)
- 같은 사진을 따로 묻는 교차 확인이다. 팬틸트 동의에서 VLM 을 허용했을 때만 돈다(`SCI_E6_VLM=1`).
- 입력: 운용자 영역이 있으면 1.5배 넓힌 영역, 없으면 전체 사진이다. 긴 변 768, temperature 0, 최대 96 토큰이다.
- 질문은 명판 종류(방 번호판·연구실 이름판·층 안내·엘리베이터·포스터·기타)와 번호·글자를 JSON 으로 답하게 한다.
- 거부되면 5 s 간격으로 60회까지 다시 시도한다.

**합치기**(SRC/sci_text_reading/semantic.py:112-181)

| 경우 | 결과 |
|---|---|
| OCR·VLM 번호 일치 | AGREE, p = max(0.9, 신뢰도) |
| OCR 만 | p = 0.8 × 신뢰도. 신뢰도 ≥ 0.85 일 때만 |
| 둘이 다름 | CONFLICT, 0.5 / 0.5 |
| VLM 만 | 라벨 없음 |

- 번호 정규식은 `^(?:[A-Z]{1,2}-?)?B?\d{3,4}(?:-\d{1,2})?(?:호|호실)?$` 다. 포스터·층 안내 낱말이나 날짜가 보이면 위치 표지가 아닌 것으로 본다.
- **확실한 판독**: 합친 라벨 확률이 0.6 이상이다(`SEM_DECISIVE_P`, L2/e3_tracker.py:60).
- **화질 경고가 붙은 사진**: OCR 신뢰도 ≥ 0.9 일 때만 쓴다(L2/e8_pair.py:261-263, 522).
- 번호를 지도 글자 중에서 골라 맞추는 퍼지 선택은 운영 코드에 없다. 번호가 의미지도의 방 번호와 정확히 같을 때만 그 층의 근거가 된다.
- 번호 없는 글자는 지도 글자 3-gram 으로 층을 뒷받침하는 데만 쓴다(B7, L2/map_text.py:112-175). 순위에만 영향을 준다.

**층 가능도**(L2/e6_semantic.py:93-103): P(f) ∝ Σ 대안 p × (그 층에 그 라벨 공간이 있으면 1, 없으면 0.02) 를 9개 층에 대해 정규화한다. 이 건물은 방 번호가 층 사이에 겹치지 않는다. 그래서 라벨 하나가 한 층에만 있으면 그 층의 가능도는 1/(1 + 8 × 0.02) = 0.862 다.

**기록**
- `state/e8_observations.jsonl` 에 사건마다 한 줄씩 쌓인다(`STORED`, `READ`, `VLM_READ`, `REJECTED`). 실험 뒤 요약·지표 계산이 읽는다.
- 실시간 판정은 메모리 안에서 쌍 판정(8장)으로 넘어가고, 결과가 `state/e8_pairs.jsonl` 에 남는다.

## 7. 명판 기준 후보 (E7 의 앞 단계)

명판 하나를 확실히 읽으면, 그 명판이 그 방 앞 복도 벽에 붙어 있다고 놓고 로봇의 변환 후보를 직접 만든다. E2 의 전 층 탐색을 기다리지 않고, 그 층 그 방 주변에서만 찾는 것이다.

### 7.1 언제 만드는가

- **조건**: 판독 하나가 `single_obs_region` 을 통과해야 한다(L2/e6e7_link.py:589-597).
  - 층 가능도가 `SINGLE_FLOOR_P` = 0.8 이상이다(L2/e8_pair.py:518, 907-909).
  - 그 층에 그 번호의 공간이 정확히 하나다(:913-919).
  - 표지판 위치(`target_odom_link_xy`)가 있다.
- **처리 방식**: 작업은 대기열 `_anchor_pending` 에 쌓이고, 스레드 하나가 하나씩 처리한다. 코어는 6, 7 이다(L2/e6e7_link.py:614-627). 스위치는 `LIVE2_ANCHOR`, 기본 1 이다.
- **사전값**(L2/semantic_anchor.py:164-177): 정답이 아니라 E2 결과에서 가져온 추정치다.
  - beta: 그 층 E2 후보들의 beta 의 원형 중앙값. 그 층 후보가 없으면 전 층 후보로 계산한다.
  - mirror: 최빈값.
  - 기준 축척 ppm0: 1/해상도(L2/e6e7_link.py:659-660). 5층은 `semantic_scale_v1.json` 의 덮어쓰기 값 31.4 px/m(불확실도 5 %)를 쓴다(L2/e8_pair.py:118-126).
  - E2 가 그 층을 아직 안 봤으면 10 s 마다 다시 보고, 300 s 가 지나면 건너뛴다(L2/e6e7_link.py:639-644).
- **LiDAR 창**: 끝은 min(최신 표본, 촬영 시각), 길이는 `WINDOW_S` = 240 s 다(L2/e6e7_link.py:645-652, L2/semantic_anchor.py:39).
  - 벽 모양 점 프레임이 8장 미만이면 건너뛴다. 프레임 형식은 E7 과 같다(8.4절).

### 7.2 벽선 점

`plate_wall_points`(L2/semantic_anchor.py:50-70)가 명판이 붙을 자리를 정한다.

1. 방 다각형의 변마다 중점에서 가장 가까운 복도 다각형 변까지의 거리를 잰다.
2. 가장 가까운 변을 고르고, 복도 쪽으로 그 간격의 절반만큼 옮긴다.
3. 그 선 위에 `WALL_STEP_M` = 0.5 m 간격으로 점을 찍는다(양 끝 포함). 502호는 8점이다.

복도 다각형이 없으면 방 경계 전체를 쓴다.

### 7.3 회전 탐색

비용은 모두 E7 비용(8.4절)이다(L2/semantic_anchor.py:73-125).

1. **거친 단계**: 벽선 가운데 점에 명판을 놓고 φ 를 0~357° 범위에서 3° 간격(120개)으로 훑는다.
2. **방향 고르기**: 비용 순으로 서로 8° 넘게 떨어진 방향을 고른다. 벽 검사가 없으면 5개, 있으면 8개다(`TOP_K` = `LIVE2_ANCHOR_TOPK` 기본 3, :34, 82-88).
3. **정밀 단계**: 고른 방향마다 φ₀ ± 3°(1° 간격) × 벽선 점 전부를 평가한다. 벽 검사가 있으면 비용 순 상위 12개까지만 검사하고, 처음 통과한 것을 고른다(:92-110).
4. **축척·beta**: φ 와 위치는 그대로 두고, 축척 {0.95, 1, 1.05} × beta {−6°, 0, +6°} 를 본다(:112-117).
5. **결과**: 비용 순 상위 3개를 낸다. 다음 순위와의 비용비(`separation_ratio`)를 붙인다(:119-125).

**벽 검사**(`LIVE2_ANCHOR_WALLCHECK` 기본 1, L2/semantic_anchor.py:145-161): 최근 30 m 경로가 벽을 뚫지 않아야 한다. 또 최근 90 s 경로가 벽 위를 지난 길이가 0.3 m 이하여야 한다.

### 7.4 경로 기준 후보

경로 기준 후보는 두 번째 명판이 들어왔을 때 만든다(`LIVE2_ANCHOR_PATH` 기본 1, L2/e6e7_link.py:143, 606-611, 646-694).

- **고르기**: 새 판독이 들어오면, 같은 층·다른 방·같은 FAST-LIO 세션의 판독 중 가장 최근 것을 고른다. 그 앞 명판의 표지판 위치를 고정점으로 삼는다.
- **회전**: 두 명판 사이 경로의 LiDAR 로 정한다. 창은 min(240, max(90, 촬영 − 앞 명판 정지 + 20)) s 다.
- **새 명판 검사**: 고정한 변환으로 새 명판을 올렸을 때 반폭 규칙(8.2.4)을 통과해야 한다.
- **기록**: `pass_id = "AP_<관측>"`, `evidence.kind = "PATH_ANCHORED"`.

**E3 편입.** 명판 기준 후보는 E3 가 매 틱 받아들인다(L2/e3_tracker.py:402-426). 1 m·10° 안에 기존 가설이 있으면 그것과 합치고, 없으면 `SEMANTIC_CONSISTENT` 로 넣는다. 기록 파일은 `state/e2/semantic_anchor.json` 이다.

## 8. E8 쌍 판정과 E7 정합, 층 확정

### 8.1 판정 대상과 순서

E8 쌍 판정은 독립된 명판 두 장 이상이 한 층의 한 자리로 설명되는지를 본다(`e8_pair.evaluate_group`, L2/e8_pair.py:318-490). 대상은 E3 가 들고 있는 살아 있는 가설 전부다. 보류(SUSPENDED)된 것도 들어간다(:743).

판독이 새로 들어오면 판독 스레드(코어 6, 7)가 판정을 돌린다. 판정이 READY 면 **같은 자리에서 곧바로 E7 을 요청한다**(:748-759, 781-795).

### 8.2 쌍 판정의 검사

**8.2.1 쓸 수 있는 관측**(:324-349). 아래에 하나라도 걸리면 그 관측은 빼고 본다.
- 화질 불합격.
- 확실하지 않음: 최대 가능도 < 0.6.
- 화질 경고를 무시하고 찍었는데 OCR 신뢰도가 0.9 미만.
- 표지판 위치 없음.
- 오독으로 빠진 판독.
- VLM 만 읽은 번호. 같은 사진의 학과명이 그 층과 맞을 때만 인정한다.

**8.2.2 증거 묶기**(:360-377).
- FAST-LIO 세션이 바뀌면 이전 증거를 버린다.
- 같은 번호이고 표지판 위치가 2 m 안이면 같은 명판이다.
- 증거가 2개 미만이면 `MORE_OBSERVATION_REQUIRED` 다.

**8.2.3 명판 사이 검사**(가설과 무관, `_pairs`, :271-315). 연속한 두 명판마다 다음을 본다.
- **주행거리 조건**(`odom_between`, :229-253): 같은 세션이어야 한다. 두 정지 사이 `lio_series` 표본 중 하나라도 8 Hz 미만이면 쓰지 않는다.
- **주행거리 오차**: 0.5 + 0.05·max(√Σ직선구간², 순변위) m 로 둔다(:221-226).
- **허용 폭**: tol = 2·3 m + 주행거리 오차 + 축척 불확실도 × dmax 다(:292-293).
- **잔차**: dmin, dmax 는 두 방 다각형 사이 최소·최대 거리, sep 은 표지판 위치 간격이다. 잔차 = max(0, dmin − sep − tol, sep − dmax − tol) 이다.
- **설명하는 층**: 잔차가 0 이고, 복도 연결(방–복도 2 m, 복도–복도 1 m)이 끊기지 않은 층이다(:45-46, 164-176, 312). 모든 쌍을 설명하는 층의 교집합을 `floors_ok` 라 한다.

**8.2.4 가설마다 명판 위치 검사 — 복도 반폭 규칙**(`LIVE2_PLATE_HALF_CORRIDOR` 기본 1, L2/e6_semantic.py:43, 125-194)
- 표지판 위치를 가설 변환으로 지도에 올린 점을 q 라 한다.
- 그 방의 복도 쪽 벽선(7.2절의 선)과 q 사이에서 두 거리를 잰다.
  - 수직 거리 `perp` 가 반폭 이하여야 한다. 반폭은 q 에 가장 가까운 복도 다각형의 면적 ÷ 둘레다.
  - 벽선 선분까지의 거리 `seg` = √(perp² + 벽선 밖으로 나간 길이²) 가 3 m 이하여야 한다.
- 복도 다각형이 없거나 시설 표지(`cat:`)이면 예전 규칙(방 다각형에서 3 m)을 쓴다.
- 이 규칙의 근거는 지도에서 나온다. 명판이 복도 가운데를 넘어가면 맞은편 방의 명판으로 봐야 하기 때문이다.

**8.2.5 한 장 빼 보기**(B6, `LIVE2_E8_DROP_MISREAD` 기본 1, :393-413): `floors_ok` 가 비었고 명판이 3장 이상일 때만 돈다. 한 장씩 빼 보고, 설명하는 층이 생기는 경우가 정확히 하나일 때만 그 장을 오독으로 둔다.

**8.2.6 후보와 경로 규칙**(:414-439)
- 후보는 모든 증거 명판이 `SEMANTIC_CONSISTENT` 이고 층이 `floors_ok` 에 있는 가설이다.
- 경로 규칙(`LIVE2_E8_PATH_RULE`, `_PATH_RANK` 기본 1)은 두 가지를 본다. 첫·마지막 명판 정지 사이 경로가 벽을 뚫는지, 최근 30 m 경로가 벽을 뚫는지다(`path_feasible`, :66-82).
- 뚫는 후보는 지우지 않고 맨 뒤로 보낸다.

**8.2.7 판정**(:445-489)

| 판정 | 조건 | 다음 동작 |
|---|---|---|
| `REGION_CANDIDATE_READY` | 경로를 통과한 후보가 모두 한 층에 있다. 통과 후보가 없으면 후보 전체가 한 층에 있다 | 후보에 `SEMANTIC_CONSISTENT`, 나머지에 `PAIR_INCONSISTENT` 를 E3 순위에 적용 → E7 요청 |
| `MORE_OBSERVATION_REQUIRED` | 사유가 `AMBIGUOUS:`(여러 층), `INCONSISTENT:`(어느 층도 설명 못 함), 관측 부족 등 | 다음 명판을 기다림 |
| `CANDIDATE_MISSING` | 번호는 지도와 맞는데 두 명판을 함께 맞추는 가설이 없다 | 명판 주변 지역 E2 를 요청. 새 후보가 E3 에 들어오면 같은 묶음을 다시 판정 |

**예산**(:84-86, 768-779): 유효 독립 명판 4장, 촬영 시도 8회, 묶음 시간 900 s 다. 하나라도 넘으면 묶음을 `NO_FIX` 로 닫는다.

### 8.3 E7 후보 고르기

`request_e7`(L2/e6e7_link.py:1126-1147)이 E7 에 넘길 후보를 고른다.
- 명판과 맞는 후보를 정렬한다. 경로를 통과한 것이 먼저이고, 그다음 E3 순위다.
- 최대 `LIVE2_E7_PLATE_CANDS` = 4개를 넘긴다.
- 경쟁 1개를 더한다. 명판 후보에 들지 않은 살아 있는 가설 중 E3 순위 1위다. 같은 층일 수도 있다(`LIVE2_E7_FAST` 기본 1).
- 이미 확정됐으면 아무것도 하지 않는다.

### 8.4 E7 정합의 계산

E7 계산은 별도 프로세스 `e7_proc.py` 에서 돈다(코어 0,1,6,7, nice 5, 제한 300 s; L2/e7_proc.py:26-131). 프로세스가 실패하면 스레드에서 계산한다.

**프레임**(`frames`, L2/e7_rematch.py:106-127)
- 최근 90 s 표본 중 최대 60장을 고르게 뽑는다.
- 장마다 벽 모양 점만 남긴다. 반경 0.30 m 안에 이웃이 5점 이상이고, 주성분 선형도 (λ₁−λ₂)/λ₁ 이 0.85 이상인 점이다(:43-70).
- 20점 미만인 장은 버리고, 장당 150점으로 줄인다.
- 창의 끝은 E7 이 시작될 때의 최신 표본이다.

**투영**(`project`, :130-146): 지도는 y축이 반전돼 있어 A(θ) = [[cos θ, −sin θ], [−sin θ, −cos θ]] 를 쓴다.
- 후보 변환을 최신 프레임에서 다시 놓는다: pₙ = pos + ppm·A(φ)(xₙ − anchor).
- 보정 d = (dx, dy, dφ, s) 를 넣는다. 각 점은 다음 위치가 된다.

  base + s·ppm·A(φ + dφ)(x_t − xₙ) + s·ppm·A(φ + dφ + yaw_t + β)·(p ⊙ [1, mirror])

  여기서 base = pₙ + ppm·[dx, −dy] 다.

**비용**(`cost`, :323-339)
- 점마다 오차 = min(벽 거리장(점)/ppm, 5 m) 이다. 벽 거리장은 1/4 화소 단위이고 쌍선형 보간으로 읽는다.
- 비용은 작은 쪽 70 % 의 평균이다.

**정합 단계**(`register`, :342-369)

| 단계 | 내용 |
|---|---|
| 격자 | dx, dy ∈ {−1, −0.5, 0, 0.5, 1} m × dφ ∈ {−4, −2, 0, 2, 4}° × s ∈ {0.98, 1, 1.02}, 375개 |
| 국소 모호성 | 격자 상위 12개 중 최소 비용의 1.05배 이내이면서 1 m 또는 5° 넘게 떨어진 점이 있으면 모호하다 |
| 최소제곱 | `least_squares`(soft_l1, f_scale 0.2, 최대 40회). 범위는 ±1.5 m, ±6°, s ∈ [0.97, 1.03] |
| 수렴 | 최적화가 성공하고 범위 끝에 닿지 않았다 |
| 그 밖의 출력 | 잔차 중앙값, 0.15 m 안 점 비율, 구간 3개 비용, 빛줄기 기준 벽 통과 |

**이어 맞추기**(L2/e7_proc.py:38-49): 명판과 맞는 후보만 대상이다. 결과가 범위 끝에 닿으면, 그 보정을 반영한 변환을 새 시작점으로 다시 맞춘다. 최대 2회다. 경쟁 후보는 이어 맞추지 않는다.

**E7 자체 판정**(:399-440): 기하만으로 내리는 판정이다.
- `ok_geom` 의 조건: 1위가 수렴했고, 국소 모호성이 없고, 경쟁과의 차가 0.05 m 이상이고, 벽 통과 거부가 없다.
- 이것만으로 확정하려면 의미 조건(`_semantic_fix`, L2/e6e7_link.py:1401-1432)까지 모두 통과해야 한다.
  - 서로 다른 방 번호 2개 + 번호 외 층 근거.
  - 겹치지 않는 창 2개가 1.5 m·6° 안에서 일치.
  - 자료 나이 60 s 이하.
- 층마다 구조가 비슷한 이 건물에서는 이 경로로 확정되기 어렵다. 그래서 다음의 층 점수 결합이 주 확정 경로다.

### 8.5 층 확정 — FUSED_FLOOR

`fused_floor.py`(`LIVE2_FUSED_FLOOR` 기본 1)가 E7 결과와 명판을 한 점수로 묶는다. E7 자체 판정이 확정이 아닐 때 이 단계가 적용된다(L2/fused_floor.py:226-227, L2/e6e7_link.py:1251-1262).

**상수**(:47-49)

| 이름 | 값 |
|---|---|
| σ_g | 0.05 m |
| σ_p | 1.0 m |
| MISS | ln 20 = 2.996 |
| Λ | ln 100 = 4.605 |
| 명판 허용 거리 | 3 m |
| 경로 벽 통과 한도 | 0.3 m |
| 최근 경로 | 30 m |

**후보마다 계산**(:134-164)
- 기하 항: geo = (그 후보의 E7 비용 − 이번 E7 최소 비용) / σ_g.
- 명판 항: 명판마다 c = min(d² / 2σ_p², ln 20) 이고, d 는 명판에서 그 번호 방 다각형까지 거리다. 3 m 밖이거나 그 층에 그 방이 없으면 ln 20 이다.
  - 여기서 d 는 반폭 규칙이 아니라 방 다각형 거리다.
- 결합 점수: fused = geo + Σc.
- 쓸 수 있는 후보: 수렴했고, E7 창 경로의 벽 위 길이가 0.3 m 이하이고, 최근 30 m 경로가 벽을 뚫지 않는다.

**층 점수**(:165-173)
- 쓸 수 있는 후보가 있는 층: 그 층 후보 중 최소 fused.
- 후보가 없는 층: 명판마다 (그 층에 같은 번호 방이 있으면 0, 없으면 ln 20) 의 합.

**확정**(:206-235)
- 조건은 네 가지다.
  - 2위 층 점수 − 1위 층 점수 ≥ ln 100.
  - 1위 층에 실제 후보가 있다.
  - FAST-LIO 가 저하되지 않았다. 120 s 안에 8 Hz 미만 표본이 5개 또는 10 % 이상이거나, 6 Hz 미만 표본이 하나라도 있으면 저하다.
  - 세션이 같다.
- 위치는 이긴 층의 후보 중 두 명판이 모두 3 m 안인 것 가운데 E7 비용이 가장 낮은 것이다(v3, :184-189).
- 기록은 `state/e7_fused_floor.jsonl` 에 E7 한 번마다 한 줄씩 남는다. 후보마다 비용·수렴·벽 통과·명판 거리·geo·fused 와 층 점수·차·이유가 들어간다.

### 8.6 명판 한 장으로 확정하는 경로 (B3)

`LIVE2_SINGLE_PLATE_E7` 기본 1이다(L2/e6e7_link.py:467-555, L2/fused_floor.py:192-220).

- **조건**: 층 가능도 0.8 이상인 판독 하나. 쌍 판정이 READY 이거나 E7 이 도는 중에는 하지 않는다.
- **후보**: 그 층의 명판 일치 가설 2개 + 다른 층마다 E3 최선 1개.
- **사전 검사**: 층마다 90 s 창에서 보정 없이 ±1 m 격자 최소 비용을 잰다. 그다음 두 경우에만 E7 을 돌린다.
  - 명판 층이 다른 모든 층보다 (ln 100 − ln 20)·0.05 = 0.0805 m 이상 좋을 때.
  - 어떤 다른 층이 0.150 m 이상 좋을 때. 이 경우는 오독 의심이다.
- **확정**: 같은 층 점수식에 단일 명판 조건을 더한다. 명판 층이 이겨야 하고, 오독 의심이 없어야 하고, 명판이 3 m 안이어야 한다.
- **오독 처리**: 오독 의심이 서로 다른 창에서 2번 나오면 그 판독을 뺀다. 기록은 `e7_single_plate.jsonl`, `semantic_misread.jsonl` 이다.

번호 없이 층을 정하는 완화 규칙(DIGITS_FALLBACK)은 운영 코드에 없다.

### 8.7 확정 뒤

- **현재 자세**: 매 틱 현재 자세 = 확정 변환 ∘ 최신 FAST-LIO 자세 다(L2/e6e7_link.py:1447-1462). 기록은 `state/fix_current.json(l)` 이다.
- **누적 오차**: 확정 뒤에는 다시 정합하지 않는다. FAST-LIO 누적 오차가 그대로 쌓인다.
- **멈추는 것**: E7 요청·단일 명판 E7·구조 검증, SigLIP(`IDLE_OK`), 정지 요청(`SUPPRESSED_AFTER_GLOBAL_FIX`)이다(L2/auto_fsm.py:169-171, 192, 227-228).
- **계속 도는 것**: E3 추적과 `e7_long`(기록 전용)은 계속 돈다.
- **표시**: 운영 화면 지도 위 표시는 `fix_overlay.json` 이 맡는다(L2/loc_overlay.py:50-114).

### 8.8 이 단계의 스위치 (기본값)

| 스위치 | 기본 | 내용 |
|---|---|---|
| `LIVE2_E8_POLICY` | `E8_PAIR` | 쌍 판정 사용 |
| `LIVE2_PLATE_HALF_CORRIDOR` | 1 | 복도 반폭 규칙(0 이면 방 다각형 3 m) |
| `LIVE2_E8_PATH_RULE` / `_PATH_RANK` / `_DROP_MISREAD` | 1 / 1 / 1 | 경로 벽 검사, 뚫는 후보 뒤로, 한 장 빼 보기 |
| `LIVE2_ANCHOR` / `_ANCHOR_TOPK` / `_ANCHOR_WALLCHECK` / `_ANCHOR_PATH` | 1 / 3 / 1 / 1 | 명판 기준 후보, 상위 수, 벽 검사, 경로 기준 후보 |
| `LIVE2_E7_FAST` / `_E7_PLATE_CANDS` / `_E7_PROC` | 1 / 4 / 1 | 명판 후보 4 + 경쟁 1, 별도 프로세스 |
| `LIVE2_FUSED_FLOOR` | 1 | 층 점수 결합 확정 |
| `LIVE2_SINGLE_PLATE_E7` / `_SEM_NEW_HYP` / `_TEXT_FLOOR` | 1 / 1 / 1 | 명판 한 장 경로, 판독 뒤 생긴 가설 표시, 번호 없는 글자(순위만) |
| `LIVE2_E3_SEM_FIRST` | 1 | 명판 일치 가설을 매 키프레임 추적 |
| `config/e7_semantic_policy.json` | ENABLED | 의미 확정 정책(없으면 SHADOW) |

## 9. 실험 한 건 따라가기 — `examples/X20261116_182124`

### 9.1 읽기 전에 알아 둘 것

**실험 조건**
- 정답 층은 5층, 명판은 502 → 504 → 505 순으로 찍었다.
- IMU 방식은 `body`, E6 는 명판 촬영 방식이다.
- 시각은 실험 시작(`experiment.json` 의 `created_wall` = 1794820884.11)부터 센 값이다.

**[현장]과 [재생]**
- **[현장]**은 기록 파일에 남은 값이다. 이 기록은 지금 운영판보다 앞선 판으로 만들어졌다. 그래서 반폭 규칙·경로 기준 후보·별도 E7 프로세스가 없고, 현장에서는 끝내 확정이 나지 않았다.
- **[재생]**은 같은 기록을 지금 운영판 코드로 노트북에서 다시 판정한 값이다. 노트북과 로봇은 같은 입력에서 46/46 같은 확정을 냈다.
- 재생에서 확정 시각은 노트북 계산 시간에 12.5 를 곱해 더한 **추정값**이다. 12.5 = 로봇/노트북 4.64 × 현장/한가할 때 2.7 이다.

### 9.2 시작 (0~58 s)

| 시각 | 일 | 볼 곳 |
|---|---|---|
| 17.4 → 45.8 s | fastlio, lio2(10.03 Hz), hub, rec, observe, e34, e2 순으로 건강 확인 통과 | `timeline.jsonl` 의 `module_started` |
| 24.2 s | 첫 자세. FAST-LIO 세션 시작 | `archive/live2_store/session.json` |
| 57.7 s | 이동 준비(`experiment_ready_to_move`). FAST-LIO 10.0 Hz, E2/E3/E4 갱신 0.90/0.18/0.18 s, 지도 9개 층 | `experiment.json` 의 `ready` |
| 58.2 s | E5 첫 장면 | `archive/observe_run/scenes.jsonl` |

### 9.3 기하만 있을 때 (66~158 s)

| 시각 | 일 | 볼 곳 |
|---|---|---|
| ~66 s | 출발. 71 s 까지 직선 2.2 m | `archive/live2_store/odom_link.csv` |
| 71.9 s | E2 첫 회차. 20 s 창(프레임 4장), β 228°, mirror +1 | `state/e2/e2_passes.jsonl` 의 `pass_start` |
| 86.3~106.0 s | 층별 거친 탐색 완료. 순서 1,2,3,4,6,8,5,7,9, 9개 층 합계 34.1 s | `coarse_floor_done` |
| 87.9 s | 첫 키프레임 평가. 1위 H001(1층) | `state/e3_history.jsonl` |
| 90.6 / 95.7 / 105.1 s | 1위가 H012(2층) → H022(3층) → H038(8층) 으로 바뀜 | 같은 파일 |
| 105.1 s | E4 `AMBIGUOUS`. 3층과 8층의 차 0.038 m | `state/e4_events.jsonl`, `state/auto_transitions.jsonl` |
| 103.8 s | 5층 거친 후보 12개 전부 보류. 축척이 틀렸다(최선 ppm 15.0, 맞는 값은 약 31) | `state/e3_events.jsonl` |
| 107~137 s | 되풀이 1회차. 맞는 축척의 5층 후보(H059~H061)가 128 s 에 E3 에 들어옴 | `refresh_floor_done`, `e3_events.jsonl` |
| 86~158 s | 명판 502 앞에 서 있어 키프레임이 없다. 순위가 멈춘다(1위 H038, 8층) | `e3_history.jsonl` 의 갱신 간격 |

**1위가 3층·8층이었던 이유.**
- 첫 창의 경로는 직선 2.2 m 뒤에 한 번 꺾인 짧은 L자였다.
- 같은 모양의 복도 모서리가 여러 층에 있어서, 벽 통과 없이 여러 층에 들어맞는다.
- H022(3층)는 벽 거리 5.5 cm, H038(8층)은 3.8 cm 였다. 5층 정답 변환도 같은 모서리에 똑같이 붙는다.

### 9.4 첫 명판 502 (81~177 s)

| 시각 | 일 | 볼 곳 |
|---|---|---|
| 81.35 s | E5 가 502 를 검출했다 → P2 VERIFIED | `archive/observe_run/scenes.jsonl`, `live2_detection_pose.jsonl` |
| 83.4 s | 정지 요청 O001 → 운영 화면 신호음 | `archive/observe_run/live2_stop_requests.jsonl` |
| 83.9 s | 촬영 도구가 명판을 기억. 몸체 기준 (2.96, 1.08, 1.15) m | `state/e6_observations.jsonl` |
| 88.1 s | 정지 확인. 검출부터 6.55 s, 2.45 m | 같은 파일 |
| 105~148 s | 사진 5장. 마지막 사진 005(위로 5° 조정)를 사용 | `state/e6_precision_results.jsonl` |
| 151.4 s | STORED. 명판 위치(저장소 좌표) (3.351, −6.147) | `state/e8_observations.jsonl` |
| 158.6 s | OCR "502" | 같은 파일 `READ` |
| 159.2 s | 이 판독으로 지역 E2 요청. 상태기계 `SEMANTIC_CANDIDATE` | `state/semantic_recovery_single.jsonl`, `auto_transitions.jsonl` |
| 163.8 s | 다시 걷자 1위가 H059(5층). 층은 맞지만 502 에서 64.6 m 떨어진 자리다 | `e3_history.jsonl` |
| 165.8 s | 명판 기준 후보 계산 완료(프레임 60장, 6.5 s) | `state/e2/semantic_anchor.json` |
| 166.4 / 168.0 s | H077~H079 가 E3 에 들어옴 → 곧 보류(GAP > 0.25 × 3) | `e3_events.jsonl` |
| 176.7 s | VLM 도 "502"(AGREE, p 0.998). 판독과 맞는 가설: H077 0.80 m, H078 0.83 m, H079 1.30 m | `e8_observations.jsonl` `VLM_READ` |
| 180.0~242.6 s | 지역 E2 탐색(반경 1030 px). 후보 12개 | `state/e2/e2_passes.jsonl` 의 `region_done` |

**81.35 s 검출 수치**: 유사도 0.1265, 평균 0.0947(5회 관측), 문턱 0.0818, 거리 3.05 m, 폭 0.68 m.

**151.4 s 위치 변환**: 저장소 좌표로 옮기면서 0.27 m 이동했다.

**158.6 s 판독**
- OCR 신뢰도 0.998, 소요 6.94 s.
- OCR 만 → p = 0.798(확실). 층 가능도 5층 0.862, 다른 층 각 0.017.
- 가설 판정: 층 불일치 68, 자리 불일치 8(5층 전부, 10.4~64.6 m), 일치 0.

**165.8 s 명판 기준 후보**
- φ 37° / 57° / 60°, 비용 0.0747 / 0.2055 / 0.2634 m, 비용비 2.75.
- 셋 다 사전값(E2 merit 중앙 1.1687)이 S 를 끌어올렸다. H077 의 S = (3·1.1687 + 5·0.281)/8 = 0.614 다. 이 때문에 보류됐다.
- 보류된 가설도 E8 판정 대상이다.

### 9.5 두 번째 명판 504 와 쌍 판정 (218~259 s)

| 시각 | 일 | 볼 곳 |
|---|---|---|
| 218.7 s | 운용자가 화면의 SigLIP 상자를 눌러 504 를 기억. 거리 4.05 m. 키 큰 상자라 높이를 1.72 m 로 맞춤 | `e6_observations.jsonl` |
| 236.5 / 242.8 s | 사진 2장. 선명도 5.6, 명판 영역 못 찾음 | `e6_precision_results.jsonl` |
| 245.7 s | STORED. 명판 위치 (10.308, −11.193) | `e8_observations.jsonl` |
| 258.5 s | OCR "504", 신뢰도 0.999, 12.8 s. 화질 경고가 붙었지만 0.999 ≥ 0.9 라 사용 | 같은 파일 `READ` |
| 259.2 s | 쌍 판정 READY | `state/e8_pairs.jsonl` 의 `LIVE2_E8_EVALUATION` |

**259.2 s 쌍 판정 계산 [현장]**

| 항목 | 값 |
|---|---|
| 두 명판 위치 간격 | 8.594 m |
| 주행 경로 | 15.8 m, 94.5 s, 8 Hz 미만 표본 0 |
| 주행거리 오차 | 0.5 + 0.05·√(12.4² + 3.4²) = 1.14 m |
| 502–504 방 다각형 거리 | [4.20, 14.08] m |
| 허용 폭 | 6 + 1.14 + 0.05·14.08 = 7.84 m |
| 잔차 | 0, 복도 연결됨 → 설명하는 층은 5층뿐 |

- 예전 규칙(방 다각형 3 m)으로는 H077·H078·H079 가 통과해 READY 였다.

**복도 반폭 규칙 [재생]**

| 가설 | 명판 | 벽선에서 수직 거리 | 반폭 | 결과 |
|---|---|---|---|---|
| H077 | 502 / 504 | 0.40 / 0.17 m | 1.31 m | 통과 |
| H078 | 504 | 2.42 m | 1.335 m | 탈락(방 안쪽) |
| H079 | 504 | 2.59 m | 1.40 m | 탈락 |

H077 하나만 남는다.

### 9.6 E7 과 확정 (259~273 s) [재생]

1. **E7 요청.** 판정이 READY 면 같은 자리에서 E7 을 요청한다(L2/e8_pair.py:748-759, 781-795). 후보는 H077 + 경쟁 1(명판 후보가 아닌 E3 1위 = H078)이다.
   - 경로 기준 후보(PAP002_1_F5)는 그다음 틱에야 만들어진다. E7 이 도는 동안에는 재판정도 열리지 않으므로, 첫 E7 에는 들어가지 않는다.
2. **E7 정합**(최근 90 s, 벽 모양 점)
   - H077 은 수렴했다. 보정은 이동 0.14 m, 회전 +1.1° 다.
   - H078 은 수렴하지 않았고, 90 s 경로가 벽 위를 8.31 m 지나 층 판정에서 빠졌다.
3. **층 점수**(`fused_floor`)
   - 5층 = 기하 0 + 명판 502 min(1.09²/2, ln 20) = 0.594 + 명판 504 min(0.39²/2, ln 20) = 0.076, 합계 0.670.
   - 다른 층은 후보가 없다. 그 층에 502·504 방이 없으므로 각 2·ln 20 = 5.991 이다.
   - 차 = 5.321 ≥ ln 100 = 4.605 → **5층 확정**, 위치는 H077.
   - 경로 벽 통과는 0.0 m, 최근 28.8 m 에 벽 뚫기가 없다.
4. **확정 변환**: pos (1156.7, 561.5) px, anchor (9.929, −13.135) m, φ 41.06°, ppm 32.12, β 228°.
   - 기록 전체 LiDAR 로 따로 맞춘 기준과의 위치 차는 0.08 m 다.
5. **확정 시각(추정)**: 259.2 + 노트북 E7 1.07 s × 12.5 ≈ **272.6 s**.
6. **확정 뒤**
   - 현재 자세 = 확정 변환 ∘ 최신 FAST-LIO 자세(`state/fix_current.json`).
   - E7 요청이 멈추고 SigLIP 은 `IDLE_OK`, 정지 요청은 억제된다.

### 9.7 이 기록에서 현장에서 일어난 일 [현장]

- **E7 결과 없음.** 264.0 s 에 E7 검증에 들어갔지만, 끝(344.9 s)까지 E7 결과가 기록되지 않았다(`e8_pairs.jsonl` 에 `LIVE2_E8_E7_RESULT` 0건). 이것이 지금 운영판에서 E7 을 별도 프로세스로 옮긴 이유다.
- **279.7 s:** VLM 이 504 사진에서 번호와 학과명을 읽었다. 이름은 이 저장소에서 가명으로 바꿨다.
- **281.7 s:** 504 기준 후보의 1위가 H077 과 1 m·10° 안이라 합쳐졌다. 두 명판이 따로 같은 자리를 가리킨 것이다.
- **311.5 s:** 세 번째 명판 505 를 판독했다.
- **314.4 s:** 기록용 구조 비교(`e7_long`)에서 4층이 0.0021 m 차로 앞섰고, 나머지 8개 층도 모두 근접했다. 구조만으로는 층을 가를 수 없다는 근거다.

## 10. 주의점과 확인하지 못한 것

### 10.1 코드에서 본 주의점

1. **E2 경쟁 목록이 빈다.** E2 회차 끝에서 `candidates` 를 그 회차 결과로 통째로 바꾼다(L2/e2_worker.py:743-744). 정밀 단계가 꺼진 지금(`e2_acc.on`)은 다른 층 E2 경쟁 목록이 빈다. E3 는 누적 흡수라 영향이 없다.
2. **E2 일부 후보가 E3 에 안 들어간다.** E3 에 넘기는 거친 후보는 그 시점 전체 상위 24개뿐이다. 늦게 끝난 층은 E3 에 덜 들어간다. REC 에서는 88개 중 49개만 들어갔다.
3. **되풀이 후보가 쌓인다.** 되풀이 후보는 모든 회차가 같은 `pass_id` 를 써서 회차 사이에 합쳐지지 않는다. 거의 같은 가설이 쌓인다.
4. **서 있으면 순위가 멈춘다.** E3 키프레임은 움직일 때만 생긴다. 명판 앞에 서 있는 동안 순위·보류 판정이 멈춘다.
5. **명판 기준 후보가 바로 보류된다.** 명판 기준 후보는 S 의 사전값(E2 merit 중앙값)이 커서 들어오자마자 보류되는 일이 많다. 쌍 판정은 보류된 가설도 보므로 확정에는 지장이 없다.
6. **층 점수와 쌍 판정의 명판 거리가 다르다.** 층 점수(`fused_floor`)의 명판 거리는 방 다각형 거리다. 반폭 규칙은 쌍 판정과 경로 기준 후보 검사에만 쓴다.
7. **VLM 번호도 판독에 들어간다.** VLM 이 읽은 번호도 판독에 쓰인다. OCR 과 같으면 확실(AGREE), 다르면 보류(CONFLICT)다. VLM 만 읽은 번호는 라벨이 되지 않는다.
8. **완화 규칙은 없다.** 번호 없이 층을 정하는 완화 규칙(DIGITS_FALLBACK)은 운영 코드에 없다.
9. **E2 층 순위는 꺼져 있다.** `~/kci_loadtest_20261113/e2_rank.on` 은 로봇에 없다. E2 층 순위를 E3 순위 앞에 붙이는 기능은 꺼져 있다.
10. **확정 뒤 오차.** 확정 뒤에는 다시 정합하지 않으므로 FAST-LIO 누적 오차가 쌓인다. 화면 그림(`fix_overlay.json`)은 추적기의 현재 변환에 E7 보정을 넣어 그린다. 그래서 확정 변환과 조금 다를 수 있다.

### 10.2 확인하지 못한 것

- 카메라–몸체 외부 파라미터. 코드에 `UNVERIFIED` 로 적혀 있다(L2/e6_core.py:23-25).
- `visual_stop.world_target` 이 원래 FAST-LIO 좌표의 거울상 성질을 보정하지 않는 것이 같은 명판 판정에 주는 영향.
- 로봇 기본 DDS 설정(hub·observe 가 쓰는 것)의 내용.
- 지도 파일의 로봇 쪽 정확한 경로와 해시. 지도는 이 저장소에 없다.
- `body` 방식에서 출력 어댑터의 QoS. 소스는 `code/home_extra/sci_ws_alt_imu_20261029_204740/src/sci_alt_imu_bridge` 에 있지만 따로 검토하지 않았다.
- 9장 재생에서 명판 기준 후보 생성 단계의 벽 검사(지금 판)는 흉내 내지 않았다. 지금 판에서는 셋째 후보 집합이 달라질 수 있다.

### 10.3 저장소의 판과 기록의 판

- 저장소 코드는 로봇 운영 트리를 그대로 읽어 온 것이다(README 의 sha256 표).
- `examples/X20261116_182124` 는 그보다 앞선 판에서 만들어졌다. 그래서 기록에 없는 필드가 있다. 예를 들어 반폭 규칙의 `place_rule`, `e7_fused_floor.jsonl`, `e7_proc` 프로세스가 없다.
