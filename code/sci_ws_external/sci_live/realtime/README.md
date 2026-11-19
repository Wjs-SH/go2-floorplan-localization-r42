# 실시간 층 판별·지도 위치 (Go2, ROS2 Foxy) — 2026-10-07

걸으면서 **지금 몇 층인지**와 **그 층 안내도 위 위치**를 계속 낸다.
지도 생성·구조점 추출·위치추정·층 판별 **알고리즘은 하나도 바꾸지 않았다.** 기존 코드를 실시간 입력으로 부르는 껍데기다.

```
Hesai XT16 /lidar_points ─┐
                          ├─▶ rt_live_node.py ──▶ a131_live(esy, 구조점) ──▶ LiveStore(/dev/shm, 기존 주행데이터 형식)
FAST-LIO  /Odometry ──────┘        │                                              │
                                   │                         ┌────────────────────┴─────────────────┐
                                   │                 loc_worker --role floor               loc_worker --role pose
                                   │                 (8개 층, 프로세스 3개로 나눔)            (다수결 층 하나 계속 재정합)
                                   │                         └──────── floor.json · pose_fit.json ──┘
                                   ▼
        /floorplan_loc/pose (FAST-LIO 가 올 때마다) · /floorplan_loc/floor · /floorplan_loc/status (1초마다)
```

| 파일 | 하는 일 |
|---|---|
| `rt_live_node.py` | ROS 노드. 점군·FAST-LIO 를 받아 1.5초 격자로 구조점 표본을 만들고(a131_live), 작업자를 띄우고, 결과를 발행 |
| `live_store.py` | 표본을 `floor_id` 가 읽는 주행 폴더 형식 그대로 `/dev/shm` 에 쓴다(원자적 쓰기, 최근 900초 + 첫 표본) |
| `loc_worker.py` | `floor`: 8개 층을 같은 창(같은 t_now)으로 맞춰 순위 → 바퀴 판정 + 최근 5바퀴 다수결. `pose`: 다수결 층 재정합 |
| `pose_tools.py` | 정합 결과로 FAST-LIO 위치를 지도 px·m 로 옮기는 식(정합 코드의 궤적 투영식과 같다) |
| `replay_samples.py` | 센서 없이: 저장된 주행 표본을 실제 시간대로 흘려 작업자 시험 |
| `tools/replay_dump_ros.py` | 센서 없이: a131_live 시험 데이터를 점군·Odometry 토픽으로 내보냄 |
| `tools/ros_selftest.sh` | 위 둘로 노드 ROS 경로 자체 시험(도메인 77 + 루프백만) |
| `tools/check_store_vs_dump.py` | 노드가 만든 표본을 esy 기준 결과와 비트 대조 |
| `tools/compare_live_offline.py` | 실시간 입력 vs 오프라인 입력 층 판별 성적 비교 |

## 로봇에서 실행

드라이버와 FAST-LIO 가 떠 있어야 한다(`~/go2_ws` — 이 패키지 밖, 띄우는 방법은 그쪽 담당 확인).
노드는 드라이버와 **같은 DDS 환경**(Unitree CycloneDDS)에서 띄운다.

```bash
cd ~/jsh_onboard_robot_20261001/realtime
bash tools/sensors_go2.sh start                   # Hesai 드라이버 + FAST-LIO(~/go2_ws 를 고치지 않고 띄우기만), 로그 ../logs/live_<월일>
source ~/unitree_ros2/setup.sh                    # Foxy + CycloneDDS(eth0)
export CYCLONEDDS_URI=file://$PWD/config/cyclonedds_live.xml PYTHONNOUSERSITE=1
../venv_gpu/bin/python rt_live_node.py --ros-args -p building:=kci
bash tools/sensors_go2.sh stop                    # 끝낼 때
```
**브라우저 실시간 보기**: 노드가 같이 띄운다(`viewer_port`, 기본 8765, 0 이면 끔). 노트북 브라우저에서 `http://<로봇 IP>:8765`
(같은 와이파이). 왼쪽 = FAST-LIO 좌표의 최근 60초 누적 구조점·경로·지금 방향(초당 5번), 오른쪽 = 다수결 층 안내도 위 같은 누적점·추정 위치·층 순위.
점이 지도 밖이면 지도 전체를 같이 보여 주고 경고를 띄운다. 왼쪽 그림은 축 규약 점검도 된다 — 돌 때 벽이 겹쳐 번지면 δ·거울상이 틀린 것.
정보관 지도처럼 `base_map.yaml` 이 없으면 `/floorplan_loc/pose` 를 픽셀 좌표(frame `floorplan_map_<층>F_px`)로 낸다(10-08 이전에는 여기서 노드가 죽었다).

**DDS 설정이 꼭 필요하다**(`config/cyclonedds_live.xml`, 2026-10-08 로봇 실측). Unitree 기본(조각 1344 B)으로는 점군 구독자가
FAST-LIO 말고 하나만 더 붙어도(reliable·best-effort 모두) 드라이버 송신이 밀려 **FAST-LIO 까지 10→6.9 Hz, 지연 0.05→2.4초**가 됐다.
조각 65000 B·소켓 버퍼 8 MB 로 드라이버·FAST-LIO·노드를 모두 띄우면 추가 구독자 10.1 Hz(스캔 100% 연속), FAST-LIO 10 Hz·지연 0.05초 그대로.

| 파라미터 | 기본 | 뜻 |
|---|---|---|
| `building` | `sw` | `floor_id/config/buildings.yaml` 의 건물 |
| `lidar_topic` / `odom_topic` | `/lidar_points` / `/Odometry` | Hesai 설정·FAST-LIO 코드에서 확인한 이름 |
| `cloud_reliability` | `auto` | 점군 발행자 QoS 를 보고 맞춤(Hesai 는 reliable) |
| `sample_dt` | `3.0` | 구조점 표본 간격(초). 스캔 번호 격자(30 스캔마다)에 맞춘다. 3초 = 위치추정 프레임 간격이라 프레임이 정확히 표본에 떨어진다. 10-08 정제를 붙이며 1.5→3.0(로봇: 구조점 1.6초 + 정제) |
| `a131_workers` / `exact_pca` | `3` / `true` | 구조점 프로세스 수 / 정확한 PCA(기준과 같은 결과) |
| `clean` / `clean_stride` | `true` / `2` | esy 정제 실시간판(`../a131_clean_live`, 10-08): 이중반사·따라오는 사람·움직이는 물체를 스캔 단계에서 NaN 으로. a131 에 넘기는 3스캔은 `clean()`, 나머지는 `observe()` 로 이력만. **별도 프로세스**로 돈다(로봇 실측 clean 98 ms·observe 44 ms/스캔 — 표본 루프와 한 줄이면 3초도 넘침). 로봇에서 esy 검증 재현율 100.00%·일치 99.40%(stride 2). 노트북 대조: 노드 = 오프라인 같은 순서 정제 82/82 비트 동일 |
| `delta_deg` / `mirror` | `45.3` / `true` | FAST-LIO → base_link 규약(esy 녹화본 추정값). 실물 확인 필요(아래) |
| `static_base_lidar` | URDF 값 | base_link←hesai_lidar. `go2_description.urdf` 의 `hesai_lidar_joint`(0.145,0,0.11, yaw 90°)와 같다 |
| `floor_procs` | `3` | 층 판별을 나눠 맡을 프로세스 수 |
| `store_root` | `/dev/shm/rt_live` | 표본·결과 폴더(시작할 때 비운다) |

끝낼 때는 Ctrl-C 또는 SIGTERM(2초 안에 작업자까지 정리된다).

### 드라이버·FAST-LIO 를 띄운 직후, 노드 켜기 전 (`tools/live_preflight.py`)

```bash
python3 tools/live_preflight.py static                                   # 세워 둔 채: 점군 10 Hz·필드·점 시각 폭·FAST-LIO·장착 TF
python3 tools/live_preflight.py motion --seconds 40 --save /dev/shm/flio_rec.npz
#   ↑ 그동안 직진 몇 m(0.25 m/s 넘게) → 제자리에서 반시계(좌회전)로 90° 넘게
```
FAST-LIO 규약(δ 45.3°·거울상)은 esy 가 **노트북 FAST-LIO 설정**에서 잰 값이다. 판정이 기본값과 다르면 노드에
`-p delta_deg:=<값> -p mirror:=<true|false>` 로 준다(코드에 부호 뒤집기를 넣지 않는다).
판정 로직은 정답을 아는 합성 주행(δ 45.3°·40°)으로 확인했다. esy 시험 데이터(S6 원본 녹화)의 자세는 축 보정(tffix) 전
/tf 라서 이 점검에 쓰면 안 된다(걷는 방향이 base_link 기준 −41.8° 로 나온다).

## 검증 (2026-10-07, 센서 없이)

**1. ROS 경로 — 노드가 만든 구조점 = a131_live 오프라인 결과(비트 단위)**
`tools/ros_selftest.sh`: esy 시험 데이터(S6 13초, 스캔 84개)를 점군·FAST-LIO 토픽으로 내보내 노드가 만든 표본을 기준과 대조.

| 환경 | 받은 스캔 | 기준과 같음 | 같은 기계 오프라인(`demo_from_dump.py`) |
|---|---|---|---|
| Go2 (Foxy · CycloneDDS · venv_gpu py3.8) | 84/84 | 49/55 | 49/55 — **다른 6개가 같은 표본·같은 칸 수** |
| 노트북 (Humble · FastDDS · py3.10) | 84/84 | 45/55 | 45/55 — 다른 10개가 같은 표본·같은 칸 수 |

노드가 더한 차이는 0이다. 기준과의 차이(칸 1~3개)는 기계별 부동소수 차이로, 오프라인에서도 똑같이 난다.

**2. 작업자 — 실시간 방식으로 불러도 결과가 같다** (노트북, 소융대 6층·3층 지도)
- 지도를 표본 1개일 때 열고 표본이 늘 때마다 다시 읽힘(장면 캐시 끝부분 무효화) = 그 시점 데이터로 새로 계산: 85·100초 위치·각·축척·벽비용·twoway 같음.
- 같은 위치추정기 객체로 여러 번 정합 = 매번 새로 연 것: 100·130·100초 같음.
- 시동 정렬(T[0]부터 15초) 유무, 주축 AX0·중심 ctr 변경: 정합 결과 같음 → 보관 기간(900초)은 결과에 영향 없음.
- 층 판별 프로세스 1개 = 3개: 같은 t_now 에서 8개 층 결과 모두 같음.

**3. 실시간 입력 vs 오프라인 입력 층 판별 성적** (`tools/compare_live_offline.py`, 소융대 10구간 S1·S6 70초부터 60초 간격,
창 60초 · dt 3초 · 장면 lidar · per_map 축척)

| 입력 | 1위 | 상위3 | 정답 순위 |
|---|---|---|---|
| 오프라인(0.1초 간격 전체 = floor_id.py, 기존 기록과 같음) | 2/10 | 4/10 | 4 6 2 1 · 8 1 5 3 5 6 |
| 실시간(1.5초 격자 표본) | 1/10 | 4/10 | 3 6 2 1 · 8 2 5 6 5 8 |

판정이 같은 구간 3/10. 표본 간격만 바뀌어도 1위 층이 쉽게 바뀐다. 실시간에서는 벽 관통 검사용 '조밀 경로'가 0.2초→1.5초 간격으로
성겨진다(위치추정기는 행마다 표본 파일이 있어야 해서 자세만 끼울 수 없다 — 알고리즘을 바꿔야 한다). 10구간이라 1/10 대 2/10 차이의 유의성은 말할 수 없다.

**4. Go2 속도** (`replay_samples.py` S6 1배속, 층 판별 3개 + 위치 작업자, venv_gpu CUDA, 구조점 추출 없음)
- 지도 준비 80초(고해상도 6~8층이 층당 37~42초). 첫 판정은 시작 뒤 **244초**.
- 바퀴 164~198초(층당 정합 37~105초, GPU 99%). 바퀴 판정은 그 바퀴 시작 시각의 창 → 판정이 나올 때 이미 ~3분 전 상황.
- 위치 보정 49~66초마다. 그 사이는 FAST-LIO 로 이어 간다(발행 10 Hz).
- 메모리: 층 판별 자식 프로세스 2.0~2.4 GB(층 지도 + CUDA·torch ~1.1 GB), 위치 작업자 0.8 GB. 가용 메모리 최저 **1.76 GB**.
- 이 시험의 판정(정답 6층): 1층 → 4층 → 1층, 다수결 1층. 틀렸다.

**5. Go2 전체 부하** (`tools/load_test_go2.sh 3`: 노드가 시험 점군을 1배속 반복 재생으로 받아 구조점 추출 + 위 4번 동시)
- 구조점 표본 172개 · 실패 0 · 표본 처리 중앙 **1.62초**(최대 2.57초) — 1.5초 예산을 조금 넘어 가끔 격자를 다시 잡는다.
  (반복 재생 경계마다 ~5초 빈틈이 있어 구조점 부하는 실제의 ~73%다. 실제 주행에서는 조금 더 느릴 수 있다.)
- 바퀴 165~193초 · 위치 보정 29~57초 — 구조점 추출이 없을 때와 비슷(병목은 GPU).
- 가용 메모리 최저 **1.67 GB**(정합 사이 torch 캐시 반환 포함). 부족은 없었지만 여유가 크지 않다 — 다른 무거운 프로세스와 같이 돌리지 말 것.
- 판정(정답 6층): 2층 → 3층 → 1층. 4번과 바퀴 시각이 달라 판정도 달랐다(같은 데이터·같은 알고리즘).

**6. 기타**: 종료(Ctrl-C·SIGTERM) Foxy 에서 2초 안, 작업자까지 정리. 표본 간격 1.498~1.501초(격자 정렬).
`/floorplan_loc/pose` 10.2 Hz(FAST-LIO 주기) `floorplan_map_<층>F` 프레임 미터 좌표.

## 한계 — 반드시 읽을 것

- **층 판별 성적은 오프라인과 같은 알고리즘 그대로다: 한 구간 1위 21%·상위3 45%(무작위 12%·40%).**
  실시간이 된다고 좋아지지 않는다. `vote`(최근 5바퀴 다수결)를 쓰되, 그것도 확정으로 믿지 말 것.
- **미터 좌표는 실측 확인 전이다**(안내도 축척 추정). `/floorplan_loc/pose` 의 m 는 base_map.yaml 기준.
- **실제 센서로는 아직 안 돌렸다.** 센서 없이 녹화 데이터로만 확인했다. 실물에서 먼저 볼 것:
  FAST-LIO 규약(δ≈45°·거울상) — esy 요청대로 직진 후 좌회전 한 번으로 확인.
- 층이 바뀐 직후(엘리베이터) 창 60초 동안 두 층 데이터가 섞이고, 다수결은 5바퀴 동안 이전 층을 끈다.
