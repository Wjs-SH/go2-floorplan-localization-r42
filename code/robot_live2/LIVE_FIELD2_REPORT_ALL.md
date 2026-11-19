# LIVE-FIELD2 — 실제 다층 위치 후보·모호성 판단 구현 보고 (2026-10-18)

- 작업공간: `/home/unitree/sci_ws/repairs/livefield2_20261018_151606`
  - `baseline/`: LIVE1 사본 (읽기 전용)
  - `candidate/`: 이번 작업 대상
- 기준선: `/home/unitree/sci_ws/repairs/livefield1_20261018_060257` (변경 0. 965개 파일의 sha 가 시작 시점 기록과 일치)
- 진입점: `scripts/live2_ctl.sh`. 절차는 `LIVE_FIELD2_START_STOP.md` 에 있습니다.

## 0. 결론 요약

| 항목 | 상태 |
|---|---|
| 실기동 (센서·FAST-LIO·GPU) | **수행하지 않음**. 현재 boot 에 유효한 승인이 없습니다 (§1) |
| E2 다층 Top-K | 구현 완료. 학생 best 와 비트 단위로 같고 2.39배 빠릅니다 (저장 입력) |
| E3 증분 갱신 | 구현 완료. 단위시험 6건을 통과했고 REPLAY 4회를 돌렸습니다 |
| E4 모호성 6상태 | 구현 완료. 마지막 run4 에서 잘못된 CONFIDENT 는 0건입니다 |
| E4→E5 연결 | 구현 완료 (예산이 구속될 때만 우선순위가 바뀝니다). 팬틸트 명령 없음. 기하 STOP 은 보류 |
| 웹 | 전역위치 UNKNOWN, 실제 Top-K, 층별 분포, E4 사유, 의미 필요, 지연·신선도 표시 |
| 산출물 | CSV 5종, 보고서, 번들, SHA256SUMS |

가장 중요한 목표인 "실제 로봇이 움직일 때 웹에서 후보와 모호성이 변하는 모습"은 저장 입력 재생(REPLAY)으로만 확인했습니다. 실주행 확인은 §8 의 승인 한 건이 필요합니다.

## 1. PHASE 0 — 점검

**승인**
- LIVE1 승인 A (`A_20261018_065543`) 는 boot `eb8dcb40…` 에 묶여 있고 07:55 에 만료됐습니다.
- 현재 boot 는 `dde2c30f-c406-4516-be04-9d26d910b2bb` 입니다. 따라서 **유효한 승인이 없습니다**.
- 지시("승인 범위가 없으면 준비와 저장 입력 시험")에 따라 실기동 없이 진행했습니다.

**ROS 그래프**
- `/lidar_points` 는 발행자 1, 구독자 0 입니다.
- `/Odometry` 는 없습니다 (FAST-LIO 꺼짐).
- ROPUS hesai·realsense·robot_state_publisher·go2_control_runner 는 각각 하나씩 동작 중이며 손대지 않았습니다.

**게이트**
- fastlio·lio2·hub·observe 시작 요청 30건이 모두 `approval_A` 에서 거부됐습니다 (`reports/gates_start_*`).
- 이 네 모듈의 로그는 0개로, 실제로 기동된 적이 없습니다.

**freedrive 코드와 hub·keeper·web**
- LIVE1 사본에 E2/E3/E4 모듈을 더했습니다.
- 바꾼 파일은 5개입니다 (ctl, common, web, observe_node, index.html). 차이는 `diffs/` 에 있습니다.

## 2. 설계

```
FAST-LIO ─/Odometry, /cloud_registered (같은 stamp)─▶ lio2 (E2Feed) ─▶ LiveStore(/dev/shm)
                                                                       │
                     e2_worker (CPU, 코어 6·7, nice 10, 층 하나씩) ◀────┤  건물 전 층 Top-K, pass 단위
                     e34_node (코어 5, 2 Hz)  ◀────────────────────────┘  E3 추적, E4 판단, 의미 필요
                                │ e4_feed.json
                     observe(SigLIP, E5) ─ e5_e4_link: 예산이 구속될 때 E4 대상 방향 ROI 우선
                     web (읽기 전용) ─ 전역 UNKNOWN + 후보 + 상태
```

- **원시 `/lidar_points` 구독자는 추가하지 않았습니다.** E2 입력은 FAST-LIO 출력 두 토픽만 씁니다.

**E2 (PHASE 1)**
- `metric_fit_v2.fit_window_v2` 는 내부에서 최대 12개 후보(`picked`)를 고르지만 best 하나만 반환합니다.
- `e2_core.fit_window_v2_topk` 는 같은 계산에서 후보 전부를 반환합니다.
  - best 는 학생 함수의 결과와 모든 필드가 같습니다 (`evidence/e2_equivalence_S6_6F_t120.json`).
- 후보마다 다음을 기록합니다: floor, px/py, x_m/y_m, yaw(phi), ppm, beta, mirror, merit, twoway, wall_pass, map_version (지도 sha), odom_session, fit_t, pass_id.
- 모든 후보는 `status=UNVERIFIED_HYPOTHESIS` 이고 `verified=False` 입니다.
- 층 순서는 1..N 고정이며, 모든 층을 같은 조건으로 탐색합니다. 정답 층을 지도 선택에 쓰지 않습니다.
- 속도
  - CPU 에서 학생 fit 은 층당 305 s 이고, 그중 90% 가 wall-fit 입니다.
  - `wallfit_cpu_fast` 는 결과가 비트 단위로 같은 대체 구현입니다. uint8 제곱거리 EDT 표, 점 우선 조회, 기수 정렬을 씁니다.
  - fit 한 번 기준 2.39배 빨라졌습니다 (305.8 s → 127.7 s).
  - 표가 정확하지 않으면 원본 함수로 자동 대체합니다.
- 자원 보호 (층을 열기 전에 확인)
  - FAST-LIO odom hz 가 기준의 90% 미만이면 HELD_FASTLIO
  - MemAvailable 3500 MB 미만이면 HELD_MEMORY
  - 80 °C 이상이면 HELD_THERMAL
  - 실행 중 2800 MB 아래로 떨어지면 감시자가 자식 프로세스를 종료하고 그 층을 다시 대기열에 넣습니다.
  - 지시대로 "새로운 무거운 정합만 보류"합니다.

**프레임·정보손실 검증**
- `/cloud_registered` 는 dense_publish_en=true 이므로 deskew 된 전체 스캔이고, stamp 가 `/Odometry` 와 같습니다.
- world→base 변환 사슬: p_imu = R_imuᵀ(p_w − t), p_lidar = R_extᵀ(p_imu − T_ext), p_base = R_bl p_lidar + t_bl
  - 단위시험(U03)에서 정확히 복원됐습니다.
- stamp 가 정확히 같은 pose 가 없으면 표본을 만들지 않습니다 (1 ms 넘는 보간 금지, U04).
- 잃는 정보: ring 번호와 점별 시간이 없습니다.
  - 그래서 학생 A131Live 구조 추출은 쓸 수 없고 `structure_proxy` 로 대체했습니다 (8 cm 셀, z −0.15..1.6 m, 2회 이상 관측).
  - 이 proxy 는 **현장에서 검증되지 않았습니다** (§7, frame_probe 필요).

**E3 (PHASE 2)**
- 키프레임: 0.5 m 이동 또는 15° 회전 시 생성합니다. 정지 중에는 증거가 늘지 않습니다.
- 증거는 가설마다 sample id 단위로 한 번만 셉니다 (이중계산 방지, U05).
- 점수 S = (3·e2_merit + 최근 20 키프레임 merit 합) / (3 + n). 낮을수록 좋습니다.
- 약한 관측 한 번으로는 보류하지 않습니다 (U06).
  - gap > 0.25 가 3 키프레임 이어지거나, 최근 5개 중 강한 위반이 2개 이상일 때만 SUSPENDED 로 갑니다.
  - gap < 0.12 가 2 키프레임 이어지면 복귀합니다 (U07).
  - DROPPED 는 상한 120 초과 때와 세션 변경 때만 일어납니다.
- 추적: 가설마다 최근 3 키프레임으로 제한된 보정을 합니다 (±0.2 m, ±1°, ±1% 척도, coarse-to-fine). 다른 가설로 점프하지 않습니다 (U10).
- 매 프레임 전 층 정합은 하지 않습니다.
- 재탐색 경로
  - E4 가 RECOVERY 이면서 사유가 조치 가능한 것일 때
  - 또는 AMBIGUOUS 상태에서 E2 window 이후 30 m 이상 이동했을 때
  - 위 두 경우에 `e2_request.json` 을 씁니다 (120 s 간격 제한). 재탐색 결과는 다른 pass 의 후보와만 병합합니다.

**E4 (PHASE 3)**
- 상태: INITIALIZING / SEARCH_INCOMPLETE / AMBIGUOUS / DOMINANT_UNVERIFIED / CONFIDENT_GEOMETRY / RECOVERY_REQUIRED
- 판단 근거: 층 다양성(gap 안의 층 수), gap, 분리도(2 m·30°), 증거 충분성(8 kf, 8 m 경로, 비용 0.35), 1위 안정성(15 kf), 신선도(odom 2 s, 입력 6 s)
- 임계값은 정답으로 조정하지 않았습니다 (`tuned_on_ground_truth=False`).
- 가짜 후보나 확률은 만들지 않습니다. score 는 m 단위 기하 불일치이며 확률이 아닙니다.
- 의미 필요도는 HIGH/MEDIUM/LOW/NONE 와 사유로 냅니다.
  - 대상은 경쟁 층 의미지도(kci_handoff 등)의 room_number_plate, stair, elevator, restroom 입니다.
  - 각 대상의 거리, 방위, 관측 가능성(OBSERVABLE_NOW / NEEDS_OPERATOR_TURN / NO_LINE_OF_SIGHT), 층 구분력을 함께 냅니다.

**E4–E5 (PHASE 4)**
- SigLIP 장면 주기는 바꾸지 않았습니다. 기존 저비용 ROI 탐색도 그대로입니다.
- 예산이 구속되고 E4 우선순위가 HIGH/MEDIUM 이며 3 s 이내로 신선할 때만 바뀝니다. 그때 OBSERVABLE_NOW 대상 방위 ±15° 의 ROI 를 먼저 처리합니다.
- 그 밖의 경우 결과는 원래와 같습니다 (U15).
- ROI, 모델 점수, 표지 거리, 관측 가능성은 `live2_e4_link` 상태에 기록합니다.
- **기하 STOP 은 보류했습니다.** 실제 Odometry frame 계약이 없기 때문입니다.
- **팬틸트 명령은 보내지 않습니다** (`pantilt=NOT_COMMANDED`).

**웹 (PHASE 5)**
- 전역 위치는 항상 **UNKNOWN** 입니다. E6/E7 이 연결되지 않았기 때문입니다.
- Top-K 표에는 "미검증 가설" 표시가 붙습니다.
- 그 밖의 표시 항목:
  - 층별 썸네일과 후보 마커
  - E3 점수 추이와 이벤트
  - E4 상태와 사유
  - 의미 필요와 대상
  - 단계별 처리시간·입력 신선도 표

## 3. 저장 입력 시험 결과 (S6 = sw 건물 6층, 380 s, 정답 6F 는 평가 전용)

| | run2 | run3 | run4 (최종) |
|---|---|---|---|
| 잘못된 CONFIDENT 행 | **14 (2F)** | 0 | **0** |
| 맞는 CONFIDENT 행 | 0 | 0 | 0 |
| 전 층 탐색 후 1위=정답층 비율 | 24% | 0% | **46%** |
| E3 top6 에 정답층이 든 비율 | 30% | 4% | **60%** |
| E4 시간 비율 (AMBIGUOUS) | 13% | 36% | 40% |
| E2 pass (s) | 483 | 414 / 403 / 141 | 400 / 396 |
| E3 키프레임 p95 | 610 ms | 2494 ms | **468 ms** |
| 마지막 상태 | CONFIDENT 2F (오판) | RECOVERY (입력 종료) | RECOVERY (입력 종료) |

학생 방식(pass 1 window 하나, twoway 와 wall_pass 순위)의 판정은 7F 였습니다. 정답 6F 는 8개 층 중 8위였습니다.

run4 의 E4 상태 전이 (`E4_AMBIGUITY_RESULTS.csv`):
1. 0 s: INITIALIZING
2. 201 s: SEARCH_INCOMPLETE
3. 542 s: AMBIGUOUS (5F 와 6F 가 0.035 m 안)
4. 589 s: 6F 가 1위가 됨. DOMINANT_UNVERIFIED 와 AMBIGUOUS 를 오감
5. 943 s: 3F 와 6F 가 경쟁
6. 995 s: replay 종료로 odom 이 끊겨 RECOVERY_REQUIRED (ODOMETRY_STALE). 이 RECOVERY 는 E2 재탐색을 일으키지 않았습니다 (조치 가능한 사유가 아님)

해석:
- S6 에서는 기하만으로 층을 가를 수 없습니다. 5F, 6F, 3F 의 구조가 사실상 같습니다.
- E4 는 이를 확정하지 않고 AMBIGUOUS + 의미 HIGH 로 올바르게 보고했습니다.
- 정답 6F 에는 의미지도(S6F, 호실 616/610/614/608 표지, 계단)가 있어 의미 대상이 나왔습니다.
- 다른 sw 층은 의미지도가 없어 NO_SEMANTIC_MAP 입니다. KCI 는 9개 층 모두 의미지도가 있습니다.

정지 유지 (run2): 2분 동안 점수와 키프레임 수가 변하지 않았습니다 (`stationary_A/B.json`).

## 4. 발견한 결함과 수정

| # | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | E2 CPU 305 s/층 | wall-fit 의 무작위 gather 와 sort | 비트 동일 `wallfit_cpu_fast` (2.4×) |
| 2 | run1 거짓 RECOVERY | replay 의 odom 간격 3 s | replay odom heartbeat 0.1 s |
| 3 | 같은 pass 의 다른 후보가 병합됨 | MERGE 1.5 m | pass 간 병합만, 학생 DUP 값 사용 |
| 4 | **run2 잘못된 CONFIDENT 2F** | 고정된 E2 변환이 경로를 따라 드리프트 | 가설별 제한 추적과 1위 안정 15 kf 조건 |
| 5 | run3 재탐색 중 INITIALIZING 으로 회귀 | pass_id>1 을 처리하지 않음 | 재탐색 중에도 기존 가설로 계속 판단 |
| 6 | run3 stale 입력이 E2 재탐색 유발 | RECOVERY 사유를 구분하지 않음 | 조치 가능한 사유만 재탐색을 요청 |
| 7 | run3 키프레임 p95 2.5 s | 추적 조합이 많음 | coarse-to-fine 40 조합, 보류 가설은 4회에 1번 |
| 8 | **학생 venv 에 .pyc 15개 생성** (15:52) | `python -I` 가 PYTHONDONTWRITEBYTECODE 를 무시 | 목록 저장 후 삭제 (`evidence/venv_pyc_created_by_live2.txt`). e2, 시험, check 실행에 `-B` 추가. 이후 신규 0 |
| 9 | 백그라운드 대기 셸 3개가 무한 대기 | `pgrep -f replay_feed.py` 가 대기 셸 자신의 명령줄과 일치 | 실제 replay 종료를 확인한 뒤 대기 셸만 정지. `tools/wait_module.py` (owner PID + /proc starttime, timeout, 신호 없음) |

결함 9 의 처리 내용:
- 실제 run4 모듈(replay rc 0, e2 run-s, e34 rc 0, web MAX_DURATION)은 이미 정상 종료한 상태였습니다.
- pkill 이나 killall 은 쓰지 않았습니다. FAST-LIO, 센서, 연구 프로세스도 건드리지 않았습니다.

## 5. 자원·지연 (`RESOURCE_AND_LATENCY.csv`)

- MemAvailable 최저 9454 MB, 온도 최고 75.0 °C, e2 RSS 최대 1.44 GB (2 프로세스 합), e34 RSS 최대 332 MB 였습니다.
- HELD 와 floor_failed 는 0건이었습니다.
- E2 층당 시간 (CPU, 2 프로세스): 1–5F 는 55–78 s, 6–8F 는 118–190 s 입니다. 9개 층 기준 한 pass 는 약 7 분입니다.
- E3 키프레임 처리 p50 172 ms, p95 468 ms. E4 판단 p50 28 ms, p95 50 ms.
- 이 수치는 REPLAY 기준입니다. FAST-LIO 와 SigLIP 을 동시에 돌린 실측은 없습니다.
  - 코어는 분리했습니다: FAST-LIO, hub 0–3 / lio2 4 / e34 5 / e2 6,7.

## 6. 보존 확인

**바뀌지 않은 곳**
- `sci_ws/src`, `config`, `maps`, `models`, `jsh_onboard_robot_20261001`, `go2_ws`, LIVE1 원본은 15:16 이후 새 파일이나 수정이 0입니다.
- LIVE1 원본은 965개 파일의 sha 가 시작 기록과 일치합니다.
- ROPUS 서비스 변경 없음, 패키지 설치 없음.

**제어 명령**
- 팬틸트, 초점, HOME, Go2 제어 명령은 0건입니다.

**sci_ws/external**
- 결함 8 의 .pyc 를 지운 디렉터리 4곳의 mtime 만 바뀌었습니다. 남은 파일은 없습니다.

## 7. 한계

1. 실기동 시험이 0회입니다. 승인이 없었기 때문입니다.
2. E2 입력 `structure_proxy` 는 학생 a1_3_1 구조점과 현장에서 비교하지 않았습니다.
   - REPLAY 는 학생 구조점을 썼습니다. 실기동에서 frame_probe 로 비교해야 합니다.
3. CPU E2 한 pass 는 8–9개 층에 7–8 분이 걸립니다.
   - 처음 몇 분은 SEARCH_INCOMPLETE 입니다.
   - GPU 를 쓰면 줄어들 수 있지만 `e2_gpu` 승인이 따로 필요하고, SigLIP 의 GPU 기준과 충돌하지 않는지 확인해야 합니다.
4. 기하만으로 층을 가를 수 없는 건물(S6 처럼)에서는 의미 관측(E5)이 필수입니다. 의미지도가 없는 층은 대상을 낼 수 없습니다.
5. 실제 Odometry frame 계약이 없어 기하 STOP 은 보류했습니다. 팬틸트 자동 명령도 보내지 않았습니다.
6. E6/E7 이 연결되지 않아 전역 위치는 UNKNOWN 으로 고정했습니다. 이는 설계 의도입니다.

## 8. 실기동 승인 요청 (한 번에)

아래 범위를 승인해 주시면 `record-approval` 로 기록한 뒤 `freedrive --building <건물>` 로 실주행 시험을 하겠습니다.

**승인 A (필수)**
- 모듈: `fastlio, lio2 (/cloud_registered 축적 포함), hub, observe (SigLIP SHADOW, GPU 작업자 1), web, e2 (CPU), e34, frame_probe`
- 실행 한도: 각 1800 s (web 3600 s, frame_probe 120 s). 승인 유효시간 60 분. 현재 boot `dde2c30f…` 에 묶이며, 재부팅하면 무효입니다.
- 건물: kci (9개 층, 의미지도 있음) 권장. sw 도 가능합니다.

**운용자 C**
- 운용자가 조이스틱으로 자유 주행합니다. 처음 1–2 분은 직진과 회전을 섞어 주십시오.
- 시스템은 로봇 제어 명령을 보내지 않습니다.

**선택**
- `e2_gpu`: E2 를 GPU 로 돌립니다. SigLIP GPU 기준(72/80 °C, 4300/2500 MB)은 그대로 적용합니다.
- 승인하지 않으면 CPU 만 씁니다.

**계속 금지하는 것**
- 팬틸트, 초점, HOME, `/cmd_vel`, ROPUS 재시작, 원시 `/lidar_points` 추가 구독

## 9. 산출물

- `LIVE_FIELD2_REPORT_ALL.md` (이 문서)
- `LIVE_FIELD2_TEST_MATRIX.csv` (40행)
- `E2_TOPK_RESULTS.csv` (482행)
- `E3_INCREMENTAL_RESULTS.csv` (2119행)
- `E4_AMBIGUITY_RESULTS.csv` (1054행)
- `RESOURCE_AND_LATENCY.csv` (7262행)
- `LIVE_FIELD2_RESPONSE_BUNDLE.zip`: 보고서, CSV, candidate 코드(live2 전체와 수정된 live1), diffs, tests, tools, evidence, run 요약
- `SHA256SUMS.txt`
