# LIVE-FIELD2 시작·중지 절차

단일 진입점은 `/home/unitree/sci_ws/repairs/livefield2_20261018_151606/scripts/live2_ctl.sh` 입니다. LIVE1 ctl 사본에 E2/E3/E4 모듈을 더했습니다.

- `check` 와 `status` 는 아무것도 기동하거나 중지하지 않습니다.
- 실기동 모듈은 이번 boot 에 기록된 사용자 승인 A 의 모듈 목록과 시간 한도를 통과해야 합니다.
- 승인이 끝나는 시각을 넘어서 실행하지 않습니다.

## 1. 모듈

| 모듈 | 하는 일 | 구독 / 자원 | 승인 |
|---|---|---|---|
| fastlio | 단일 SCI FAST-LIO. LIVE1 과 같은 빌드와 설정이며, Log 는 네임스페이스로 우회합니다 | `/lidar_points` 를 구독하는 유일한 프로세스 | 필요 |
| lio2 | Odometry 상태, 2D 축적, **E2 입력 생성** | `/Odometry`, `/cloud_registered` 두 가지만 (LIVE1 A-acc 와 같음). 코어 4 | 필요 |
| hub | RGB, depth, camera_info, Odometry 를 공유 링에 씁니다 | 카메라 3개, `/Odometry`. 코어 2,3 | 필요 |
| observe (keep) | SigLIP SHADOW, GPU 작업자 1개. **E4 파일 입력이 연결됩니다** | hub 링만 읽습니다. 코어 0,1. 72/80 °C, 4300/2500 MB 기준 | 필요 |
| e2 | 학생 metric_fit_v2 의 내부 후보로 건물 전체 Top-K 를 만듭니다. CPU 전용 | 파일만 다룹니다. 코어 6,7, nice 10. 층 하나만 메모리에 둡니다(~1.2 GB × 2) | 필요 (`e2`). GPU 는 `e2_gpu` 가 따로 필요 |
| e34 | E3 후보 갱신과 E4 모호성 판단, E5 로 보내는 피드 | 파일만 다룹니다. 코어 5 | 필요 |
| web | 읽기 전용 패널 `0.0.0.0:8770`. POST 는 405 | 파일만 다룹니다 | 필요 |
| rec | 현장 기록 (LIVE1 과 같음) | 파일만 다룹니다 | 불필요 |
| replay | **저장 입력 시험 전용**: 녹화 데이터셋을 LiveStore 로 재생합니다 | 파일만 다룹니다. `state_REPLAY` 와 `/dev/shm/live2_store_replay` 에만 씁니다 | 불필요 |

`e2`, `e34`, `web` 에 `--test-only` 나 `--replay` 를 주면 REPLAY 트리만 사용하고 화면에 TEST_ONLY 로 표시됩니다.

## 2. 실기동 순서 (승인 기록 후)

```bash
C=/home/unitree/sci_ws/repairs/livefield2_20261018_151606/scripts/live2_ctl.sh
$C record-approval A --text-file <사용자 원문 파일> --minutes 60 \
   --modules fastlio,lio2,hub,observe,web,e2,e34,frame_probe \
   --limits '{"fastlio":{"max_run_s":1800},"lio2":{"max_run_s":1800,"accum_cloud_registered":true},"hub":{"max_run_s":1800},
              "observe":{"max_run_s":1800},"web":{"max_run_s":3600},"e2":{"max_run_s":1800},"e34":{"max_run_s":1800},
              "frame_probe":{"max_run_s":120}}'
$C check --graph                      # /lidar_points 구독자 0, /Odometry 0 인지 확인
$C freedrive --building kci           # fastlio → lio2 → hub → web → rec → observe(keep) → e34 → e2
#   로봇은 FAST-LIO IMU 초기화 동안 서 있습니다. 이후 운용자가 조이스틱으로 자유롭게 이동합니다.
#   E2 첫 정합은 표본 70 s 가 쌓인 뒤 시작합니다. 첫 1–2 분은 직진과 회전이 섞이도록 움직여 주십시오.
$C status
$C freedrive-end                      # LIVE2 소유 프로세스만 정지(PID + starttime 확인). 기록은 디스크에 남습니다
```

## 3. 저장 입력 시험 (승인 불필요, 이번 작업에서 실제로 실행)

```bash
$C start replay --data /home/unitree/jsh_onboard_robot_20261001/testdata/S6 --rate 0.5 --hold-end-s 600 --run-s 1500
$C start e2 --test-only --building sw --procs 2 --run-s 2400
$C start e34 --test-only --building sw --run-s 2400
$C start web --replay --host 127.0.0.1 --run-s 2400
python3 /home/unitree/sci_ws/repairs/livefield2_20261018_151606/tools/wait_module.py replay e2 e34 --timeout-s 3000
$C stop all
```

종료를 기다릴 때는 `wait_module.py` 를 씁니다. 이 도구는 owner 기록의 PID 와 /proc starttime 을 비교하고, 제한 시간이 지나면 rc 2 로 끝나며, 어떤 신호도 보내지 않습니다.
`pgrep -f replay_feed.py` 로 기다리지 마십시오. 대기하는 셸의 명령줄에도 같은 문자열이 들어 있어서 셸이 자기 자신을 찾아내고, 대기가 끝나지 않습니다.
이 문제가 2026-10-18 run 3·4 대기 중에 실제로 일어났습니다.

## 4. 금지

LIVE1 과 같습니다.
- `kci_field_up.sh` 와 `kci_field_down.sh`
- ROPUS 서비스 stop/restart
- `rt_live_node.py` (두 번째 원시 구독자)
- `/lidar_points` 를 echo/hz 로 보는 것
- 팬틸트, 초점, HOME, `/cmd_vel`, `/api/sport/request`
- 패턴으로 프로세스를 종료하는 것

## 5. 운용자 웹 콘솔 (2026-10-18 추가)

8770 웹에서 운용자가 실험을 직접 시작·일시정지·재개·종료합니다. 화면은 기존 패널에 제어 카드, GT 카드, 검출 평가 카드를 더한 것입니다.

### 준비 (Claude Code, 승인 A 가 유효할 때만)

```bash
C=/home/unitree/sci_ws/repairs/livefield2_20261018_151606/scripts/live2_ctl.sh
$C record-approval A --text-file <사용자 원문> --minutes 60 --modules fastlio,lio2,hub,observe,web,e2,e34,frame_probe --limits '{...}'
$C console-token                  # 운용자 토큰 (runtime/ctl/operator_token, 0600). 필요할 때마다 새로 만들 수 있습니다
$C start console --run-s 3500     # 'web' 승인으로 기동합니다. supervisor 밖에서 detached 로 실행됩니다
```

### 운용자 (브라우저 http://<로봇 IP>:8770)

1. 토큰으로 로그인합니다.
2. 건물을 고르고 SigLIP 사용 여부를 정한 뒤 **실험 시작**을 누릅니다.
   - 순서: fastlio → lio2(축적) → hub → rec → observe → e34 → e2
   - 각 모듈은 실제 health 신호를 확인한 다음에 다음 모듈로 넘어갑니다.
3. 조이스틱으로 자유 주행합니다.
   - E2 는 window 안에서 2 m 이상 움직여야 계산을 시작합니다 (`WAITING_MOTION`).
4. **일시정지**: E2/E3 의 새 분석을 보류하고 SigLIP 을 정지합니다. FAST-LIO·축적·hub 는 계속 동작합니다.
5. **재개**: 일시정지를 풀고, 일시정지 전에 SigLIP 이 켜져 있었다면 다시 켭니다.
6. **실험 종료·저장**: LIVE2 소유 모듈만 정지합니다 (PID+starttime 확인). 그 뒤 `experiments/<id>/` 에 보관하고 표시를 비웁니다.

### 동작 규칙

- 콘솔이 종료되어도 실험 모듈은 멈추지 않습니다. 각 모듈은 자기 실행 한도로 끝납니다.
- 승인이 만료되거나 boot 가 바뀌면 콘솔이 실험을 자동으로 종료하고 보관합니다.

### 할 수 없는 것

- 임의 명령, 경로 입력
- 승인 기록과 승인 연장
- 팬틸트, 초점, HOME, Go2 이동, ROPUS 조작
- 원시 `/lidar_points` 구독
