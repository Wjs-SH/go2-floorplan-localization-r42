# LIVE-FIELD2 체크포인트 — 2026-10-18 작업 종료 (22:47 KST)

이 문서는 내일 작업을 다시 시작할 때 쓰는 기준 문서입니다.
기록된 boot_id, PID, 승인, Odometry 세션은 **오늘 boot 에서만 의미가 있습니다**.

## 1. 오늘 종료 시점 상태

| 항목 | 값 |
|---|---|
| 작업 경로 | `/home/unitree/sci_ws/repairs/livefield2_20261018_151606` |
| 실행 진입점 | `scripts/live2_ctl.sh` (candidate 의 `live1/ctl.py` 를 LIVE1_ROOT=작업 경로로 실행) |
| 마지막 boot_id | `677fbdc4-0454-4b6a-83ab-0d0545815510` (배터리 교체·재부팅 후) |
| 마지막 승인 A | `runtime/approvals/A_20261018_224209.json` — **2026-10-18 23:42:09 만료**, boot `677fbdc4` 전용. 내일은 무효이며 사용하지 않습니다 |
| 웹 콘솔 | 22:42 실제 기동, 22:47 정상 종료. 8770 포트 닫힘 |
| 태블릿 접속 | `http://<ROBOT_TAILSCALE_IP>:8770/` 에서 page_view 200 (22:44:27, tablet `tab-s8` <TABLET_TAILSCALE_IP>) |
| 실험 상태 | IDLE. 웹 [실험 시작]은 한 번도 눌리지 않았습니다 (`runtime/ctl/experiment.json` 없음) |
| FAST-LIO·SigLIP·E2/E3/E4 | 콘솔을 통해서는 미기동. 19:12–19:42 에 Claude Code 로 직접 기동한 필드 세션이 한 번 있었습니다 (§4) |
| 남은 LIVE2 프로세스 | 없음 (22:47 확인). ROPUS 4종(hesai, realsense, robot_state_publisher, go2_control_runner)은 각각 1개씩 동작 중이며 손대지 않았습니다 |
| 보호 대상 | LIVE1 원본 965개 파일 sha 일치. sci_ws/src·config·maps·models, jsh_onboard_robot, go2_ws, external 에 15:16 이후 새 파일 0개 |

## 2. 재부팅 후 반드시 지킬 것

- **이전 boot_id, PID, 승인 기록, Odometry 세션, 로그인 토큰을 재사용하지 않습니다.**
- 승인 기록은 boot_id 에 묶여 있어 재부팅하면 `APPROVAL_FOR_OTHER_BOOT` 로 자동 무효가 됩니다.
  - 새 승인 A 를 사용자 원문으로 다시 기록해야 합니다.
- 소유 프로세스 기록(`runtime/owned/*.json`)은 PID 와 /proc starttime 이 함께 맞아야 "살아 있음" 으로 판단합니다.
  - 재부팅 후에는 모두 dead 로 판정되며 신호를 보내지 않습니다. 지우지 말고 그대로 두십시오.
- Odometry 세션은 FAST-LIO 를 새로 시작할 때마다 새로 생성됩니다 (`FIRST_ODOMETRY`). 이전 E2/E3 후보는 이어받지 않습니다.
- 운용자 토큰은 콘솔을 기동할 때마다 `console-token` 으로 새로 발급합니다.

## 3. 내일 실행 순서

```bash
W=/home/unitree/sci_ws/repairs/livefield2_20261018_151606
C=$W/scripts/live2_ctl.sh
```

1. **배터리 교체와 부팅 확인** (읽기만 합니다)
   - `cat /proc/sys/kernel/random/boot_id` — 새 boot 인지 확인
   - `$C status` — 모든 모듈이 alive=False 인지 확인
   - `$C check --graph`
     - ROPUS 4종이 각각 1개
     - `/lidar_points` 발행 1 / 구독 0, `/Odometry` 0
     - 최고 온도 72 °C 미만, MemAvailable 4300 MB 이상
   - `tailscale status` — 로봇 `ubuntu` <ROBOT_TAILSCALE_IP> 과 태블릿 `tab-s8` 가 active 인지 확인
   - **로봇 데스크톱의 Firefox 를 닫습니다.** 오늘 브라우저 탭이 CPU 1코어를 100% 쓰면서 80 °C 보호 중단의 원인이 되었습니다.
2. **새 승인 A 기록** — 사용자가 채팅에 쓴 원문만 씁니다.
   ```bash
   $C record-approval A --text-file <원문 파일> --minutes 60 --modules web,fastlio,lio2,hub,observe,e2,e34,rec \
     --limits '{"web":{"max_run_s":3600},"fastlio":{"max_run_s":1800},"lio2":{"max_run_s":1800,"accum_cloud_registered":true},
                "hub":{"max_run_s":1800},"observe":{"max_run_s":1800},"e2":{"max_run_s":1800},"e34":{"max_run_s":1800}}'
   ```
3. **콘솔만 기동**
   - `$C console-token` — 새 토큰
   - `$C start console --run-s 3500`
   - 확인 항목:
     - `ss -ltnp | grep 8770` → `0.0.0.0:8770` LISTEN
     - `curl http://<ROBOT_TAILSCALE_IP>:8770/api/session` → 200
4. **태블릿 접속**
   - URL: **`http://<ROBOT_TAILSCALE_IP>:8770/`** (또는 `http://<ROBOT_TAILNET_HOST>:8770/`). 반드시 `http://` 로 입력합니다.
   - `runtime/ctl/audit.jsonl` 에 tablet IP 의 `page_view` 가 들어오는지 확인합니다.
5. **로그인** → 건물 `kci` 선택, SigLIP 사용 여부 선택 → **[실험 시작]**
   - 순서: fastlio → lio2(축적) → hub → rec → observe → e34 → e2
   - 단계마다 실제 health 를 확인합니다.
   - 실패하면 사유를 표시하고 자동 재시도는 하지 않습니다.
6. **FAST-LIO 초기화** — "map kdtree" 가 뜰 때까지 로봇을 10–20 s 세워 둡니다.
7. **조이스틱 자유주행** — 사용자가 직접 합니다. AI 는 이동 명령을 보내지 않습니다.
   - E2 는 window 안에서 2 m 이상 움직여야 시작합니다 (`WAITING_MOTION`).
   - 처음 1–2 분은 직진과 회전을 섞어 주십시오.
   - CPU E2 한 pass 는 9개 층에 약 7–8 분 걸립니다.
8. **관찰 항목** (웹 화면)
   - E2 Top-K
   - E3 갱신(키프레임, ACTIVE/SUSPENDED)
   - E4 상태와 근거
   - SigLIP 프레임과 점수
   - 신선도, 온도
   - 필요하면 [일시정지]/[재개], GT 클릭, 검출 평가
9. **[실험 종료·저장]**
   - LIVE2 소유 모듈만 정지합니다.
   - 기록은 `experiments/<X날짜_시각>/` 에, 평가 자료는 `evaluation/<id>/` 에 남습니다.
10. **마무리 (Claude Code)**
    - `$C stop all`
    - `pgrep -af 'console.py|fastlio_mapping|live2/|live1/'` 결과가 없는지 확인
    - 결과 CSV 생성: `tools/make_live2_csv.py`

## 4. 오늘 확인된 사실과 주의점

- **19:12–19:42 실기동** (승인 A_20261018_191134, 이전 boot)
  - FAST-LIO 9.9–10.3 Hz, RGB 8 Hz, depth 14 Hz, SigLIP LIVE 정상
  - 로봇이 거의 움직이지 않아 (1.4–1.7 m) 학생 calibration 이 모든 층에서 실패했습니다 (`CalibrationUnavailable`)
    - E2 pass 4회 모두 후보 0개
    - E4 는 `RECOVERY_REQUIRED (E2_ALL_FLOORS_FAILED)` 로 정직하게 표시했습니다
  - **온도 80 °C 도달**
    - lio2 가 supervisor 보호로 중단됐습니다 (ABORT_TEMP)
    - FAST-LIO 를 보호하기 위해 e2 와 observe 를 운용자 판단으로 정지했습니다
  - 기록: `runtime/state_FIELD_20261018_1912/`
- 이 경험을 반영해 수정한 내용
  - E2 `WAITING_MOTION` (window 경로 2 m 미만이면 대기)
  - 콘솔 모듈 생존 감시 (`ENDED_UNEXPECTEDLY` 이면 PARTIAL)
  - MagicDNS 호스트 허용
  - page_view 접속 로그
- **열 여유가 작습니다.** 오늘 대기 상태에서도 62–70 °C 였습니다.
  - realsense + FAST-LIO + hub + SigLIP 을 함께 돌리면 79–80 °C 까지 갑니다.
  - 온도가 79 °C 에 가까워지면 웹에서 [SigLIP 끄기] 를 먼저 누르는 것을 권장합니다.
- 실제 Odometry frame 계약이 없어 기하 STOP 은 보류 상태입니다. 팬틸트·초점·HOME 은 금지입니다.
- 전역 위치는 E6/E7 이 연결되지 않아 계속 UNKNOWN 입니다.

## 5. 보존된 결과와 파일

| 내용 | 경로 (작업 경로 기준) |
|---|---|
| 코드 (수정본) | `candidate/live2/*.py` (console, e2_worker, e34_node, e3_tracker, e4_ambiguity, …), `candidate/live1/ctl.py`, `live1_common.py`, `live1_web.py`, `observe_node.py`, `web/index.html` |
| 코드 해시 (종료 시점) | `manifests/candidate_code_sha256_20261018_end.txt` (24개 파일) |
| 기준선 대비 diff | `diffs/live1_baseline_to_candidate.diff` (기준선 `baseline/` 은 읽기 전용) |
| 보고서 | `LIVE_FIELD2_REPORT_ALL.md`, `LIVE_FIELD2_CONSOLE_REPORT.md`, `LIVE_FIELD2_START_STOP.md` |
| 결과 CSV | `E2_TOPK_RESULTS.csv`, `E3_INCREMENTAL_RESULTS.csv`, `E4_AMBIGUITY_RESULTS.csv`, `RESOURCE_AND_LATENCY.csv`, `LIVE_FIELD2_TEST_MATRIX.csv` |
| 번들·해시 | `LIVE_FIELD2_RESPONSE_BUNDLE.zip`, `SHA256SUMS.txt` |
| REPLAY 결과 | `runtime/state_REPLAY_run1`–`run4`, `evidence/replay_runs_summary.json` |
| 실기동 결과 | `runtime/state_FIELD_20261018_1912/`, `reports/freedrive_session_20261018_201123.json` |
| 승인 기록 | `runtime/approvals/` (원문 txt, sha256 이 담긴 json) |
| 콘솔 감사 기록 | `runtime/ctl/audit.jsonl` (12행: 21:20 보안 점검, 22:44 태블릿 page_view) |
| 로그 | `logs/` (모듈별 `.out`, `.sup.jsonl`) |
| 시험 | `tests/test_live2.py` (19건), `tests/test_console_fake.py` (FAKE 18건), 실행은 `tools/run_live2_tests.sh` |
| 증거 화면 | `evidence/web_REPLAY/`, `evidence/web_CONSOLE_FAKE/` |

## 6. 남은 과제

1. 웹 콘솔로 실제 실험을 한 번 끝까지 진행합니다 (내일).
2. 실주행에서 E2 후보 생성, E3 갱신, E4 모호성 판단, SigLIP 병행 동작을 확인합니다.
3. frame_probe 로 Odometry frame 계약 초안을 만듭니다 (별도 승인 필요, 보정값은 자동 확정하지 않음).
4. `structure_proxy` 와 학생 a1_3_1 구조점을 현장에서 비교합니다.
5. 웹 승인 UI 와 HTTPS 는 미구현입니다.
