# LIVE-FIELD2 운용자 웹 콘솔 — 구현 보고 (2026-10-18)

- 작업공간: `/home/unitree/sci_ws/repairs/livefield2_20261018_151606` (candidate 에만 변경)
- 연구 원본과 학생 운영판은 변경하지 않았습니다.

## 1. 웹 주소

- `http://192.168.0.10:8770`
- `http://192.168.123.18:8770`
- Tailscale `http://<ROBOT_TAILSCALE_IP>:8770`

콘솔은 Claude Code 가 승인 A 를 기록한 뒤 `ctl start console` 로 띄웁니다. 승인이 없으면 기동이 거부됩니다 (2026-10-18 20:23 실제 확인: `APPROVAL_EXPIRED`).

## 2. 구현 상태

| 기능 | 상태 |
|---|---|
| 실험 시작 (모듈 순서 기동, 단계별 health 확인) | 구현, FAKE 검증 |
| 일시정지 / 재개 (E2·E3 분석 보류, SigLIP 정지·재기동) | 구현, FAKE 검증 |
| 실험 종료 및 기록 저장 (소유 모듈만 정지, 상태 보관, 표시 초기화) | 구현, FAKE 검증 |
| E2 초기 후보 계산 / E2 재탐색 요청 | 구현, FAKE 검증 |
| SigLIP 켜기 / 끄기 | 구현, FAKE 검증 |
| 미기동 모듈 재시도 (부분 기동 후) | 구현, FAKE 검증 |
| 모듈 상태 · 승인 잔여시간 · 안전 조건 · 자원 | 구현 (모든 모듈 표에 승인·실험 단계·프로세스·감독·사유) |
| 실시간 패널 (FAST-LIO, 축적 지도·궤적, E2 Top-K, E3, E4 근거, SigLIP 영상·후보·점수, 최신성) | 기존 패널을 그대로 사용 |
| 실험이 끝나면 과거 데이터를 LIVE 로 표시하지 않음 | 구현. 종료 시 상태를 보관 폴더로 옮겨 DATA_UNAVAILABLE 로 표시하고, 배너에 "실험 없음" 을 띄웁니다 |
| GT 클릭 저장 (시작 위치 / 확인 지점, 선택적 방향) | 구현, FAKE 검증 (실제 kci 의미지도 사용) |
| 검출 평가 (올바른 / 오검출 / 놓친 / 애매) | 구현, FAKE 검증 |
| 웹 승인 UI | **미구현 (설계만)**. 승인은 지금도 Claude Code 의 `record-approval` 로만 기록합니다 |

## 3. 승인과 인증

**승인**
- 기존 승인 A 기록을 그대로 씁니다: 사용자 원문 해시, boot_id, 만료 시각, 모듈 목록, 모듈별 실행 한도.
- 콘솔은 사전 점검과 표시만 합니다. 실제 판정은 매번 ctl 의 게이트가 합니다.
- 승인되지 않은 모듈, 만료된 승인, 다른 boot 의 승인, 실행 한도 초과는 모두 거부하고 이유를 표시합니다.

**인증**
- 운용자 토큰: `runtime/ctl/operator_token` (0600, `ctl console-token` 으로 다시 만들 수 있음)
- 로그인하면 HttpOnly·SameSite=Strict 세션 쿠키가 발급되고, 세션마다 CSRF 토큰 헤더를 요구합니다.
- 토큰을 5회 틀리면 60 s 동안 잠깁니다.

**요청 검사**
- Origin / Sec-Fetch-Site 검사로 교차 출처 요청을 거부합니다. Host 허용 목록으로 DNS rebinding 을 막습니다.
- JSON 본문만 받습니다.
- 모든 제어 요청에 request_id 가 필요합니다. 같은 ID 는 다시 실행하지 않고 이전 결과를 돌려줍니다.
- 제어 작업은 한 번에 하나만 실행하고, 나머지는 `BUSY` 로 거부합니다.
- 실험 세션은 한 세션만 소유합니다. 다른 세션은 423 을 받고, 제어권 인수에는 토큰이 다시 필요합니다.

**감사 기록**
- `runtime/ctl/audit.jsonl` 에 시각, IP, 세션 해시, action, request_id, 결과, 거부 사유를 남깁니다. 토큰은 기록하지 않습니다.

**한계**
- HTTP 평문이라 같은 LAN 에서는 토큰을 엿볼 수 있습니다. Tailscale 주소로 접속하기를 권장합니다. HTTPS 는 미구현입니다.

## 4. 시작·정지·재개 동작

**시작**
1. 이전 상태 파일을 `experiments/<id>/pre_start_leftovers` 로 옮깁니다. 옛 데이터가 LIVE 로 보이지 않게 하기 위해서입니다.
2. 각 모듈을 `ctl start <m>` (고정 argv) 로 기동합니다.
3. 모듈마다 실제 health 를 확인합니다.
   - FAST-LIO: map kdtree 초기화
   - lio2: odometry 수신
   - hub: index 갱신
   - SigLIP: keeper RUNNING
   - E4·E2: 상태 파일 기록

**일시정지**
- `state/control/pause.json` 을 씁니다.
- E2 는 새 pass 와 새 층 정합을 보류합니다. 이미 돌고 있는 층은 끝까지 합니다.
- E3 는 새 키프레임을 평가하지 않고, 재탐색 요청도 보내지 않습니다.
- SigLIP 은 정지합니다.
- FAST-LIO·lio2·hub 는 계속 동작합니다. Go2 에는 아무 명령도 보내지 않습니다.

**재개**
- pause 를 해제하고, 필요하면 SigLIP 을 재기동합니다. 이때도 승인과 게이트를 다시 통과해야 합니다.

**종료**
- observe → e2 → e34 → rec → hub → lio2 → fastlio 순서로 `ctl stop` 합니다.
- 소유권은 PID + /proc starttime 으로 확인합니다. PID 가 재사용되었으면 신호를 보내지 않습니다.
- 콘솔 자체는 살아 있습니다.

## 5. 정상 종료와 실패 복구

**정상 종료**
- 실험이 ENDED 가 되고 `experiments/<id>/` 에 다음을 남깁니다: 상태 전체, 감독 로그, sup_state, timeline, experiment.json

**시작 실패**
- 핵심 모듈이 아닌 모듈(hub, rec, observe, e34, e2)이 실패하면 PARTIAL 이 됩니다. 실패 사유(ctl 게이트 결과)를 표시하고, 나머지 모듈은 계속 기동합니다. "미기동 모듈 재시도" 로 복구할 수 있습니다.
- 핵심 모듈(fastlio, lio2)이 실패하면 그 뒤 모듈은 SKIPPED 처리하고 기동하지 않습니다.

**실행 중 종료**
- 온도·메모리 중단, 실행 한도 도달, 충돌로 모듈이 끝나면 2 s 안에 `ENDED_UNEXPECTEDLY` 와 감독의 종료 사유를 표시하고 실험을 PARTIAL 로 바꿉니다.
- 이 기능은 FAKE 시험 중 발견해 추가했습니다. 2026-10-18 실주행에서 lio2 가 80 °C 로 중단된 것과 같은 상황입니다.

**자동 종료**
- 승인이 만료되거나 boot_id 가 바뀌면 자동으로 종료하고 보관합니다 (`AUTO_END APPROVAL_EXPIRED`).

**콘솔이 죽었을 때**
- 콘솔은 supervisor 밖에서 실행합니다. live1_supervisor 는 자기가 본 모든 후손을 정리하기 때문입니다.
- 따라서 콘솔이 종료되어도 모듈은 각자의 한도로 계속 동작합니다.
- 콘솔을 다시 띄우면 기존 실험 상태를 PARTIAL 로 보여 줍니다. 제어하려면 토큰으로 제어권을 인수합니다.

## 6. 실험 기록 경로

| 내용 | 경로 |
|---|---|
| 실험 상태(현재) | `runtime/ctl/experiment.json` |
| 감사 기록 | `runtime/ctl/audit.jsonl` |
| 실험별 기록 | `experiments/<X날짜_시각>/` (timeline.jsonl, experiment.json, state/, *.sup.jsonl, *.sup_state.json) |
| GT | `evaluation/<실험 id>/gt.jsonl` |
| 검출 평가 | `evaluation/<실험 id>/detections.jsonl`, `snapshots/<id>.jpg/.json` |
| 실험 밖에서 남긴 평가 | `evaluation/NO_EXPERIMENT_<날짜>/` |

**GT 기록 항목**
- px, py, 지도 m 좌표(의미지도 해상도 기준), 선택적 방향
- 지도 파일과 sha256, 그 순간의 odom
- `source=HUMAN_CLICK`, `used_by_inference=false`

**검출 기록 항목**
- run_id, frame (`scene_stamp`, `n_scene`, stamp 로 맞춘 hub seq), SigLIP 상자·점수·threshold, odom
- 이미지와 status 파일의 시각 차와 일치 여부
- `review=HUMAN_LABELED`
- "애매한 관측" 은 `ground_truth_confirmed=false` 입니다. 사람이 고정한 프레임만 평가 대상이 됩니다.

**격리 확인**
- E2–E7 과 E5 코드 어디에서도 evaluation 저장소 경로를 참조하지 않습니다. 시험으로 검사합니다.

## 7. 검증

**FAKE 검증: `tests/test_console_fake.py` 18건 통과**
- 진짜 ctl, 진짜 live1_supervisor, 진짜 콘솔을 쓰고, 모듈 자식만 fake 입니다.
- 승인 관련: 승인 없음, 웹 승인만 있음, 미승인 모듈, 승인 만료, 다른 boot, 실행 한도 초과
- 보안 관련: CSRF, 교차 출처, 비 JSON, request_id 누락·중복, 허용되지 않은 action, Host 위조
- 실험 흐름: 중복 시작, 다른 세션 제어 거부, 순서 기동과 소유권, 일시정지·재개, SigLIP 토글, E2 재탐색
- 기록: GT·검출 평가 저장소
- 실패와 복구: 종료·보관, 시작 실패 PARTIAL·복구, 기동 직후 종료 감지, 핵심 실패 시 의존 모듈 SKIPPED, 승인 만료 시 자동 종료
- 소유권: PID 재사용 시 신호 없음, 콘솔이 종료되어도 모듈 유지

**FAKE 모드 안전장치**
- `LIVE2_FAKE_WORKSPACE.json` 만 있는 tmp 작업공간에서만 켜집니다.
- 실제 작업공간에서는 꺼짐이 시험으로 확인됩니다. fake_module 도 실제 작업공간에서는 실행을 거부합니다.

**기존 단위시험**
- 18건 + E2 운동·일시정지 1건 = 19건 통과

**실제 LIVE 검증: 하지 않음**
- 승인 A 가 20:11:34 에 만료되었습니다. 실제 작업공간에서 콘솔 기동이 거부되는 것만 확인했습니다.

## 8. 함께 바꾼 연구 모듈 동작 (candidate)

- `e2_worker`
  - window 안의 odom 경로가 2 m 미만이면 `WAITING_MOTION` 으로 대기합니다. 2026-10-18 실주행에서 정지 상태로 4번의 pass 가 모든 층에서 실패하고 열만 낸 문제를 막기 위한 것입니다.
  - 운용자 일시정지(`HELD_PAUSED`, `PAUSED`)를 지원합니다.
- `e34_node`
  - 일시정지 중에는 새 키프레임을 평가하지 않고 재탐색 요청도 보내지 않습니다. E4 출력에 `paused` 를 표시합니다.

## 9. 미구현 항목

- 웹 승인 UI (설계만 함). 승인 기록 구조는 그대로 재사용할 수 있습니다.
- HTTPS
- 웹에서 frame_probe 실행
- 웹에서 E2 GPU 선택 (최초 실험은 CPU)
- 팬틸트·초점·HOME (이번 범위에서 금지)
- 실제 장치를 기동한 LIVE 검증

## 10. 다음 실험 조건

1. **새 승인 A** (현재 boot 에 한정)
   - 모듈: `web`(콘솔), `fastlio`, `lio2`(축적), `hub`, `observe`, `e2`, `e34` (선택: `frame_probe`)
   - 실행 한도와 유효시간을 함께 정해 주십시오.
2. Claude Code 가 승인을 기록하고 `ctl start console` 을 실행합니다.
3. 운용자가 브라우저에서 토큰으로 로그인한 뒤 **실험 시작**을 누릅니다.
4. 시작 시 최고 온도가 72 °C 미만이어야 합니다 (ctl 게이트).
   - 로봇 데스크톱에 웹 패널을 띄우지 마십시오. 2026-10-18 실주행에서 브라우저 탭이 CPU 코어 하나를 100% 쓰면서 80 °C 보호 중단의 원인이 되었습니다.
5. **로봇이 실제로 움직여야 E2 가 시작됩니다** (window 안에서 2 m 이상).
