# LIVE-FIELD2 자동 실행 · 단축명령어 · 웹 승인 (2026-10-19 21:00)

## 1. 단축명령어 `live2` (설치됨)
`~/.local/bin/live2` → `scripts/live2` → `scripts/live2_cli.py`
- 이미 검증된 `scripts/live2_ctl.sh` 를 그대로 사용합니다.
- 다루는 것은 웹 콘솔뿐이고, 연구 모듈은 시작하지 않습니다.

| 명령 | 동작 |
|---|---|
| `live2 start` | 웹(8770)만 기동합니다 (서비스가 있으면 서비스로). 이미 실행 중이면 중복 실행하지 않습니다 |
| `live2 status` | boot_id, 웹 PID·포트, 서비스 상태, Tailscale 주소, 승인 A, 실험 상태, 연구 모듈, 온도·메모리를 보여 줍니다 |
| `live2 stop` | 웹 서비스와 콘솔을 정지합니다. **실험 진행 중이거나 연구 모듈이 살아 있으면 거부**하고, 태블릿에서 [실험 종료·저장] 하라고 안내합니다 |
| `live2 restart` | 웹 콘솔만 재시작합니다. 연구 모듈은 콘솔과 분리되어 있어 영향이 없습니다. 모듈을 켜거나 끄는 중(STARTING/STOPPING)에는 거부합니다 |
| `live2 logs [-n N] [-f]` | 웹 콘솔 로그, 자동 실행 감시 로그, 웹 요청 기록, 실험 사건을 보여 줍니다 |

## 2. 부팅 시 웹 자동 실행 (설치·활성화됨)
**`~/.config/systemd/user/live2-console.service`** (enabled, active)
- systemd user 서비스입니다.
- **linger=yes** 가 이미 켜져 있어 로그인 없이 부팅 때 시작됩니다. sudo 나 시스템 변경은 필요 없었습니다.
- `live2 serve` 가 Tailscale 주소를 최대 180 s 기다린 뒤 웹 콘솔만 기동합니다. 웹이 끝나면(12 h 상한, 장애) 웹만 다시 켭니다.
  - Tailscale 보다 먼저 떴다면, 유휴 상태에서 한 번 재기동해 tailnet 기기를 인식하게 합니다.
- `KillMode=process`: 서비스를 정지·재시작해도 웹에서 시작한 연구 모듈은 죽지 않습니다.
- `ExecStop` 은 실험 중이면 콘솔을 남겨 둡니다.
- 로그: `logs/live2_serve.log`, `journalctl --user -u live2-console`
- ROPUS user 서비스(`ropus-go2-*`)는 건드리지 않았습니다 (active 그대로).
- **자동 실행 대상은 웹뿐**입니다. FAST-LIO, E2–E7, SigLIP, 팬틸트, IMX477, HOME 은 자동으로 시작하지 않습니다.

## 3. 검증 (실제 시스템, 연구 모듈 미기동)
| 항목 | 결과 |
|---|---|
| 서비스 설치·활성화 → 웹 자동 기동 | PASS. Tailscale 확인 후 1 s 안에 기동, 8770 LISTEN, Tailscale HTTP 200 |
| 중복 실행 방지 | PASS. `live2 start` 와 직접 ctl 기동 모두 거부, 콘솔 1개 |
| `live2 restart` | PASS. 새 PID, 콘솔 1개 |
| 웹이 죽으면 감시기가 복구 | PASS. 약 10 s |
| `live2 stop` / `start` | PASS. 서비스·포트가 정상으로 닫혔다 열림 |
| 실험 중 stop 거부, STARTING 중 restart 거부 | PASS (상태 모의) |
| **발견·수정** | 콘솔 중복 검사가 명령줄 문자열만 보고 판단해, `live2/console.py` 를 언급한 셸이나 편집기만 있어도 기동을 막았습니다. 이제 실제로 console.py 를 실행하는 python 프로세스만 셉니다 (`ctl.python_runs`) |
| 실제 재부팅 검증 | **NOT_RUN — 사용자 승인 필요** |

## 4. 웹 승인 A — 구현 완료(V2 후보판), **운영 8770 미반영**
- **누가**: 허용 장치(tab-s8)와 로봇 자신만 할 수 있습니다. 보기 전용 Tailscale 기기와 LAN 은 거부합니다 (실제 whois 로 검증).
- **범위 (고정)**
  - 현재 boot 만, 최대 60 분 (30/45/60 선택)
  - fastlio, lio2(축적), hub, observe(SigLIP 1 GPU SHADOW), e2(CPU), e34: 각 1800 s, rec
  - 기존 온도·메모리 보호 그대로
  - **e2_gpu, frame_probe, 팬틸트, IMX477, HOME, Go2 명령은 웹으로 승인할 수 없습니다.**
- **방법**: 승인 카드에서 범위를 확인하고, 체크박스를 켠 뒤 [승인 기록] 을 누릅니다.
  - 서버는 다음을 모두 검증합니다: ack, boot_id 일치, 모듈 허용 목록, 시간 범위, 실험 중이 아님.
- **기록**
  - 승인 원문(범위, 기기 이름·StableID, 시각)과 SHA-256 을 기존 승인 A 형식으로 남깁니다. `recorded_by=web:<기기>`
  - 감사 로그에 `approval_grant` 를 남깁니다.
- **취소**: [승인 취소] 를 누르면 즉시 만료됩니다. 실행 중인 실험은 감시기가 `AUTO_END APPROVAL_EXPIRED` 로 정리합니다.
- **시험**: `tests/test_web_approval_fake.py` 6건 통과. V2 전체 67건 통과.
- **운영 반영**: 승인 판단 주체가 바뀌는 보안 변경이므로 **사용자 승인 후** `switch_8770_to_v2.sh --apply` 로 반영합니다 (백업·롤백 포함).
