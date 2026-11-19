#!/bin/bash
# apply_plate_merge.sh — 명판 촬영(케이스 1) 방식을 운영판에 넣거나 뺀다 (학생 스테이징, 2026-11-13).
# 운영판에서 바뀌는 것: (1) candidate/live2/web_v2/index.html  운용 탭 안의 명판 촬영 카드  (2) candidate/live2/console.py  스위치 한 덩어리(8줄)
#                      (3) runtime/plate_case1.on  스위치 파일 — 있으면 '팬틸트 카메라 사용'에 동의한 실험에서 E6 자동 촬영 대신 명판 촬영 탭이 팬틸트를 쓴다
#                      (4) runtime/student_start_hook  실험을 시작할 때 콘솔이 돌릴 학생 도구 시작 스크립트의 경로 (촬영 도구·전용 코어 도구)
#   bash apply_plate_merge.sh --check                 조건만 확인, 아무것도 안 바꿈
#   bash apply_plate_merge.sh --apply                 원본 2개를 운영폴더/rollback/student_plate_<시각>/ 에 백업 → 끼워 넣기 → 스위치 켜기 → 콘솔만 다시 띄움 → 8770 확인(안 뜨면 자동으로 되돌림)
#   bash apply_plate_merge.sh --apply-tab             화면만 바꾼다 (console.py·스위치·콘솔 재시작 없음)
#   bash apply_plate_merge.sh --switch on|off         스위치 파일만 만들거나 지운다 (다음 실험 시작부터 적용, 재시작 없음)
#   bash apply_plate_merge.sh --hook on|off           실험 시작 때 학생 도구 시작만 켜거나 끈다 (재시작 없음)
#   bash apply_plate_merge.sh --other-account on|off  추가 Tailscale 계정 허용 덩어리를 console.py 에 넣거나 뺀다 (백업 → 바꿈 → 콘솔 재시작).
#                                                     넣은 뒤에도 runtime/extra_tailscale_accounts 파일(한 줄에 계정 하나)이 없으면 동작은 그대로다
#   bash apply_plate_merge.sh --rollback <백업 폴더>    끼워 넣은 덩어리만 빼고 스위치를 지우고 콘솔을 다시 띄운다 (그 뒤 다른 사람이 고친 내용은 남긴다)
set -u
O=$HOME/sci_ws/repairs/livefield2_20261018_151606
T=$(cd "$(dirname "$0")" && pwd)
PAGE=candidate/live2/web_v2/index.html
CONS=candidate/live2/console.py
SW=$O/runtime/plate_case1.on
HOOK=$O/runtime/student_start_hook
HOOK_SCRIPT=$T/on_experiment_start.sh
EXP=$O/runtime/ctl/experiment.json
state() { python3 -c "import json; print(json.load(open('$EXP')).get('state'))" 2>/dev/null; }
idle() {
  st=$(state)
  case "$st" in RUNNING|PARTIAL|PAUSED|STARTING|STOPPING) echo "중단: 실험 상태 $st — 실험이 없을 때만 바꿉니다"; return 1 ;; esac
  busy=$(pgrep -af '^[^ ]*python3?[^ ]* .*(e34_node|e2_worker|lio_monitor2|sensor_hub|e6_precision_bridge)\.py|^[^ ]*(bridge_node|fastlio_mapping|l1_adapter_node)( |$)|^bash [^ ]*body_fastlio_live2\.sh')
  if [ -n "$busy" ]; then echo "중단: 실험 프로세스가 떠 있습니다"; echo "$busy" | cut -c1-120; return 1; fi
  echo "실험 상태 $st · 실험 프로세스 없음"
}
tool() { case "$1" in "$PAGE") echo "$T/patch_plate_tab.py" ;; "$CONS") echo "$T/patch_console_plate.py" ;; esac; }
precheck() {      # $@ = 바꿀 파일들
  idle || return 1
  for f in "$@"; do
    if [ ! -f "$O/$f" ] || [ -L "$O/$f" ]; then echo "중단: $O/$f 가 없거나 링크입니다"; return 1; fi
    python3 "$(tool "$f")" "$O/$f" --check || return 1
  done
  free=$(df -Pk "$O" | awk 'NR==2{print int($4/1024)}')
  if [ "$free" -lt 1024 ]; then echo "중단: 디스크 여유 ${free} MB"; return 1; fi
  echo "디스크 여유 ${free} MB"
}
put() {           # $1 = 파일, $2 = --revert 또는 빈칸 : 끼우거나 뺀 내용을 같은 폴더의 임시 파일에 만들고 한 번에 바꾼다 (권한 유지)
  f=$O/$1; tmp=$(mktemp "$(dirname "$f")/.plate_merge.XXXXXX") || return 1
  python3 "$(tool "$1")" "$f" ${2:+$2} --out "$tmp" >/dev/null || { rm -f "$tmp"; return 1; }
  case "$1" in *.py) python3 -c "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())" "$tmp" 2>/dev/null || { echo "중단: $1 문법 오류"; rm -f "$tmp"; return 1; } ;; esac
  chmod --reference="$f" "$tmp" && mv -f "$tmp" "$f"
}
restart_console() {
  "$O/scripts/live2" restart >/dev/null 2>&1
  for i in $(seq 1 60); do ss -ltn | grep -q ':8770 ' && return 0; sleep 1; done
  return 1
}
backup() {        # $@ = 파일들 -> 백업 폴더 이름을 B 에
  B=$O/rollback/student_plate_$(date +%Y%m%d_%H%M%S)
  for f in "$@"; do mkdir -p "$B/$(dirname "$f")" && cp -p "$O/$f" "$B/$f" || return 1; done
  (cd "$B" && sha256sum "$@" > SHA256SUMS.txt && printf '%s\n' "$@" > FILES.txt)
}
undo() {          # 백업 원본으로 되돌린다 (방금 넣은 것이 잘못됐을 때만)
  for f in $(cat "$B/FILES.txt"); do cp -p "$B/$f" "$O/$f"; done; rm -f "$SW" "$HOOK"
}
case "${1:-}" in
  --check)
    precheck "$PAGE" "$CONS" && echo "스위치 파일: $([ -e "$SW" ] && echo 있음 || echo 없음) · 학생 도구 시작 파일: $([ -e "$HOOK" ] && echo 있음 || echo 없음)" && echo "PRECHECK OK — 아무것도 바꾸지 않았습니다" ;;
  --apply-tab)
    precheck "$PAGE" || exit 1
    backup "$PAGE" || { echo "중단: 백업 실패"; exit 1; }
    put "$PAGE" || { echo "중단: 화면에 넣기 실패 — 아무것도 바꾸지 않았습니다"; exit 1; }
    sha256sum "$O/$PAGE" | tee "$B/AFTER.sha256"
    echo "화면에 넣었습니다(콘솔 재시작 없음). 되돌리기: bash $T/apply_plate_merge.sh --rollback $B" ;;
  --apply)
    precheck "$PAGE" "$CONS" || exit 1
    [ -x "$O/scripts/live2" ] || { echo "중단: $O/scripts/live2 가 없습니다"; exit 1; }
    backup "$PAGE" "$CONS" || { echo "중단: 백업 실패"; exit 1; }
    put "$PAGE" && put "$CONS" || { echo "중단: 끼워 넣기 실패 → 되돌림"; undo; exit 1; }
    printf '명판 촬영(케이스 1) 방식 스위치. 이 파일이 있으면 팬틸트 사용에 동의한 실험에서 E6 자동 촬영 대신 명판 촬영 탭이 팬틸트를 쓴다.\n만든 때: %s (학생 병합, tools/plate_merge)\n' "$(date '+%F %T')" > "$SW"
    [ -f "$HOOK_SCRIPT" ] && printf '%s\n' "$HOOK_SCRIPT" > "$HOOK"
    (cd "$O" && sha256sum "$PAGE" "$CONS") | tee "$B/AFTER.sha256"
    if restart_console; then echo "8770 UP"; else echo "8770 이 뜨지 않음 → 되돌림"; undo; restart_console && echo "되돌린 뒤 8770 UP" || echo "되돌린 뒤에도 8770 이 뜨지 않습니다 — 직접 확인 필요"; exit 5; fi
    echo "넣었습니다. 스위치 켜짐, 실험 시작 때 학생 도구 시작: $([ -f "$HOOK" ] && cat "$HOOK" || echo 없음)"
    echo "되돌리기: bash $T/apply_plate_merge.sh --rollback $B" ;;
  --switch)
    case "${2:-}" in
      on) grep -q "PLATE_MERGE_BEGIN" "$O/$CONS" || { echo "중단: console.py 에 스위치 덩어리가 없습니다 (--apply 먼저)"; exit 1; }
          printf '명판 촬영(케이스 1) 방식 스위치.\n만든 때: %s\n' "$(date '+%F %T')" > "$SW"; echo "스위치 켬 — 다음 실험 시작부터 적용" ;;
      off) rm -f "$SW"; echo "스위치 끔 — 다음 실험부터 예전 방식(동의하면 E6 자동 촬영)" ;;
      *) echo "on 또는 off"; exit 1 ;;
    esac ;;
  --hook)
    case "${2:-}" in
      on) grep -q "student_start_hook" "$O/$CONS" || { echo "중단: console.py 에 학생 도구 시작 덩어리가 없습니다 (--apply 먼저)"; exit 1; }
          printf '%s\n' "$HOOK_SCRIPT" > "$HOOK"; echo "실험 시작 때 학생 도구 시작: 켬 ($HOOK_SCRIPT)" ;;
      off) rm -f "$HOOK"; echo "실험 시작 때 학생 도구 시작: 끔" ;;
      *) echo "on 또는 off"; exit 1 ;;
    esac ;;
  --other-account)
    case "${2:-}" in on) FLAG=--other-account ;; off) FLAG=--no-other-account ;; *) echo "on 또는 off"; exit 1 ;; esac
    precheck "$CONS" || exit 1
    [ -x "$O/scripts/live2" ] || { echo "중단: $O/scripts/live2 가 없습니다"; exit 1; }
    grep -q "PLATE_MERGE_BEGIN" "$O/$CONS" || { echo "중단: console.py 에 병합 덩어리가 없습니다 (--apply 먼저)"; exit 1; }
    backup "$CONS" || { echo "중단: 백업 실패"; exit 1; }
    put "$CONS" "$FLAG" || { echo "중단: 바꾸기 실패 → 백업으로 되돌림"; cp -p "$B/$CONS" "$O/$CONS"; exit 1; }
    (cd "$O" && sha256sum "$CONS") | tee "$B/AFTER.sha256"
    if restart_console; then echo "8770 UP"; else echo "8770 이 뜨지 않음 → 백업으로 되돌림"; cp -p "$B/$CONS" "$O/$CONS"; restart_console && echo "되돌린 뒤 8770 UP" || echo "되돌린 뒤에도 8770 이 뜨지 않습니다 — 직접 확인 필요"; exit 5; fi
    ACC=$O/runtime/extra_tailscale_accounts
    if grep -q "추가 Tailscale 계정 허용" "$O/$CONS"; then echo "추가 계정 허용 덩어리: 들어 있음 · 계정 파일 $ACC: $([ -s "$ACC" ] && grep -vc '^[[:space:]]*\(#\|$\)' "$ACC" || echo 0)개 계정"; else echo "추가 계정 허용 덩어리: 없음 (원래 규칙)"; fi
    echo "되돌리기: bash $T/apply_plate_merge.sh --other-account $([ "$2" = on ] && echo off || echo on)   (백업: $B)" ;;
  --rollback)
    B=${2:?백업 폴더를 주세요}
    [ -f "$B/FILES.txt" ] || { echo "중단: $B/FILES.txt 가 없습니다"; exit 1; }
    idle || exit 1
    (cd "$B" && sha256sum -c SHA256SUMS.txt >/dev/null) || { echo "중단: 백업 파일이 변했습니다"; exit 1; }
    had_cons=0
    for f in $(cat "$B/FILES.txt"); do
      [ "$f" = "$CONS" ] && grep -q "PLATE_MERGE_BEGIN" "$O/$f" && had_cons=1
      put "$f" --revert || { echo "중단: $f 에서 빼기 실패"; exit 1; }
      cmp -s "$O/$f" "$B/$f" && echo "$f: 백업 원본과 바이트까지 같음" || echo "$f: 넣은 뒤에 따로 바뀐 내용이 있어 그것은 남기고 끼운 덩어리만 뺌"
    done
    rm -f "$SW" "$HOOK"; echo "스위치 파일·학생 도구 시작 파일 지움"
    if [ "$had_cons" = 1 ]; then restart_console && echo "8770 UP" || { echo "8770 이 뜨지 않습니다 — 직접 확인 필요"; exit 5; }; fi
    echo "뺐습니다. 화면을 새로 고치세요." ;;
  *)
    sed -n 2,13p "$0" ;;
esac
