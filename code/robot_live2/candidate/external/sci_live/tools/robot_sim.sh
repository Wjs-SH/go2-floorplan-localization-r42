#!/usr/bin/env bash
# 로봇 흉내 격리 실행(개발용 검증 도구, 로봇에는 불필요).
# /home 전체를 비우고 패키지·데이터·출력·사용자 파이썬 라이브러리만 보이게 해서, 패키지가 밖의 파일에
# 기대지 않는지 확인한다(2026-10-01 이 방법으로 숨은 절대경로 3개를 찾았다). bwrap 필요.
#   tools/robot_sim.sh <패키지폴더> <데이터폴더|-> <출력폴더> -- <명령...>
#   EXTRA_RO=<폴더>  → 읽기전용으로 /home/robot/extra 에 붙인다(예: PaddleOCR 설치 폴더)
# 안에서 보이는 경로: 패키지 /home/robot/pkg(작업 폴더), 데이터 /home/robot/data, 출력 /home/robot/out
PKG=$1; DATA=$2; OUT=$3; shift 4
args=(--ro-bind / / --tmpfs /home --tmpfs /media --tmpfs /tmp --dev /dev --proc /proc
      --bind "$PKG" /home/robot/pkg --bind "$OUT" /home/robot/out)
[ -d "$HOME/.local/lib" ] && args+=(--ro-bind "$HOME/.local/lib" "$HOME/.local/lib")
[ "$DATA" != - ] && args+=(--ro-bind "$DATA" /home/robot/data)
[ -n "${EXTRA_RO:-}" ] && args+=(--ro-bind "$EXTRA_RO" /home/robot/extra)
exec bwrap "${args[@]}" --chdir /home/robot/pkg "$@"
