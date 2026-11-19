#!/bin/bash
# on_experiment_start.sh — 실험이 시작될 때 콘솔이 부르는 학생 도구 시작 스크립트 (학생 스테이징, 2026-11-13).
# 운영폴더 runtime/student_start_hook 에 이 파일의 경로가 적혀 있을 때만 불린다(콘솔의 a_start 에 끼운 덩어리).
#   * 촬영 도구(plate_site, 8791): 운용 탭의 명판 촬영 카드가 띄우는 화면
#   * 전용 코어 도구(core_pin, 모든 실험 적용): FAST-LIO 본체 스레드에 3번 코어 (CORE_LAYOUT_FASTLIO_DEDICATED.md)
# 이미 떠 있는 도구는 건드리지 않는다(번갈아 하는 비교 시험 방식으로 떠 있는 전용 코어 도구만 '모든 실험' 방식으로 바꿔 띄운다).
# 각 도구를 떼어서 띄우고 바로 돌아온다. 운영 폴더에는 아무것도 쓰지 않는다.
S=$HOME/sci_ws_student_staging_20261109
LOG=$S/_runs/start_hook.log
mkdir -p "$S/_runs"

pid_of() {             # $1 = 스크립트 이름. 떠 있으면 pid 를 찍고 0
    for d in /proc/[0-9]*; do
        c=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null )
        case "$c" in
            python3\ *"/$1 "*|python3\ *"/$1") echo "${d#/proc/}"; return 0 ;;
        esac
    done
    return 1
}
launch() {             # $1 = 폴더, $2 = 스크립트 이름, 나머지 = 인자
    dir=$1; name=$2; shift 2
    if pid=$(pid_of "$name"); then echo "  $name: 이미 떠 있음 (pid $pid)"; return; fi
    # 콘솔이 넘겨준 환경 변수(ROS·DDS 설정 등)를 물려받지 않게 정해진 환경으로 띄운다 — ssh 로 손으로 띄울 때(arm_all.sh)와 같은 조건.
    env -i HOME="$HOME" USER="${USER:-unitree}" LOGNAME="${USER:-unitree}" SHELL=/bin/bash LANG=en_US.UTF-8 \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        nohup setsid python3 "$dir/$name" "$@" > "$S/_runs/hook_${name%.py}.out" 2>&1 < /dev/null &
    sleep 1
    if pid=$(pid_of "$name"); then echo "  $name: 띄움 (pid $pid)"; else echo "  $name: 뜨지 않음 — $S/_runs/hook_${name%.py}.out 확인"; fi
}
{
    echo "$(date '+%F %T') 실험 시작 신호"
    launch "$S/tools/plate_site" plate_site.py --port 8791 --arm-s 86400
    if pid=$(pid_of core_pin.py); then
        c=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        case "$c" in
            *"--first always"*) ;;
            *"tools/field1112/core_pin.py"*"--wait-experiment"*) kill "$pid" 2>/dev/null; sleep 1; echo "  core_pin.py: 비교 시험 방식(pid $pid)을 끝냄(종료하면서 코어 지정을 되돌린다)" ;;
        esac
    fi
    launch "$S/tools/field1112" core_pin.py --wait-experiment --first always --arm-s 86400
} >> "$LOG" 2>&1
