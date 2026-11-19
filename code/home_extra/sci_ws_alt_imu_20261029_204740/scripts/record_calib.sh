#!/bin/bash
# record_calib.sh APPROVAL_TEXT_FILE [--duration S]
# Synchronous raw recording for LiDAR-IMU calibration: /lidar_points + /lowstate (+ /utlidar/imu, /utlidar/lidar_state,
# /tf_static for the L1 chain cross-check). This adds a HIGH-BANDWIDTH /lidar_points subscriber, which the operational rule
# forbids ("Never: /lidar_points subscription"); it runs ONLY with a file containing the user's explicit one-time exception.
# No FAST-LIO, no E2/SigLIP/E6 may run. Robot motion is done by the operator with the joystick; this script never commands it.
source "$(dirname "$0")/common.sh"
appr="${1:?APPROVAL_TEXT_FILE with the user verbatim exception text}"; shift
dur=240
while [ $# -gt 0 ]; do case "$1" in --duration) dur="$2"; shift 2;; *) die "unknown arg $1";; esac; done
[ -s "$appr" ] || die "no approval text file"
grep -q "lidar_points" "$appr" || die "approval text must name /lidar_points explicitly"
[ "$dur" -le 300 ] || die "duration capped at 300 s"
run=$(new_run_dir calib_record); cp -p "$appr" "$run/APPROVAL_TEXT.txt"; sha256sum "$appr" > "$run/APPROVAL_TEXT.sha256"
"$(dirname "$0")/preflight.sh" --live "$run" || die "live preflight failed (see $run/preflight.txt)"
for p in e2_worker observe_keeper siglip_worker e6_precision_bridge; do [ "$(pgrep -fc "python3 .*$p" || true)" = 0 ] || die "$p running"; done
ros_env live "$run"
TOPICS="/lidar_points /lowstate"   # 2026-10-29 user exception scope: exactly these two topics
# 1) 10 s load test
# 2026-10-29: one Foxy rosbag2 process writing 16 MB/s of /lidar_points dropped ~15 % of /lowstate (tick gaps up to 54 ms,
# run calib_record_20261029_212806). -> one recorder process PER TOPIC (same Jetson receive clock for both bags).
owned_start "$run" rec_lt_lidar "$run/rec_lt_lidar.out" bash -c "cd '$run' && exec ros2 bag record -o bag_loadtest_lidar /lidar_points"
owned_start "$run" rec_lt_imu "$run/rec_lt_imu.out" bash -c "cd '$run' && exec ros2 bag record -o bag_loadtest_imu /lowstate"
cpu=0; for i in $(seq 1 10); do sleep 1; c=0; for n in rec_lt_lidar rec_lt_imu; do read -r p _ < "$run/owned/$n.pid"; c=$(( c + $(ps -o pcpu= --ppid "$p" -p "$p" 2>/dev/null | awk '{s+=$1} END{print int(s)}') )); done; [ "$c" -gt "$cpu" ] && cpu=$c; done
owned_stop "$run"; rm -f "$run/owned/rec_lt_lidar.pid" "$run/owned/rec_lt_imu.pid"
size=$(du -smc "$run/bag_loadtest_lidar" "$run/bag_loadtest_imu" | tail -1 | cut -f1)
cnt() { python3 -c 'import re,sys; t=open(sys.argv[1]).read(); print(sum(int(c) for c in re.findall(r"message_count: (\d+)", t)[1:]) or re.findall(r"message_count: (\d+)", t)[0])' "$1/metadata.yaml"; }
lp=$(cnt "$run/bag_loadtest_lidar"); ls_=$(cnt "$run/bag_loadtest_imu")
echo "/lidar_points $lp" > "$run/loadtest_counts.txt"; echo "/lowstate $ls_" >> "$run/loadtest_counts.txt"
say "load test: recorders cpu<=${cpu}% size ${size}MB lidar_points=${lp:-0} lowstate=${ls_:-0} (10 s)"
[ "${lp:-0}" -ge 80 ] && [ "${ls_:-0}" -ge 4500 ] || die "load test: rates too low (lidar ${lp:-0}/100, lowstate ${ls_:-0}/5000) — not recording"
[ "$cpu" -le 150 ] || die "load test: recorder CPU ${cpu}% too high"
need=$(( size * dur / 10 * 2 )); free=$(df -BM --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free" -gt $(( need + 5000 )) ] || die "disk: need ~${need}MB + 5GB margin, free ${free}MB"
# 2) bounded main recording (split files of 1 GB)
say "RECORDING ${dur}s -> $run/bag_calib_{lidar,imu}. Operator follows the runbook phases (static 30 s, slow yaw both ways, pitch/roll posture, short straight/curve, static 30 s)."
date '+START %F %T.%N' > "$run/phases.txt"
owned_start "$run" rec_main_lidar "$run/rec_main_lidar.out" bash -c "cd '$run' && exec timeout -s INT $dur ros2 bag record -b 1000000000 -o bag_calib_lidar /lidar_points"
owned_start "$run" rec_main_imu "$run/rec_main_imu.out" bash -c "cd '$run' && exec timeout -s INT $dur ros2 bag record -o bag_calib_imu /lowstate"
read -r p _ < "$run/owned/rec_main_lidar.pid"
while [ -n "$(proc_start_time "$p")" ]; do
  sleep 2; t=$(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1); m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  echo "$(date +%T) temp=$((t/1000))C mem=${m}MB lidar_MB=$(du -sm "$run/bag_calib_lidar" 2>/dev/null | cut -f1) imu_MB=$(du -sm "$run/bag_calib_imu" 2>/dev/null | cut -f1)" >> "$run/resources.log"
  if [ $((t/1000)) -ge 80 ] || [ "$m" -lt 2500 ]; then say "ABORT: temp ${t} mem ${m}"; owned_stop "$run"; break; fi
done
sleep 3; owned_stop "$run"
date '+END %F %T.%N' >> "$run/phases.txt"
for b in bag_calib_lidar bag_calib_imu; do (cd "$run/$b" && sha256sum * > "../$b.SHA256SUMS" && chmod a-w *); done
say "bags sealed read-only: $run/bag_calib_lidar $run/bag_calib_imu (hashes *.SHA256SUMS)"
