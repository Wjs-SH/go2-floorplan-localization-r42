#!/bin/bash
# live_walk_diag.sh CALIB.yaml APPROVAL_FILE [--duration S] [--record-cloud] [--cloud-window MB]
# LIVE WALK DIAGNOSTIC of the body-IMU FAST-LIO path (domain 0): ONE short slow walk driven by the OPERATOR's joystick (nothing
# here commands the robot). Same isolation as live_static_diag.sh (owned processes, outputs under /sci_alt_imu, Log/PCD bind
# mount, no E2-E8). Differences: approval file must name 'live_walk_diag'; config mode 'live-walk-diag' (data-derived candidate
# only); walk_diag_watchdog.py (motion allowed; aborts on IMU/LiDAR loss, FAST-LIO stall, speed/jump/height divergence, resources;
# default 180 s, cap 300 s). FAST-LIO must initialise with the robot STILL: start standing still, walk after "WALK NOW".
# --record-cloud adds /sci_alt_imu/cloud_registered (~9 MB/s, FAST-LIO output, not a /lidar_points subscription) for map checks.
# walk_tap.py always runs: /sci_alt_imu/Odometry -> odom.csv (the ONE path source of tablet + report). --cloud-window MB also stores
# /sci_alt_imu/cloud_registered (undecoded, = the E2 input format) ONLY while RUN/CLOUD_WINDOW is 'on' (tablet steps), capped at MB.
source "$(dirname "$0")/common.sh"
calib="${1:?CALIB.yaml}"; appr="${2:?APPROVAL_FILE (user approval text for this one walk)}"; shift 2
dur=180; cloud=0; cwin=0
while [ $# -gt 0 ]; do case "$1" in --duration) dur="$2"; shift 2;; --record-cloud) cloud=1; shift;; --cloud-window) cwin="$2"; shift 2;; *) die "unknown arg $1";; esac; done
[ "$dur" -ge 30 ] && [ "$dur" -le 300 ] || die "duration must be 30..300 s"
[ -s "$appr" ] && grep -q "live_walk_diag" "$appr" || die "approval file must exist and name 'live_walk_diag'"
for p in e2_worker observe_keeper siglip_worker e6_precision_bridge lio_monitor2 e34_node; do
  [ "$(pgrep -fc "python3 .*$p" || true)" = 0 ] || die "$p is running — stop the operational stack first"; done
pgrep -f "ros2 bag record .*lidar_points" >/dev/null && die "a /lidar_points recording is running"
need=3000; [ $cloud = 1 ] && need=$(( 3000 + dur * 12 ))
[ "$cwin" -ge 0 ] && [ "$cwin" -le 1500 ] || die "--cloud-window MB must be 0..1500"
need=$(( need + cwin ))
free=$(df -BM --output=avail "$HOME" | tail -1 | tr -dc 0-9); [ "$free" -gt "$need" ] || die "disk: ${free} MB free, need > ${need} MB"
run=$(new_run_dir live_walk_diag); say "run dir $run"
cp -p "$appr" "$run/APPROVAL_TEXT.txt"; sha256sum "$calib" > "$run/calib.sha256"; cp -p "$calib" "$run/"
python3 -I "$(dirname "$0")/render_fastlio_cfg.py" "$calib" "$run/fastlio_alt.yaml" --mode live-walk-diag || die "render refused"
"$(dirname "$0")/preflight.sh" --live "$run" || die "live preflight failed (see $run/preflight.txt)"
ros_env live "$run"
trap 'owned_stop "$run"' EXIT
owned_start "$run" calib_monitor "$run/calib_monitor.out" ros2 run sci_alt_imu_bridge calib_monitor --ros-args -p status_file:="$run/calib_monitor.json"
sleep 4
python3 - "$run/calib_monitor.json" <<'PY' || die "robot must be STANDING and STILL at start (FAST-LIO initialisation)"
import json, sys, time
s = json.load(open(sys.argv[1])); p = s["posture"]
ok = time.time() - s["wall"] < 1.5 and s["rate_hz_1s"] >= 400 and s["stick_max_abs_1s"] < 0.2 and s["gyro_norm_p95_1s"] < 0.25 \
     and -1.9 < p["calf"] < -1.1 and p["tau_mean"] > 1.0
print("pre-start /lowstate %s Hz stick %.3f gyro_p95 %.3f calf %.2f tau %.2f -> %s" % (s["rate_hz_1s"], s["stick_max_abs_1s"], s["gyro_norm_p95_1s"], p["calf"], p["tau_mean"], "OK" if ok else "REFUSE"))
sys.exit(0 if ok else 1)
PY
bridge_start "$run"
ok=0; for _ in $(seq 1 30); do sleep 0.5; tail -1 "$run/bridge_health.jsonl" 2>/dev/null | grep -q '"health":"READY"' && { ok=1; break; }; done
[ $ok = 1 ] || die "bridge not READY within 15 s (see $run/bridge_health.jsonl)"
fastlio_alt_start "$run" "$run/fastlio_alt.yaml"
topics="/sci_alt_imu/imu /sci_alt_imu/health /sci_alt_imu/Odometry"; [ $cloud = 1 ] && topics="$topics /sci_alt_imu/cloud_registered"
owned_start "$run" recorder "$run/recorder.out" bash -c "cd '$run' && exec ros2 bag record -o out_bag $topics"
owned_start "$run" walk_tap "$run/walk_tap.out" python3 -u "$(dirname "$0")/walk_tap.py" "$run" --cloud-max-mb "$cwin" --cloud-max-s 90
say "WALK DIAGNOSTIC ${dur}s — keep STILL until FAST-LIO initialises (watch for scans>0), then walk slowly with the joystick."
python3 -I "$(dirname "$0")/walk_diag_watchdog.py" "$run" --duration "$dur" | tee "$run/watchdog.out"; wd=${PIPESTATUS[0]}
trap - EXIT
"$(dirname "$0")/stop_archive_alt.sh" "$run"
python3 -I "$ALT_WS/calib_tools/static_diag_report.py" "$run" > "$run/diag_report.txt" 2>&1
python3 -I "$ALT_WS/calib_tools/walk_diag_report.py" "$run" > "$run/walk_report.txt" 2>&1; cat "$run/walk_report.txt"
[ "$wd" = 0 ] && say "DONE (one walk diagnostic — compare with the operator's measured distance/turns): $run" || say "ABORTED by watchdog (see $run/verdict.json): $run"
exit "$wd"
