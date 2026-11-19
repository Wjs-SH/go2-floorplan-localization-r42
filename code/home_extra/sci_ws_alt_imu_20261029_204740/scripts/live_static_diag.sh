#!/bin/bash
# live_static_diag.sh CALIB.yaml APPROVAL_FILE [--duration S]
# LIVE STATIC DIAGNOSTIC of the body-IMU FAST-LIO path (domain 0). Robot must stay still; nothing here commands the robot.
# Starts ONLY (owned, pid+starttime): calib_monitor (motion guard, /lowstate read-only), the bridge, FAST-LIO (unchanged binary,
# rendered 'live-static-diag' config, outputs remapped under /sci_alt_imu, Log/PCD bind-mounted into the run dir), a small output
# recorder (no LiDAR). static_diag_watchdog.py aborts on robot input/motion, IMU/LiDAR loss, FAST-LIO stall/drift/jump,
# temperature/memory, and ends at the duration (default 60 s, cap 300 s). Everything is then stopped and archived.
# Refuses unless: approval file names 'live_static_diag' (one explicit user approval per run), live preflight PASS (no other
# FAST-LIO, protected hashes), no E2/SigLIP/E6/lio2 and no running calibration recording.
source "$(dirname "$0")/common.sh"
calib="${1:?CALIB.yaml}"; appr="${2:?APPROVAL_FILE (user approval text for this one static run)}"; shift 2
dur=60
while [ $# -gt 0 ]; do case "$1" in --duration) dur="$2"; shift 2;; *) die "unknown arg $1";; esac; done
[ "$dur" -ge 10 ] && [ "$dur" -le 300 ] || die "duration must be 10..300 s"
[ -s "$appr" ] && grep -q "live_static_diag" "$appr" || die "approval file must exist and name 'live_static_diag'"
for p in e2_worker observe_keeper siglip_worker e6_precision_bridge lio_monitor2 e34_node; do
  [ "$(pgrep -fc "python3 .*$p" || true)" = 0 ] || die "$p is running — stop the operational stack first"; done
pgrep -f "ros2 bag record .*lidar_points" >/dev/null && die "a /lidar_points recording is running"
run=$(new_run_dir live_static_diag); say "run dir $run"
cp -p "$appr" "$run/APPROVAL_TEXT.txt"; sha256sum "$calib" > "$run/calib.sha256"; cp -p "$calib" "$run/"
python3 -I "$(dirname "$0")/render_fastlio_cfg.py" "$calib" "$run/fastlio_alt.yaml" --mode live-static-diag || die "render refused"
"$(dirname "$0")/preflight.sh" --live "$run" || die "live preflight failed (see $run/preflight.txt)"
free=$(df -BM --output=avail "$HOME" | tail -1 | tr -dc 0-9); [ "$free" -gt 3000 ] || die "disk: ${free} MB free (< 3 GB)"
ros_env live "$run"
trap 'owned_stop "$run"' EXIT
owned_start "$run" calib_monitor "$run/calib_monitor.out" ros2 run sci_alt_imu_bridge calib_monitor --ros-args -p status_file:="$run/calib_monitor.json"
sleep 4
python3 - "$run/calib_monitor.json" <<'PY' || die "robot not still / /lowstate not live before start (see above)"
import json, sys, time
s = json.load(open(sys.argv[1]))
ok = time.time() - s["wall"] < 1.5 and s["rate_hz_1s"] >= 400 and s["stick_max_abs_1s"] < 0.2 and s["gyro_norm_p95_1s"] < 0.25
print("pre-start /lowstate %s Hz stick %.3f gyro_p95 %.3f posture calf %.2f -> %s" % (s["rate_hz_1s"], s["stick_max_abs_1s"], s["gyro_norm_p95_1s"], s["posture"]["calf"], "OK" if ok else "REFUSE"))
sys.exit(0 if ok else 1)
PY
bridge_start "$run"
ok=0; for _ in $(seq 1 30); do sleep 0.5; tail -1 "$run/bridge_health.jsonl" 2>/dev/null | grep -q '"health":"READY"' && { ok=1; break; }; done
[ $ok = 1 ] || die "bridge not READY within 15 s (see $run/bridge_health.jsonl)"
fastlio_alt_start "$run" "$run/fastlio_alt.yaml"
owned_start "$run" recorder "$run/recorder.out" bash -c "cd '$run' && exec ros2 bag record -o out_bag /sci_alt_imu/imu /sci_alt_imu/health /sci_alt_imu/Odometry"
say "STATIC DIAGNOSTIC RUNNING ${dur}s — robot must stay still. PROVISIONAL extrinsic. Not a walk permission."
python3 -I "$(dirname "$0")/static_diag_watchdog.py" "$run" --duration "$dur" | tee "$run/watchdog.out"; wd=${PIPESTATUS[0]}
trap - EXIT
"$(dirname "$0")/stop_archive_alt.sh" "$run"
python3 -I "$ALT_WS/calib_tools/static_diag_report.py" "$run" > "$run/static_diag_report.txt" 2>&1; cat "$run/static_diag_report.txt"
[ "$wd" = 0 ] && say "DONE (static only — NOT verified for walking): $run" || say "ABORTED by watchdog (see $run/verdict.json): $run"
exit "$wd"
