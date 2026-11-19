#!/bin/bash
# start_alt_static.sh CALIB.yaml — LIVE (domain 0) alt FAST-LIO for the static check. Starts ONLY: bridge, FAST-LIO(alt cfg,
# remapped under /sci_alt_imu), an output recorder (no LiDAR) and the monitor. Never moves the robot. Refuses unless:
#   calibration status CALIBRATION_VERIFIED + live_walk_allowed, live preflight PASS (no other FAST-LIO, protected hashes ok),
#   bridge health READY within 15 s.
source "$(dirname "$0")/common.sh"
calib="${1:?CALIB.yaml (must be CALIBRATION_VERIFIED)}"
run=$(new_run_dir live_static)
python3 -I "$(dirname "$0")/render_fastlio_cfg.py" "$calib" "$run/fastlio_alt.yaml" --mode live || die "calibration gate (see message above); run dir $run"
"$(dirname "$0")/preflight.sh" --live "$run" || die "live preflight failed (see $run/preflight.txt)"
ros_env live "$run"
bridge_start "$run"
ok=0; for _ in $(seq 1 30); do sleep 0.5; tail -1 "$run/bridge_health.jsonl" 2>/dev/null | grep -q '"health":"READY"' && { ok=1; break; }; done
[ $ok = 1 ] || { owned_stop "$run"; die "bridge not READY within 15 s (see $run/bridge_health.jsonl)"; }
fastlio_alt_start "$run" "$run/fastlio_alt.yaml"
owned_start "$run" recorder "$run/recorder.out" bash -c "cd '$run' && exec ros2 bag record -o out_bag /sci_alt_imu/imu /sci_alt_imu/health /sci_alt_imu/Odometry"
owned_start "$run" monitor "$run/monitor.log" "$(dirname "$0")/monitor_alt.sh" "$run"
say "RUNNING. Watch:  tail -f $run/monitor.log"
say "Robot must stay still for the static check. Stop with:  $(dirname "$0")/stop_archive_alt.sh $run"
