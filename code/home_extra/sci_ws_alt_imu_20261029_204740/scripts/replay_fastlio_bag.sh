#!/bin/bash
# replay_fastlio_bag.sh REPLAY_BAG CALIB.yaml TAG [--extrinsic-est]
# Offline FAST-LIO on a prebuilt bag (/lidar_points blobs + /sci_alt_imu/imu from offline_stamp). Domain 42 + loopback DDS,
# unchanged FAST-LIO binary with a rendered offline-plumbing config, Log/PCD bind-mounted into the run dir. Stops only what it started.
source "$(dirname "$0")/common.sh"
bag="${1:?REPLAY_BAG}"; calib="${2:?CALIB.yaml}"; tag="${3:?TAG}"; est="${4:-}"
[ -f "$bag/metadata.yaml" ] || die "not a rosbag2 dir: $bag"
run=$(new_run_dir "offline_fastlio_$tag"); say "run dir $run"
"$(dirname "$0")/preflight.sh" --offline "$run" >/dev/null || die "preflight failed (see $run/preflight.txt)"
ros_env offline "$run"
mat_before=$(sha256sum "$FASTLIO_SRC/Log/mat_pre.txt" 2>/dev/null | cut -d' ' -f1)
{ echo "bag $bag"; (cd "$bag" && sha256sum *.db3); echo "calib $calib"; sha256sum "$calib"; } > "$run/inputs.txt"
trap 'owned_stop "$run"' EXIT
python3 -I "$(dirname "$0")/render_fastlio_cfg.py" "$calib" "$run/fastlio_alt.yaml" --mode offline-plumbing $est || die "render failed"
fastlio_alt_start "$run" "$run/fastlio_alt.yaml"; sleep 4
owned_start "$run" recorder "$run/recorder.out" bash -c "cd '$run' && exec ros2 bag record -o out_bag /sci_alt_imu/Odometry"; sleep 3
ros2 bag play "$bag" > "$run/play.out" 2>&1
sleep 3
owned_stop "$run"; trap - EXIT
mat_after=$(sha256sum "$FASTLIO_SRC/Log/mat_pre.txt" 2>/dev/null | cut -d' ' -f1)
echo "orig_mat_pre_before=$mat_before after=$mat_after" >> "$run/inputs.txt"
[ "$mat_before" = "$mat_after" ] && say "original FAST-LIO Log/mat_pre.txt unchanged" || say "WARNING original Log/mat_pre.txt changed!"
(cd "$run" && find . -type f -not -path "./roslog/*" | sort | xargs sha256sum > SHA256SUMS.txt)
say "done: $run"
