#!/bin/bash
# replay_offline.sh BAG_DIR [--fastlio CALIB.yaml] [--rate R]
# Domain 42 + loopback-only Cyclone profile: live sensors are NOT visible; only the bag is played. Runs the bridge (and optionally
# the unchanged FAST-LIO binary with a rendered offline-plumbing config), records outputs, stops ONLY what it started.
source "$(dirname "$0")/common.sh"
bag="${1:?BAG_DIR}"; shift
calib=""; rate=1.0
while [ $# -gt 0 ]; do case "$1" in --fastlio) calib="$2"; shift 2;; --rate) rate="$2"; shift 2;; *) die "unknown arg $1";; esac; done
[ -f "$bag/metadata.yaml" ] || die "not a rosbag2 dir: $bag"
run=$(new_run_dir offline); say "run dir $run"
"$(dirname "$0")/preflight.sh" --offline "$run" >/dev/null || die "preflight failed (see $run/preflight.txt)"
ros_env offline "$run"
mat_before=$(sha256sum "$FASTLIO_SRC/Log/mat_pre.txt" 2>/dev/null | cut -d' ' -f1)
topics=$(grep -E "^\s+name:" "$bag/metadata.yaml" | awk '{print $2}' | tr '\n' ' ')
echo "bag $bag topics: $topics" > "$run/inputs.txt"; (cd "$bag" && sha256sum *.db3) >> "$run/inputs.txt"
trap 'owned_stop "$run"' EXIT
bridge_start "$run"; sleep 4
rec_topics="/sci_alt_imu/imu /sci_alt_imu/health"
if [ -n "$calib" ]; then
  python3 -I "$(dirname "$0")/render_fastlio_cfg.py" "$calib" "$run/fastlio_alt.yaml" --mode offline-plumbing || die "render failed"
  fastlio_alt_start "$run" "$run/fastlio_alt.yaml"; sleep 4
  rec_topics="$rec_topics /sci_alt_imu/Odometry /sci_alt_imu/cloud_registered /sci_alt_imu/tf"
fi
owned_start "$run" recorder "$run/recorder.out" bash -c "cd '$run' && exec ros2 bag record -o out_bag $rec_topics"; sleep 3
play_topics=""; for t in /lowstate /lidar_points; do echo " $topics " | grep -q " $t " && play_topics="$play_topics $t"; done
say "playing:$play_topics"
ros2 bag play "$bag" --rate "$rate" --topics $play_topics > "$run/play.out" 2>&1
sleep 3
timeout 15 ros2 node list > "$run/node_list.txt" 2>&1
timeout 15 ros2 topic list -t > "$run/topic_list.txt" 2>&1
owned_stop "$run"; trap - EXIT
mat_after=$(sha256sum "$FASTLIO_SRC/Log/mat_pre.txt" 2>/dev/null | cut -d' ' -f1)
[ "$mat_before" = "$mat_after" ] && say "original FAST-LIO Log/mat_pre.txt unchanged" || say "WARNING original Log/mat_pre.txt changed!"
echo "orig_mat_pre_before=$mat_before after=$mat_after" >> "$run/inputs.txt"
python3 -I "$ALT_WS/tests/verify_bridge_output.py" "$run/out_bag" "$bag" "$run/verify_bridge_output.json" > /dev/null 2>&1 && say "bridge output verified -> $run/verify_bridge_output.json"
(cd "$run" && find . -type f -not -path "./roslog/*" | sort | xargs sha256sum > SHA256SUMS.txt)
say "done: $run"
