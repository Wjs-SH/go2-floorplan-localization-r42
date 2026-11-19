#!/usr/bin/env bash
# record_e5_dataset.sh <name> [extra topics...] — record a NEW E5 evaluation/development drive (camera + odometry only).
# Never records /lidar_points (an extra point-cloud subscriber drops FAST-LIO to ~6.9 Hz, measured 2026-10-14).
# Output: ~/sci_ws/datasets/raw/<name>_<time>/ (sqlite3 bag). Then: python3 tools/extract_bag.py <db3> datasets/bags/<name> --hz 10
# Labelling: run e5_node replay on it, then tools/rescore.py spots ... --eligible-only and tools/label_review.py.
# Data-split rule: decide BEFORE looking at results whether a recording is DEV or FINAL-TEST (write it to datasets/raw/SPLITS.md).
set -euo pipefail
NAME=${1:?name}; shift || true
for t in "$@"; do case "$t" in *lidar*|*points*|*cloud*) echo "refusing point-cloud topic $t"; exit 2;; esac; done
set +u; source /opt/ros/foxy/setup.bash; set -u
OUT=$HOME/sci_ws/datasets/raw/${NAME}_$(date +%Y%m%d_%H%M%S); mkdir -p $(dirname $OUT)
ODOM=/utlidar/robot_odom; ros2 topic list | grep -qx /Odometry && ODOM="$ODOM /Odometry"
TOPICS="/camera/color/image_raw /camera/color/camera_info /camera/aligned_depth_to_color/image_raw /camera/aligned_depth_to_color/camera_info $ODOM /tf /tf_static $*"
echo "recording to $OUT : $TOPICS  (Ctrl+C to stop)"; df -h $HOME | tail -1
exec ros2 bag record -o $OUT $TOPICS
