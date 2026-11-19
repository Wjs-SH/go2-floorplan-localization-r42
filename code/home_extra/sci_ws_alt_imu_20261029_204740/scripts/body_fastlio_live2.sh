#!/bin/bash
# body_fastlio_live2.sh RUN_DIR [CALIB.yaml] — the "body" input mode of the LIVE2 'fastlio' module (8770 ctl), foreground.
# Go2 body IMU (/lowstate) -> sci_alt_imu bridge -> UNCHANGED FAST-LIO binary (alt config, outputs remapped to /sci_alt_imu/*)
#   -> l1_adapter_node -> /Odometry + /cloud_registered in the L1-IMU frame contract -> UNCHANGED lio2 / hub / E2..E8 (L1 yaml).
# Called by ctl INSIDE its own `unshare -Urm` + Log/PCD bind mount (same as the L1 path), with ctl's live ROS env. Stops everything
# it started when it gets SIGINT/SIGTERM or when any child exits (the supervisor then sees the module fail, as with L1 FAST-LIO).
# Never publishes to the robot. Refuses (exit 3) unless: calibration status allowed by render_fastlio_cfg --mode live-e2-body,
# a /lowstate publisher exists, no /Odometry or /sci_alt_imu/* publisher exists, the bridge becomes READY within 15 s.
source "$(dirname "$0")/common.sh"
run="${1:?RUN_DIR}"; calib="${2:-$ALT_WS/config/calib/alt_imu_extrinsic_DYNFIELD_20261030_CANDIDATE.yaml}"
mkdir -p "$run"
ros_env live "$run"
python3 -I "$ALT_WS/scripts/render_fastlio_cfg.py" "$calib" "$run/fastlio_alt.yaml" --mode live-e2-body || die "render refused"
sha256sum "$calib" "$run/fastlio_alt.yaml" "$OPS_FASTLIO_CFG" > "$run/inputs.sha256"
python3 - "$run/fastlio_alt.yaml" "$OPS_FASTLIO_CFG" "$run/adapter_params.yaml" "$ALT_WS/e2_compat" <<'PY' || die "adapter transform failed"
import sys, hashlib
sys.path.insert(0, sys.argv[4]); import alt_to_l1_adapter as AD
R, t = AD.l1_from_body(sys.argv[1], sys.argv[2])
tag = hashlib.sha256((repr(R.round(12).tolist()) + repr(t.round(12).tolist())).encode()).hexdigest()[:12]
f = lambda v: "[" + ", ".join("%.12f" % x for x in v) + "]"
open(sys.argv[3], "w").write("sci_alt_l1_adapter:\n  ros__parameters:\n    R_1b: %s\n    t_1b: %s\n    transform_tag: \"%s\"\n" % (f(R.ravel()), f(t), tag))
print("adapter T_1b tag", tag)
PY
graph=$(ros2 topic info /lowstate 2>/dev/null | awk '/Publisher count/ {print $3}')
[ "${graph:-0}" -ge 1 ] || die "no /lowstate publisher (Go2 body IMU)"
for tp in /Odometry /sci_alt_imu/Odometry /sci_alt_imu/imu; do
  c=$(ros2 topic info "$tp" 2>/dev/null | awk '/Publisher count/ {print $3}'); [ "${c:-0}" = 0 ] || die "$tp already has a publisher"; done
pids=()
cleanup() { trap - EXIT INT TERM; for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 2
            for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done; say "body mode stopped"; }
trap cleanup EXIT; trap 'exit 130' INT TERM
setsid ros2 launch sci_alt_imu_bridge alt_imu_bridge.launch.py log_dir:="$run" > "$run/bridge.out" 2>&1 < /dev/null & pids+=($!)
ok=0; for _ in $(seq 1 30); do sleep 0.5; tail -1 "$run/bridge_health.jsonl" 2>/dev/null | grep -q '"health":"READY"' && { ok=1; break; }; done
[ $ok = 1 ] || die "bridge not READY within 15 s (see $run/bridge_health.jsonl)"
setsid ros2 run sci_alt_imu_bridge l1_adapter_node --ros-args --params-file "$run/adapter_params.yaml" -p status_file:="$run/adapter_status.json" \
  > "$run/adapter.out" 2>&1 < /dev/null & pids+=($!)
say "body mode: bridge READY, adapter started, starting FAST-LIO (alt config) — log follows"
setsid ros2 run fast_lio fastlio_mapping --ros-args -r __node:=sci_alt_laser_mapping --params-file "$run/fastlio_alt.yaml" "${ALT_TOPICS_REMAP[@]}" \
  < /dev/null & pids+=($!)
wait -n "${pids[@]}"; rc=$?
say "a body-mode child exited (rc $rc) — stopping the others"
exit 4
