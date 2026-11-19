#!/bin/bash
# LIVE2 unit tests: student venv python (read-only), tmp/cache under LIVE2 runtime, no ROS env, CUDA hidden.
R="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$R/runtime/tmp" "$R/test_artifacts"
cd "$R/tests" || exit 2
env -i PATH=/usr/bin:/bin HOME="$R/runtime/home" LANG=C.UTF-8 TMPDIR="$R/runtime/tmp" CUDA_VISIBLE_DEVICES= \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  /home/unitree/sci_ws/external/sci_live/venv_gpu/bin/python -B -m pytest -p no:cacheprovider -q --basetemp="$R/runtime/tmp/pytest" "$@" \
  2>&1 | tee "$R/test_artifacts/live2_tests.log"
exit "${PIPESTATUS[0]}"
