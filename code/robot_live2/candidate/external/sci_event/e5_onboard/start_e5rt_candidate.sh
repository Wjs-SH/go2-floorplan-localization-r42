#!/usr/bin/env bash
# E5-RT candidate launcher — NOT EXECUTED in this work (no device / ROS start was approved). Review before use.
# Differences to start_e5_onboard.sh: (1) SCI_WS is forced to the candidate tree AFTER sci_env.sh (which resets it to ~/sci_ws),
# (2) overrides come from explicit profile files, (3) the import path is verified before anything starts.
#   SCI_E5RT_TREE=<tree> bash start_e5rt_candidate.sh [--pantilt-execute]      (pan/tilt still needs SCI_PANTILT_APPROVED=1)
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
TREE="${SCI_E5RT_TREE:-$(cd "$HERE/../../.." && pwd)}"
set -a; . "$TREE/config/e5rt_profiles/kci_20261017_field.env"; . "$TREE/config/e5rt_profiles/e5rt_candidate.env"; set +a
export SCI_WS_DEPLOY="$TREE"
# import check: the candidate e5_onboard must load e5_node/e5rt from $TREE (fails loudly if sci_env.sh or PYTHONPATH mixes trees)
( set +u; source "$HOME/sci_ws/src/sci_bringup/sci_env.sh" >/dev/null; export SCI_WS="$TREE"
  "$SCI_PY" - <<PY
import os, sys
sys.path.insert(0, os.path.join("$TREE", "src", "sci_visual_observation"))
import e5rt, importlib.util as u
spec = u.find_spec("e5_node")
assert spec.origin.startswith("$TREE/"), spec.origin
assert e5rt.__file__.startswith("$TREE/"), e5rt.__file__
print("import check OK:", spec.origin, e5rt.__file__)
PY
)
exec bash "$HERE/start_e5_onboard.sh" "$@"
