# source me: SCI runtime environment (does not modify system state)
set +u; source /opt/ros/foxy/setup.bash; set -u 2>/dev/null || true
export SCI_WS=$HOME/sci_ws
export SCI_PY=$SCI_WS/.venv/bin/python
unset PYTHONNOUSERSITE            # handoff trap: would hide ~/.local transformers
export SCI_SHM=/dev/shm/sci_e5
mkdir -p $SCI_WS/state/pids $SCI_WS/logs
