#!/bin/bash
# LIVE-FIELD2 single control entry (LIVE1 ctl + E2/E3/E4 modules). No argument / check / status: starts and stops NOTHING.
# Never calls kci_field_up/down, never touches ROPUS sensor services, pan-tilt, focus, HOME, serial, IMX477 or Go2 motion.
R="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$R/LIVE_FIELD2_WORKSPACE.json" ] || { echo "[live2_ctl] REFUSED: not a LIVE-FIELD2 workspace"; exit 4; }
export LIVE1_ROOT="$R"
exec /usr/bin/python3 -I -B "$R/candidate/live1/ctl.py" "${@:-check}"
