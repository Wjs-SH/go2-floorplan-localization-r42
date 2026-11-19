#!/usr/bin/env bash
# stop_event_console.sh --trial <dir> — SIGINT the console started by start_event_console.sh (its own pid file only).
# aim_console handles KeyboardInterrupt by sending F0 (focus motor stop) and releasing the camera. No servo command.
set -uo pipefail
[ "${1:-}" = --trial ] && TRIAL="$2" || { echo "usage: $0 --trial <dir>"; exit 2; }
P="$TRIAL/console/console.pid"
[ -f "$P" ] || { echo "no pid file $P"; exit 0; }
PID=$(cat "$P")
ps -o args= -p "$PID" | grep -q aim_console.py || { echo "pid $PID is not aim_console — not signalling"; exit 4; }
kill -INT "$PID"
for i in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -0 "$PID" 2>/dev/null && { echo "console still alive after 10 s (pid $PID) — left running, inspect manually"; exit 5; }
date -Is > "$TRIAL/console/stopped_at.txt"; rm -f "$P"; echo "console stopped"
# copy the console's own outputs (bundle out/aim) into the trial, never moving originals
mkdir -p "$TRIAL/console/aim"
cp -n "$HOME/kci_event_handoff_20261009/out/aim/aim_log.jsonl" "$TRIAL/console/aim_log.jsonl" 2>/dev/null || true
cp -n "$HOME"/kci_event_handoff_20261009/out/aim/aim_*.jpg "$TRIAL/console/aim/" 2>/dev/null || true
