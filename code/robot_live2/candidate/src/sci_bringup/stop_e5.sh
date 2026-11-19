#!/usr/bin/env bash
# Stop ONLY processes started by start_e5.sh (PID files). Does not touch drivers, FAST-LIO, ROPUS or anything else.
source "$(dirname "$0")/sci_env.sh"
for n in reader e5_node ui llama_server; do
  f=$SCI_WS/state/pids/$n.pid; [ -f $f ] || continue; p=$(cat $f)
  if kill -0 $p 2>/dev/null && grep -q sci_ws /proc/$p/cmdline 2>/dev/null; then kill -TERM $p && echo "stopped $n ($p)"; fi
  rm -f $f
done
pkill -TERM -f "$SCI_WS/external/llama.cpp/build/bin/llama-server" 2>/dev/null && echo "stopped llama-server" || true
