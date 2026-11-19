#!/bin/bash
# stop_archive_alt.sh RUN_DIR — stops ONLY processes recorded in RUN_DIR/owned (pid + starttime), then summarises and hashes.
source "$(dirname "$0")/common.sh"
run="${1:?RUN_DIR}"; [ -d "$run/owned" ] || die "not an alt run dir: $run"
owned_stop "$run"
watch="$HOME/sci_ws/reports/fastlio_preflight_20261028_192027/scripts/mat_pre_watch.py"
[ -s "$run/fastlio_Log/mat_pre.txt" ] && python3 -I "$watch" "$run/fastlio_Log/mat_pre.txt" --out "$run/mat_pre_per_second.csv" > "$run/mat_pre_summary.txt" 2>&1
[ -d "$run/out_bag" ] && python3 -I "$ALT_WS/tests/verify_bridge_output.py" "$run/out_bag" "$run/out_bag" "$run/verify_bridge_output.json" >/dev/null 2>&1
(cd "$run" && find . -type f -not -path "./roslog/*" ! -name SHA256SUMS.txt | sort | xargs sha256sum > SHA256SUMS.txt)
if protected_hash_check > "$run/protected_check.txt"; then say "protected files unchanged"; else say "WARNING: protected file change, see $run/protected_check.txt"; fi
say "archived in place: $run (SHA256SUMS.txt)"
