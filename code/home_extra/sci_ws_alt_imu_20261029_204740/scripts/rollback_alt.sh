#!/bin/bash
# rollback_alt.sh [--dry-run] — stop every still-running alt process (ownership-verified), verify protected originals.
# The alt path never modified operational files, so "rollback" = stop the alt processes; nothing is deleted.
source "$(dirname "$0")/common.sh"
dry=0; [ "${1:-}" = --dry-run ] && dry=1
for run in "$ALT_WS"/runs/*/; do
  for f in "$run"owned/*.pid; do
    [ -e "$f" ] || continue; read -r p s _ < "$f"
    if [ "$(proc_start_time "$p")" = "$s" ]; then
      if [ $dry = 1 ]; then say "WOULD STOP $(basename "$f" .pid) pid $p ($run)"; else owned_stop "${run%/}"; break; fi
    fi
  done
done
if bad=$(protected_hash_check); then say "protected originals unchanged (FAST-LIO src/install/config, ctl/console/lio2/e2_feed, bashrc, units)";
else say "PROTECTED CHANGE DETECTED:"; echo "$bad"; say "restore from manifest $ALT_REPORT_DIR/01_ORIGINAL_SNAPSHOT_MANIFEST.sha256 + rollback/ backups"; exit 4; fi
say "operational FAST-LIO path unaffected: ctl.py still uses $OPS_FASTLIO_CFG ($(sha256sum "$OPS_FASTLIO_CFG" | cut -c1-16)…)"
[ $dry = 1 ] && say "dry-run: nothing was signalled"
