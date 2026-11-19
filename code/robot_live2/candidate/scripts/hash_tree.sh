#!/usr/bin/env bash
# hash_tree.sh <root> <name> <out.sha256> : sha256 of every regular file under <root>/<name> (symlinks NOT followed), relative paths
set -euo pipefail
cd "$1"; find "$2" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$3"
