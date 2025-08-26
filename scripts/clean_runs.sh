#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "This will REMOVE all run artifacts and app logs under:"
echo "  $ROOT/runs  and  $ROOT/logs"
read -p "Type 'YES' to proceed: " ans
if [[ "$ans" != "YES" ]]; then echo "Abort."; exit 1; fi
rm -rf "$ROOT/runs"/* "$ROOT/logs"/* || true
mkdir -p "$ROOT/runs" "$ROOT/logs"
echo "✓ Cleaned."
