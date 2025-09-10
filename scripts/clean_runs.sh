#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$ROOT/runs" "$ROOT/logs"
mkdir -p "$ROOT/runs" "$ROOT/logs"
echo "✓ Cleaned runs/ and logs/"
