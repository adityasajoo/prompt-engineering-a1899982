#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
mkdir -p runs
pip freeze > runs/ENV.txt
echo "Saved environment to runs/ENV.txt"
