#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONUNBUFFERED=1
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --reload
