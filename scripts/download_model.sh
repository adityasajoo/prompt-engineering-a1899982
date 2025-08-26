#!/usr/bin/env bash
set -euo pipefail

# Requires: huggingface-cli login (first time only)
# Example models (choose ONE to start):
# 1) Llama 3 8B Instruct Q4_K_M (license acceptance required on HF)
# 2) Mistral 7B Instruct Q4_K_M (usually no gating)

MODEL_DIR="./models"
mkdir -p "$MODEL_DIR"

case "${1:-mistral}" in
  llama3)
    echo "→ Downloading Llama 3 8B Instruct (Q4_K_M)…"
    # After accepting the license on HF website, run: huggingface-cli login
    hf download  bartowski/Llama-3.2-3B-Instruct-GGUF \
      --include "Llama-3.2-3B-Instruct-Q6_K.gguf" --local-dir "$MODEL_DIR"
    ;;
  mistral)
    echo "→ Downloading Mistral 7B Instruct (Q4_K_M)…"
    hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
      mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir "$MODEL_DIR"
    ;;
  qwen25)
    echo "→ Downloading Qwen2.5 3B Instruct (Q4_K_M)…"
    hf download Qwen/Qwen2.5-3B-Instruct-GGUF \
      Qwen2.5-3B-Instruct-Q4_K_M.gguf --local-dir "$MODEL_DIR"
    ;;
  *)
    echo "Usage: $0 {mistral|llama3|qwen25}"
    exit 1
    ;;
esac

echo "Done. Files in $MODEL_DIR:"
ls -lh "$MODEL_DIR"
