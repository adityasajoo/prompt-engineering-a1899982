#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

# Phase A: baseline (no model)
python -m app.cli.run_baseline --datasets samsum xsum cnn_dailymail --n 15 --gens 3 --pop 6 --topk 3

# Phase B: train non-LLM classifier on logs
python -m app.cli.train_model --runs_dir runs --out policies/mutation_classifier.pkl

# Phase C: learned mode (uses trained classifier)
python -m app.cli.run_learned --datasets samsum xsum cnn_dailymail --n 15 ---gens 3 --pop 6 --topk 3

# Optional: quick plot comparison
python -m app.plots.plot_compare_benchmarks runs/benchmark_baseline.json runs/benchmark_learned.json
