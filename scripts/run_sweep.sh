#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

# 1) many baseline runs
python -m app.cli.run_sweep --phase baseline --repeats 5 --n 12 --gens 3 --pop 8 --topk 3

# 2) train the non-LLM model on all those logs
python -m app.cli.train_model --runs_dir runs --out policies/mutation_classifier.pkl

# 3) many learned runs
python -m app.cli.run_sweep --phase learned --repeats 5 --n 12 --gens 3 --pop 8 --topk 3

# 4) generate plots from all runs
python -m app.plots.plot_operator_winrate runs
python -m app.plots.plot_improvement_hist runs
python -m app.plots.plot_seed_contributions runs
python -m app.plots.plot_violin_by_dataset runs
python -m app.plots.plot_confusion_matrix runs

# 5) per-run progress plots already produced by earlier scripts, but if you need:
# for d in runs/*; do [ -d "$d" ] && python -m app.plots.plot_run_progress "$d"; done

# 6) final HTML report
python -m app.reports.make_report runs
