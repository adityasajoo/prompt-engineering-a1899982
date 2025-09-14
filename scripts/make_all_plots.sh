#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true

# Rebuild benchmarks
[ -f runs/benchmark_baseline.json ] && python -m app.plots.plot_benchmark runs/benchmark_baseline.json || true
[ -f runs/benchmark_learned.json ]  && python -m app.plots.plot_benchmark runs/benchmark_learned.json  || true
if [ -f runs/benchmark_baseline.json ] && [ -f runs/benchmark_learned.json ]; then
  python -m app.plots.plot_compare_benchmarks runs/benchmark_baseline.json runs/benchmark_learned.json
fi

# Aggregates across all runs
python -m app.plots.plot_operator_winrate runs || true
python -m app.plots.plot_improvement_hist runs || true
python -m app.plots.plot_seed_contributions runs || true
python -m app.plots.plot_violin_by_dataset runs || true
python -m app.plots.plot_confusion_matrix runs || true

python -m app.plots.plot_metric_correlations runs || true
python -m app.plots.plot_length_vs_score runs || true
python -m app.plots.plot_box_by_dataset_phase runs || true
python -m app.plots.plot_pipeline_performance runs || true
python -m app.plots.plot_op_metric_deltas runs || true
python -m app.plots.plot_best_mean_median_by_gen runs || true
python -m app.plots.plot_metric_histograms runs || true


# Finally, the report
python -m app.reports.make_report runs
echo "Open: runs/report.html"
