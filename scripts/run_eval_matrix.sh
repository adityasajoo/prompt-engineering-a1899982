#!/usr/bin/env bash
set -euo pipefail

# Usage (defaults shown):
#   bash scripts/run_eval_matrix.sh
#   ENGINES="extractive llama3" TOPK=3 bash scripts/run_eval_matrix.sh
#   OFFLINE_ONLY=1 FRACTIONS="0.1 0.25 0.5 1.0" bash scripts/run_eval_matrix.sh

# Activate venv if present
source .venv/bin/activate 2>/dev/null || true

# Config (env‑overridable)
: "${DATASETS:=samsum xsum cnn_dailymail}"
: "${FRACTIONS:=0.1 0.25 0.5 1.0}"
: "${ENGINES:=extractive}"
: "${GENS:=4}"
: "${POP:=10}"
: "${TOPK:=4}"
: "${PATIENCE:=2}"
: "${SEED:=42}"
: "${RUNS_DIR:=runs}"

# Optional: offline tiny datasets only (no HF downloads)
: "${OFFLINE_ONLY:=0}"
export OFFLINE_ONLY

# Headless plotting
export MPLBACKEND=Agg
# Limit math library threads for snappier small jobs (tweak as needed)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

echo "▶ Running sweep:"
echo "  datasets:  ${DATASETS}"
echo "  fractions: ${FRACTIONS}"
echo "  engines:   ${ENGINES}"
echo "  gens/pop/topk/patience: ${GENS}/${POP}/${TOPK}/${PATIENCE}"
echo "  runs_dir:  ${RUNS_DIR}"

python -m app.cli.run_fraction_matrix \
  --datasets ${DATASETS} \
  --fractions ${FRACTIONS} \
  --engines ${ENGINES} \
  --gens ${GENS} --pop ${POP} --topk ${TOPK} --patience ${PATIENCE} \
  --seed ${SEED} \
  --runs_dir "${RUNS_DIR}"

SUMMARY_JSON="${RUNS_DIR}/summary_matrix.json"
echo "▶ Generating summary plots from ${SUMMARY_JSON}"
python -m app.plots.plot_fraction_curves_from_summary      "${SUMMARY_JSON}" || true
python -m app.plots.plot_scores_by_dataset_from_summary    "${SUMMARY_JSON}" || true
python -m app.plots.plot_scores_by_engine_from_summary     "${SUMMARY_JSON}" || true
python -m app.plots.plot_combo_heatmap_from_summary        "${SUMMARY_JSON}" || true

echo "▶ Building HTML report"
python -m app.reports.make_report "${RUNS_DIR}" || true
echo "Open: ${RUNS_DIR}/report.html"
