# app/reports/make_report.py
import sys
from pathlib import Path
from datetime import datetime
from jinja2 import Template

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prompt-Eng Report</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
    h1, h2 { margin: 0.5em 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 18px; }
    figure { margin: 0; border: 1px solid #eee; padding: 8px; border-radius: 8px; background: #fff; }
    figcaption { font-size: 0.9em; color: #444; margin-top: 6px; }
    .muted { color:#666; }
  </style>
</head>
<body>
  <h1>Prompt-Eng: Baseline → Model → Learned</h1>
  <p class="muted"><b>Generated:</b> {{ timestamp }} • <b>Project:</b> {{ cwd }}</p>

  <h2>Key Benchmarks</h2>
  {% if bench_imgs %}
  <div class="grid">
    {% for img, cap in bench_imgs %}
    <figure>
      <img src="{{ img }}" width="100%" />
      <figcaption>{{ cap }}</figcaption>
    </figure>
    {% endfor %}
  </div>
  {% else %}
  <p class="muted">No benchmark plots found. Run the plotting commands first.</p>
  {% endif %}

  <h2>Run Diagnostics</h2>
  {% if diag_imgs %}
  <div class="grid">
    {% for img, cap in diag_imgs %}
    <figure>
      <img src="{{ img }}" width="100%" />
      <figcaption>{{ cap }}</figcaption>
    </figure>
    {% endfor %}
  </div>
  {% else %}
  <p class="muted">No diagnostic plots found yet.</p>
  {% endif %}
</body>
</html>
"""

def collect_images(runs_dir: Path):
    bench, diag = [], []

    # Benchmarks
    for p in sorted(runs_dir.glob("plot_benchmark_baseline_*.png")):
        ds = p.stem.replace("plot_benchmark_baseline_", "")
        bench.append((str(p), f"Baseline benchmark — {ds}"))
    for p in sorted(runs_dir.glob("plot_benchmark_learned_*.png")):
        ds = p.stem.replace("plot_benchmark_learned_", "")
        bench.append((str(p), f"Learned benchmark — {ds}"))
    for p in sorted(runs_dir.glob("compare_*.png")):
        ds = p.stem.replace("compare_", "")
        bench.append((str(p), f"Baseline vs Learned — {ds}"))

    # Global diagnostics & new eval visuals
    for name, cap in [
        ("plot_operator_winrate.png",        "Mutation operator win-rate"),
        ("plot_improvement_hist.png",        "Improvement distribution across all mutations"),
        ("plot_seed_contributions.png",      "Seed strategy contributions (Gen 0)"),
        ("plot_violin_by_dataset.png",       "Composite distribution by dataset"),
        ("plot_confusion_matrix.png",        "Predicted vs Best op (Learned)"),
        ("plot_metric_correlations.png",     "Metric & length correlations"),
        ("plot_length_vs_composite.png",     "Length vs Composite"),
        ("plot_box_dataset_phase.png",       "Composite distribution by dataset & phase"),
        ("plot_pipeline_performance.png",    "Pipeline performance"),
        ("plot_op_metric_deltas.png",        "Average metric deltas by operator"),
        ("plot_best_mean_median_by_gen.png", "Aggregate learning curves"),
        ("plot_hist_rougeL.png",             "ROUGE-L distribution"),
        ("plot_hist_BLEU.png",               "BLEU distribution"),
        ("plot_hist_BERTScore.png",          "BERTScore distribution"),
        ("plot_hist_composite.png",          "Composite distribution"),
        ("plot_strategy_bars_overall.png",        "Strategy — average composite (overall)"),
        ("plot_strategy_boxplots.png",            "Strategy — distribution (overall)"),
        ("plot_strategy_progress_by_gen.png",     "Strategy — learning curves (avg by gen)"),
        ("plot_dataset_strategy_panels_samsum.png","Strategy panels — SAMSum"),
        ("plot_dataset_strategy_panels_xsum.png",  "Strategy panels — XSum"),
        ("plot_dataset_strategy_panels_cnn_dailymail.png","Strategy panels — CNN/DM"),
        ("fraction_curve_overall.png",        "Composite vs fraction — overall"),
        ("fraction_curve_samsum.png",         "Composite vs fraction — SAMSum"),
        ("fraction_curve_xsum.png",           "Composite vs fraction — XSum"),
        ("fraction_curve_cnn_dailymail.png",  "Composite vs fraction — CNN/DM"),
        ("scores_by_dataset_100.png",         "Per-dataset scores @ 100%"),
        ("scores_by_engine_100.png",          "Per-engine average @ 100%"),
        ("heatmap_combo_100.png",             "Heatmap: dataset × engine @ 100%"),
    ]:
        p = runs_dir / name
        if p.exists():
            diag.append((str(p), cap))

    # Per-run learning curves
    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir(): continue
        for name, cap in [("plot_best_composite.png","Best composite vs generation"),
                          ("plot_best_metrics.png","Best ROUGE/ BLEU/ BERTScore vs generation")]:
            p = rd / name
            if p.exists():
                diag.append((str(p), f"{cap} — {rd.name}"))

    return bench, diag

def main(runs_path: str):
    runs_dir = Path(runs_path)
    bench_imgs, diag_imgs = collect_images(runs_dir)
    html = Template(TEMPLATE).render(
        cwd=str(Path().cwd()),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        bench_imgs=bench_imgs,
        diag_imgs=diag_imgs
    )
    out = runs_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print("Saved:", out)
    print(f"Bench images found: {len(bench_imgs)} • Diagnostic images found: {len(diag_imgs)}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
