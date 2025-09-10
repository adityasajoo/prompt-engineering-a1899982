import sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from app.analysis.loader import load_runs, load_mutations, aggregate_improvements_all_metrics

def main(runs_dir: str):
    evals = load_runs(runs_dir)
    muts = load_mutations(runs_dir)
    df = aggregate_improvements_all_metrics(evals, muts)
    if df.empty:
        print("No mutation data"); return
    # average delta per op
    cols = ["delta_rougeL","delta_bleu","delta_bertscore","delta_composite"]
    g = df.groupby("op")[cols].mean().sort_values("delta_composite", ascending=False)
    # plot grouped bars (one plot)
    xs = np.arange(len(g.index))
    w  = 0.2
    plt.figure(figsize=(max(8, len(xs)*0.6), 5))
    for i,c in enumerate(cols):
        plt.bar(xs + i*w - w*1.5, g[c].values, width=w, label=c.replace("delta_","Δ "))
    plt.axhline(0, color="#333", linewidth=0.8)
    plt.xticks(xs, g.index, rotation=25, ha="right")
    plt.ylabel("Average delta")
    plt.title("Average metric deltas by mutation operator")
    plt.legend()
    plt.tight_layout()
    out = Path(runs_dir)/"plot_op_metric_deltas.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")

