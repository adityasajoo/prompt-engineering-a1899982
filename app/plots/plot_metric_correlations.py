import sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from app.analysis.loader import load_runs

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data"); return
    cols = ["rougeL","bleu","bertscore","composite","output_len_words"]
    M = df[cols].corr().values
    plt.figure(figsize=(6,5))
    im = plt.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=25, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Metric & length correlations")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_metric_correlations.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
