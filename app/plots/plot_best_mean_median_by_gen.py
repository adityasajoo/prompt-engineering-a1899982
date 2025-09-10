import sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from app.analysis.loader import load_runs

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data"); return
    # mean per (run, gen)
    g = df.groupby(["run_id","gen"])["composite"].agg(["mean","median","max"]).reset_index()
    # aggregate across runs (mean of means/medians/maxes)
    G = g.groupby("gen")[["mean","median","max"]].mean().reset_index()
    plt.figure(figsize=(7,4))
    plt.plot(G["gen"], G["max"], marker="o", label="best (avg of per-run bests)")
    plt.plot(G["gen"], G["mean"], marker="o", label="mean")
    plt.plot(G["gen"], G["median"], marker="o", label="median")
    plt.xlabel("Generation")
    plt.ylabel("Composite")
    plt.ylim(0,1)
    plt.title("Aggregate learning curves (across all runs)")
    plt.legend()
    plt.tight_layout()
    out = Path(runs_dir)/"plot_best_mean_median_by_gen.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
