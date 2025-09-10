import sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from app.analysis.loader import load_runs

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data"); return
    g = df.groupby("pipeline")["composite"].mean().sort_values(ascending=False)
    plt.figure(figsize=(7,4))
    g.plot(kind="bar")
    plt.ylim(0,1)
    plt.ylabel("Avg composite")
    plt.title("Pipeline performance (avg over all runs)")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_pipeline_performance.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
