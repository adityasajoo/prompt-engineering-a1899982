import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from app.analysis.loader import load_runs

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data found"); return
    # violin per dataset, composite distribution across all runs/gens/examples
    datasets = sorted(df["dataset"].dropna().unique())
    data = [df[df["dataset"]==d]["composite"].values for d in datasets]
    if not data:
        print("No data for violins"); return
    plt.figure(figsize=(8,4))
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(range(1, len(datasets)+1), datasets, rotation=15)
    plt.ylabel("Composite")
    plt.title("Composite distribution by dataset")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_violin_by_dataset.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
