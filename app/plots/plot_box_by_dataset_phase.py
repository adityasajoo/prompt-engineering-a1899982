import sys
from pathlib import Path
import matplotlib.pyplot as plt
from app.analysis.loader import load_runs

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data"); return
    df = df.dropna(subset=["dataset","phase"])
    datasets = sorted(df["dataset"].unique())
    phases = ["baseline","learned"]
    data = []
    labels = []
    for d in datasets:
        for p in phases:
            vals = df[(df["dataset"]==d) & (df["phase"]==p)]["composite"].values
            if len(vals)==0: continue
            data.append(vals); labels.append(f"{d}\n{p}")
    if not data:
        print("No data for boxplot"); return
    plt.figure(figsize=(max(8, len(labels)*0.9), 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1,len(labels)+1), labels, rotation=25, ha="right")
    plt.ylabel("Composite")
    plt.title("Composite distribution by dataset & phase")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_box_dataset_phase.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
