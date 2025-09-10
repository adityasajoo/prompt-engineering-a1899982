import sys
from pathlib import Path
import matplotlib.pyplot as plt
from app.analysis.loader import load_runs, load_mutations, aggregate_improvements

def main(runs_dir: str):
    evals = load_runs(runs_dir)
    muts = load_mutations(runs_dir)
    df = aggregate_improvements(evals, muts).dropna(subset=["improvement"])
    if df.empty:
        print("No data for improvement histogram"); return
    plt.figure(figsize=(7,4))
    plt.hist(df["improvement"], bins=40)
    plt.xlabel("Composite improvement (child - parent)")
    plt.ylabel("Count")
    plt.title("Distribution of improvements across all mutations")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_improvement_hist.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
