import sys
from pathlib import Path
import matplotlib.pyplot as plt
from app.analysis.loader import load_runs, load_mutations, aggregate_improvements
import numpy as np

def main(runs_dir: str):
    evals = load_runs(runs_dir)
    muts = load_mutations(runs_dir)
    df = aggregate_improvements(evals, muts).dropna(subset=["improvement"])
    if df.empty:
        print("No data for win-rate plot"); return
    df["win"] = (df["improvement"] > 0.0).astype(int)
    wr = df.groupby("op")["win"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    wr.plot(kind="bar")
    plt.ylim(0,1); plt.ylabel("Win-rate (improvement>0)"); plt.title("Mutation operator win-rate")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_operator_winrate.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
