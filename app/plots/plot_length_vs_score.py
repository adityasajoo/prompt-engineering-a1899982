import sys
from pathlib import Path
import matplotlib.pyplot as plt
from app.analysis.loader import load_runs

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data"); return
    plt.figure(figsize=(7,4))
    plt.scatter(df["output_len_words"], df["composite"], s=8, alpha=0.35)
    plt.xlabel("Output length (words)")
    plt.ylabel("Composite")
    plt.title("Length vs Composite (all runs)")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_length_vs_composite.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
