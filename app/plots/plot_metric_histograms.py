import sys
from pathlib import Path
import matplotlib.pyplot as plt
from app.analysis.loader import load_runs

def _hist(vals, title, out_path):
    plt.figure(figsize=(7,4))
    plt.hist(vals, bins=40)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()
    print("Saved:", out_path)

def main(runs_dir: str):
    df = load_runs(runs_dir)
    if df.empty:
        print("No eval data"); return
    p = Path(runs_dir)
    _hist(df["rougeL"].values,   "ROUGE-L distribution (all evals)", p/"plot_hist_rougeL.png")
    _hist(df["bleu"].values,     "BLEU distribution (all evals)",    p/"plot_hist_BLEU.png")
    if (df["bertscore"]>0).any():
        _hist(df["bertscore"].values, "BERTScore distribution (all evals)", p/"plot_hist_BERTScore.png")
    _hist(df["composite"].values,"Composite distribution (all evals)", p/"plot_hist_composite.png")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
