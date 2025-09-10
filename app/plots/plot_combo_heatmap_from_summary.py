import sys, json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

def main(summary_path: str, fraction_target: float = 1.0):
    data = json.loads(Path(summary_path).read_text())
    if not data:
        print("No data in summary"); return
    datasets = sorted({r["dataset"] for r in data})
    engines  = sorted({r["engine"]  for r in data})
    # build table with closest-to-target fraction
    M = np.zeros((len(datasets), len(engines)), dtype=float)
    for i, ds in enumerate(datasets):
        for j, eng in enumerate(engines):
            arr = [ (r["fraction"], r["composite"]) for r in data if r["dataset"]==ds and r["engine"]==eng ]
            if not arr: continue
            f, s = min(arr, key=lambda t: abs(t[0]-fraction_target))
            M[i,j] = s
    plt.figure(figsize=(max(6, len(engines)*0.8), max(4, len(datasets)*0.5)))
    im = plt.imshow(M, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(engines)), engines, rotation=25, ha="right")
    plt.yticks(range(len(datasets)), datasets)
    for i in range(len(datasets)):
        for j in range(len(engines)):
            plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(f"Composite heatmap — dataset × engine @ ~{int(fraction_target*100)}%")
    plt.tight_layout()
    out = Path(summary_path).parent / f"heatmap_combo_{int(fraction_target*100)}.png"
    plt.savefig(out, dpi=160); plt.close(); print("Saved:", out)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv)>1 else "runs/summary_matrix.json"
    frac = float(sys.argv[2]) if len(sys.argv)>2 else 1.0
    main(p, frac)
