import sys, json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def main(summary_path: str, fraction_target: float = 1.0):
    data = json.loads(Path(summary_path).read_text())
    if not data: 
        print("No data in summary"); return
    # choose rows nearest to requested fraction per dataset × engine
    by = defaultdict(lambda: defaultdict(list))
    for r in data:
        by[r["dataset"]][r["engine"]].append((r["fraction"], r["composite"]))
    out_dir = Path(summary_path).parent
    # overall: average across engines or plot per-engine stacks per dataset
    engines = sorted({r["engine"] for r in data})
    datasets = sorted(by.keys())

    # Build matrix [dataset x engine]
    M = np.zeros((len(datasets), len(engines)), dtype=float)
    for i, ds in enumerate(datasets):
        for j, eng in enumerate(engines):
            arr = by[ds][eng]
            if not arr: continue
            # pick entry with fraction closest to target
            f, s = min(arr, key=lambda t: abs(t[0]-fraction_target))
            M[i,j] = s

    # bars per dataset with engine legend
    idx = np.arange(len(datasets))
    w = 0.8 / max(1, len(engines))
    plt.figure(figsize=(max(8, len(datasets)*0.8), 4.5))
    for j, eng in enumerate(engines):
        plt.bar(idx + j*w - (len(engines)-1)*w/2, M[:,j], width=w, label=eng)
    plt.xticks(idx, datasets, rotation=25, ha="right")
    plt.ylim(0,1); plt.ylabel("Best composite"); plt.title(f"Per-dataset scores @ ~{int(fraction_target*100)}%")
    plt.legend()
    plt.tight_layout()
    out = out_dir / f"scores_by_dataset_{int(fraction_target*100)}.png"
    plt.savefig(out, dpi=160); plt.close(); print("Saved:", out)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv)>1 else "runs/summary_matrix.json"
    frac = float(sys.argv[2]) if len(sys.argv)>2 else 1.0
    main(p, frac)
