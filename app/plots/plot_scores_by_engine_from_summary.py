import sys, json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def main(summary_path: str, fraction_target: float = 1.0):
    data = json.loads(Path(summary_path).read_text())
    if not data: 
        print("No data in summary"); return
    by = defaultdict(lambda: defaultdict(list))  # engine -> dataset -> [(fraction, score)]
    for r in data:
        by[r["engine"]][r["dataset"]].append((r["fraction"], r["composite"]))
    engines = sorted(by.keys())
    # pick closest to target fraction per dataset then average across datasets
    def pick_closest(arr, tgt):
        return min(arr, key=lambda t: abs(t[0]-tgt))[1]
    scores = []
    for eng in engines:
        vals = []
        for ds, arr in by[eng].items():
            vals.append(pick_closest(arr, fraction_target))
        scores.append(sum(vals)/len(vals))
    # plot
    idx = np.arange(len(engines))
    plt.figure(figsize=(max(7, len(engines)*0.8), 4))
    plt.bar(idx, scores)
    plt.xticks(idx, engines, rotation=25, ha="right")
    plt.ylim(0,1); plt.ylabel("Avg best composite"); plt.title(f"Per-engine average @ ~{int(fraction_target*100)}% (across datasets)")
    plt.tight_layout()
    out = Path(summary_path).parent / f"scores_by_engine_{int(fraction_target*100)}.png"
    plt.savefig(out, dpi=160); plt.close(); print("Saved:", out)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv)>1 else "runs/summary_matrix.json"
    frac = float(sys.argv[2]) if len(sys.argv)>2 else 1.0
    main(p, frac)
