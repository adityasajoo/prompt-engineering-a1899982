import sys, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def main(summary_path: str):
    data = json.loads(Path(summary_path).read_text())
    if not data:
        print("No data in summary"); return
    # group by dataset → engine → (fraction → scores)
    G = defaultdict(lambda: defaultdict(list))
    for r in data:
        G[r["dataset"]][r["engine"]].append((r["fraction"], r["composite"]))
    out_dir = Path(summary_path).parent

    for ds, by_eng in G.items():
        plt.figure(figsize=(7,4))
        for eng, arr in by_eng.items():
            arr.sort(key=lambda t: t[0])
            xs = [a for a,_ in arr]
            ys = [b for _,b in arr]
            plt.plot(xs, ys, marker="o", label=eng)
        plt.xlabel("Fraction of dataset")
        plt.ylabel("Best composite")
        plt.ylim(0,1)
        plt.title(f"Composite vs fraction — {ds}")
        plt.legend()
        plt.tight_layout()
        out = out_dir / f"fraction_curve_{ds}.png"
        plt.savefig(out, dpi=160); plt.close()
        print("Saved:", out)

    # overall average across datasets per engine
    H = defaultdict(lambda: defaultdict(list))
    for r in data:
        H[r["engine"]][r["fraction"]].append(r["composite"])
    plt.figure(figsize=(7,4))
    for eng, frac_map in H.items():
        xs = sorted(frac_map.keys())
        ys = [sum(frac_map[f])/len(frac_map[f]) for f in xs]
        plt.plot(xs, ys, marker="o", label=eng)
    plt.xlabel("Fraction of dataset"); plt.ylabel("Avg best composite (across datasets)")
    plt.ylim(0,1); plt.title("Overall composite vs fraction")
    plt.legend(); plt.tight_layout()
    out = out_dir / "fraction_curve_overall.png"
    plt.savefig(out, dpi=160); plt.close(); print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs/summary_matrix.json")
