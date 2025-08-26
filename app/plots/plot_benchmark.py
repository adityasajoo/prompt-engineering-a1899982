import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main(json_path):
    p = Path(json_path)
    data = json.loads(p.read_text())
    # group by dataset then model
    datasets = sorted({x["dataset"] for x in data})
    models   = sorted({x["model"] for x in data})
    for ds in datasets:
        items = [x for x in data if x["dataset"] == ds]
        by_model = {x["model"]: x["scores"]["composite"] if "composite" in x["scores"] else x["composite"] for x in items}
        vals = [by_model.get(m, 0.0) for m in models]
        plt.figure(figsize=(8,4))
        xs = np.arange(len(models))
        plt.bar(xs, vals)
        plt.xticks(xs, models, rotation=25, ha="right")
        plt.ylabel("Composite"); plt.ylim(0,1)
        plt.title(f"Composite by model — {ds}")
        plt.tight_layout()
        out = p.parent / f"plot_{p.stem}_{ds}.png"
        plt.savefig(out, dpi=160); plt.close()
        print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1])

# Usage: python -m app.plots.plot_benchmark runs/benchmark_*.json
