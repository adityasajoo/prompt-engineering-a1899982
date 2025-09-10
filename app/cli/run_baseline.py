from __future__ import annotations
import argparse, json
from pathlib import Path
from ..data.datasets import load_many
from ..optimize.baseline import run_baseline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["samsum","xsum","cnn_dailymail"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    bench = []
    for ds in args.datasets:
        xs, ys = load_many(ds, n=args.n, start=args.start)
        out = run_baseline(ds, xs, ys, generations=args.gens, population_size=args.pop, top_k=args.topk, patience=args.patience)
        print(f"[baseline:{ds}] run_id={out['run_id']} best_composite={out['best']['metrics']['composite']:.3f}")
        bench.append({"dataset": ds, "label": "baseline", "run_id": out["run_id"], "best_cfg": out["best"]["cfg"], "composite": out["best"]["metrics"]["composite"]})
    Path("runs/benchmark_baseline.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    print("Saved: runs/benchmark_baseline.json")

if __name__ == "__main__":
    main()
