from __future__ import annotations
import argparse, json, random
from pathlib import Path
from ..data.datasets import load_many
from ..optimize.baseline import run_baseline
from ..optimize.learned import run_learned
from ..core.config import GLOBAL_SEED

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["samsum","xsum","cnn_dailymail"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3, help="number of runs per dataset")
    ap.add_argument("--offset_step", type=int, default=12, help="shift start index each repeat to get different slices")
    ap.add_argument("--phase", choices=["baseline","learned","both"], default="both")
    args = ap.parse_args()

    random.seed(GLOBAL_SEED)

    for ds in args.datasets:
        for r in range(args.repeats):
            start = r * args.offset_step
            xs, ys = load_many(ds, n=args.n, start=start)
            if args.phase in ("baseline","both"):
                out = run_baseline(ds, xs, ys, generations=args.gens, population_size=args.pop, top_k=args.topk, patience=args.patience)
                print(f"[baseline:{ds} r{r}] run_id={out['run_id']} best={out['best']['metrics']['composite']:.3f}")
            if args.phase in ("learned","both"):
                out = run_learned(ds, xs, ys, generations=args.gens, population_size=args.pop, top_k=args.topk, patience=args.patience)
                print(f"[learned:{ds} r{r}] run_id={out['run_id']} best={out['best']['metrics']['composite']:.3f}")

if __name__ == "__main__":
    main()
