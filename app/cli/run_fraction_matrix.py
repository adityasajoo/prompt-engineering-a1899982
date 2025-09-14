# app/cli/run_fraction_matrix.py
from __future__ import annotations
import argparse, math, json, random, os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from app.data.datasets import load_many, offline_size
from app.optimize.baseline import run_baseline
 

def write_manifest_patch(run_dir: Path, patch: Dict[str, Any]):
    man = run_dir / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
    else:
        m = {}
    m.update(patch)
    man.write_text(json.dumps(m, indent=2), encoding="utf-8")

def best_from_run(run_dir: Path) -> Dict[str, Any]:
    best_p = run_dir / "bests.jsonl"
    if best_p.exists():
        last = None
        for line in best_p.read_text().splitlines():
            o = json.loads(line)
            if o.get("event") == "best":
                last = o
        if last:
            return {"cfg": last.get("cfg"),
                    "metrics": last.get("metrics"),
                    "gen": last.get("gen")}
    man = json.loads((run_dir / "manifest.json").read_text())
    return {"cfg": man.get("best_cfg"), "metrics": man.get("best_metrics"), "gen": man.get("best_gen")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets",  nargs="+", default=["samsum","xsum","cnn_dailymail"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1,0.25,0.5,1.0])
    ap.add_argument("--engines",   nargs="+", default=["extractive"], help="labels only; used to tag runs")
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--pop",  type=int, default=10)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    runs_dir = Path(args.runs_dir); runs_dir.mkdir(exist_ok=True)

    summary_rows: List[Dict[str,Any]] = []
    total_steps = len(args.engines) * len(args.datasets) * len(args.fractions)
    step = 0

    for eng in args.engines:
        for ds in args.datasets:
            offline = os.environ.get("OFFLINE_ONLY", "0") == "1"
            if offline:
                total = offline_size(ds)
            else:
                # Fallback: sample a modest upper bound then measure size
                xs_probe, ys_probe = load_many(ds, n=4096, start=0)
                total = len(xs_probe)
            if total == 0:
                step += len(args.fractions)
                continue
            for frac in args.fractions:
                step += 1

                n = max(1, math.ceil(total * frac))
                if offline:
                    xs, ys = load_many(ds, n=n, start=0)
                else:
                    # For online/HF, load exactly n per fraction to avoid huge preloads
                    xs, ys = load_many(ds, n=n, start=0)

                out = run_baseline(ds, xs, ys,
                                   generations=args.gens,
                                   population_size=args.pop,
                                   top_k=args.topk,
                                   patience=args.patience)
                run_id = out["run_id"]
                rd = runs_dir / run_id

                write_manifest_patch(rd, {
                    "engine": eng,
                    "fraction": float(frac),
                    "n_used": n,
                    "generations": args.gens,
                    "population_size": args.pop,
                    "top_k": args.topk
                })

                best = best_from_run(rd)
                metrics = best.get("metrics") or {}
                row = {
                    "run_id": run_id,
                    "dataset": ds,
                    "engine": eng,
                    "fraction": float(frac),
                    "n_used": n,
                    "gens": args.gens,
                    "pop": args.pop,
                    "topk": args.topk,
                    "best_gen": best.get("gen"),
                    "rougeL":    metrics.get("rougeL", 0.0),
                    "bleu":      metrics.get("bleu", 0.0),
                    "bertscore": metrics.get("bertscore", 0.0),
                    "composite": metrics.get("composite", 0.0),
                }
                summary_rows.append(row)
                print(f"\n[{eng} | {ds} | {frac:.2f}] best={row['composite']:.3f} (n={n}) → run_id={run_id}")

    # End of run matrix
    out_json = runs_dir / "summary_matrix.json"
    out_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("Saved:", out_json)

if __name__ == "__main__":
    main()
