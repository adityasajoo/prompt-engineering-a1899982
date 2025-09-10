from __future__ import annotations
import json, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from .features import cfg_features

def _load_runs(runs_dir: Path):
    data = []
    for rd in runs_dir.iterdir():
        if not rd.is_dir(): continue
        evals_f = rd / "evals.jsonl"
        muts_f  = rd / "mutations.jsonl"
        if not evals_f.exists() or not muts_f.exists(): continue

        # Aggregate composite per (gen, cfg)
        agg = {}
        for line in evals_f.read_text().splitlines():
            o = json.loads(line)
            if o.get("event") != "eval": continue
            gen = int(o.get("gen") or o.get("generation") or 0)
            cfg = o.get("prompt")  # cfg dict stored in logs
            key = (gen, json.dumps(cfg, sort_keys=True))
            agg.setdefault(key, []).append(o["metrics"]["composite"])
        agg = {k: float(np.mean(v)) for k,v in agg.items()}

        # For each mutation row, compute improvement child(gen) - parent(gen-1)
        for line in muts_f.read_text().splitlines():
            o = json.loads(line)
            if o.get("event") != "mutation": continue
            gen = int(o.get("gen") or o.get("generation") or 1)
            parent = o.get("parent"); child = o.get("child")
            op = o.get("op") or "unknown"
            pkey = (gen-1, json.dumps(parent, sort_keys=True))
            ckey = (gen,   json.dumps(child,  sort_keys=True))
            if pkey in agg and ckey in agg:
                imp = agg[ckey] - agg[pkey]
                data.append((parent, op, imp))
    return data

def train_classifier(runs_dir: str, out_path: str):
    rows = _load_runs(Path(runs_dir))
    if not rows:
        print("No training data found. Run baseline first."); return
    X, y, w = [], [], []
    for parent_cfg, op, imp in rows:
        X.append(cfg_features(parent_cfg))
        y.append(op)
        w.append(max(0.1, 1.0 + 2.0*imp))  # weight positives more
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, multi_class="auto"))
    ])
    pipe.fit(np.stack(X), y, lr__sample_weight=np.array(w))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"Saved classifier to: {out_path} (n={len(X)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="policies/mutation_classifier.pkl")
    args = ap.parse_args()
    train_classifier(args.runs_dir, args.out)
