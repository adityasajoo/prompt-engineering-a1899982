from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Tuple, Dict
from app.engine.model_runner import LlamaCppRunner
from app.engine.datasets import load_sample
from app.engine.optimizer import optimize, OptConfig
from app.core.config import MODELS_DIR, RUNS_DIR

# ---- Which local models to use (GGUF files in models/)
DEFAULT_MODELS = [
    # (filename, family, nickname)
    ("mistral-7b-instruct-v0.2.Q4_K_M.gguf", "llama3", "Mistral-7B-Q4"),
    ("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", "llama3", "Llama3-8B-Q4"),
    ("Qwen2.5-3B-Instruct-Q4_K_M.gguf",      "llama3", "Qwen2.5-3B-Q4"),
]

# ---- Datasets and how many examples per dataset
DEFAULT_DATASETS = [
    ("samsum", 0, 3),         # (name, start_index, n_examples)
    ("xsum",   0, 3),
    ("cnn_dailymail", 0, 3),
]

def load_many(dataset_name: str, start: int, n: int) -> Tuple[List[str], List[str]]:
    inputs, refs = [], []
    for i in range(start, start+n):
        x, y, _ = load_sample(dataset_name, index=i)
        inputs.append(x); refs.append(y)
    return inputs, refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="Model nicknames or filenames under models/")
    ap.add_argument("--datasets", nargs="*", default=[], help="Dataset names (samsum|xsum|cnn_dailymail)")
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--w", nargs=3, type=float, default=[1/3,1/3,1/3], help="Weights wRouge wBLEU wBERT")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--bertscore_model", type=str, default=os.environ.get("BERTSCORE_MODEL","roberta-large"))
    args = ap.parse_args()

    os.environ["BERTSCORE_MODEL"] = args.bertscore_model

    models = DEFAULT_MODELS if not args.models else [
        # Allow user to pass nicknames or filenames; try to match
        next((m for m in DEFAULT_MODELS if m[2] == mname or m[0] == mname), (mname, "llama3", Path(mname).stem))
        for mname in args.models
    ]

    datasets = DEFAULT_DATASETS if not args.datasets else [(d, args.start, args.n) for d in args.datasets]

    results: List[Dict] = []
    for fname, family, nickname in models:
        mpath = MODELS_DIR / fname
        if not mpath.exists():
            print(f"!! Missing model file: {mpath}. Skipping.")
            continue
        runner = LlamaCppRunner(str(mpath), n_ctx=4096, n_threads=0, n_gpu_layers=-1, family=family)

        for dname, start, n in datasets:
            try:
                inputs, refs = load_many(dname, start, n)
            except Exception as e:
                print(f"Dataset load failed for {dname}: {e}")
                continue

            cfg = OptConfig(
                generations=args.gens, population_size=args.pop, top_k=args.topk,
                max_new_tokens=args.max_new, temperature=args.temp,
                w_rouge=args.w[0], w_bleu=args.w[1], w_bert=args.w[2],
            )
            print(f"\n=== Running {nickname} on {dname} (n={len(inputs)}) ===")
            out = optimize(inputs, refs, runner, cfg, dataset_name=dname)

            res = {
                "model_nickname": nickname,
                "model_path": str(mpath),
                "dataset": dname,
                "best_prompt": out["best"]["prompt_text"],
                "scores": out["best"]["metrics"],
                "composite": out["best"]["score"],
                "gen_found": out["best"]["gen"],
                "summary_preview": out["best"]["summary"],
                "logs_dir": out["logs_dir"],
                "run_id": out["run_id"],
                "weights": {"rouge":cfg.w_rouge,"bleu":cfg.w_bleu,"bertscore":cfg.w_bert},
                "config": {"gens":cfg.generations,"pop":cfg.population_size,"topk":cfg.top_k,"max_new":cfg.max_new_tokens}
            }
            results.append(res)

    # Save one JSON file for the whole benchmark
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RUNS_DIR / f"multi_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()
