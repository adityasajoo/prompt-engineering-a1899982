from __future__ import annotations
import json, numpy as np
from pathlib import Path
from typing import Dict, List
from ..core.config import RUNS_DIR, new_run_id, W_ROUGE, W_BLEU, W_BERT, write_manifest
from ..utils.logging import log_jsonl
from ..mutate.space import clip_cfg
from ..mutate.ops import all_children_with_ops
from ..summarize.extractive import summarise_extractive
from ..eval.metrics import evaluate_summary
from ..strategies.seeds import generate_seeds

from app.core.simple_progress import print_pct, newline
import sys 


import os
MAX_CHILDREN = int(os.environ.get("MAX_CHILDREN_PER_PARENT", "6"))

def _eval_cfg(dataset: str, inputs: List[str], refs: List[str], cfg: Dict, gen: int, pid: str, evals_path: Path, weights: Dict[str,float]):
    per = []
    for i,(x,ref) in enumerate(zip(inputs, refs)):
        pred = summarise_extractive(x, cfg)
        m = evaluate_summary(pred, ref, weights)
        log_jsonl(evals_path, {"event":"eval","gen":gen,"prompt_id":pid,"prompt":cfg,"dataset":dataset,"example_index":i,"metrics":m,"output":pred})
        per.append(m)
    agg = {k: float(np.mean([r[k] for r in per])) for k in per[0].keys()}
    return agg

def run_baseline(dataset_name: str, inputs: List[str], refs: List[str],
                 generations: int=3, population_size: int=8, top_k: int=3, patience: int=2) -> Dict:
    run_id = new_run_id()
    out_dir = RUNS_DIR / run_id
    seeds_path, muts_path, evals_path, bests_path = (out_dir/"seeds.jsonl", out_dir/"mutations.jsonl", out_dir/"evals.jsonl", out_dir/"bests.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {"rouge": W_ROUGE, "bleu": W_BLEU, "bertscore": W_BERT}

    write_manifest(out_dir, {
        "phase": "baseline", "dataset": dataset_name, "generations": generations,
        "population_size": population_size, "top_k": top_k, "patience": patience,
        "weights": weights
    })

    # Gen 0: strategy-driven seeds
    wanted = ["default","cot","map_reduce","cluster","fewshot_fit"]
    pool = generate_seeds(inputs[:6], refs[:6], weights, include=wanted)
    seed_cfgs = [{"id": f"seed_{i}", "name": n, "cfg": clip_cfg(c)} for i,(n,c) in enumerate(pool.items())]
    for s in seed_cfgs: log_jsonl(seeds_path, {"event":"seed", **s})

    scored0 = []
    for s in seed_cfgs:
        agg = _eval_cfg(dataset_name, inputs, refs, s["cfg"], 0, s["id"], evals_path, weights)
        scored0.append((s, agg["composite"], agg))
    scored0.sort(key=lambda t: t[1], reverse=True)
    best = {"gen": 0, "cfg": scored0[0][0]["cfg"], "metrics": scored0[0][2]}
    log_jsonl(bests_path, {"event":"best", **best})
    survivors = [t[0] for t in scored0[:max(1, top_k)]]

    last_best = best["metrics"]["composite"]; since = 0

    # Generations ≥1
    for gen in range(1, generations+1):
        print_pct(f"[{dataset_name}] generations", gen, generations)

        scored = []
        for pi, s in enumerate(survivors):
            print_pct(f"  gen {gen} • parents", pi + 1, len(survivors))

            for (kcfg, op) in all_children_with_ops(s["cfg"], [x["cfg"] for x in survivors], n_random=1):
                agg = _eval_cfg(dataset_name, inputs, refs, kcfg, gen, f"{s['id']}_g{gen}", evals_path, weights)
                scored.append((kcfg, agg["composite"], agg, op, s["cfg"]))
                log_jsonl(muts_path, {"event":"mutation","gen":gen,"parent":s["cfg"],"child":kcfg,"op":op})
                if (ci + 1 == len(children)) or ((ci + 1) % max(1, len(children)//10) == 0):
                    print_pct(f"    gen {gen} • children", ci + 1, len(children))
        scored.sort(key=lambda t: t[1], reverse=True)
        survivors = [{"id": f"g{gen}_s{i}", "name":"survivor", "cfg": clip_cfg(scored[i][0])}
                     for i in range(min(top_k, len(scored)))]
        best_score = (
        best["metrics"]["composite"]
            if isinstance(best, dict) and "metrics" in best else 0.0
        )
        sys.stdout.write(
            f"\r[{dataset_name}] gen {gen+1}/{generations} done | best={best_score:.3f}\n"
        ); sys.stdout.flush()
        if scored and scored[0][1] > last_best + 1e-6:
            last_best = scored[0][1]
            best = {"gen": gen, "cfg": scored[0][0], "metrics": scored[0][2]}
            log_jsonl(bests_path, {"event":"best", **best})
            since = 0
        else:
            since += 1
        if since >= patience: break

    return {"run_id": str(out_dir.name), "best": best, "out_dir": str(out_dir)}
