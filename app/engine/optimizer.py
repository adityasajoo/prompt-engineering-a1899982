from __future__ import annotations
import json, time, uuid, hashlib, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .strategies import seed_strategies
from .mutations import mutate_pool
from .model_runner import LlamaCppRunner
from .metrics import compute_triplet, aggregate_triplets, minmax_normalize, composite_from_normalized
from ..core.config import RUNS_DIR

@dataclass
class OptConfig:
    generations: int = 2
    population_size: int = 6
    top_k: int = 2
    max_new_tokens: int = 160
    temperature: float = 0.0
    # composite weights (ROUGE-L, BLEU, BERTScore)
    w_rouge: float = 1/3
    w_bleu: float = 1/3
    w_bert: float = 1/3
    # number of examples to eval within a dataset (handled by caller)
    # optimizer handles per-example accumulation provided to it

def _log(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def optimize(
    inputs: List[str],
    refs: List[str],
    runner: LlamaCppRunner,
    cfg: OptConfig,
    dataset_name: str = "unknown",
    hard_k: int = 2,               # how many hard examples to add each gen
    patience: int = 2,             # gens without composite improvement
) -> Dict:
    assert len(inputs) == len(refs) and len(inputs) > 0

    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_path, mutations_path, evals_path, bests_path = (out_dir / "seeds.jsonl", out_dir / "mutations.jsonl",
                                                          out_dir / "evals.jsonl", out_dir / "bests.jsonl")

    pop: List[Dict] = seed_strategies()
    for s in pop: _log(seeds_path, {"event":"seed", **s})

    # expand to target pop
    while len(pop) < cfg.population_size:
        base = random.choice(pop)
        pop.extend(mutate_pool(base, parents=pop, runner=None))
        pop = list({p["text"]: p for p in pop}.values())
    pop = pop[:cfg.population_size]

    manifest = {
        "run_id": run_id, "dataset": dataset_name,
        "inputs_sha1": [hashlib.sha1(x.encode()).hexdigest() for x in inputs],
        "refs_sha1": [hashlib.sha1(x.encode()).hexdigest() for x in refs],
        "config": cfg.__dict__, "model": {"path": runner.model_path, "family": runner.family},
        "files": {k: v.name for k,v in dict(seeds=seeds_path, mutations=mutations_path, evals=evals_path, bests=bests_path).items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best = {"prompt_id": None, "prompt_text": None, "score": -1.0, "summary": None, "gen": 0,
            "metrics": {"rougeL":0.0,"bleu":0.0,"bertscore":0.0,"composite":0.0}}
    since_improve, last_best = 0, -1.0

    # start with a small core slice for speed; grow with hard cases
    core_idx = list(range(min( max(1, len(inputs)//3), len(inputs) )))
    hard_pool = []  # indices of tough examples seen so far

    for gen in range(1, cfg.generations + 1):
        # make evaluation set = core + top-k hard cases (unique, clipped)
        eval_idx = list(dict.fromkeys(core_idx + hard_pool[:hard_k]))[:len(inputs)]
        eval_inputs = [inputs[i] for i in eval_idx]
        eval_refs   = [refs[i]   for i in eval_idx]

        scored = []
        raw_agg_list = []

        # optional: collect brief issues from worst outputs to feed critic
        gen_issues = []

        for p in pop:
            per_example = []
            preview = None
            for i, (inp, ref) in enumerate(zip(eval_inputs, eval_refs)):
                full = runner.render_prompt(p["text"], inp)
                g = runner.generate(full, max_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
                if i == 0: preview = g["text"]
                trip = compute_triplet(g["text"], ref)
                _log(evals_path, {"event":"eval","generation":gen,"prompt_id":p["id"],
                                  "strategy":p.get("strategy"),"provenance":p.get("provenance"),
                                  "prompt_text":p["text"],"output":g["text"],"metrics":trip,
                                  "perf":{"latency_ms":g["latency_ms"]}, "dataset":dataset_name,
                                  "example_index": int(eval_idx[i])})
                per_example.append(trip)

            agg = aggregate_triplets(per_example)
            raw_agg_list.append((p, agg, preview or ""))

        mins = {"rougeL":1.0,"bleu":1.0,"bertscore":1.0}
        maxs = {"rougeL":0.0,"bleu":0.0,"bertscore":0.0}
        for _, agg, _ in raw_agg_list:
            for k in mins: mins[k] = min(mins[k], agg[k]); maxs[k] = max(maxs[k], agg[k])

        weights = (cfg.w_rouge, cfg.w_bleu, cfg.w_bert)
        for p, agg, preview in raw_agg_list:
            norm = {m: minmax_normalize(agg[m], mins[m], maxs[m]) for m in ("rougeL","bleu","bertscore")}
            comp = composite_from_normalized(norm, weights)
            scored.append((p, comp, agg, preview))
            if comp > best["score"]:
                best = {"prompt_id": p["id"], "prompt_text": p["text"], "score": comp,
                        "summary": preview, "gen": gen,
                        "metrics": {"rougeL":agg["rougeL"],"bleu":agg["bleu"],"bertscore":agg["bertscore"],"composite":comp}}
                _log(bests_path, {"event":"best_update", **best})

        # identify hardest examples this gen (for mining)
        # compute per-example average over pop
        per_ex_scores = []
        for j in range(len(eval_inputs)):
            # average composite across prompts using that example
            preds = []
            # quick proxy: use aggregated rougeL only to rank hardness (fast)
            # (could log full triplets per example if needed)
            rs = []
            for p, comp, agg, _ in scored:
                rs.append(agg["rougeL"])
            per_ex_scores.append((eval_idx[j], sum(rs)/len(rs)))
        per_ex_scores.sort(key=lambda x: x[1])  # small = hard
        hard_pool = [idx for idx,_ in per_ex_scores] + hard_pool
        hard_pool = list(dict.fromkeys(hard_pool))  # unique, keep order

        # selection
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors = [p for p,_,_,_ in scored[:max(1, cfg.top_k)]]

        # build issues summary from worst prompts to feed critic
        worst = scored[-1]
        gen_issues = (f"Low coverage (ROUGE-L={worst[2]['rougeL']:.2f}); " 
                      f"BLEU={worst[2]['bleu']:.2f}; BERTScore={worst[2]['bertscore']:.2f}")

        new_pop = survivors[:]
        for s in survivors:
            kids = mutate_pool(s, parents=survivors, runner=runner, last_issues=gen_issues)
            for k in kids: _log(mutations_path, {"event":"mutation", **k})
            new_pop.extend(kids)

        while len(new_pop) < cfg.population_size:
            new_pop.extend(mutate_pool(random.choice(survivors), parents=survivors, runner=runner, last_issues=gen_issues))

        pop = list({p["text"]: p for p in new_pop}.values())[:cfg.population_size]

        print(f"[Gen {gen}] Best composite={scored[0][1]:.3f} | Global best={best['score']:.3f} "
              f"(R={best['metrics']['rougeL']:.3f} B={best['metrics']['bleu']:.3f} S={best['metrics']['bertscore']:.3f})")

        # early stop by patience
        if best["score"] <= last_best + 1e-4:
            since_improve += 1
        else:
            since_improve, last_best = 0, best["score"]
        if since_improve >= patience:
            print(f"[Early stop] No composite improvement for {patience} generations.")
            break

    return {"best": best, "run_id": run_id, "logs_dir": str(out_dir)}