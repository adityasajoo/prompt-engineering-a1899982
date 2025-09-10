from __future__ import annotations
from typing import Dict, List, Tuple
import random
from .space import clip_cfg, crossover

def mutate_length(cfg: Dict) -> Dict:
    d = dict(cfg); d["target_sentences"] = d.get("target_sentences",3) + random.choice([-1,1]); return clip_cfg(d)

def mutate_format(cfg: Dict) -> Dict:
    d = dict(cfg); d["bullet"] = not d.get("bullet", False); return d

def mutate_position_bias(cfg: Dict) -> Dict:
    d = dict(cfg); d["position_bias"] = d.get("position_bias",0.3) + random.choice([-0.2,-0.1,0.1,0.2]); return clip_cfg(d)

def mutate_keyword_boost(cfg: Dict) -> Dict:
    d = dict(cfg); d["keyword_boost"] = d.get("keyword_boost",0.3) + random.choice([-0.2,-0.1,0.1,0.2]); return clip_cfg(d)

def mutate_redundancy(cfg: Dict) -> Dict:
    d = dict(cfg); d["redundancy_penalty"] = d.get("redundancy_penalty",0.5) + random.choice([-0.2,-0.1,0.1,0.2]); return clip_cfg(d)

def mutate_pipeline_params(cfg: Dict) -> Dict:
    d = dict(cfg)
    p = d.get("pipeline","basic")
    if p == "cot":
        d["cot_keypoints"] = max(1, min(8, d.get("cot_keypoints",3) + random.choice([-1,1])))
        d["cot_final_sentences"] = max(1, min(4, d.get("cot_final_sentences",2) + random.choice([-1,1])))
    elif p == "map_reduce":
        d["chunk_size"]   = max(3, min(20, d.get("chunk_size",8) + random.choice([-2,-1,1,2])))
        d["chunk_stride"] = max(1,  min(d["chunk_size"], d.get("chunk_stride",6) + random.choice([-2,-1,1,2])))
        d["chunk_k"]      = max(1, min(5,  d.get("chunk_k",2) + random.choice([-1,1])))
        d["final_k"]      = max(1, min(5,  d.get("final_k",2) + random.choice([-1,1])))
    elif p == "cluster":
        d["cluster_k"]    = max(2, min(8, d.get("cluster_k",3) + random.choice([-1,1])))
    return clip_cfg(d)

MUTATION_OPS = {
    "length": mutate_length,
    "format": mutate_format,
    "position_bias": mutate_position_bias,
    "keyword_boost": mutate_keyword_boost,
    "redundancy": mutate_redundancy,
    "pipeline_params": mutate_pipeline_params,
}

def all_children_with_ops(parent: Dict, parents: List[Dict], n_random: int = 1) -> List[Tuple[Dict,str]]:
    kids = []
    for name, fn in MUTATION_OPS.items():
        kids.append((fn(parent), name))
    for _ in range(n_random):
        name, fn = random.choice(list(MUTATION_OPS.items()))
        kids.append((fn(parent), name))
    if parents and random.random() < 0.5:
        mate = random.choice(parents)
        kids.append((crossover(parent, mate), "crossover"))
    uniq, seen = [], set()
    for cfg, op in kids:
        t = tuple(sorted(cfg.items()))
        if t not in seen:
            seen.add(t); uniq.append((cfg, op))
    return uniq
