from __future__ import annotations
from typing import Dict
import random

def clip_cfg(cfg: Dict) -> Dict:
    cfg["target_sentences"] = int(max(1, min(8, cfg.get("target_sentences", 3))))
    for k in ("position_bias","keyword_boost","redundancy_penalty"):
        ub = 2.0 if k == "keyword_boost" else 1.0
        cfg[k] = float(max(0.0, min(ub, cfg.get(k, 0.3))))
    cfg["bullet"] = bool(cfg.get("bullet", False))
    if cfg.get("pipeline") == "cot":
        cfg["cot_keypoints"] = int(max(1, min(8, cfg.get("cot_keypoints", 3))))
        cfg["cot_final_sentences"] = int(max(1, min(4, cfg.get("cot_final_sentences", 2))))
    if cfg.get("pipeline") == "map_reduce":
        cfg["chunk_size"]   = int(max(3, min(20, cfg.get("chunk_size", 8))))
        cfg["chunk_stride"] = int(max(1,  min(cfg["chunk_size"], cfg.get("chunk_stride", 6))))
        cfg["chunk_k"]      = int(max(1, min(5,  cfg.get("chunk_k", 2))))
        cfg["final_k"]      = int(max(1, min(5,  cfg.get("final_k", 2))))
    if cfg.get("pipeline") == "cluster":
        cfg["cluster_k"] = int(max(2, min(8, cfg.get("cluster_k", 3))))
    return cfg

def crossover(a: Dict, b: Dict) -> Dict:
    c = {}
    for k in ("pipeline","target_sentences","position_bias","keyword_boost","redundancy_penalty","bullet",
              "cot_keypoints","cot_final_sentences","chunk_size","chunk_stride","chunk_k","final_k","cluster_k"):
        if k in a or k in b:
            c[k] = a.get(k, b.get(k))
            if random.random() < 0.5: c[k] = b.get(k, a.get(k))
    return clip_cfg(c)
