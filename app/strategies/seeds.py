from __future__ import annotations
import random
from typing import Dict, List
from statistics import mean
from ..summarize.extractive import summarise_extractive
from ..eval.metrics import evaluate_summary

def _clip(cfg: Dict) -> Dict:
    cfg["target_sentences"] = int(max(1, min(8, cfg.get("target_sentences", 3))))
    for k in ("position_bias","keyword_boost","redundancy_penalty"):
        ub = 2.0 if k=="keyword_boost" else 1.0
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
        cfg["cluster_k"]    = int(max(2, min(8, cfg.get("cluster_k", 3))))
    return cfg

def _default() -> Dict[str, Dict]:
    return {
        "concise_paragraph": dict(pipeline="basic", target_sentences=2, position_bias=0.2, keyword_boost=0.3, redundancy_penalty=0.5, bullet=False),
        "bullet_keypoints":  dict(pipeline="basic", target_sentences=3, position_bias=0.1, keyword_boost=0.6, redundancy_penalty=0.6, bullet=True),
        "headline_focus":    dict(pipeline="basic", target_sentences=1, position_bias=0.6, keyword_boost=0.5, redundancy_penalty=0.4, bullet=False),
        "balanced_brief":    dict(pipeline="basic", target_sentences=3, position_bias=0.3, keyword_boost=0.3, redundancy_penalty=0.5, bullet=False),
        "coverage_lean":     dict(pipeline="basic", target_sentences=4, position_bias=0.1, keyword_boost=0.2, redundancy_penalty=0.7, bullet=False),
    }

def _cot_seed() -> Dict:
    return _clip(dict(pipeline="cot", cot_keypoints=3, cot_final_sentences=2,
                      target_sentences=3, position_bias=0.3, keyword_boost=0.4, redundancy_penalty=0.5, bullet=False))

def _map_reduce_seed() -> Dict:
    return _clip(dict(pipeline="map_reduce", chunk_size=8, chunk_stride=6, chunk_k=2, final_k=2,
                      target_sentences=3, position_bias=0.2, keyword_boost=0.4, redundancy_penalty=0.5, bullet=False))

def _cluster_seed() -> Dict:
    return _clip(dict(pipeline="cluster", cluster_k=3,
                      target_sentences=3, position_bias=0.2, keyword_boost=0.5, redundancy_penalty=0.6, bullet=False))

def _rand_cfg() -> Dict:
    return _clip(dict(pipeline="basic",
                      target_sentences=random.randint(1,4),
                      position_bias=round(random.uniform(0.0, 0.8), 2),
                      keyword_boost=round(random.uniform(0.0, 1.5), 2),
                      redundancy_penalty=round(random.uniform(0.2, 0.9), 2),
                      bullet=random.choice([False, True])))

def _score_cfg(cfg: Dict, xs: List[str], ys: List[str], weights: Dict[str,float]) -> float:
    scores = []
    for x,y in zip(xs, ys):
        pred = summarise_extractive(x, cfg)
        m = evaluate_summary(pred, y, weights)
        scores.append(m["composite"])
    return float(mean(scores)) if scores else 0.0

def fit_fewshot_seed(xs: List[str], ys: List[str], weights: Dict[str,float], trials: int = 40) -> Dict:
    best, best_s = None, -1.0
    for _ in range(trials):
        cfg = _rand_cfg()
        s = _score_cfg(cfg, xs, ys, weights)
        if s > best_s:
            best, best_s = cfg, s
    best["strategy_tag"] = "few_shot_fit"
    return best

def generate_seeds(xs: List[str], ys: List[str], weights: Dict[str,float], include: list) -> Dict[str, Dict]:
    out = {}
    if "default" in include:
        for k,v in _default().items(): out[k] = _clip(v)
    if "cot" in include:        out["cot_seed"] = _cot_seed()
    if "map_reduce" in include: out["map_reduce_seed"] = _map_reduce_seed()
    if "cluster" in include:    out["cluster_seed"] = _cluster_seed()
    if "fewshot_fit" in include and xs and ys:
        out["fewshot_fit_seed"] = fit_fewshot_seed(xs[:6], ys[:6], weights, trials=40)
    return out
