import numpy as np
from typing import Dict

def cfg_features(cfg: Dict) -> np.ndarray:
    return np.array([
        min(1.0, max(0.0, cfg.get("target_sentences",3)/8.0)),
        float(cfg.get("position_bias",0.3)),
        min(1.0, float(cfg.get("keyword_boost",0.3))/2.0),
        float(cfg.get("redundancy_penalty",0.5)),
        1.0 if cfg.get("bullet", False) else 0.0,
        # pipeline one-hots
        1.0 if cfg.get("pipeline")=="basic" else 0.0,
        1.0 if cfg.get("pipeline")=="cot" else 0.0,
        1.0 if cfg.get("pipeline")=="map_reduce" else 0.0,
        1.0 if cfg.get("pipeline")=="cluster" else 0.0,
    ], dtype=float)
