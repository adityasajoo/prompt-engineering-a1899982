import joblib
from pathlib import Path
from typing import Dict, Optional
from .features import cfg_features

def load_classifier(path: str):
    p = Path(path)
    if not p.exists(): return None
    return joblib.load(p)

def predict_op(clf, cfg: Dict) -> Optional[str]:
    try:
        x = cfg_features(cfg).reshape(1, -1)
        return clf.predict(x)[0]
    except Exception:
        return None
