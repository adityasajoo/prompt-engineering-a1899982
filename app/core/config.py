from pathlib import Path
import os, time, uuid, json, subprocess, sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"
POLICIES_DIR = PROJECT_ROOT / "policies"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
POLICIES_DIR.mkdir(parents=True, exist_ok=True)

# Metric weights (sum ≈ 1)
W_ROUGE = float(os.environ.get("W_ROUGE", "0.34"))
W_BLEU  = float(os.environ.get("W_BLEU",  "0.33"))
W_BERT  = float(os.environ.get("W_BERT",  "0.33"))
USE_BERTSCORE = bool(int(os.environ.get("USE_BERTSCORE", "0")))

DEFAULT_POLICY_PATH = str(POLICIES_DIR / "mutation_classifier.pkl")

# Global RNG seed for reproducibility
GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", "42"))

def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

def _pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return out
    except Exception:
        return ""

def write_manifest(out_dir: Path, manifest: dict):
    manifest = dict(manifest)
    manifest["env"] = {
        "python": sys.version.replace("\n"," "),
        "pip_freeze": _pip_freeze()[:20000]  # truncate to keep file reasonable
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
