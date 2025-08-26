from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Default model (change to your file name)
DEFAULT_MODEL_PATH = str(next(MODELS_DIR.glob("*.gguf"), MODELS_DIR / "PUT_MODEL_HERE.gguf"))

# Llama.cpp defaults tuned for Apple Silicon (Metal)
DEFAULT_CTX = 4096
DEFAULT_THREADS = 0       # 0 = auto (use all)
DEFAULT_GPU_LAYERS = -1   # -1 = "as many as possible" on Metal GPU
DEFAULT_FAMILY = "llama3" # "llama3" | "llama2" | "plain"


POLICIES_DIR = PROJECT_ROOT / "policies"
POLICIES_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_POLICY_PATH = str(POLICIES_DIR / "mutation_policy.pkl")
