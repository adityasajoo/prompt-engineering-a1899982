from pydantic import BaseModel, Field
from typing import Optional

class OptimizePromptRequest(BaseModel):
    dataset: str = Field(..., description="samsum | xsum | cnn_dailymail | builtin")
    index: int = 0
    model_path: Optional[str] = None
    generations: int = 2
    population_size: int = 6
    top_k: int = 2
    max_new_tokens: int = 160
    temperature: float = 0.0
    family: str = "llama3"   # llama3|llama2|plain
    n_ctx: int = 4096
    n_threads: int = 0
    n_gpu_layers: int = -1

class OptimizeResponse(BaseModel):
    run_id: str
    dataset: str
    index: int
    best_prompt: str
    best_score: float
    best_generation: int
    best_summary: str
    logs_dir: str
