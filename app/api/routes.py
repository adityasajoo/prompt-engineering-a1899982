from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from .schemas import OptimizePromptRequest, OptimizeResponse
from ..core.config import DEFAULT_MODEL_PATH, MODELS_DIR
from ..core.logging import logger
from ..engine.datasets import load_sample
from ..engine.model_runner import LlamaCppRunner
from ..engine.optimizer import optimize, OptConfig
from typing import List, Optional, Dict


app = FastAPI(title="Prompt Optimizer (Mac)", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/models")
def list_models():
    files = [str(p.name) for p in MODELS_DIR.glob("*.gguf")]
    return {"count": len(files), "files": files}

@app.post("/optimize_prompt", response_model=OptimizeResponse)
def optimize_prompt(req: OptimizePromptRequest):
    logger.info(f"optimize_prompt: {req}")

    text, ref, ds = load_sample(req.dataset, index=req.index)

    runner = LlamaCppRunner(
        model_path=req.model_path or DEFAULT_MODEL_PATH,
        n_ctx=req.n_ctx,
        n_threads=req.n_threads,
        n_gpu_layers=req.n_gpu_layers,
        family=req.family
    )

    cfg = OptConfig(
        generations=req.generations,
        population_size=req.population_size,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    result = optimize(text, ref, runner, cfg)
    best = result["best"]
    return OptimizeResponse(
        run_id=result["run_id"],
        dataset=ds,
        index=req.index,
        best_prompt=best["prompt_text"],
        best_score=best["score"],
        best_generation=best["gen"],
        best_summary=best["summary"],
        logs_dir=result["logs_dir"],
    )


@app.post("/multi_model_optimize_prompt")
def multi_model_optimize_prompt(
    input_texts: List[str] = Body(..., description="List of input documents/texts"),
    references: List[str] = Body(..., description="Aligned list of gold references (same length as input_texts)"),
    model_files: Optional[List[str]] = Body(None, description="List of .gguf filenames under models/. If None, use defaults."),
    generations: int = Body(2),
    population_size: int = Body(6),
    top_k: int = Body(2),
    max_new_tokens: int = Body(160),
    temperature: float = Body(0.0),
    weights: Dict[str,float] = Body({"rouge":1/3,"bleu":1/3,"bertscore":1/3}),
    family: str = Body("llama3"),
    dataset_name: str = Body("custom"),
):
    assert len(input_texts) == len(references) and len(input_texts) > 0, "inputs/refs length mismatch"

    defaults = [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    ]
    model_files = model_files or defaults

    results = []
    for mf in model_files:
        mpath = MODELS_DIR / mf
        if not mpath.exists():
            results.append({"model": mf, "error": "missing model file"})
            continue

        runner = LlamaCppRunner(str(mpath), n_ctx=4096, n_threads=0, n_gpu_layers=-1, family=family)
        cfg = OptConfig(
            generations=generations, population_size=population_size, top_k=top_k,
            max_new_tokens=max_new_tokens, temperature=temperature,
            w_rouge=weights.get("rouge", 1/3), w_bleu=weights.get("bleu", 1/3), w_bert=weights.get("bertscore", 1/3),
        )
        out = optimize(input_texts, references, runner, cfg, dataset_name=dataset_name)
        results.append({
            "model": mf,
            "best_prompt": out["best"]["prompt_text"],
            "scores": out["best"]["metrics"],
            "composite": out["best"]["score"],
            "gen_found": out["best"]["gen"],
            "summary_preview": out["best"]["summary"],
            "logs_dir": out["logs_dir"],
            "run_id": out["run_id"],
        })
    return {"dataset": dataset_name, "count": len(results), "results": results}
