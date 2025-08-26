from app.engine.model_runner import LlamaCppRunner
from app.engine.optimizer import optimize, OptConfig

def test_smoke():
    runner = LlamaCppRunner("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    text = "A: Bring the slides?\nB: Yes, I'll print them now.\nA: Thanks!"
    ref  = "B agrees to print the slides now."
    cfg = OptConfig(generations=1, population_size=4, top_k=1, max_new_tokens=80)
    res = optimize(text, ref, runner, cfg)
    assert res["best"]["score"] >= 0.0