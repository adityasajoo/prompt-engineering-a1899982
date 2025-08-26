from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
from llama_cpp import Llama
import time
import re

LLAMA3_SYS = "You are a helpful, concise assistant."
def llama3_wrap(user_text: str, system_text: str = LLAMA3_SYS) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n" + system_text.strip() + "\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"   + user_text.strip()   + "\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

LLAMA2_SYS = "You are a helpful assistant."
def llama2_wrap(user_text: str, system_text: str = LLAMA2_SYS) -> str:
    return f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text} [/INST]"

class LlamaCppRunner:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 0,
        n_gpu_layers: int = -1,
        family: str = "llama3",
    ) -> None:
        self.model_path = str(Path(model_path))
        self.family = family
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or None,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    @staticmethod
    def render_prompt(template: str, input_text: str) -> str:
        return template.replace("{INPUT}", input_text.strip())

    def _apply_chat_template(self, user_prompt: str) -> str:
        if self.family == "llama3":
            return llama3_wrap(user_prompt)
        if self.family == "llama2":
            return llama2_wrap(user_prompt)
        return user_prompt

    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 160,
        temperature: Optional[float] = 0.0,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        wrapped = self._apply_chat_template(user_prompt)
        params = dict(
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop or ["<|eot_id|>"],
            repeat_penalty=1.05,
        )
        if temperature and temperature > 0.0:
            params["temperature"] = temperature
        else:
            params["temperature"] = 0.0  # greedy

        t0 = time.time()
        out = self.llm(wrapped, **params)
        dt_ms = int((time.time() - t0) * 1000)

        text = out["choices"][0]["text"]
        text = re.sub(r"\s+", " ", text).strip()

        usage = out.get("usage", {})
        return {
            "text": text,
            "latency_ms": dt_ms,
            "usage": usage,
        }
