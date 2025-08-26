from __future__ import annotations
from typing import Dict
from .model_runner import LlamaCppRunner

_CRITIC_SYS = (
  "You are a strict summarization prompt critic. "
  "You will propose SMALL edits to the instruction that improve: "
  "faithfulness (no fabricated facts), relevance/coverage, and brevity."
)

_CRITIC_TMPL = """Task: Summarization.
Current prompt (between <<< >>>):
<<<
{PROMPT}
>>>

Observed issues in model outputs (if any):
- {ISSUES}

Propose a minimally-edited prompt that keeps the same intent but improves:
- Faithfulness (avoid hallucinations)
- Coverage of key points
- Brevity and clarity
- Structure (if helpful: key points → 1–2 sentence summary)

Return ONLY the revised prompt text, nothing else.
"""

def critic_edit(runner: LlamaCppRunner, prompt_text: str, issues: str = "") -> str:
    user = _CRITIC_TMPL.replace("{PROMPT}", prompt_text).replace("{ISSUES}", issues or "N/A")
    wrapped = runner._apply_chat_template(user_prompt=user)
    out = runner.llm(wrapped, max_tokens=200, temperature=0.2, top_p=0.95, stop=["<|eot_id|>"])
    text = out["choices"][0]["text"].strip()
    return text if text else prompt_text
