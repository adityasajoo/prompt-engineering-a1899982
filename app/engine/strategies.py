from __future__ import annotations
import uuid
from typing import List, Dict

def seed_strategies() -> List[Dict]:
    items = [
        ("zero_shot", "Summarize the following text in 1–2 sentences. Be concise and faithful.\n\n{INPUT}\n\nSummary:"),
        ("instructional", "You are an expert editor. Write a brief, clear summary (max 2 sentences).\n\n{INPUT}\n\nSummary:"),
        ("cot", "Think step-by-step to find key points, then give a short 2-sentence summary.\n\n{INPUT}\n\nSummary:"),
        ("extract_then_generate", "First list the 3 main points, then write a short 1–2 sentence summary.\n\n{INPUT}\n\nKey points:\n1)\n2)\n3)\n\nSummary:"),
        ("length_focus", "Summarize the main outcome and actions in exactly 2 sentences, simple language.\n\n{INPUT}\n\nSummary:"),
    ]
    out = []
    for strat, text in items:
        out.append({"id": f"seed-{uuid.uuid4().hex[:8]}", "strategy": strat, "text": text})
    return out
