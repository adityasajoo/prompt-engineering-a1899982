from __future__ import annotations
from typing import Dict
from rouge_score import rouge_scorer
import sacrebleu
import os

USE_BERTSCORE = bool(int(os.environ.get("USE_BERTSCORE", "0")))

def rouge_l(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ref, pred)['rougeL'].fmeasure

def bleu(pred: str, ref: str) -> float:
    return sacrebleu.corpus_bleu([pred], [[ref]]).score / 100.0

def bertscore_f1(pred: str, ref: str) -> float:
    if not USE_BERTSCORE: return 0.0
    from bert_score import score as bs
    P, R, F1 = bs([pred], [ref], lang="en", rescale_with_baseline=True, verbose=False)
    return float(F1[0])

def composite(r: float, b: float, s: float, w_r: float, w_b: float, w_s: float) -> float:
    return w_r * r + w_b * b + w_s * s

def evaluate_summary(pred: str, ref: str, weights: Dict[str, float]) -> Dict[str, float]:
    r = rouge_l(pred, ref); bl = bleu(pred, ref)
    s = bertscore_f1(pred, ref) if USE_BERTSCORE else 0.0
    c = composite(r, bl, s, weights["rouge"], weights["bleu"], weights["bertscore"])
    return {"rougeL": r, "bleu": bl, "bertscore": s, "composite": c}
