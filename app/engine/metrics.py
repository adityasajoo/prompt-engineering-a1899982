from __future__ import annotations
from typing import Dict, List, Tuple
from rouge_score import rouge_scorer
import sacrebleu
import os

# ---- ROUGE-L (F1) ----
_rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def rouge_l_f1(pred: str, ref: str) -> float:
    return float(_rs.score(ref, pred)["rougeL"].fmeasure)

# ---- BLEU (sacrebleu) ----
def bleu_score(preds: List[str], refs: List[str]) -> float:
    # sacrebleu expects list of refs lists: [[ref1, ref2,...]] for multi-refs; we have single-ref
    return float(sacrebleu.corpus_bleu(preds, [refs]).score / 100.0)

# ---- BERTScore ----
# You can change the model by env var: BERTSCORE_MODEL=bert-base-uncased
_BERT_MODEL = os.environ.get("BERTSCORE_MODEL", "roberta-large")
_bert_scorer = None

def bertscore_f1(preds: List[str], refs: List[str]) -> float:
    global _bert_scorer
    if _bert_scorer is None:
        # lazy import to avoid heavy init if user disabled BERTScore
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(model_type=_BERT_MODEL, lang="en", rescale_with_baseline=True)
    P, R, F = _bert_scorer.score(preds, refs)
    # Return mean F1
    return float(F.mean().item())

# ---- Per-sample + aggregate helpers ----
def compute_triplet(pred: str, ref: str) -> Dict[str, float]:
    # BLEU is unstable at single-sample granularity; we still compute it for completeness
    return {
        "rougeL": rouge_l_f1(pred, ref),
        "bleu": bleu_score([pred], [ref]),
        "bertscore": bertscore_f1([pred], [ref]),
    }

def aggregate_triplets(triplets: List[Dict[str,float]]) -> Dict[str,float]:
    if not triplets:
        return {"rougeL": 0.0, "bleu": 0.0, "bertscore": 0.0}
    k = len(triplets)
    s = {"rougeL":0.0, "bleu":0.0, "bertscore":0.0}
    for t in triplets:
        s["rougeL"] += t["rougeL"]
        s["bleu"] += t["bleu"]
        s["bertscore"] += t["bertscore"]
    return {m: s[m]/k for m in s}

# ---- Composite score ----
# Default equal weights (you can tune in CLI/config)
def composite_from_normalized(norm: Dict[str,float], w=(1/3,1/3,1/3)) -> float:
    wr, wb, ws = w
    return float(wr*norm["rougeL"] + wb*norm["bleu"] + ws*norm["bertscore"])

def minmax_normalize(x: float, mn: float, mx: float) -> float:
    if mx <= mn: return 0.0
    return (x - mn) / (mx - mn)
