from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l_f1(pred: str, ref: str) -> float:
    score = _rouge.score(ref, pred)["rougeL"].fmeasure
    return float(score)
