from __future__ import annotations
from typing import Tuple
from datasets import load_dataset

def load_sample(dataset: str, index: int = 0) -> Tuple[str, str, str]:
    """
    Returns: (input_text, reference, dataset_name)
    Supported: samsum, xsum, cnn_dailymail
    """
    ds = dataset.lower().strip()
    try:
        if ds == "samsum":
            d = load_dataset("samsum", split="test")
            r = d[index % len(d)]
            return r["dialogue"], r["summary"], "samsum"
        elif ds == "xsum":
            d = load_dataset("xsum", split="validation")
            r = d[index % len(d)]
            return r["document"], r["summary"], "xsum"
        elif ds in ("cnn", "cnn_dailymail", "cnndm"):
            d = load_dataset("cnn_dailymail", "3.0.0", split="validation")
            r = d[index % len(d)]
            return r["article"], r["highlights"], "cnn_dailymail"
    except Exception:
        pass

    # Built-in fallback (no internet required)
    input_text = (
        "A: Hey, did you finish the report?\n"
        "B: Almost, I'll send it tonight.\n"
        "A: Great, the deadline is tomorrow morning.\n"
        "B: Got it, I'll make sure it's in your inbox by 9 AM."
    )
    reference = "B will finish and send the report tonight to meet tomorrow morning's deadline."
    return input_text, reference, "builtin"
