from __future__ import annotations
from typing import List, Tuple
import os

_TINY = {
    "samsum": [
        ("A: Hey, are we still on for dinner tonight?\nB: Yes, 7 pm at the new Italian place.\nA: Perfect, see you!", "They confirm dinner at 7 pm at a new Italian restaurant."),
        ("A: Did you finish the report?\nB: Not yet, need another day.\nA: OK, send a draft by evening.", "B needs one more day; a draft is requested by evening."),
        ("A: Where's the meeting?\nB: Room 301 at 2 PM.\nA: Got it, thanks.", "Meeting is at 2 PM in Room 301."),
    ],
    "xsum": [
        ("The government announced new climate targets aiming to cut emissions by 45% by 2030, pledging investments in renewable energy.", "Government sets 2030 climate target of 45% emissions cut with renewable investments."),
        ("A rare comet will be visible this weekend, with astronomers advising viewers to find dark skies and use binoculars for best results.", "A rare comet is visible this weekend; dark skies and binoculars are recommended."),
        ("The central bank kept interest rates unchanged, citing stable inflation and moderate growth outlook.", "Central bank holds rates amid stable inflation and moderate growth."),
    ],
    "cnn_dailymail": [
        ("The underdogs triumphed after a tense match, overcoming injuries and staging a late comeback to win the championship.", "Underdogs win championship with a late comeback despite injuries."),
        ("Health officials reported progress on vaccination campaigns, noting higher uptake among older adults and plans to expand mobile clinics.", "Vaccination rates improved among seniors; mobile clinics to expand."),
        ("Scientists unveiled a new battery design promising faster charging and longer life for consumer electronics.", "New battery design promises faster charging and longer lifespan."),
    ],
}

def _try_hf(ds_name: str, n: int, start: int):
    try:
        from datasets import load_dataset
        if ds_name == "cnn_dailymail":
            ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
            xs = [ds[i]["article"] for i in range(start, start+n)]
            ys = [ds[i]["highlights"] for i in range(start, start+n)]
        else:
            ds = load_dataset(ds_name, split="test")
            if ds_name == "samsum":
                xs = [ds[i]["dialogue"] for i in range(start, start+n)]
                ys = [ds[i]["summary"] for i in range(start, start+n)]
            elif ds_name == "xsum":
                xs = [ds[i]["document"] for i in range(start, start+n)]
                ys = [ds[i]["summary"] for i in range(start, start+n)]
            else:
                raise ValueError("Unsupported HF dataset")
        return xs, ys
    except Exception:
        return None

def load_many(dataset: str, n: int = 8, start: int = 0) -> Tuple[List[str], List[str]]:
    dataset = dataset.lower()
    if os.environ.get("OFFLINE_ONLY", "0") == "1":
        data = _TINY.get(dataset)
        if not data: raise ValueError(f"No tiny data for {dataset}")
        xs, ys = zip(*data * ((n + len(data) - 1)//len(data)))
        return list(xs)[:n], list(ys)[:n]
    res = _try_hf("cnn_dailymail" if dataset=="cnn_dailymail" else dataset, n, start)
    if res is not None: return res
    data = _TINY.get(dataset)
    if not data: raise ValueError(f"Dataset {dataset} not found; install 'datasets' or set OFFLINE_ONLY=1")
    xs, ys = zip(*data * ((n + len(data) - 1)//len(data)))
    return list(xs)[:n], list(ys)[:n]
