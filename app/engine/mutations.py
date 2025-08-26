from __future__ import annotations
import re, uuid, random
from typing import Dict, List, Optional
from .critic import critic_edit
from .model_runner import LlamaCppRunner

LEX = {
    "concise": ["brief","succinct","short"],
    "summary": ["outline","synopsis","digest"],
    "main":    ["key","primary","central"],
    "simple":  ["plain","clear","straightforward"],
    "write":   ["compose","provide","produce"],
}

def mutate_lexical(p: Dict) -> Optional[Dict]:
    txt = p["text"]
    toks = re.findall(r"\w+|\W+", txt)
    cand = [i for i,t in enumerate(toks) if t.isalpha() and t.lower() in LEX]
    if not cand: return None
    i = random.choice(cand)
    orig = toks[i].lower()
    rep = random.choice(LEX[orig])
    toks[i] = rep if toks[i].islower() else rep.capitalize()
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": "".join(toks),
        "provenance": {"type":"lexical","parents":[p["id"]],"target":orig,"replacement":rep}
    }

def mutate_reorder(p: Dict) -> Optional[Dict]:
    txt = p["text"]
    parts = re.split(r'(\n\n|\.\s+)', txt)
    sents = [parts[i] for i in range(0,len(parts),2)]
    seps  = [parts[i] for i in range(1,len(parts),2)]
    if len(sents) < 2: return None
    i,j = random.sample(range(len(sents)), 2)
    sents[i], sents[j] = sents[j], sents[i]
    out=[]
    for k,s in enumerate(sents):
        out.append(s)
        if k < len(seps): out.append(seps[k])
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": "".join(out),
        "provenance": {"type":"reorder","parents":[p["id"]],"swap":[i,j]}
    }

def mutate_format(p: Dict) -> Dict:
    txt = p["text"]
    if "bullet" in txt.lower() or "list" in txt.lower():
        new_txt = re.sub(r"(bullet points?|list)", "one short paragraph", txt, flags=re.I)
    else:
        new_txt = txt + "\n\nFormat: Use bullet points."
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": new_txt,
        "provenance": {"type":"format","parents":[p["id"]]}
    }

def crossover(a: Dict, b: Dict) -> Dict:
    txt = a["text"][:len(a["text"])//2].rstrip() + "\n" + b["text"][len(b["text"])//2:].lstrip()
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": txt,
        "provenance": {"type":"crossover","parents":[a["id"], b["id"]]}
    }

def mutate_length_tweak(p: Dict) -> Dict:
    t = p["text"]
    if re.search(r"\bsentences?\b", t, re.I):
        new = re.sub(r"exactly\s*\d+\s*sentences?|(\d+[\-\–]\d+\s*sentences?)",
                     "no more than 60 words", t, flags=re.I)
        if new == t: new = t + "\n\nLength limit: no more than 60 words."
    else:
        new = t + "\n\nLimit the summary to at most 2 sentences."
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": new,
        "provenance": {"type":"length_tweak","parents":[p["id"]]}
    }

def mutate_critic_edit(p: Dict, runner: LlamaCppRunner, last_issues: str = "") -> Dict:
    new_txt = critic_edit(runner, p["text"], last_issues)
    if not new_txt or new_txt == p["text"]:
        return None
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": new_txt,
        "provenance": {"type":"critic_edit","parents":[p["id"]],"notes":"llm-as-critic"}
    }

def mutate_decompose(p: Dict) -> Dict:
    t = p["text"]
    if "Key points" in t or "key points" in t.lower():
        return None
    new = (t.rstrip() + 
           "\n\nFirst list 2–4 key points from the text, " +
           "then write a faithful 1–2 sentence summary.")
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": new,
        "provenance": {"type":"decompose","parents":[p["id"]]}
    }

def mutate_negative_guidance(p: Dict) -> Dict:
    t = p["text"]
    add = ("\n\nConstraints: Avoid invented facts, do not quote verbatim unless necessary, " +
           "prefer concise wording, preserve numbers only if present in the text.")
    if add.strip().lower() in t.lower():
        return None
    return {
        "id": f"mut-{uuid.uuid4().hex[:8]}",
        "text": t.rstrip() + add,
        "provenance": {"type":"negative_guidance","parents":[p["id"]]}
    }

def mutate_pool(prompt: Dict, parents: List[Dict], runner: LlamaCppRunner = None, last_issues: str = "") -> List[Dict]:
    outs = []
    for fn in (mutate_lexical, mutate_reorder, mutate_format, mutate_length_tweak, mutate_decompose, mutate_negative_guidance):
        c = fn(prompt)
        if c: outs.append(c)
    if runner is not None:
        ce = mutate_critic_edit(prompt, runner, last_issues)
        if ce: outs.append(ce)
    if len(parents) >= 2:
        import random as _r
        pa, pb = _r.sample(parents, 2)
        outs.append(crossover(pa, pb))
    seen=set(); uniq=[]
    for c in outs:
        if c["text"] not in seen:
            seen.add(c["text"]); uniq.append(c)
    return uniq