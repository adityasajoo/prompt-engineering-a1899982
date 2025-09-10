from __future__ import annotations
import re
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from functools import lru_cache
import hashlib

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def sentence_split(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents: sents = [text.strip()]
    return sents

def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

@lru_cache(maxsize=512)
def _precompute(text_hash: str, text: str) -> Tuple[List[str], np.ndarray, np.ndarray, 'scipy.sparse.csr_matrix']:
    """
    Returns:
      sentences,
      base_scores (TextRank),
      tf_sums (per-sentence),
      E (TF-IDF unigram matrix for sentences)
    All cached per-article (by hash).
    """
    sents = sentence_split(text)

    # Unigram TF-IDF for MMR and keyword weights
    vect_uni = TfidfVectorizer(min_df=1)
    E = vect_uni.fit_transform(sents)   # shape: [n_sents, vocab]
    tf_sums = E.sum(1).A.ravel()

    # Bi-gram TF-IDF for TextRank similarity
    vect_bi = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vect_bi.fit_transform(sents)    # csr
    sim = (X @ X.T)                     # csr matrix
    # zero the diagonal without densifying
    sim.setdiag(0.0)
    sim.eliminate_zeros()

    # Build graph from sparse matrix (weights carried)
    g = nx.from_scipy_sparse_array(sim)
    pr = nx.pagerank(g, alpha=0.85, weight="weight", max_iter=100)
    base_scores = np.array([pr.get(i, 0.0) for i in range(len(sents))], dtype=float)
    if base_scores.max() > 0:
        base_scores = base_scores / base_scores.max()
    return sents, base_scores, tf_sums, E

def _mmr_select_with_E(sentences: List[str], scores: np.ndarray, E, k: int, lambda_div=0.6) -> List[int]:
    from scipy.sparse import csr_matrix
    selected, cand = [], list(range(len(sentences)))
    if k >= len(sentences): return list(range(len(sentences)))
    # normalize rows to use dot as cosine proxy (fast-enough)
    # (optional — E is already TF-IDF; good proxy even without explicit norm)
    while len(selected) < k and cand:
        best_i, best_val = None, -1e9
        for i in cand:
            if not selected:
                val = scores[i]
            else:
                # max similarity to already selected
                sim_to_sel = 0.0
                Ei = E[i]
                for j in selected:
                    val_ij = (Ei.multiply(E[j])).sum()
                    if val_ij > sim_to_sel: sim_to_sel = val_ij
                val = lambda_div * scores[i] - (1 - lambda_div) * float(sim_to_sel)
            if val > best_val: best_i, best_val = i, val
        selected.append(best_i); cand.remove(best_i)
    return selected

def _summarise_basic(article: str, cfg: Dict) -> str:
    sents, base_scores, tf_sums, E = _precompute(_hash(article), article)
    if len(sents) == 1:
        return sents[0]

    pos_bias = np.linspace(1.0, 1.0 + cfg.get("position_bias", 0.0), num=len(sents))
    if tf_sums.max() > 0:
        kw = 1.0 + cfg.get("keyword_boost", 0.0) * (tf_sums / tf_sums.max())
    else:
        kw = np.ones_like(tf_sums)
    scores = base_scores * pos_bias * kw

    k = max(1, int(cfg.get("target_sentences", 3)))
    ridx = _mmr_select_with_E(sents, scores, E, k=k, lambda_div=1.0 - cfg.get("redundancy_penalty", 0.4))
    ridx.sort()
    if cfg.get("bullet", False):
        return "\n".join(["- " + sents[i] for i in ridx])
    return " ".join([sents[i] for i in ridx])

def _summarise_cot(article: str, cfg: Dict) -> str:
    base_cfg = dict(cfg); base_cfg["target_sentences"] = cfg.get("cot_keypoints", 3)
    kp_text = _summarise_basic(article, base_cfg)
    comp_cfg = dict(cfg); comp_cfg["target_sentences"] = cfg.get("cot_final_sentences", 2)
    return _summarise_basic(kp_text, comp_cfg)

def _summarise_map_reduce(article: str, cfg: Dict) -> str:
    sents = sentence_split(article)
    cs = int(cfg.get("chunk_size", 8)); st = int(cfg.get("chunk_stride", 6))
    ck = int(cfg.get("chunk_k", 2)); fk = int(cfg.get("final_k", 2))
    if len(sents) <= cs:
        return _summarise_basic(article, cfg)
    chunks = []
    for start in range(0, len(sents), st):
        window = sents[start:start+cs]
        if not window: break
        sub_cfg = dict(cfg); sub_cfg["target_sentences"] = ck
        chunks.append(_summarise_basic(" ".join(window), sub_cfg))
        if start + cs >= len(sents): break
    final_cfg = dict(cfg); final_cfg["target_sentences"] = fk
    return _summarise_basic(" ".join(chunks), final_cfg)

def _summarise_cluster(article: str, cfg: Dict) -> str:
    sents = sentence_split(article)
    if len(sents) <= 2: return _summarise_basic(article, cfg)
    k = int(cfg.get("cluster_k", 3))
    vect = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vect.fit_transform(sents)
    km = KMeans(n_clusters=min(k, len(sents)), n_init=5, random_state=42)  # n_init 5 for speed
    labels = km.fit_predict(X)
    # Reuse TextRank (from cache) on the full article
    _, tr, _, _ = _precompute(_hash(article), article)
    picks = []
    for c in sorted(set(labels)):
        idxs = [i for i,l in enumerate(labels) if l==c]
        picks.append(max(idxs, key=lambda i: tr[i]))
    picks.sort()
    out = " ".join([sents[i] for i in picks])
    return _summarise_basic(out, cfg)

def summarise_extractive(article: str, cfg: Dict) -> str:
    p = cfg.get("pipeline", "basic")
    if p == "cot":        return _summarise_cot(article, cfg)
    if p == "map_reduce": return _summarise_map_reduce(article, cfg)
    if p == "cluster":    return _summarise_cluster(article, cfg)
    return _summarise_basic(article, cfg)
