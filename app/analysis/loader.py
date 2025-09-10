# app/analysis/loader.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict

def _cfg_flat(cfg: dict) -> dict:
    d = {
        "pipeline": cfg.get("pipeline","basic"),
        "target_sentences": cfg.get("target_sentences",3),
        "position_bias": cfg.get("position_bias",0.0),
        "keyword_boost": cfg.get("keyword_boost",0.0),
        "redundancy_penalty": cfg.get("redundancy_penalty",0.0),
        "bullet": bool(cfg.get("bullet", False)),
        "cot_keypoints": cfg.get("cot_keypoints"),
        "cot_final_sentences": cfg.get("cot_final_sentences"),
        "chunk_size": cfg.get("chunk_size"),
        "chunk_stride": cfg.get("chunk_stride"),
        "chunk_k": cfg.get("chunk_k"),
        "final_k": cfg.get("final_k"),
        "cluster_k": cfg.get("cluster_k"),
    }
    return d

def load_runs(runs_dir: str) -> pd.DataFrame:
    """Return one row per eval event (incl metrics)."""
    rows = []
    for rd in Path(runs_dir).iterdir():
        if not rd.is_dir(): continue
        manifest = (rd/"manifest.json")
        phase = None; dataset = None
        if manifest.exists():
            m = json.loads(manifest.read_text())
            phase = m.get("phase"); dataset = m.get("dataset")
        evals = rd/"evals.jsonl"
        if not evals.exists(): continue
        for line in evals.read_text().splitlines():
            o = json.loads(line)
            if o.get("event") != "eval": continue
            cfg = o.get("prompt") or {}
            output = o.get("output","") or ""
            flat = _cfg_flat(cfg)
            rows.append({
                "run_id": rd.name,
                "phase": phase or o.get("phase"),
                "dataset": dataset or o.get("dataset"),
                "gen": o.get("gen"),
                "example_index": o.get("example_index"),
                "rougeL": o["metrics"]["rougeL"],
                "bleu": o["metrics"]["bleu"],
                "bertscore": o["metrics"]["bertscore"],
                "composite": o["metrics"]["composite"],
                "output_len_words": len(output.split()),
                "output_len_chars": len(output),
                **flat,
                "cfg_json": json.dumps(cfg, sort_keys=True)
            })
    return pd.DataFrame(rows)

def load_mutations(runs_dir: str) -> pd.DataFrame:
    rows = []
    for rd in Path(runs_dir).iterdir():
        if not rd.is_dir(): continue
        muts = rd/"mutations.jsonl"
        if not muts.exists(): continue
        for line in muts.read_text().splitlines():
            o = json.loads(line)
            if o.get("event") != "mutation": continue
            rows.append({
                "run_id": rd.name,
                "gen": o.get("gen"),
                "op": o.get("op") or "unknown",
                "parent_json": json.dumps(o.get("parent") or {}, sort_keys=True),
                "child_json": json.dumps(o.get("child") or {}, sort_keys=True),
            })
    return pd.DataFrame(rows)

def aggregate_improvements_all_metrics(evals_df: pd.DataFrame, muts_df: pd.DataFrame) -> pd.DataFrame:
    """Return deltas (child-parent) for rougeL/bleu/bertscore/composite."""
    if evals_df.empty or muts_df.empty:
        return pd.DataFrame()
    mcols = ["rougeL","bleu","bertscore","composite"]
    avg = (evals_df.groupby(["run_id","gen","cfg_json"])[mcols]
           .mean().reset_index().rename(columns={c:f"{c}_mean" for c in mcols}))
    j = (muts_df
         .merge(avg.rename(columns={"gen":"gen_child","cfg_json":"child_json"}), on=["run_id","child_json"], how="left")
         .merge(avg.rename(columns={"gen":"gen_parent","cfg_json":"parent_json"}), on=["run_id","parent_json"], how="left", suffixes=("","_parent")))
    for c in mcols:
        j[f"delta_{c}"] = j[f"{c}_mean"] - j[f"{c}_mean_parent"]
    return j

# ----- seed origin lineage helpers -----

def _seed_maps_for_run(run_dir: Path):
    """Return (cfg_json -> seed_name) and child->parent cfg map."""
    seeds = {}
    sp = run_dir/"seeds.jsonl"
    if sp.exists():
        for line in sp.read_text().splitlines():
            o = json.loads(line)
            if o.get("event")=="seed":
                cfg = json.dumps(o.get("cfg") or {}, sort_keys=True)
                seeds[cfg] = o.get("name","seed")
    lineage = {}
    mp = run_dir/"mutations.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            o = json.loads(line)
            if o.get("event")=="mutation":
                parent = json.dumps(o.get("parent") or {}, sort_keys=True)
                child  = json.dumps(o.get("child")  or {}, sort_keys=True)
                lineage[child] = parent
    return seeds, lineage

def _origin_for_cfg(cfg_json: str, seeds_map: dict, lineage_map: dict, max_hops: int = 50) -> str:
    cur = cfg_json
    hops = 0
    while hops < max_hops and cur in lineage_map:
        cur = lineage_map[cur]
        hops += 1
    return seeds_map.get(cur, "unknown")

def annotate_with_origin(evals_df: pd.DataFrame, runs_dir: str) -> pd.DataFrame:
    if evals_df.empty: 
        evals_df["origin_seed"] = "unknown"
        return evals_df
    out = evals_df.copy()
    out["origin_seed"] = "unknown"
    runs_path = Path(runs_dir)
    for run_id in out["run_id"].unique():
        rd = runs_path/run_id
        seeds, lineage = _seed_maps_for_run(rd)
        mask = out["run_id"]==run_id
        cfgs = out.loc[mask, "cfg_json"].tolist()
        origins = [ _origin_for_cfg(c, seeds, lineage) for c in cfgs ]
        out.loc[mask, "origin_seed"] = origins
    return out

def try_add_engine_column(evals_df: pd.DataFrame, runs_dir: str) -> pd.DataFrame:
    if "engine" in evals_df.columns:
        return evals_df
    out = evals_df.copy()
    out["engine"] = "N/A"
    return out

# ----- manifest enrichment (engine, fraction, etc.) -----

def enrich_from_manifest(evals_df: pd.DataFrame, runs_dir: str) -> pd.DataFrame:
    if evals_df.empty:
        return evals_df
    out = evals_df.copy()
    out["engine"] = "extractive"
    out["fraction"] = 1.0
    out["n_used"] = None
    out["gens"] = None
    out["pop"] = None
    out["topk"] = None

    base = Path(runs_dir)
    for run_id in out["run_id"].unique():
        man = base / run_id / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        eng = m.get("engine", "extractive")
        frac = float(m.get("fraction", 1.0))
        n_used = m.get("n_used")
        gens = m.get("generations")
        pop  = m.get("population_size")
        topk = m.get("top_k")
        mask = out["run_id"] == run_id
        out.loc[mask, "engine"]   = eng
        out.loc[mask, "fraction"] = frac
        out.loc[mask, "n_used"]   = n_used
        out.loc[mask, "gens"]     = gens
        out.loc[mask, "pop"]      = pop
        out.loc[mask, "topk"]     = topk
    return out
