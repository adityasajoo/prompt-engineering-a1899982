import sys, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def main(runs_dir: str):
    # Approximate: for each parent in a generation, "best_op" = op of highest-scoring child.
    # "pred_op" = op of the *first* child logged for that parent (learned mode places predicted first).
    mats = defaultdict(lambda: defaultdict(int))
    for rd in Path(runs_dir).iterdir():
        if not rd.is_dir(): continue
        manifest = rd/"manifest.json"
        if not manifest.exists(): continue
        phase = json.loads(manifest.read_text()).get("phase")
        if phase != "learned":  # only meaningful for learned mode
            continue
        ev = rd/"evals.jsonl"; mu = rd/"mutations.jsonl"
        if not (ev.exists() and mu.exists()): continue

        # gather child comp per (gen, child_cfg)
        child_comp = {}
        for line in ev.read_text().splitlines():
            o = json.loads(line)
            if o.get("event")!="eval": continue
            gen = o.get("gen"); cfg = json.dumps(o.get("prompt"), sort_keys=True)
            child_comp[(gen, cfg)] = child_comp.get((gen,cfg), []) + [o["metrics"]["composite"]]
        child_mean = {k: sum(v)/len(v) for k,v in child_comp.items()}

        # group mutations by (gen,parent_cfg)
        groups = defaultdict(list)
        for line in mu.read_text().splitlines():
            o = json.loads(line)
            if o.get("event")!="mutation": continue
            gen = o.get("gen")
            parent = json.dumps(o.get("parent"), sort_keys=True)
            child  = json.dumps(o.get("child"), sort_keys=True)
            op = o.get("op") or "unknown"
            groups[(gen,parent)].append((op, child))

        for (gen,parent), lst in groups.items():
            if not lst: continue
            # predicted op assumed = first logged
            pred_op = lst[0][0]
            # best op by child mean score
            best_op = max(lst, key=lambda t: child_mean.get((gen, t[1]), -1.0))[0]
            mats[pred_op][best_op] += 1

    if not mats:
        print("No learned runs to plot confusion matrix"); return

    ops = sorted(set(list(mats.keys()) + [k for d in mats.values() for k in d.keys()]))
    M = np.zeros((len(ops), len(ops)), dtype=int)
    for i, po in enumerate(ops):
        for j, bo in enumerate(ops):
            M[i,j] = mats[po].get(bo, 0)

    plt.figure(figsize=(6,5))
    plt.imshow(M, interpolation="nearest")
    plt.xticks(range(len(ops)), ops, rotation=25, ha="right")
    plt.yticks(range(len(ops)), ops)
    plt.xlabel("Best op"); plt.ylabel("Predicted op (first child)")
    plt.title("Confusion matrix of predicted vs. best op (learned)")
    for i in range(len(ops)):
        for j in range(len(ops)):
            plt.text(j, i, M[i,j], ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out = Path(runs_dir)/"plot_confusion_matrix.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
