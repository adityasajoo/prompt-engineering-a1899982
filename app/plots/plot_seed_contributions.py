import sys, json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def main(runs_dir: str):
    rows = []
    for rd in Path(runs_dir).iterdir():
        sp = rd/"seeds.jsonl"
        ev = rd/"evals.jsonl"
        if not (rd.is_dir() and sp.exists() and ev.exists()): continue
        # map id->name
        id2name = {}
        for line in sp.read_text().splitlines():
            o = json.loads(line)
            if o.get("event")=="seed":
                id2name[o["id"]] = o.get("name","seed")
        # collect gen0 scores per seed id
        gen0 = defaultdict(list)
        for line in ev.read_text().splitlines():
            o = json.loads(line)
            if o.get("event")=="eval" and o.get("gen")==0:
                gen0[o["prompt_id"]].append(o["metrics"]["composite"])
        for pid, arr in gen0.items():
            rows.append((rd.name, id2name.get(pid,pid), sum(arr)/len(arr)))
    if not rows:
        print("No seed data found"); return
    # average per seed name across runs
    from statistics import mean
    seed2score = defaultdict(list)
    for run, name, s in rows:
        seed2score[name].append(s)
    names = sorted(seed2score.keys())
    vals = [mean(seed2score[n]) for n in names]
    xs = range(len(names))
    plt.figure(figsize=(9,4))
    plt.bar(xs, vals)
    plt.xticks(xs, names, rotation=25, ha="right")
    plt.ylim(0,1); plt.ylabel("Avg Gen0 composite"); plt.title("Seed strategy contributions (Gen 0)")
    plt.tight_layout()
    out = Path(runs_dir)/"plot_seed_contributions.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs")
