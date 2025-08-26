import json, sys, re
from pathlib import Path
import matplotlib.pyplot as plt

def load_best_series(run_dir: Path):
    best_file = run_dir / "bests.jsonl"
    xs, comp, r, b, s = [], [], [], [], []
    if not best_file.exists(): return xs, comp, r, b, s
    for line in best_file.read_text().splitlines():
        obj = json.loads(line)
        if obj.get("event") == "best_update":
            xs.append(obj["gen"])
            m = obj["metrics"]
            comp.append(obj.get("score", m.get("composite", 0.0)))
            r.append(m.get("rougeL", 0.0))
            b.append(m.get("bleu", 0.0))
            s.append(m.get("bertscore", 0.0))
    return xs, comp, r, b, s

def main(run_dir):
    run_path = Path(run_dir)
    xs, comp, r, b, s = load_best_series(run_path)
    if not xs:
        print("No bests.jsonl found or empty."); return
    # Composite
    plt.figure(figsize=(7,4))
    plt.plot(xs, comp, marker="o")
    plt.xlabel("Generation"); plt.ylabel("Composite score")
    plt.title(f"Best composite over generations\n{run_path.name}")
    plt.tight_layout()
    out1 = run_path / "plot_best_composite.png"
    plt.savefig(out1, dpi=160); plt.close()
    # Metrics
    plt.figure(figsize=(7,4))
    plt.plot(xs, r, marker="o", label="ROUGE-L")
    plt.plot(xs, b, marker="o", label="BLEU")
    plt.plot(xs, s, marker="o", label="BERTScore")
    plt.xlabel("Generation"); plt.ylabel("Score")
    plt.title("Best metrics over generations")
    plt.legend(); plt.tight_layout()
    out2 = run_path / "plot_best_metrics.png"
    plt.savefig(out2, dpi=160); plt.close()
    print(f"Saved:\n  {out1}\n  {out2}")

if __name__ == "__main__":
    main(sys.argv[1])

# Usage: python -m app.plots.plot_run_progress runs/<RUN_ID>
