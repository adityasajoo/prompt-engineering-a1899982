from __future__ import annotations
import argparse
from ..model.train_classifier import train_classifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default="policies/mutation_classifier.pkl")
    args = ap.parse_args()
    train_classifier(args.runs_dir, args.out)

if __name__ == "__main__":
    main()
