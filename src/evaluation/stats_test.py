#!/usr/bin/env python3
"""
Statistical comparison between two classifiers on the same labeled dataset.

Implements McNemar's test using the 2x2 table:
  b = A correct, B wrong
  c = A wrong,  B correct

Usage:
  python -m src.evaluation.stat_tests \
    --pred_a outputs/preds_custom.csv --name_a CustomCNN \
    --pred_b outputs/preds_effnet.csv --name_b EfficientNetV2B0 \
    --out_csv reports/evaluation/mcnemar.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def load_labeled(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    needed = {"path", "true_label", "pred_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{pred_csv} missing columns: {sorted(missing)}")
    df = df[df["true_label"].notna()].copy()
    df["correct"] = df["true_label"] == df["pred_label"]
    return df[["path", "correct"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", type=Path, required=True)
    ap.add_argument("--name_a", type=str, required=True)
    ap.add_argument("--pred_b", type=Path, required=True)
    ap.add_argument("--name_b", type=str, required=True)
    ap.add_argument("--out_csv", type=Path, default=Path("reports/evaluation/mcnemar.csv"))
    ap.add_argument("--exact", action="store_true", help="Use exact binomial test (slower).")
    args = ap.parse_args()

    a = load_labeled(args.pred_a.expanduser().resolve())
    b = load_labeled(args.pred_b.expanduser().resolve())

    # Join on path (same samples)
    merged = a.merge(b, on="path", suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("No overlapping labeled samples between the two prediction files (by path).")

    # Build contingency
    # b_count: A correct, B wrong
    # c_count: A wrong, B correct
    b_count = int(((merged["correct_a"] == True) & (merged["correct_b"] == False)).sum())
    c_count = int(((merged["correct_a"] == False) & (merged["correct_b"] == True)).sum())
    both_correct = int(((merged["correct_a"] == True) & (merged["correct_b"] == True)).sum())
    both_wrong = int(((merged["correct_a"] == False) & (merged["correct_b"] == False)).sum())

    table = [[both_correct, b_count],
             [c_count, both_wrong]]

    result = mcnemar(table, exact=bool(args.exact), correction=True)

    out = pd.DataFrame(
        [{
            "model_a": args.name_a,
            "model_b": args.name_b,
            "n_samples": int(len(merged)),
            "both_correct": both_correct,
            "a_correct_b_wrong": b_count,
            "a_wrong_b_correct": c_count,
            "both_wrong": both_wrong,
            "statistic": float(result.statistic) if result.statistic is not None else None,
            "p_value": float(result.pvalue),
            "exact": bool(args.exact),
        }]
    )

    out_path = args.out_csv.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("✅ McNemar test complete")
    print(out.to_string(index=False))
    print(f"📄 Saved: {out_path}")


if __name__ == "__main__":
    main()
