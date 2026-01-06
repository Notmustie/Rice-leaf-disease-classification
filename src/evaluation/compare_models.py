# src/evaluation/compare_models.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.evaluation.metrics import compute_metrics, classification_report_df
from src.evaluation.confusion_matrix import save_confusion_matrices


def load_preds(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "true_label" not in df.columns:
        raise ValueError(f"{csv_path} has no true_label column.")
    df = df[df["true_label"].notna()].copy()
    if df.empty:
        raise ValueError(f"{csv_path} has no labeled rows (true_label is empty).")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple models on labeled prediction CSVs.")
    ap.add_argument("--pred_csvs", type=Path, nargs="+", required=True,
                    help="Prediction CSVs (from predict.py)")
    ap.add_argument("--model_names", type=str, nargs="+", required=True,
                    help="Names corresponding to each CSV")
    ap.add_argument("--class_names", type=Path, default=Path("configs/class_names.txt"))
    ap.add_argument("--out_dir", type=Path, default=Path("reports/evaluation"))
    ap.add_argument("--write_latex", action="store_true", help="Also write a LaTeX table for the summary.")
    args = ap.parse_args()

    if len(args.pred_csvs) != len(args.model_names):
        raise ValueError("Number of --pred_csvs must match --model_names.")

    labels = [ln.strip() for ln in args.class_names.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for csv_path, model_name in zip(args.pred_csvs, args.model_names):
        df = load_preds(csv_path.expanduser().resolve())

        y_true = df["true_label"].tolist()
        y_pred = df["pred_label"].tolist()

        metrics = compute_metrics(y_true, y_pred, labels)

        summary_rows.append(
            {
                "model": model_name,
                "accuracy": metrics.accuracy,
                "f1_macro": metrics.f1_macro,
                "f1_weighted": metrics.f1_weighted,
                "precision_macro": metrics.precision_macro,
                "recall_macro": metrics.recall_macro,
            }
        )

        # Confusion matrices per model
        save_confusion_matrices(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            out_dir=out_dir,
            prefix=model_name,
        )

        # Per-class report
        report_df = classification_report_df(y_true, y_pred, labels)
        report_df.to_csv(out_dir / f"{model_name}_classification_report.csv", index=True)

    summary_df = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)
    summary_csv = out_dir / "model_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("✅ Model comparison complete")
    print(summary_df.to_string(index=False))
    print(f"📄 Summary saved to: {summary_csv}")

    if args.write_latex:
        latex_path = out_dir / "model_comparison_summary.tex"
        latex = summary_df.to_latex(index=False, float_format="%.4f")
        latex_path.write_text(latex, encoding="utf-8")
        print(f"📄 LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
