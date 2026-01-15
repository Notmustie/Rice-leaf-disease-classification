"""
Confidence-based rejection policy.

Purpose:
- Keep rejection logic out of predict.py so it can be reused consistently in:
  - batch inference
  - evaluation
  - XAI analysis
  - analytics / warehousing

Key concept:
- "rejected" is NOT the same as "misclassified"
  - rejected: pred_conf < threshold
  - misclassified: pred_label != true_label (only defined when true_label exists)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RejectionPolicy:
    threshold: float = 0.60

    def reject(self, confidence: float) -> bool:
        """Return True if prediction should be rejected."""
        return float(confidence) < float(self.threshold)


def apply_rejection_to_df(
    df: pd.DataFrame,
    conf_col: str = "pred_conf",
    out_col: str = "rejected",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Adds/overwrites a boolean 'rejected' column based on pred_conf < threshold.
    Returns a COPY (does not mutate input df).
    """
    if conf_col not in df.columns:
        raise ValueError(f"Missing required confidence column: {conf_col}")

    thr = float(threshold) if threshold is not None else float(df.get("reject_threshold", np.nan).dropna().iloc[0]
                                                              if "reject_threshold" in df.columns and df["reject_threshold"].notna().any()
                                                              else 0.60)

    out = df.copy()
    out[out_col] = out[conf_col].astype(float) < thr
    out["reject_threshold"] = thr
    return out


def add_correctness_columns(
    df: pd.DataFrame,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    out_col: str = "is_correct",
) -> pd.DataFrame:
    """
    Adds is_correct where labels exist; otherwise NaN.
    Returns a COPY.
    """
    if pred_col not in df.columns:
        raise ValueError(f"Missing required pred label column: {pred_col}")
    if true_col not in df.columns:
        # unlabeled dataset, just add NaN column
        out = df.copy()
        out[out_col] = np.nan
        return out

    out = df.copy()
    has_true = out[true_col].notna()
    out[out_col] = np.where(has_true, out[true_col] == out[pred_col], np.nan)
    return out


def decision_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a 2x2 decision table over labeled rows:
      - accepted vs rejected
      - correct vs misclassified

    Returns a DataFrame with counts + rates.
    """
    required = {"rejected", "is_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labeled = df[df["is_correct"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(
            [{"note": "No labeled rows available to compute correctness vs rejection buckets."}]
        )

    labeled["accepted"] = ~labeled["rejected"]
    labeled["misclassified"] = ~labeled["is_correct"].astype(bool)

    # counts
    table = (
        labeled.groupby(["accepted", "is_correct"])
        .size()
        .reset_index(name="count")
    )

    # nicer labels
    table["decision"] = np.where(table["accepted"], "ACCEPTED", "REJECTED")
    table["outcome"] = np.where(table["is_correct"], "CORRECT", "MISCLASSIFIED")

    total = len(labeled)
    table["rate_over_labeled"] = table["count"] / total
    return table[["decision", "outcome", "count", "rate_over_labeled"]].sort_values(
        ["decision", "outcome"]
    )


def summarize_by_rejection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates numeric columns by accepted/rejected:
      - mean confidence
      - counts
    Extend this later with focus_score / entropy once XAI joins are available.
    """
    if "rejected" not in df.columns:
        raise ValueError("Missing 'rejected' column.")

    out = df.copy()
    out["decision"] = np.where(out["rejected"], "REJECTED", "ACCEPTED")

    agg = out.groupby("decision").agg(
        samples=("path", "count") if "path" in out.columns else ("pred_label", "count"),
        mean_conf=("pred_conf", "mean") if "pred_conf" in out.columns else ("decision", "size"),
    ).reset_index()

    # optional: accuracy on labeled subset within each decision
    if "is_correct" in out.columns and out["is_correct"].notna().any():
        acc = (
            out[out["is_correct"].notna()]
            .groupby("decision")["is_correct"]
            .mean()
            .reset_index(name="accuracy_labeled_subset")
        )
        agg = agg.merge(acc, on="decision", how="left")

    return agg
