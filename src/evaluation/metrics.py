# src/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


@dataclass
class MetricsResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> MetricsResult:
    acc = accuracy_score(y_true, y_pred)

    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    return MetricsResult(
        accuracy=float(acc),
        precision_macro=float(p_m),
        recall_macro=float(r_m),
        f1_macro=float(f_m),
        precision_weighted=float(p_w),
        recall_weighted=float(r_w),
        f1_weighted=float(f_w),
    )


def classification_report_df(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> pd.DataFrame:
    """
    Per-class precision / recall / F1 table (DataFrame-friendly).
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).transpose()
    return df
