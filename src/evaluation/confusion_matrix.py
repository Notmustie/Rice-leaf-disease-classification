# src/evaluation/confusion_matrix.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_confusion(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Returns confusion matrix as a DataFrame.
    If normalize=True, rows sum to 1.
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    return pd.DataFrame(cm, index=labels, columns=labels)


def save_confusion_matrices(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    out_dir: Path,
    prefix: str,
) -> Tuple[Path, Path]:
    """
    Saves raw and normalized confusion matrices as CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = compute_confusion(y_true, y_pred, labels, normalize=False)
    norm_df = compute_confusion(y_true, y_pred, labels, normalize=True)

    raw_path = out_dir / f"{prefix}_confusion_raw.csv"
    norm_path = out_dir / f"{prefix}_confusion_normalized.csv"

    raw_df.to_csv(raw_path)
    norm_df.to_csv(norm_path)

    return raw_path, norm_path
