#!/usr/bin/env python3
"""
Plot confusion matrix CSVs (raw or normalized) into PNG images.

Input: CSV created by save_confusion_matrices (index=labels, columns=labels)
Output: PNG heatmap

Usage:
  python -m src.evaluation.plot_confusion_matrix \
    --cm_csv reports/evaluation/CustomCNN_confusion_normalized.csv \
    --out_png reports/evaluation/CustomCNN_confusion_normalized.png \
    --title "CustomCNN (Normalized Confusion Matrix)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm_csv", type=Path, required=True)
    ap.add_argument("--out_png", type=Path, required=True)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--annotate", action="store_true", help="Write values into cells (can be busy).")
    args = ap.parse_args()

    cm_csv = args.cm_csv.expanduser().resolve()
    out_png = args.out_png.expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cm_csv, index_col=0)
    labels = df.index.tolist()
    mat = df.values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if args.title:
        ax.set_title(args.title)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if args.annotate:
        # annotate values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"✅ Saved: {out_png}")


if __name__ == "__main__":
    main()
