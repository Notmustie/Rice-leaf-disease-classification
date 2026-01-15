#!/usr/bin/env python3
"""
Batch inference on a folder of images.

Inputs:
- model_path: .keras model saved by training
- input_dir: folder containing images (optionally in subfolders by class)
- class_names.txt: fixed label order used at training time

Outputs:
- CSV with: path,true_label,pred_label,pred_conf,rejected,topk_labels,topk_confs,preprocess_mode

Usage:
  python -m src.inference.predict \
    --model_path artifacts/custom_cnn_xxx/best.keras \
    --input_dir data/predict \
    --class_names configs/class_names.txt \
    --output_csv outputs/predictions.csv \
    --preprocess custom \
    --reject_threshold 0.60

  python -m src.inference.predict \
    --model_path artifacts/efficientnet_xxx/best.keras \
    --input_dir data/predict \
    --class_names configs/class_names.txt \
    --output_csv outputs/predictions.csv \
    --preprocess efficientnet \
    --reject_threshold 0.60
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from src.inference.rejection import RejectionPolicy

import numpy as np
import pandas as pd
import tensorflow as tf


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_class_names(path: Path) -> List[str]:
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise ValueError(f"class_names file is empty: {path}")
    return names


def iter_images(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def infer_true_label(path: Path, input_dir: Path, class_names: List[str]) -> Optional[str]:
    """
    If input_dir has structure: input_dir/<label>/<file>, return <label> if it matches class_names.
    Otherwise return None.
    """
    try:
        rel = path.relative_to(input_dir)
    except ValueError:
        return None
    if len(rel.parts) >= 2:
        maybe_label = rel.parts[0]
        if maybe_label in class_names:
            return maybe_label
    return None


def preprocess_custom(x: tf.Tensor) -> tf.Tensor:
    # x is float32 [0..255] after decode+resize; scale to [0..1]
    return x / 255.0


def preprocess_efficientnetv2(x: tf.Tensor) -> tf.Tensor:
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    return preprocess_input(x)


def load_and_preprocess(path: tf.Tensor, img_size: Tuple[int, int], preprocess_mode: str) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)

    if preprocess_mode == "custom":
        img = preprocess_custom(img)
    elif preprocess_mode == "efficientnet":
        img = preprocess_efficientnetv2(img)
    else:
        raise ValueError("preprocess_mode must be 'custom' or 'efficientnet'")

    return img


@dataclass
class PredictConfig:
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    top_k: int = 3
    reject_threshold: float = 0.60  # 0..1
    preprocess_mode: str = "custom"  # custom | efficientnet


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--class_names", type=Path, default=Path("configs/class_names.txt"))
    ap.add_argument("--output_csv", type=Path, default=Path("outputs/predictions.csv"))

    ap.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--reject_threshold", type=float, default=0.60)
    ap.add_argument("--preprocess", choices=["custom", "efficientnet"], required=True)

    ap.add_argument("--save_json_summary", action="store_true",
                    help="Also write a small JSON summary next to the CSV.")

    args = ap.parse_args()

    model_path = args.model_path.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    class_names_file = args.class_names.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input_dir: {input_dir}")
    if not class_names_file.exists():
        raise FileNotFoundError(f"Missing class_names file: {class_names_file}")

    class_names = read_class_names(class_names_file)
    img_size = (int(args.img_size[0]), int(args.img_size[1]))

    cfg = PredictConfig(
        img_size=img_size,
        batch_size=int(args.batch_size),
        top_k=int(args.top_k),
        reject_threshold=float(args.reject_threshold),
        preprocess_mode=str(args.preprocess),
    )

    # Gather images
    paths = iter_images(input_dir)
    if not paths:
        raise SystemExit(f"No images found under: {input_dir}")

    # Create tf.data pipeline
    path_strs = [str(p) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices(path_strs)

    def _map_fn(p):
        img = load_and_preprocess(p, cfg.img_size, cfg.preprocess_mode)
        return p, img

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Predict
    rows = []
    for batch_paths, batch_imgs in ds:
        probs = model.predict(batch_imgs, verbose=0)  # (B, C)

        # top-k
        topk_idx = np.argsort(probs, axis=1)[:, ::-1][:, : cfg.top_k]  # (B, K)
        topk_probs = np.take_along_axis(probs, topk_idx, axis=1)       # (B, K)

        pred_idx = topk_idx[:, 0]
        pred_conf = topk_probs[:, 0]

        for i in range(len(pred_idx)):
            p = batch_paths[i].numpy().decode("utf-8")
            pred_label = class_names[int(pred_idx[i])]
            conf = float(pred_conf[i])

            policy = RejectionPolicy(threshold=cfg.reject_threshold)

            rejected = policy.reject(conf)

            labels_k = [class_names[int(j)] for j in topk_idx[i].tolist()]
            confs_k = [float(x) for x in topk_probs[i].tolist()]

            true_label = infer_true_label(Path(p), input_dir, class_names)

            rows.append(
                {
                    "path": p,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_conf": conf,
                    "rejected": bool(rejected),
                    "topk_labels": "|".join(labels_k),
                    "topk_confs": "|".join([f"{c:.6f}" for c in confs_k]),
                    "preprocess_mode": cfg.preprocess_mode,
                    "reject_threshold": cfg.reject_threshold,
                }
            )

    df = pd.DataFrame(rows)

    # Add quick correctness flag if true_label exists
    df["is_correct"] = np.where(df["true_label"].notna(), df["true_label"] == df["pred_label"], np.nan)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Optional JSON summary
    if args.save_json_summary:
        summary = {
            "model_path": str(model_path),
            "input_dir": str(input_dir),
            "num_images": int(len(df)),
            "preprocess_mode": cfg.preprocess_mode,
            "reject_threshold": cfg.reject_threshold,
            "top_k": cfg.top_k,
            "has_true_labels": bool(df["true_label"].notna().any()),
        }
        if summary["has_true_labels"]:
            acc = df.loc[df["true_label"].notna(), "is_correct"].mean()
            summary["accuracy_on_labeled_subset"] = None if pd.isna(acc) else float(acc)

        out_json = output_csv.with_suffix(".summary.json")
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"✅ Wrote predictions: {output_csv}")
    if df["true_label"].notna().any():
        labeled = df[df["true_label"].notna()]
        acc = labeled["is_correct"].mean()
        print(f"📌 Labeled subset accuracy: {acc:.4f}  (n={len(labeled)})")
    print(f"📌 Rejected: {int(df['rejected'].sum())} / {len(df)}  (threshold={cfg.reject_threshold})")


if __name__ == "__main__":
    main()
