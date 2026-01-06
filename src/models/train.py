#!/usr/bin/env python3
"""
Train entry-point for rice leaf disease models (Custom CNN / EfficientNetV2B0)
using the pipeline outputs (splits.csv + base image directory).

Expected inputs:
- splits.csv with columns: path,label,split   (split in {train,val,test})
- base_dir where the "path" entries in splits.csv are relative to
  (typically data/interim/images if your split.py was based on interim)

Also supports a fixed class_names file for consistent label mapping.

Outputs (per run):
- artifacts/<run_name>/
    - best.keras
    - history_head.csv (for efficientnet head training) OR history.csv (custom cnn)
    - history_ft.csv   (for efficientnet fine-tuning, if enabled)
    - run_config.json
    - class_names.txt  (copied)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.features.preprocessing import (
    PreprocessConfig,
    build_dataset_from_splits_csv,
    preprocess_for_custom_cnn,
    preprocess_for_efficientnetv2,
)
from src.models.custom_cnn import CustomCNNConfig, build_custom_cnn
from src.models.efficientnet import EfficientNetV2Config, build_efficientnetv2b0, set_fine_tuning


# -------------------------
# Utilities
# -------------------------

def read_class_names(path: Path) -> List[str]:
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise ValueError(f"class_names file is empty: {path}")
    return names


def infer_class_names_from_splits(splits_csv: Path) -> List[str]:
    df = pd.read_csv(splits_csv)
    if "label" not in df.columns:
        raise ValueError("splits.csv must contain a 'label' column.")
    return sorted(df["label"].dropna().unique().tolist())


def make_augmentation() -> keras.Model:
    # Keep it modest: matches your custom CNN spirit, works for EfficientNet too
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.06),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.08),
        ],
        name="data_aug",
    )


def add_augmentation(ds: tf.data.Dataset, aug: keras.Model) -> tf.data.Dataset:
    # ds yields (image, onehot)
    def _map(x, y):
        x = aug(x, training=True)
        return x, y
    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def save_history(history: keras.callbacks.History, out_csv: Path) -> None:
    hist = pd.DataFrame(history.history)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out_csv, index=False)


@dataclass
class RunConfig:
    model: str
    img_size: Tuple[int, int]
    batch_size: int
    seed: int
    epochs: int
    lr: float
    label_smoothing: float
    splits_csv: str
    base_dir: str
    class_names_file: str
    use_augmentation: bool

    # efficientnet extras
    epochs_head: int = 0
    epochs_ft: int = 0
    lr_head: float = 0.0
    lr_ft: float = 0.0
    fine_tune_fraction: float = 0.0


# -------------------------
# Main train logic
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", choices=["custom_cnn", "efficientnet"], required=True)

    ap.add_argument("--splits_csv", type=Path, default=Path("data/processed/splits.csv"))
    ap.add_argument("--base_dir", type=Path, default=Path("data/interim/images"))

    ap.add_argument("--class_names", type=Path, default=Path("configs/class_names.txt"),
                    help="Fixed class order. If missing, will infer from splits.csv and write it.")

    ap.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_augmentation", action="store_true")

    # Custom CNN training params (single-phase)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    # EfficientNet two-phase params
    ap.add_argument("--epochs_head", type=int, default=15)
    ap.add_argument("--epochs_ft", type=int, default=10)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-5)
    ap.add_argument("--fine_tune_fraction", type=float, default=0.30,
                    help="Unfreeze top fraction of backbone in fine-tuning (0.30 = top 30%)")

    # Outputs
    ap.add_argument("--run_name", type=str, default="",
                    help="If empty, a timestamped name is created.")
    ap.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))

    args = ap.parse_args()

    splits_csv = args.splits_csv.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    class_names_file = args.class_names.expanduser().resolve()

    if not splits_csv.exists():
        raise FileNotFoundError(f"Missing splits_csv: {splits_csv}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing base_dir: {base_dir}")

    # Ensure class names exist (fixed mapping)
    class_names_file.parent.mkdir(parents=True, exist_ok=True)
    if class_names_file.exists():
        class_names = read_class_names(class_names_file)
    else:
        class_names = infer_class_names_from_splits(splits_csv)
        class_names_file.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    img_size = (int(args.img_size[0]), int(args.img_size[1]))
    num_classes = len(class_names)

    # Decide preprocess function
    if args.model == "custom_cnn":
        preprocess_fn = preprocess_for_custom_cnn
    else:
        preprocess_fn = preprocess_for_efficientnetv2

    # Build datasets from splits.csv (one-hot labels)
    cfg = PreprocessConfig(img_size=img_size, batch_size=args.batch_size, shuffle=True, seed=args.seed)

    train_ds = build_dataset_from_splits_csv(
        splits_csv=splits_csv,
        base_dir=base_dir,
        split="train",
        class_names=class_names,
        preprocess_fn=preprocess_fn,
        cfg=cfg,
    )

    # For eval splits, shuffle=False
    eval_cfg = PreprocessConfig(img_size=img_size, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    val_ds = build_dataset_from_splits_csv(
        splits_csv=splits_csv,
        base_dir=base_dir,
        split="val",
        class_names=class_names,
        preprocess_fn=preprocess_fn,
        cfg=eval_cfg,
    )

    test_ds = build_dataset_from_splits_csv(
        splits_csv=splits_csv,
        base_dir=base_dir,
        split="test",
        class_names=class_names,
        preprocess_fn=preprocess_fn,
        cfg=eval_cfg,
    )

    # Optional augmentation
    if args.use_augmentation:
        aug = make_augmentation()
        train_ds = add_augmentation(train_ds, aug)

    # Run naming + artifacts
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name.strip() or f"{args.model}_{stamp}"
    out_dir = (args.artifacts_dir.expanduser().resolve() / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common callbacks
    ckpt_path = out_dir / "best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]

    # -------------------------
    # Train Custom CNN (single-phase)
    # -------------------------
    if args.model == "custom_cnn":
        model_cfg = CustomCNNConfig(img_size=img_size, num_classes=num_classes)
        model = build_custom_cnn(model_cfg)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
        )

        save_history(history, out_dir / "history.csv")

        test_metrics = model.evaluate(test_ds, verbose=1)
        print(f"[OK] Test metrics: {dict(zip(model.metrics_names, test_metrics))}")

        run_cfg = RunConfig(
            model=args.model,
            img_size=img_size,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            label_smoothing=args.label_smoothing,
            splits_csv=str(splits_csv),
            base_dir=str(base_dir),
            class_names_file=str(class_names_file),
            use_augmentation=bool(args.use_augmentation),
        )

    # -------------------------
    # Train EfficientNet (two-phase)
    # -------------------------
    else:
        # Phase 1: frozen backbone
        eff_cfg = EfficientNetV2Config(
            img_size=img_size,
            num_classes=num_classes,
            dropout=0.30,
            weights="imagenet",
            freeze_backbone=True,
            fine_tune_fraction=float(args.fine_tune_fraction),
        )
        model = build_efficientnetv2b0(eff_cfg)

        # Use categorical CE to stay consistent with one-hot labels in pipeline
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr_head),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
            metrics=["accuracy"],
        )

        history_head = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_head,
            callbacks=callbacks,
        )
        save_history(history_head, out_dir / "history_head.csv")

        # Phase 2: fine-tune top layers
        # Unfreeze + keep bottom portion frozen
        set_fine_tuning(model, fine_tune_fraction=float(args.fine_tune_fraction))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr_ft),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
            metrics=["accuracy"],
        )

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_ft,
            callbacks=callbacks,
        )
        save_history(history_ft, out_dir / "history_ft.csv")

        test_metrics = model.evaluate(test_ds, verbose=1)
        print(f"[OK] Test metrics: {dict(zip(model.metrics_names, test_metrics))}")

        run_cfg = RunConfig(
            model=args.model,
            img_size=img_size,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs_head + args.epochs_ft,
            lr=args.lr_head,
            label_smoothing=0.0,
            splits_csv=str(splits_csv),
            base_dir=str(base_dir),
            class_names_file=str(class_names_file),
            use_augmentation=bool(args.use_augmentation),
            epochs_head=args.epochs_head,
            epochs_ft=args.epochs_ft,
            lr_head=args.lr_head,
            lr_ft=args.lr_ft,
            fine_tune_fraction=float(args.fine_tune_fraction),
        )

    # Save run config + copy class_names into artifacts
    (out_dir / "run_config.json").write_text(json.dumps(asdict(run_cfg), indent=2), encoding="utf-8")
    (out_dir / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    print(f"\n✅ Saved best model: {ckpt_path}")
    print(f"🧾 Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
