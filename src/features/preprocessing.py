"""
TensorFlow preprocessing utilities for:
- Custom CNN (simple rescaling)
- EfficientNetV2 (model-specific preprocess_input)

This file is meant to be imported by training/inference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Optional

import tensorflow as tf


@dataclass
class PreprocessConfig:
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 42


def preprocess_for_custom_cnn(img: tf.Tensor) -> tf.Tensor:
    # assumes img is uint8 [0,255]
    img = tf.cast(img, tf.float32) / 255.0
    return img


def preprocess_for_efficientnetv2(img: tf.Tensor) -> tf.Tensor:
    # EfficientNetV2 expects preprocess_input
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    img = tf.cast(img, tf.float32)
    return preprocess_input(img)


def decode_and_resize(path: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img.set_shape((*img_size, 3))
    return img


def build_dataset_from_splits_csv(
    splits_csv: str | Path,
    base_dir: str | Path,
    split: str,
    class_names: Optional[List[str]],
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
    cfg: PreprocessConfig = PreprocessConfig(),
) -> tf.data.Dataset:
    """
    Reads data/processed/splits.csv and builds a tf.data.Dataset for the requested split.

    splits.csv columns: path,label,split
    base_dir: typically "data/interim/images" OR can be "data/processed/<split>" if you adapt.
    This implementation assumes `path` is relative to base_dir (like in split.py).
    """
    import pandas as pd

    splits_csv = Path(splits_csv)
    base_dir = Path(base_dir)

    df = pd.read_csv(splits_csv)
    df = df[df["split"] == split].copy()

    paths = [str((base_dir / p).resolve()) for p in df["path"].tolist()]
    labels_str = df["label"].tolist()

    if class_names is None:
        class_names = sorted(df["label"].unique().tolist())

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = [label_to_idx[s] for s in labels_str]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if cfg.shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=cfg.seed, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = decode_and_resize(path, cfg.img_size)
        img = preprocess_fn(img)
        return img, tf.one_hot(label, depth=len(class_names))

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
