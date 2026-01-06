# src/models/efficientnet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class EfficientNetV2Config:
    img_size: Tuple[int, int] = (224, 224)
    num_classes: int = 6
    dropout: float = 0.30

    weights: Optional[str] = "imagenet"  # set None for training from scratch
    freeze_backbone: bool = True         # Phase 1: True, Phase 2: False
    fine_tune_fraction: float = 0.30     # when unfreezing, keep bottom (1 - frac) frozen


def build_efficientnetv2b0(cfg: EfficientNetV2Config) -> keras.Model:
    """
    EfficientNetV2B0 classifier head:
      - Backbone: EfficientNetV2B0(include_top=False)
      - GAP -> Dropout -> Dense(num_classes, softmax)

    Note: preprocessing (preprocess_input) and augmentation should be done upstream
    (e.g., in src/features/preprocessing.py or train pipeline).
    """
    inputs = keras.Input(shape=(*cfg.img_size, 3), name="image")

    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=cfg.weights,
        input_tensor=inputs,
        pooling=None,
    )
    base.trainable = not cfg.freeze_backbone  # if freeze_backbone=True => base frozen

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="probs")(x)

    model = keras.Model(inputs, outputs, name="efficientnetv2b0")
    return model


def set_fine_tuning(model: keras.Model, fine_tune_fraction: float = 0.30) -> None:
    """
    Unfreeze only the top fraction of EfficientNet backbone layers.
    Example: fine_tune_fraction=0.30 => unfreeze top 30%, freeze bottom 70%.

    Call this BEFORE recompiling for fine-tuning.
    """
    if not (0.0 <= fine_tune_fraction <= 1.0):
        raise ValueError("fine_tune_fraction must be in [0, 1].")

    # Locate backbone by name pattern (robust enough for this repo)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        # Fallback: often the backbone is the second layer as a Model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break

    if backbone is None:
        raise RuntimeError("Could not locate EfficientNet backbone inside the model.")

    backbone.trainable = True
    n = len(backbone.layers)
    fine_tune_at = int(n * (1.0 - fine_tune_fraction))  # freeze bottom (1-frac)

    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= fine_tune_at
