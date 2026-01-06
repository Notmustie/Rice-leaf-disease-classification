"""
Custom CNN architecture for rice leaf disease classification.

Designed to:
- balance performance and computational efficiency
- serve as a comparison baseline against EfficientNet
- operate in resource-constrained agricultural settings
"""
# src/models/custom_cnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class CustomCNNConfig:
    img_size: Tuple[int, int] = (224, 224)
    num_classes: int = 6

    # regularization / head
    dense_units: int = 256

    # dropout (matches your code)
    dropout_b2: float = 0.25
    dropout_b3: float = 0.30
    dropout_b4: float = 0.40
    dropout_head: float = 0.30


def build_custom_cnn(cfg: CustomCNNConfig) -> keras.Model:
    """
    Your custom CNN:
      - 4 conv blocks: (32, 64, 128, 256) each with 2x Conv + BN + ReLU + MaxPool
      - Dropout after blocks 2/3/4
      - Head: GAP -> Dense(256, relu) -> Dropout -> Softmax(num_classes)

    Note: expects inputs already scaled to [0,1] OR any consistent preprocessing done upstream.
    """
    inputs = keras.Input(shape=(*cfg.img_size, 3), name="image")

    x = inputs

    # --- Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # --- Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(cfg.dropout_b2)(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(cfg.dropout_b3)(x)

    # --- Block 4 ---
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(cfg.dropout_b4)(x)

    # --- Head ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout_head)(x)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="probs")(x)

    model = keras.Model(inputs, outputs, name="custom_cnn_6class_2nd")
    return model
