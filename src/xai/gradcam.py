from __future__ import annotations
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple

def preprocess_image_array(x: np.ndarray, preprocess: str) -> np.ndarray:
    if preprocess == "rescale":
        return x / 255.0
    if preprocess == "efficientnetv2":
        return tf.keras.applications.efficientnet_v2.preprocess_input(x)
    raise ValueError(f"Unknown preprocess={preprocess}")

def gradcam_from_feature_layer(
    model: tf.keras.Model,
    x_in: np.ndarray,                     # shape (1,H,W,C)
    feature_layer_name: str,              # e.g. "conv2d_15" or "efficientnetv2-b0"
    class_index: int,
    out_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Stable Grad-CAM that taps a TOP-LEVEL feature layer output.
    For your EfficientNet saved model, use feature_layer_name="efficientnetv2-b0"
    (backbone output) to avoid nested-tensor KeyError bugs.
    """
    x = tf.convert_to_tensor(x_in, dtype=tf.float32)
    feat_layer = model.get_layer(feature_layer_name)

    # IMPORTANT: use model.input (not model.inputs list wrapping)
    grad_model = tf.keras.Model(inputs=model.input, outputs=[feat_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are None (graph disconnected).")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # (1,C)
    conv_out = conv_out[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maxv = tf.reduce_max(heatmap)
    heatmap = tf.cond(maxv > 0, lambda: heatmap / maxv, lambda: heatmap)

    heatmap = tf.image.resize(heatmap[..., tf.newaxis], out_size)
    heatmap = tf.squeeze(heatmap).numpy()
    return np.clip(heatmap, 0, 1)

def overlay_heatmap(pil_rgb_image, heatmap: np.ndarray, alpha: float = 0.45):
    import cv2
    from PIL import Image
    img = np.array(pil_rgb_image)  # RGB
    hm = np.uint8(255 * heatmap)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = (img * (1 - alpha) + hm_color * alpha).astype(np.uint8)
    return Image.fromarray(overlay)
