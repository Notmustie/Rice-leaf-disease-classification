import numpy as np
from typing import Tuple

def focus_and_entropy(heatmap: np.ndarray, thr: float = 0.60) -> Tuple[float, float]:
    """
    focus_score = fraction of pixels >= thr
    entropy = -sum(p log p) over normalized heatmap intensities
    """
    h = heatmap.astype(np.float64)
    focus = float((h >= thr).mean())

    s = h.sum()
    if s <= 1e-12:
        return focus, 0.0

    p = (h / s).flatten()
    p = np.clip(p, 1e-12, 1.0)
    ent = float(-(p * np.log(p)).sum())
    return focus, ent
