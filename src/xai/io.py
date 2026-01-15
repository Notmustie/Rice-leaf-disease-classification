import os
from typing import List, Dict, Optional, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def collect_one_level(predict_root: str, max_per_class: Optional[int] = None) -> List[Dict[str, str]]:
    """
    predict_root/<class_name>/*.jpg
    Returns: [{"path": ..., "true_label": ...}, ...]
    """
    items: List[Dict[str, str]] = []
    for cls in sorted(os.listdir(predict_root)):
        cls_dir = os.path.join(predict_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(IMG_EXTS)]
        if max_per_class is not None:
            files = files[:max_per_class]

        for f in files:
            items.append({"path": os.path.join(cls_dir, f), "true_label": cls})
    return items
