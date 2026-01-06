# src/data/standardize.py
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def discover_class_dirs(raw_dir: Path) -> List[Path]:
    # class-dir heuristic: directory that directly contains images
    class_dirs = []
    for d in raw_dir.rglob("*"):
        if d.is_dir():
            imgs = [p for p in d.iterdir() if is_image_file(p)]
            if imgs:
                class_dirs.append(d)
    # sort by depth (prefer higher-level)
    class_dirs.sort(key=lambda p: len(p.parts))
    return class_dirs


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = dst.with_name(f"{stem}_{i}{suf}")
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Standardize Kaggle dataset to data/processed/images/<class> layout.")
    ap.add_argument("--raw-dir", required=True, help="Raw dataset directory from Kaggle download")
    ap.add_argument("--out-dir", default="data/processed/images", help="Output standardized images directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = discover_class_dirs(raw_dir)
    if not class_dirs:
        raise RuntimeError(f"No class folders with images found under: {raw_dir}")

    total = 0
    for class_dir in class_dirs:
        cls = slugify(class_dir.name)
        for img in class_dir.iterdir():
            if not is_image_file(img):
                continue
            dst = out_dir / cls / img.name
            if args.dry_run:
                print(f"[DRY] {img} -> {dst}")
            else:
                safe_copy(img, dst)
            total += 1

    print(f"✅ Standardization complete. Copied {total} images into: {out_dir}")


if __name__ == "__main__":
    main()
