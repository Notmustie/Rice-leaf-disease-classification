# src/data/split.py
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def collect_by_class(images_dir: Path) -> Dict[str, List[Path]]:
    by_class: Dict[str, List[Path]] = {}
    for cls_dir in sorted([d for d in images_dir.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        files = sorted([p for p in cls_dir.rglob("*") if is_image_file(p)])
        if files:
            by_class[cls] = files
    return by_class


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_manifest(rows: List[Tuple[str, str, str]], out_csv: Path) -> None:
    # rows: (split, class, path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "path"])
        for r in rows:
            w.writerow(list(r))


def main():
    ap = argparse.ArgumentParser(description="Create stratified train/val/test splits from class-folders.")
    ap.add_argument("--images-dir", required=True, help="Input directory: images/<class>/*.jpg")
    ap.add_argument("--out-dir", required=True, help="Output directory for split folders + manifest.")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy",
                    help="copy (safe) or symlink (fast). symlink may not work on some systems.")
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    images_dir = Path(args.images_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    rng = random.Random(args.seed)

    by_class = collect_by_class(images_dir)
    if not by_class:
        raise RuntimeError(f"No class folders found in {images_dir}")

    splits = {"train": args.train, "val": args.val, "test": args.test}
    split_dirs = {k: out_dir / k for k in splits.keys()}
    for d in split_dirs.values():
        ensure_dir(d)

    manifest_rows: List[Tuple[str, str, str]] = []

    for cls, files in by_class.items():
        files = files[:]  # copy
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(n * args.train))
        n_val = int(round(n * args.val))
        # rest to test
        n_test = n - n_train - n_val
        if n_test < 0:
            # rounding safety
            n_test = max(0, n_test)
            n_val = n - n_train - n_test

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for src in split_files:
                rel_name = src.name
                dst = split_dirs[split_name] / cls / rel_name
                dst.parent.mkdir(parents=True, exist_ok=True)

                if args.copy_mode == "copy":
                    # avoid overwriting by adding suffix
                    if dst.exists():
                        stem, suf = dst.stem, dst.suffix
                        i = 1
                        while True:
                            candidate = dst.with_name(f"{stem}_{i}{suf}")
                            if not candidate.exists():
                                dst = candidate
                                break
                            i += 1
                    dst.write_bytes(src.read_bytes())
                else:
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(src)

                manifest_rows.append((split_name, cls, str(dst)))

        print(f"✅ {cls}: n={n} -> train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    print(f"🧾 Manifest: {manifest_path}")
    print(f"📁 Splits created at: {out_dir}")


if __name__ == "__main__":
    main()
