# src/data/deduplicate.py
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def average_hash(img: Image.Image, hash_size: int = 8) -> int:
    """
    aHash: resize -> grayscale -> compare pixels to average.
    Returns a 64-bit integer (for hash_size=8).
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32)
    avg = pixels.mean()
    bits = pixels > avg
    # pack bits into int
    h = 0
    for b in bits.flatten():
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class DupRecord:
    kept: str
    removed: str
    kept_hash: str
    removed_hash: str
    ham: int
    reason: str


def iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def main():
    ap = argparse.ArgumentParser(description="Deduplicate images using aHash + Hamming distance.")
    ap.add_argument("--images-dir", required=True, help="Directory containing class folders of images.")
    ap.add_argument("--report-dir", required=True, help="Where to write dedup reports (CSV).")
    ap.add_argument("--hash-size", type=int, default=8, help="aHash size (8 -> 64-bit hash).")
    ap.add_argument("--max-hamming", type=int, default=0, help="0 = exact-ish duplicates; 1-5 = near duplicates.")
    ap.add_argument("--mode", choices=["move", "delete"], default="move",
                    help="What to do with duplicates. move -> quarantine folder; delete -> remove files.")
    ap.add_argument("--quarantine-dir", default=None, help="Where to move duplicates (default: <report-dir>/quarantine).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    quarantine_dir = Path(args.quarantine_dir).expanduser().resolve() if args.quarantine_dir else (report_dir / "quarantine")
    if args.mode == "move":
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_images(images_dir)
    if not paths:
        print(f"⚠️ No images found in {images_dir}")
        return

    # Hash all images
    hashes: Dict[Path, int] = {}
    failed: List[Tuple[str, str]] = []

    for p in paths:
        try:
            with Image.open(p) as im:
                hashes[p] = average_hash(im, hash_size=args.hash_size)
        except Exception as e:
            failed.append((str(p), repr(e)))

    # Group duplicates by comparing hashes (O(n^2) worst-case).
    # For typical Kaggle sizes this is OK; for huge datasets you'd bucket by hash prefixes.
    kept: List[Path] = []
    removed: List[DupRecord] = []

    for p in paths:
        if p not in hashes:
            continue  # failed to hash
        hp = hashes[p]
        duplicate_of = None
        duplicate_ham = None

        for k in kept:
            hk = hashes[k]
            ham = hamming(hp, hk)
            if ham <= args.max_hamming:
                duplicate_of = k
                duplicate_ham = ham
                break

        if duplicate_of is None:
            kept.append(p)
        else:
            rec = DupRecord(
                kept=str(duplicate_of),
                removed=str(p),
                kept_hash=hex(hashes[duplicate_of]),
                removed_hash=hex(hp),
                ham=int(duplicate_ham),
                reason=f"aHash_hamming<= {args.max_hamming}",
            )
            removed.append(rec)

            if args.dry_run:
                print(f"[DRY] DUP: {p} -> (kept {duplicate_of}) ham={duplicate_ham}")
                continue

            if args.mode == "delete":
                try:
                    p.unlink()
                except Exception as e:
                    failed.append((str(p), f"delete_failed: {repr(e)}"))
            else:
                # move to quarantine while preserving relative path
                rel = p.relative_to(images_dir)
                dst = quarantine_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    p.replace(dst)
                except Exception as e:
                    failed.append((str(p), f"move_failed: {repr(e)}"))

    # Write reports
    dup_csv = report_dir / "duplicates.csv"
    fail_csv = report_dir / "dedup_failures.csv"

    with dup_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kept", "removed", "kept_hash", "removed_hash", "hamming", "reason"])
        for r in removed:
            w.writerow([r.kept, r.removed, r.kept_hash, r.removed_hash, r.ham, r.reason])

    with fail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "error"])
        for p, e in failed:
            w.writerow([p, e])

    print(f"✅ Dedup done. Kept: {len(kept)} | Removed: {len(removed)} | Failures: {len(failed)}")
    print(f"🧾 Report: {dup_csv}")
    if failed:
        print(f"⚠️ Failures report: {fail_csv}")
    if args.mode == "move":
        print(f"📦 Quarantine: {quarantine_dir}")


if __name__ == "__main__":
    main()
