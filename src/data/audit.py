#!/usr/bin/env python3
"""
Dataset audit: counts, sanity checks, and cross-split leakage detection.

Usage examples:
  python -m src.data.audit --data-root /path/to/dataset --out-dir reports/tables
  python -m src.data.audit --data-root data/processed --splits train valid test

Assumptions:
  data-root contains split folders (train/valid/test), each containing class subfolders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: Path, splits: List[str]) -> Iterable[Tuple[str, str, Path]]:
    """Yield (split, class_name, image_path)."""
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            cls = cls_dir.name
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    yield split, cls, p


def file_sha1(path: Path, max_bytes: int = 256 * 1024) -> str:
    """
    Fast-ish hash: sha1 of first max_bytes.
    Good enough to catch identical files across splits without reading huge files fully.
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="Root folder containing split dirs (train/valid/test).")
    ap.add_argument("--out-dir", type=str, default="reports/tables", help="Where to write audit CSV outputs.")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Split folder names.")
    ap.add_argument("--hash", action="store_true", help="Compute fast SHA1 to detect identical files across splits.")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    splits = args.splits

    if not root.exists():
        raise SystemExit(f"[audit] data-root not found: {root}")

    # Count images per split/class
    counts: Dict[Tuple[str, str], int] = {}
    all_items: List[Tuple[str, str, Path]] = list(iter_images(root, splits))

    for split, cls, _ in all_items:
        counts[(split, cls)] = counts.get((split, cls), 0) + 1

    # Prepare split summary table
    split_summary_rows: List[List[str]] = []
    for (split, cls), n in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
        split_summary_rows.append([split, cls, str(n)])

    write_csv(
        out_dir / "split_summary.csv",
        header=["split", "class", "count"],
        rows=split_summary_rows,
    )

    # Total counts per split
    totals: Dict[str, int] = {}
    for (split, _cls), n in counts.items():
        totals[split] = totals.get(split, 0) + n

    totals_rows = [[s, str(totals.get(s, 0))] for s in splits]
    write_csv(out_dir / "split_totals.csv", header=["split", "total_images"], rows=totals_rows)

    # Basic warnings: empty split/class folders
    warnings: List[List[str]] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            warnings.append(["missing_split_dir", split, str(split_dir)])
            continue
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            warnings.append(["no_class_dirs", split, str(split_dir)])
        for cls_dir in class_dirs:
            # count images (fast)
            n = sum(1 for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if n == 0:
                warnings.append(["empty_class_dir", f"{split}/{cls_dir.name}", str(cls_dir)])

    # Cross-split leakage detection
    # Method A: filename collision across splits (cheap, catches many leaks)
    by_name: Dict[str, List[Tuple[str, str, str]]] = {}
    for split, cls, p in all_items:
        key = p.name.lower()
        by_name.setdefault(key, []).append((split, cls, str(p)))

    name_collisions: List[List[str]] = []
    for fname, items in by_name.items():
        split_set = sorted(set(i[0] for i in items))
        if len(split_set) >= 2:
            # Same filename appears in multiple splits -> suspicious
            for split, cls, fullpath in items:
                name_collisions.append([fname, ",".join(split_set), split, cls, fullpath])

    write_csv(
        out_dir / "cross_split_filename_collisions.csv",
        header=["filename", "splits_present", "split", "class", "path"],
        rows=name_collisions,
    )

    if name_collisions:
        warnings.append(["cross_split_filename_collision", str(len(name_collisions)), str(out_dir / "cross_split_filename_collisions.csv")])

    # Method B: fast hash collisions across splits (stronger, optional)
    if args.hash:
        by_hash: Dict[str, List[Tuple[str, str, str]]] = {}
        for split, cls, p in all_items:
            try:
                h = file_sha1(p)
            except Exception as e:
                warnings.append(["hash_failed", f"{split}/{cls}", f"{p} :: {e}"])
                continue
            by_hash.setdefault(h, []).append((split, cls, str(p)))

        hash_collisions: List[List[str]] = []
        for h, items in by_hash.items():
            split_set = sorted(set(i[0] for i in items))
            if len(split_set) >= 2:
                for split, cls, fullpath in items:
                    hash_collisions.append([h, ",".join(split_set), split, cls, fullpath])

        write_csv(
            out_dir / "cross_split_hash_collisions.csv",
            header=["sha1_first256kb", "splits_present", "split", "class", "path"],
            rows=hash_collisions,
        )

        if hash_collisions:
            warnings.append(["cross_split_hash_collision", str(len(hash_collisions)), str(out_dir / "cross_split_hash_collisions.csv")])

    write_csv(out_dir / "audit_warnings.csv", header=["type", "where", "detail"], rows=warnings)

    print(f"[audit] wrote:")
    print(f"  - {out_dir / 'split_summary.csv'}")
    print(f"  - {out_dir / 'split_totals.csv'}")
    print(f"  - {out_dir / 'cross_split_filename_collisions.csv'}")
    if args.hash:
        print(f"  - {out_dir / 'cross_split_hash_collisions.csv'}")
    print(f"  - {out_dir / 'audit_warnings.csv'}")
    print(f"[audit] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
