# src/data/validate.py
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image, UnidentifiedImageError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def verify_images(paths: List[Path]) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()  # checks file integrity
        except (UnidentifiedImageError, OSError, ValueError) as e:
            failures.append((str(p), repr(e)))
        except Exception as e:
            failures.append((str(p), repr(e)))
    return failures


def label_from_path(p: Path) -> str:
    # expects .../<split>/<label>/<file> or .../<label>/<file>
    return p.parent.name


def sha1_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate image dataset and/or split folders.")
    ap.add_argument("--root-dir", required=True, help="Directory to validate (e.g., data/processed).")
    ap.add_argument("--expect-splits", action="store_true",
                    help="If set, expects train/val/test folders under root-dir and checks overlap.")
    ap.add_argument("--report-dir", required=True, help="Write validation reports here.")
    ap.add_argument("--strong-overlap", action="store_true",
                    help="Use SHA1 hashing for overlap checks (slower but more reliable).")
    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.expect_splits:
        split_names = ["train", "val", "test"]
        split_paths = {s: root / s for s in split_names}
        for s, p in split_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing split folder: {p}")

        split_files: Dict[str, List[Path]] = {s: iter_images(p) for s, p in split_paths.items()}

        # Overlap check
        # Fast heuristic: (basename, size)
        # Strong check: SHA1 of bytes
        def key_fast(p: Path) -> Tuple[str, int]:
            try:
                return (p.name, p.stat().st_size)
            except Exception:
                return (p.name, -1)

        def key_strong(p: Path) -> str:
            try:
                return sha1_of_file(p)
            except Exception:
                return f"ERR:{p.name}"

        if args.strong_overlap:
            keysets: Dict[str, Set[str]] = {s: set(key_strong(p) for p in paths) for s, paths in split_files.items()}
        else:
            keysets = {s: set(key_fast(p) for p in paths) for s, paths in split_files.items()}

        overlaps = []
        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                inter = keysets[a].intersection(keysets[b])
                if inter:
                    overlaps.append((a, b, len(inter)))

        # Class balance
        balances = {}
        for s, paths in split_files.items():
            labels = [label_from_path(p) for p in paths]
            balances[s] = Counter(labels)

        # Corruption check
        failures = []
        for s, paths in split_files.items():
            fails = verify_images(paths)
            failures.extend([(s, fp, err) for fp, err in fails])

        # Write reports
        balance_csv = report_dir / "split_class_balance.csv"
        with balance_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "label", "count"])
            for s in split_names:
                for label, cnt in balances[s].most_common():
                    w.writerow([s, label, cnt])

        overlap_csv = report_dir / "split_overlaps.csv"
        with overlap_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split_a", "split_b", "overlap_count", "method"])
            method = "sha1" if args.strong_overlap else "name+size"
            for a, b, n in overlaps:
                w.writerow([a, b, n, method])

        corrupt_csv = report_dir / "corrupt_images.csv"
        with corrupt_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "path", "error"])
            for s, p, e in failures:
                w.writerow([s, p, e])

        # Console summary
        print("✅ Validation complete (splits mode).")
        for s in split_names:
            print(f" - {s}: {len(split_files[s])} images")

        if overlaps:
            print("⚠️ Overlaps detected (see report):", overlap_csv)
        else:
            print("✅ No split overlaps detected.")

        if failures:
            print("⚠️ Corrupt images detected (see report):", corrupt_csv)
        else:
            print("✅ No corrupt images detected.")

        print(f"🧾 Reports in: {report_dir}")

    else:
        paths = iter_images(root)
        failures = verify_images(paths)
        labels = [label_from_path(p) for p in paths]
        balance = Counter(labels)

        balance_csv = report_dir / "class_balance.csv"
        with balance_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "count"])
            for label, cnt in balance.most_common():
                w.writerow([label, cnt])

        corrupt_csv = report_dir / "corrupt_images.csv"
        with corrupt_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "error"])
            for p, e in failures:
                w.writerow([p, e])

        print("✅ Validation complete (single-folder mode).")
        print(f" - Images found: {len(paths)}")
        print(f" - Corrupt/unreadable: {len(failures)}")
        print(f"🧾 Reports in: {report_dir}")
        if failures:
            print("⚠️ Corrupt images detected (see report):", corrupt_csv)


if __name__ == "__main__":
    main()
