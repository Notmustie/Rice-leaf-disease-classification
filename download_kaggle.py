#!/usr/bin/env python3
"""
Download and unzip a Kaggle dataset into the repo data/ directory.

Requirements:
- Kaggle account
- kaggle.json API token configured (see README section below)
- pip install kaggle

Example:
  python src/data/download_kaggle.py \
    --dataset "vipoooool/new-plant-diseases-dataset" \
    --out_dir "data/raw/kaggle" \
    --unzip
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a Kaggle dataset into data/raw/")
    ap.add_argument("--dataset", required=True, help='Kaggle dataset slug e.g. "owner/dataset-name"')
    ap.add_argument("--out_dir", default="data/raw/kaggle", help="Destination root directory")
    ap.add_argument("--unzip", action="store_true", help="Unzip after download")
    args = ap.parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Dataset slug -> folder name
    slug = args.dataset.split("/")[-1].strip()
    dst = out_root / slug
    dst.mkdir(parents=True, exist_ok=True)

    # Download zip(s)
    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(dst)]
    run(cmd)

    # Optionally unzip everything
    if args.unzip:
        # unzip all zips in dst
        zips = list(dst.glob("*.zip"))
        if not zips:
            print(f"Downloaded to {dst} (no zip found to unzip)")
        else:
            for z in zips:
                run(["unzip", "-o", str(z), "-d", str(dst)])
            print(f"Downloaded and unzipped to: {dst}")
    else:
        print(f"Downloaded to: {dst}")
        print("Tip: rerun with --unzip to extract archives.")

    print("\nNext step:")
    print(f"  Use standardize.py with kaggle_root: {dst}")


if __name__ == "__main__":
    main()
