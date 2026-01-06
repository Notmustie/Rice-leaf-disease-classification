"""
Download a dataset from Kaggle using the Kaggle API.

Example:
python src/data/download_kaggle.py \
  --dataset "OWNER/DATASET" \
  --out_dir "data/raw/kaggle" \
  --unzip
"""

import argparse
import subprocess
from pathlib import Path
import sys


def check_kaggle_installed():
    try:
        subprocess.run(
            ["kaggle", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        sys.exit(
            "Kaggle CLI not found. Install it with:\n"
            "pip install kaggle\n"
            "and ensure kaggle.json is configured."
        )


def download_dataset(dataset: str, out_dir: Path, unzip: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(out_dir),
    ]

    if unzip:
        cmd.append("--unzip")

    subprocess.run(cmd, check=True)
    print(f"Dataset downloaded to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset identifier, e.g. OWNER/DATASET",
    )
    parser.add_argument(
        "--out_dir",
        default="data/raw/kaggle",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip dataset after download",
    )

    args = parser.parse_args()

    check_kaggle_installed()
    download_dataset(args.dataset, Path(args.out_dir), args.unzip)


if __name__ == "__main__":
    main()
