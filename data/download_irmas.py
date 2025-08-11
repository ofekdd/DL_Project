#!/usr/bin/env python3
"""Download the IRMAS dataset (≈2 GB) and extract it.

Example:
    python data/download_irmas.py --out_dir data/raw
"""
from __future__ import annotations

import argparse, hashlib, urllib.request, pathlib as Path
import zipfile
import pathlib

IRMAS_URL = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1"
IRMAS_TESTING_PART1_URL = "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1"
IRMAS_TESTING_PART2_URL = "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1"
IRMAS_TESTING_PART3_URL = "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1"
MD5 = "4fd9f5ed5a18d8e2687e6360b5f60afe"  # expected archive checksum

def download_and_extract_zip(url: str, out_path: pathlib.Path):
    zip_path = out_path / url.split("/")[-1].split("?")[0]
    if not zip_path.exists():
        print(f"⬇️  Downloading {zip_path.name} ...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"✅ {zip_path.name} already exists, skipping download.")

    print(f"📦 Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)
    print(f"✅ Extracted to {out_path}")

def main(out_dir: str):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Downloading IRMAS dataset (training + testing)...")

    # Download and extract training set
    download_and_extract_zip(IRMAS_URL, out_dir)

    # Download and extract testing parts
    download_and_extract_zip(IRMAS_TESTING_PART1_URL, out_dir)
    download_and_extract_zip(IRMAS_TESTING_PART2_URL, out_dir)
    download_and_extract_zip(IRMAS_TESTING_PART3_URL, out_dir)

    print(f"\n✅ All data is ready under {out_dir.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw", help="Destination directory")
    args = p.parse_args()
    main(args.out_dir)