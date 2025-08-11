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

def md5(fname, chunk=2 ** 20):
    m = hashlib.md5()
    with open(fname, 'rb') as fh:
        while True:
            data = fh.read(chunk)
            if not data: break
            m.update(data)
    return m.hexdigest()

def _pick_training_root(p: Path) -> Path:
    """
    Return the IRMAS-TrainingData folder (never the base folder).
    """
    # caller already gave the exact folder
    if p.name == "IRMAS-TrainingData":
        return p

    # otherwise look *only* for that child
    cand = p / "IRMAS-TrainingData"
    if cand.exists() and any(cand.rglob("*.wav")):
        return cand

    raise FileNotFoundError(f"No training WAVs found under {p}")

def _pick_testing_root(p: Path) -> Path:
    """
    Return a path that *really* contains the testing WAV-and-TXT pairs.
    Accepts:
        • …/IRMAS-TestingData-Part1
        • …/IRMAS-TestingData      (single-folder variant)
        • …/IRMAS                  (base)  ← only if that has the testing folder
    """
    # case 1: caller already gave the right folder
    if any(p.glob("*.txt")) and any(p.glob("*.wav")):
        return p

    # case 2: base_root → jump into the main testing folder
    for cand in [
        p / "IRMAS-TestingData",
        p / "IRMAS-TestingData-Part1",
        p / "IRMAS-TestingData-Part2",
        p / "IRMAS-TestingData-Part3",
    ]:
        if cand.exists() and any(cand.rglob("*.txt")):
            return cand

    raise FileNotFoundError(f"Could not locate testing data under {p}")


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


def find_irmas_root() -> pathlib.Path | None:
    """Return the first existing path that contains IRMAS WAVs."""
    # Check user's home directory first
    home_path = pathlib.Path.home() / "datasets" / "irmas"

    candidates = [
        home_path / "IRMAS-TrainingData",  # User's home directory (extracted)
        home_path,  # User's home directory (fallback)
        pathlib.Path("/content/IRMAS/IRMAS-TrainingData"),  # Colab scratch
        pathlib.Path("data/raw/IRMAS/IRMAS-TrainingData"),  # Project directory (extracted)
        pathlib.Path("data/raw/IRMAS-TrainingData"),  # Project directory alt
        pathlib.Path("data/raw"),  # Project directory fallback
    ]

    for p in candidates:
        if p.exists() and any(p.rglob("*.wav")):
            print(f"Found IRMAS dataset at: {p}")
            return p

    # If no extracted data found, check if we have zip files and extract them
    zip_candidates = [
        home_path / "IRMAS.zip",
        pathlib.Path("data/raw/IRMAS.zip"),
        pathlib.Path("/content/drive/MyDrive/datasets/IRMAS/IRMAS.zip")
    ]

    for zip_path in zip_candidates:
        if zip_path.exists():
            print(f"Found IRMAS zip at {zip_path}, extracting...")
            import zipfile
            extract_dir = zip_path.parent
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Try to find the extracted data
            extracted_path = extract_dir / "IRMAS-TrainingData"
            if extracted_path.exists() and any(extracted_path.rglob("*.wav")):
                print(f"Successfully extracted to: {extracted_path}")
                return extracted_path

    print("No IRMAS dataset found in any of the expected locations")
    return None

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw", help="Destination directory")
    args = p.parse_args()
    main(args.out_dir)