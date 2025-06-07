#!/usr/bin/env python3
"""Download the IRMAS dataset (≈2 GB) and extract it.

Example:
    python data/download_irmas.py --out_dir data/raw
"""
from __future__ import annotations

import argparse, hashlib, urllib.request, sys, pathlib

IRMAS_URL = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1"
MD5 = "4fd9f5ed5a18d8e2687e6360b5f60afe"  # expected archive checksum


def md5(fname, chunk=2 ** 20):
    m = hashlib.md5()
    with open(fname, 'rb') as fh:
        while True:
            data = fh.read(chunk)
            if not data: break
            m.update(data)
    return m.hexdigest()


def main(out_dir: str):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "IRMAS.zip"

    if not archive_path.exists():
        print("Downloading IRMAS ...")
        urllib.request.urlretrieve(IRMAS_URL, archive_path)
    else:
        print("Archive already exists, skipping download")

    print("Verifying checksum ...")
    if md5(archive_path) != MD5:
        print("Checksum mismatch!", file=sys.stderr)
        sys.exit(1)

    print("Extracting ...")
    import zipfile
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(out_dir)
    print("Done. Data at", out_dir)


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