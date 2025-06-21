#!/usr/bin/env python3
"""Download the IRMAS dataset (≈2 GB) and extract it.

Example:
    python data/download_irmas.py --out_dir data/raw
"""
from __future__ import annotations

import argparse, hashlib, urllib.request, sys, pathlib
import random
import zipfile

import librosa
import torch
from pathlib import Path

from var import LABELS, IRMAS_TO_LABEL_MAP

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


import pathlib
import torch
import random
import librosa

from var import LABELS


def load_irmas_audio_dataset(irmas_root, cfg, max_samples=None):
    """
    Load training (single-label) audio from IRMAS dataset.
    """
    irmas_path = _pick_training_root(Path(irmas_root))
    label_map = {label: i for i, label in enumerate(LABELS)}
    dataset = []

    # Limit sample count
    if max_samples is None:
        max_samples = cfg.get('max_original_samples', 100)
    elif isinstance(max_samples, str) and max_samples.lower() == 'none':
        max_samples = None

    wav_files = list(irmas_path.rglob("*.wav"))
    if max_samples and len(wav_files) > max_samples:
        wav_files = random.sample(wav_files, max_samples)

    print(f"Loading {len(wav_files)} training audio files...")

    for wav_file in wav_files:
        try:
            irmas_label = wav_file.parent.name.lower()
            our_label = IRMAS_TO_LABEL_MAP.get(irmas_label, irmas_label)

            if our_label not in label_map:
                continue

            y, _ = librosa.load(wav_file, sr=cfg['sample_rate'], mono=True)
            label_vec = torch.zeros(len(LABELS), dtype=torch.long)
            label_vec[label_map[our_label]] = 1
            dataset.append((torch.tensor(y, dtype=torch.float32), label_vec))

        except Exception as e:
            print(f"Error loading {wav_file}: {e}")

    print(f"✅ Loaded {len(dataset)} training samples.")
    return dataset


def load_irmas_testing_dataset(test_dir, cfg):
    """
    Load testing (multi-label) IRMAS dataset.
    Each audio file has a corresponding .txt file listing instruments.
    """
    test_path = _pick_testing_root(Path(test_dir))
    label_map = {label: i for i, label in enumerate(LABELS)}
    dataset = []

    wav_files = list(test_path.rglob("*.wav"))

    cap = cfg.get("max_test_samples")
    cap = None if cap in (None, "None") else int(cap)
    if cap is not None and len(wav_files) > cap:
        random.shuffle(wav_files)
        wav_files = wav_files[:cap]
        print(f"⚖️  Capped to {cap} test WAVs for quick run")

    print(f"Loading {len(wav_files)} testing audio files…")

    for wav_file in wav_files:
        try:
            txt_path = wav_file.with_suffix('.txt')
            if not txt_path.exists():
                print(f"⚠️  Missing label file for {wav_file.name}")
                continue

            with open(txt_path, 'r') as f:
                label_lines = f.readlines()

            label_vec = torch.zeros(len(LABELS), dtype=torch.long)
            for line in label_lines:
                raw_label = line.strip().lower()
                if raw_label in IRMAS_TO_LABEL_MAP:
                    mapped = IRMAS_TO_LABEL_MAP[raw_label]
                else:
                    mapped = raw_label

                if mapped in label_map:
                    label_vec[label_map[mapped]] = 1

            y, _ = librosa.load(wav_file, sr=cfg['sample_rate'], mono=True)
            dataset.append((torch.tensor(y, dtype=torch.float32), label_vec))

        except Exception as e:
            print(f"Error loading {wav_file}: {e}")

    print(f"✅ Loaded {len(dataset)} testing samples.")
    return dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw", help="Destination directory")
    args = p.parse_args()
    main(args.out_dir)