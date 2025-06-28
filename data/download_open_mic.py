#!/usr/bin/env python3
"""Download the OpenMIC‑2018 dataset and prepare it for training.

This mirrors the API of `data/download_irmas.py` so you can drop it into the
same pipeline.  Key points:

* **Only the raw tarball (≈ 55 MB) is cached on Drive** when running in Colab.
* Extraction happens each session to `/content/OpenMIC` (ephemeral)
  so Drive usage stays minimal.
* Provides two convenience loaders:
    • `load_openmic_split(split, cfg)` – returns clip‑level (mel) tensors + labels
    • `find_openmic_root()` – autodetects dataset location like the IRMAS helper.

Example in Colab
----------------
```python
!python data/download_openmic.py             # zip → Drive, extract → /content/OpenMIC
```

Example local usage
-------------------
```bash
python data/download_openmic.py --out_dir data/raw/OpenMIC
```
"""
from __future__ import annotations

import argparse, hashlib, os, pathlib, sys, tarfile, urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torchaudio

OPENMIC_URL = "https://zenodo.org/records/1432913/files/openmic-2018-v1.0.0.tgz?download=1"
TGZ_MD5 = "891ae5dfd9e67de4333846f5912e5dfe"  # reported by Zenodo
CLIP_SEC = 10
SR = 44100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inside_colab() -> bool:
    return "google.colab" in sys.modules


def ensure_drive_mounted():
    if not inside_colab():
        return
    from google.colab import drive  # type: ignore

    if not pathlib.Path("/content/drive").is_dir():
        print("Mounting Google Drive …")
        drive.mount("/content/drive")


def md5(path: pathlib.Path, chunk: int = 2 ** 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def download_tgz(tgz_path: pathlib.Path):
    print("⬇️  Downloading OpenMIC‑2018 …")
    urllib.request.urlretrieve(OPENMIC_URL, tgz_path)


# ---------------------------------------------------------------------------
# Main download / extract routine
# ---------------------------------------------------------------------------

def main(cache_dir: pathlib.Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = cache_dir / "openmic-2018.tgz"

    # 1) download once ------------------------------------------------------
    if tgz_path.exists():
        print("Tarball already cached – skip download")
    else:
        download_tgz(tgz_path)

    # 2) checksum -----------------------------------------------------------
    print("Verifying checksum …")
    if md5(tgz_path) != TGZ_MD5:
        print("❌  MD5 mismatch – delete the tgz and retry", file=sys.stderr)
        sys.exit(1)

    # 3) extract each runtime (Colab) or once (local) ----------------------
    extract_root = pathlib.Path("/content/OpenMIC") if inside_colab() else cache_dir / "openmic-2018"

    if extract_root.is_dir():
        print("Data already extracted – skipping untar")
        print("Dataset root:", extract_root)
        return extract_root

    print(f"Extracting to {extract_root} …")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(extract_root)
    print("✅  Extracted dataset to", extract_root)
    return extract_root


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
@lru_cache
def _labels_list(root: pathlib.Path):
    csv = root / "openmic-2018-train.csv"
    return list(pd.read_csv(csv, nrows=0).columns[1:])  # 20 labels


def load_openmic_split(split: str, cfg: dict, root: pathlib.Path | None = None):
    """Return a list of (log‑mel tensor, multi‑hot label) for *split*.
    split ∈ {"train", "test"}
    """
    if root is None:
        root = find_openmic_root()
    if root is None:
        raise FileNotFoundError("OpenMIC root not found; run download_openmic.py first")

    csv_path = root / f"openmic-2018-{split}.csv"
    audio_dir = root / "audio" / split

    df = pd.read_csv(csv_path)
    label_cols = _labels_list(root)

    dataset = []
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.get("sample_rate", SR),
                                               n_mels=cfg.get("n_mels", 64))

    for key, row in df.iterrows():
        wav_path = audio_dir / f"{row['sample_key']}.ogg"
        if not wav_path.exists():
            continue
        wav, _ = torchaudio.load(wav_path)
        feat = mel(wav.mean(0, keepdim=True)).log2()  # 1×64×T
        label = torch.tensor(row[label_cols].values.astype(np.float32))
        dataset.append((feat.squeeze(0), label))

    print(f"✅  Loaded {len(dataset)} {split} clips from OpenMIC")
    return dataset


def find_openmic_root() -> pathlib.Path | None:
    """Try typical locations for the extracted dataset."""
    candidates = [
        pathlib.Path("/content/OpenMIC"),
        pathlib.Path("/content/drive/MyDrive/datasets/OpenMIC"),
        pathlib.Path("data/raw/OpenMIC"),
    ]
    for p in candidates:
        if (p / "openmic-2018-train.csv").exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    default_dir = (
        "/content/drive/MyDrive/datasets/OpenMIC" if inside_colab() else "data/raw/OpenMIC"
    )

    parser = argparse.ArgumentParser(
        description="Download and cache the OpenMIC‑2018 dataset (tarball on Drive," " extract in /content).")
    parser.add_argument("--out_dir", default=default_dir,
                        help="Cache directory for openmic-2018.tgz (Drive in Colab)")

    args, _ = parser.parse_known_args()

    ensure_drive_mounted()
    main(pathlib.Path(args.out_dir))
