# /data/preprocess.py
#!/usr/bin/env python3
"""Prepare IRMAS into train/val/test folders storing only waveforms as .npy."""

import argparse
import pathlib
import random
import itertools
import yaml
import numpy as np
import librosa
import tqdm


def load_waveform(path, target_sr, mono=True):
    y, sr = librosa.load(path, sr=target_sr, mono=mono)
    return y, sr


def standardize_waveform(
    y: np.ndarray,
    sr: int,
    max_duration_sec: float | None = None,
    peak_normalize: bool = True,
    eps: float = 1e-8
):
    """Optionally trim/pad to max_duration_sec and peak-normalize."""
    if max_duration_sec is not None and max_duration_sec > 0:
        max_len = int(round(sr * max_duration_sec))
        if y.shape[0] > max_len:
            y = y[:max_len]
        elif y.shape[0] < max_len:
            y = np.pad(y, (0, max_len - y.shape[0]))

    if peak_normalize:
        peak = np.max(np.abs(y)) + eps
        y = y / peak

    return y.astype(np.float32)


def save_waveform(sample_dir: pathlib.Path, y: np.ndarray):
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Single file named waveform.npy
    np.save(sample_dir / "waveform.npy", y)


def preprocess_data(irmas_root, out_dir, cfg, original_data_percentage=1.0):
    """
    Build processed folders with only waveforms saved as waveform.npy
    under per-sample directories (train/val/test).
    """
    irmas_root = pathlib.Path(irmas_root)
    out_dir = pathlib.Path(out_dir)
    train_dir, val_dir, test_dir = (out_dir / s for s in ("train", "val", "test"))
    for p in (train_dir, val_dir, test_dir):
        p.mkdir(parents=True, exist_ok=True)

    target_sr = int(cfg.get("sample_rate", 22050))
    max_duration_sec = cfg.get("max_duration_sec", None)       # e.g. 6.0
    peak_normalize = bool(cfg.get("peak_normalize", True))

    # -----------------------------
    # 1) TRAIN / VAL
    # -----------------------------
    train_src = irmas_root / "IRMAS-TrainingData"
    if not train_src.exists():
        raise FileNotFoundError(f"Expected training data at {train_src}")

    wav_files = list(train_src.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wavs found under {train_src}")

    print(f"🎻 Found {len(wav_files)} training wavs")
    if 0 < original_data_percentage < 1.0:
        k = int(len(wav_files) * original_data_percentage)
        wav_files = random.sample(wav_files, k)
        print(f"⚖️ Sub-sampling: {k} files ({original_data_percentage*100:.1f}%)")

    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * 0.8)
    tr_files, val_files = wav_files[:split_idx], wav_files[split_idx:]
    print(f"📊 Train/val split → {len(tr_files)} train | {len(val_files)} val")

    def _process_split(files, split_dir, prefix):
        for i, wav in enumerate(tqdm.tqdm(files, desc=f"→ {split_dir.name} ({len(files)})")):
            try:
                y, _ = load_waveform(wav, target_sr, mono=True)
                y = standardize_waveform(
                    y, target_sr,
                    max_duration_sec=max_duration_sec,
                    peak_normalize=peak_normalize
                )
                irmas_label = wav.parent.name  # single-label folders
                base = wav.stem
                sample_dir = split_dir / f"{prefix}_{irmas_label}_{i:04d}_{base}"
                save_waveform(sample_dir, y)
            except Exception as e:
                print(f"❌ {wav}: {e}")

    _process_split(tr_files, train_dir, "original")
    _process_split(val_files, val_dir, "original")

    # -----------------------------
    # 2) TEST (all parts)
    # -----------------------------
    test_roots = [p for p in irmas_root.glob("IRMAS-TestingData-Part*") if p.is_dir()]
    if not test_roots:
        sf = irmas_root / "IRMAS-TestingData"
        if sf.exists():
            test_roots = [sf]

    test_wavs = list(itertools.chain.from_iterable(tr.rglob("*.wav") for tr in test_roots))
    cap = cfg.get("max_test_samples")
    if cap is not None and len(test_wavs) > cap:
        random.shuffle(test_wavs)
        test_wavs = test_wavs[:cap]
        print(f"⚖️  Capping test set to {cap} samples (quick-run mode)")

    for i, wav in enumerate(tqdm.tqdm(test_wavs, desc="→ test")):
        try:
            y, _ = load_waveform(wav, target_sr, mono=True)
            y = standardize_waveform(
                y, target_sr,
                max_duration_sec=max_duration_sec,
                peak_normalize=peak_normalize
            )

            # gather multi-labels from sidecar .txt, if present
            txt = wav.with_suffix(".txt")
            if txt.exists():
                with open(txt) as fh:
                    lbls = [ln.strip() for ln in fh if ln.strip()]
                lbl_tag = "+".join(lbls) if lbls else "unknown"
            else:
                lbl_tag = "unknown"

            base = wav.stem
            sample_dir = test_dir / f"irmasTest_{lbl_tag}_{i:04d}_{base}"
            save_waveform(sample_dir, y)
        except Exception as e:
            print(f"❌ {wav}: {e}")

    # -----------------------------
    # 3) summary
    # -----------------------------
    def _count(split_dir, tag):
        return len([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith(tag)])

    print("\n📋 Final counts")
    for split, p in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if p.exists():
            print(f"   {split:<5}: {len(list(p.iterdir())):>5} samples"
                  f"  (original={_count(p,'original')}, irmasTest={_count(p,'irmasTest')})")
    print(f"\n✅ Preprocessing finished. Waveforms saved to {out_dir}")


def main(in_dir, out_dir, config):
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(config, "r"))

    wav_files = list(in_dir.rglob("*.wav"))
    for wav in tqdm.tqdm(wav_files, desc="→ exporting"):
        y, sr = load_waveform(wav, target_sr=cfg.get("sample_rate", 22050), mono=True)
        y = standardize_waveform(
            y, sr,
            max_duration_sec=cfg.get("max_duration_sec", None),
            peak_normalize=bool(cfg.get("peak_normalize", True))
        )
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = out_dir / rel_dir
        save_waveform(file_out_dir, y)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=False, help="Optional direct export mode")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()

    if args.in_dir:
        main(args.in_dir, args.out_dir, args.config)
    else:
        # typical IRMAS preprocessing path (expects IRMAS-TrainingData / IRMAS-TestingData*)
        cfg = yaml.safe_load(open(args.config, "r"))
        irmas_root = cfg.get("irmas_root", None) or "."
        original_data_percentage = cfg.get("original_data_percentage", 1.0)
        preprocess_data(
            irmas_root=irmas_root,
            out_dir=args.out_dir,
            cfg=cfg,
            original_data_percentage=original_data_percentage
        )
