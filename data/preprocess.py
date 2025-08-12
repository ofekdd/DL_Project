# !/usr/bin/env python3
"""Convert wav clips to multi-band STFT spectrograms saved as .npy.

Usage:
    python data/preprocess.py --in_dir data/raw/IRMAS --out_dir data/processed
"""
import argparse, librosa, numpy as np, pathlib, tqdm, yaml, random
from var import n_ffts, band_ranges_as_tuples


# Audio-level augmentation functions
def time_stretch(y, rate_range=(0.8, 1.2)):
    """Speed up or slow down audio"""
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y, sr, n_steps_range=(-3, 3)):
    """Shift pitch up or down"""
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y, noise_factor=0.02):
    """Add Gaussian noise"""
    noise = np.random.normal(0, noise_factor, y.shape)
    return y + noise


def volume_change(y, gain_range=(0.5, 1.5)):
    """Change volume/amplitude"""
    gain = np.random.uniform(*gain_range)
    return y * gain


# Spectrogram-level augmentation functions
def freq_mask(spec, freq_mask_param=15, num_masks=2):
    """Mask random frequency bands"""
    spec = spec.copy()
    freq_dim = spec.shape[0]

    for _ in range(num_masks):
        f = np.random.uniform(0, freq_mask_param)
        f = int(f)
        f0 = np.random.uniform(0, freq_dim - f)
        f0 = int(f0)
        spec[f0:f0+f, :] = 0
    return spec


def time_mask(spec, time_mask_param=25, num_masks=2):
    """Mask random time segments"""
    spec = spec.copy()
    time_dim = spec.shape[1]

    for _ in range(num_masks):
        t = np.random.uniform(0, time_mask_param)
        t = int(t)
        t0 = np.random.uniform(0, time_dim - t)
        t0 = int(t0)
        spec[:, t0:t0+t] = 0
    return spec


def augment_audio(y, sr, augmentation_prob=0.5):
    """Apply random audio augmentations"""
    if np.random.random() < augmentation_prob:
        # Time stretch
        if np.random.random() < 0.3:
            y = time_stretch(y, rate_range=(0.9, 1.1))

        # Pitch shift
        if np.random.random() < 0.3:
            y = pitch_shift(y, sr, n_steps_range=(-2, 2))

        # Add noise
        if np.random.random() < 0.4:
            y = add_noise(y, noise_factor=0.01)

        # Volume change
        if np.random.random() < 0.3:
            y = volume_change(y, gain_range=(0.8, 1.2))

    return y


def augment_spectrogram(spec, augmentation_prob=0.5):
    """Apply SpecAugment to spectrogram"""
    if np.random.random() < augmentation_prob:
        # Frequency masking
        if np.random.random() < 0.5:
            spec = freq_mask(spec, freq_mask_param=8, num_masks=1)

        # Time masking
        if np.random.random() < 0.5:
            spec = time_mask(spec, time_mask_param=15, num_masks=1)

    return spec


def generate_multi_stft(
        y: np.ndarray,
        sr: int,
        apply_augmentation=False
):
    """
    Generates 3 spectrograms: 3 window sizes × 3 frequency bands.
    Optionally applies augmentation during training.

    Parameters:
        y (np.ndarray): Audio waveform
        sr (int): Sampling rate
        apply_augmentation (bool): Whether to apply audio and spectrogram augmentations

    Returns:
        dict: { (band_label, n_fft): spectrogram }
    """
    # Apply audio-level augmentation during training
    if apply_augmentation:
        y = augment_audio(y, sr)

    result = {}

    # Define the 3 optimal STFT settings
    optimized_stfts = [
        ((0, 1000), 1024),  # low freq + long window
        ((1000, 4000), 512),  # mid freq + medium window
        ((4000, 11025), 256),  # high freq + short window
    ]

    for (f_low, f_high), n_fft in optimized_stfts:
        hop_length = n_fft // 4
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        band_mask = (freqs >= f_low) & (freqs < f_high)
        band_spec = mag[band_mask, :]
        log_spec = librosa.power_to_db(band_spec, ref=np.max).astype(np.float32)

        # Apply spectrogram-level augmentation during training
        if apply_augmentation:
            log_spec = augment_spectrogram(log_spec)

        band_label = f"{f_low}-{f_high}Hz"
        result[(band_label, n_fft)] = log_spec

    return result


    def process_file(wav_path, cfg, apply_augmentation=False):
    y, sr = librosa.load(wav_path, sr=cfg['sample_rate'], mono=True)
    return generate_multi_stft(y, sr, apply_augmentation=apply_augmentation)


def preprocess_data(irmas_root, out_dir, cfg, original_data_percentage=1.0):
    """
    Preprocess IRMAS training **and** testing data.

    Args:
        irmas_root (str | Path): Path pointing at the base directory that
                                 contains IRMAS-TrainingData *and* the
                                 IRMAS-TestingData* folders / zips you extracted.
        out_dir (str | Path):  Destination directory for spectrogram features.
        cfg (dict):            Config dict (expects 'sample_rate' key).
        original_data_percentage (float): Fraction of training data to keep.
    """
    # --------------------------------------------------------------------- #
    # Imports (kept local so the function is self-contained)
    # --------------------------------------------------------------------- #
    import os, random, itertools
    from pathlib import Path
    from tqdm import tqdm
    import librosa, numpy as np

    # --------------------------------------------------------------------- #
    # Helper
    # --------------------------------------------------------------------- #
    def generate_specs(y, sr, apply_augmentation=False):
        """Wrapper around your generate_multi_stft() helper."""
        return generate_multi_stft(y, sr, apply_augmentation=apply_augmentation)

    # --------------------------------------------------------------------- #
    # Prep output directories
    # --------------------------------------------------------------------- #
    out_dir = Path(out_dir)
    train_dir, val_dir, test_dir = (out_dir / s for s in ("train", "val", "test"))
    for p in (train_dir, val_dir, test_dir):
        p.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    # 1) TRAIN / VAL  (IRMAS-TrainingData)
    # --------------------------------------------------------------------- #
    irmas_root = Path(irmas_root)
    train_src = irmas_root / "IRMAS-TrainingData"
    if not train_src.exists():
        raise FileNotFoundError(f"Expected training data at {train_src}")

    wav_files = list(train_src.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wavs found under {train_src}")

    print(f"🎻 Found {len(wav_files)} training wavs")
    # percentage filter ---------------------------------------------------- #
    if 0 < original_data_percentage < 1.0:
        k = int(len(wav_files) * original_data_percentage)
        wav_files = random.sample(wav_files, k)
        print(f"⚖️  Sub-sampling: {k} files ({original_data_percentage*100:.1f}% )")

    # shuffle + split 80/20 ------------------------------------------------ #
    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * 0.8)
    tr_files, val_files = wav_files[:split_idx], wav_files[split_idx:]

            def _process_split(files, split_dir, prefix, apply_augmentation=False):
        for i, wav in enumerate(
            tqdm(files, desc=f"→ {split_dir.name} ({len(files)})")
        ):
            try:
                y, sr = librosa.load(wav, sr=cfg["sample_rate"], mono=True)
                irmas_label = wav.parent.name  # single-label folders
                base = wav.stem
                sample_dir = split_dir / f"{prefix}_{irmas_label}_{i:04d}_{base}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Process original version
                for (band, n_fft), spec in generate_specs(y, sr, apply_augmentation=False).items():
                    np.save(sample_dir / f"{band}_fft{n_fft}.npy", spec)

                # Create augmented version if requested
                if apply_augmentation and prefix == "original" and split_dir.name == "train":
                    # Create augmented version for training data only
                    aug_sample_dir = split_dir / f"augmented_{irmas_label}_{i:04d}_{base}"
                    aug_sample_dir.mkdir(parents=True, exist_ok=True)

                    for (band, n_fft), spec in generate_specs(y, sr, apply_augmentation=True).items():
                        np.save(aug_sample_dir / f"{band}_fft{n_fft}.npy", spec)
            except Exception as e:
                print(f"❌ {wav}: {e}")

    print(f"📊 Train/val split → {len(tr_files)} train | {len(val_files)} val")

    # Check if augmentation is enabled in config
    augmentation_enabled = cfg.get('augmentation', {}).get('enabled', True)
    print(f"🎛️ Data augmentation: {'enabled' if augmentation_enabled else 'disabled'}")

    _process_split(tr_files, train_dir, "original", apply_augmentation=augmentation_enabled)
    _process_split(val_files, val_dir, "original", apply_augmentation=False)  # No augmentation for validation

    # --------------------------------------------------------------------- #
    # 2) TEST  (all IRMAS-TestingData* folders)  – 100 %
    # --------------------------------------------------------------------- #
    test_roots = [
        p for p in irmas_root.glob("IRMAS-TestingData-Part*")
        if p.is_dir() and any(p.rglob("*.wav"))
    ]
    if not test_roots:
        sf = irmas_root / "IRMAS-TestingData"
        if sf.exists():
            test_roots = [sf]
    else:
        test_wavs = list(
            itertools.chain.from_iterable(tr.rglob("*.wav") for tr in test_roots)
        )
        print(f"🎯 Found {len(test_wavs)} testing wavs across {len(test_roots)} parts.")

        # NEW ▸ quick-run cap
        cap = cfg.get("max_test_samples")
        if cap is not None and len(test_wavs) > cap:
            random.shuffle(test_wavs)
            test_wavs = test_wavs[:cap]
            print(f"⚖️  Capping test set to {cap} samples (quick-run mode)")

        for i, wav in enumerate(tqdm(test_wavs, desc="→ test")):
            try:
                y, sr = librosa.load(wav, sr=cfg["sample_rate"], mono=True)

                # concat multi-labels from .txt (if available)
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    with open(txt) as fh:
                        lbls = [ln.strip() for ln in fh if ln.strip()]
                    lbl_tag = "+".join(lbls) if lbls else "unknown"
                else:
                    lbl_tag = "unknown"

                base = wav.stem
                sample_dir = test_dir / f"irmasTest_{lbl_tag}_{i:04d}_{base}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                for (band, n_fft), spec in generate_specs(y, sr).items():
                    np.save(sample_dir / f"{band}_fft{n_fft}.npy", spec)
            except Exception as e:
                print(f"❌ {wav}: {e}")

    # --------------------------------------------------------------------- #
    # 3) SUMMARY
    # --------------------------------------------------------------------- #
    def _count(split_dir, tag):
        return len(
            [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith(tag)]
        )

    print("\n📋  Final counts")
    for split, p in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if p.exists():
            print(
                f"   {split:<5}: {len(list(p.iterdir())):>5} samples"
                f"  (original={_count(p,'original')}, irmasTest={_count(p,'irmasTest')})"
            )

    print(f"\n✅ Preprocessing finished. Features saved to {out_dir}")


def main(in_dir, out_dir, config):
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(config))

    wav_files = list(in_dir.rglob("*.wav"))
    for wav in tqdm.tqdm(wav_files):
        specs_dict = process_file(wav, cfg)

        # Create a directory for this audio file
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = out_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)

        # Save each spectrogram with band and FFT size information in the filename
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.in_dir, args.out_dir, args.config)