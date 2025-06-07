# !/usr/bin/env python3
"""Convert wav clips to multi-band STFT spectrograms saved as .npy.

Usage:
    python data/preprocess.py --in_dir data/raw/IRMAS --out_dir data/processed
"""
import argparse, librosa, numpy as np, pathlib, tqdm, yaml, os, torch, tempfile
from var import n_ffts, band_ranges_as_tuples, LABELS


def generate_multi_stft(
        y: np.ndarray,
        sr: int,
        n_ffts=n_ffts,
        band_ranges=band_ranges_as_tuples
):
    """
    Generates 9 spectrograms: 3 window sizes × 3 frequency bands.

    Parameters:
        y (np.ndarray): Audio waveform
        sr (int): Sampling rate
        n_ffts (tuple): FFT window sizes
        band_ranges (tuple): Frequency band ranges (Hz)

    Returns:
        dict: { (band_label, n_fft): spectrogram }
    """
    result = {}

    for n_fft in n_ffts:
        hop_length = n_fft // 4
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(stft)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        for (f_low, f_high) in band_ranges:
            band_label = f"{f_low}-{f_high}Hz"
            # Get frequency indices within this band
            band_mask = (freqs >= f_low) & (freqs < f_high)
            band_spec = mag[band_mask, :]
            # Convert to log scale
            log_spec = librosa.power_to_db(band_spec, ref=np.max).astype(np.float32)
            result[(band_label, n_fft)] = log_spec

    return result


def process_file(wav_path, cfg):
    y, sr = librosa.load(wav_path, sr=cfg['sample_rate'], mono=True)
    return generate_multi_stft(y, sr)


def process_mixed_audio(audio_tensor, labels, cfg, sample_id):
    """
    Process a mixed audio tensor and return spectrograms with multi-label.

    Args:
        audio_tensor: torch.Tensor of audio samples
        labels: torch.Tensor of multi-label vector
        cfg: Configuration dictionary
        sample_id: Unique identifier for this sample

    Returns:
        dict: { (band_label, n_fft): spectrogram }, labels_dict
    """
    # Convert tensor to numpy
    y = audio_tensor.numpy()
    sr = cfg['sample_rate']

    # Generate spectrograms
    specs_dict = generate_multi_stft(y, sr)

    # Create labels dictionary
    active_labels = [LABELS[i] for i, val in enumerate(labels) if val == 1]
    labels_dict = {
        'multi_label_vector': labels.numpy(),
        'active_labels': active_labels,
        'sample_id': sample_id
    }

    return specs_dict, labels_dict


def preprocess_mixed_data(irmas_root, mixed_dataset, out_dir, cfg, original_data_percentage=1.0):
    """
    Preprocess both original IRMAS data and mixed multi-label data.

    Args:
        irmas_root: Path to IRMAS dataset root directory
        mixed_dataset: List of (audio_tensor, label_vector) for mixed samples
        out_dir: Output directory for processed features
        cfg: Configuration dictionary
        original_data_percentage: Percentage of original data to use (0.0 to 1.0)
    """
    import os
    import random
    from pathlib import Path
    from tqdm import tqdm
    import librosa
    import numpy as np

    out_dir = Path(out_dir)

    # Create split directories
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # 1. PROCESS ORIGINAL IRMAS DATA
    # ============================================================================

    if original_data_percentage > 0:
        print(f"Using {original_data_percentage * 100:.1f}% of original IRMAS data (from config)")

        irmas_path = Path(irmas_root) / "IRMAS-TrainingData"

        if irmas_path.exists():
            print("Processing original IRMAS data...")

            # Get all WAV files
            wav_files = list(irmas_path.rglob("*.wav"))

            # Apply percentage filter
            if original_data_percentage < 1.0:
                num_files = int(len(wav_files) * original_data_percentage)
                wav_files = random.sample(wav_files, num_files)
                print(f"Using {original_data_percentage * 100:.1f}% of data: {len(wav_files)} files")
            else:
                print(f"Using all original data: {len(wav_files)} files")

            # Split original data
            random.shuffle(wav_files)
            train_split = int(len(wav_files) * 0.8)  # 80% train
            val_split = int(len(wav_files) * 0.9)  # 10% val, 10% test

            original_splits = {
                'train': wav_files[:train_split],
                'val': wav_files[train_split:val_split],
                'test': wav_files[val_split:]
            }

            print(
                f"Original data split: {len(original_splits['train'])} train, {len(original_splits['val'])} val, {len(original_splits['test'])} test")

            # Process each split
            for split_name, files in original_splits.items():
                if not files:
                    continue

                print(f"Processing {len(files)} original {split_name} files...")
                split_dir = out_dir / split_name

                for i, wav_file in enumerate(tqdm(files)):
                    try:
                        # Load audio
                        y, sr = librosa.load(wav_file, sr=cfg['sample_rate'], mono=True)

                        # Extract instrument label from parent directory
                        irmas_label = wav_file.parent.name

                        # Create UNIQUE directory name for each sample
                        # Format: original_{instrument}_{index}_{filename_without_ext}
                        filename_base = wav_file.stem  # filename without extension
                        sample_dir_name = f"original_{irmas_label}_{i:04d}_{filename_base}"
                        sample_dir = split_dir / sample_dir_name
                        sample_dir.mkdir(parents=True, exist_ok=True)

                        # Generate and save spectrograms
                        specs = generate_multi_stft(y, sr)

                        for (band_label, n_fft), spec in specs.items():
                            spec_filename = f"{band_label}_fft{n_fft}.npy"
                            np.save(sample_dir / spec_filename, spec)

                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")
                        continue
        else:
            print(f"Warning: IRMAS training data not found at {irmas_path}")

    # ============================================================================
    # 2. PROCESS MIXED MULTI-LABEL DATA
    # ============================================================================

    if mixed_dataset:
        print(f"\nProcessing {len(mixed_dataset)} mixed multi-label samples...")

        # Split mixed data
        random.shuffle(mixed_dataset)
        mixed_train_split = int(len(mixed_dataset) * 0.8)  # 80% train
        mixed_val_split = int(len(mixed_dataset) * 0.9)  # 10% val, 10% test

        mixed_splits = {
            'train': mixed_dataset[:mixed_train_split],
            'val': mixed_dataset[mixed_train_split:mixed_val_split],
            'test': mixed_dataset[mixed_val_split:]
        }

        print(
            f"Mixed data split: {len(mixed_splits['train'])} train, {len(mixed_splits['val'])} val, {len(mixed_splits['test'])} test")

        # Process each split
        for split_name, samples in mixed_splits.items():
            if not samples:
                continue

            split_dir = out_dir / split_name

            for i, (audio_tensor, label_vector) in enumerate(
                    tqdm(samples, desc=f"Processing {split_name} mixed samples")):
                try:
                    # Convert audio tensor to numpy
                    y = audio_tensor.numpy()

                    # Get active instrument labels for directory naming
                    from var import LABELS
                    active_labels = [LABELS[j] for j, val in enumerate(label_vector) if val == 1]

                    # Create directory name
                    label_str = "_".join(active_labels)
                    sample_dir_name = f"mixed_{i}_{label_str}"
                    sample_dir = split_dir / sample_dir_name
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    # Generate and save spectrograms
                    specs = generate_multi_stft(y, cfg['sample_rate'])

                    for (band_label, n_fft), spec in specs.items():
                        spec_filename = f"{band_label}_fft{n_fft}.npy"
                        np.save(sample_dir / spec_filename, spec)

                except Exception as e:
                    print(f"Error processing mixed sample {i}: {e}")
                    continue

    # ============================================================================
    # 3. SUMMARY
    # ============================================================================

    print("✅ Preprocessing complete with 80/10/10 split!")

    # Count final samples
    for split in ['train', 'val', 'test']:
        split_path = out_dir / split
        if split_path.exists():
            all_dirs = [d for d in split_path.iterdir() if d.is_dir()]
            original_count = len([d for d in all_dirs if d.name.startswith('original_')])
            mixed_count = len([d for d in all_dirs if d.name.startswith('mixed_')])
            print(
                f"   {split.capitalize()} data: {original_count} original + {mixed_count} mixed = {len(all_dirs)} total")

    print(f"✅ Preprocessing complete with mixed labels. Features saved to {out_dir}")

def preprocess_data(in_dir, out_dir, cfg):
    """Standard preprocessing with 80/10/10 split."""
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create train, validation, and test directories
    train_dir = out_dir / 'train'
    val_dir = out_dir / 'val'
    test_dir = out_dir / 'test'
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Get all WAV files
    wav_files = list(in_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    # Split into train/val/test sets (80/10/10 split)
    np.random.shuffle(wav_files)
    train_split = int(len(wav_files) * 0.8)
    val_split = int(len(wav_files) * 0.9)
    
    train_files = wav_files[:train_split]
    val_files = wav_files[train_split:val_split]
    test_files = wav_files[val_split:]

    print(f"Data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Process training files
    print("Processing training files...")
    for wav in tqdm.tqdm(train_files):
        specs_dict = process_file(wav, cfg)
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = train_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

    # Process validation files
    print("Processing validation files...")
    for wav in tqdm.tqdm(val_files):
        specs_dict = process_file(wav, cfg)
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = val_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

    # Process test files
    print("Processing test files...")
    for wav in tqdm.tqdm(test_files):
        specs_dict = process_file(wav, cfg)
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = test_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

    print(f"✅ Processed {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test files")


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