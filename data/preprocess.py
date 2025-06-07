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


def preprocess_mixed_data(irmas_root, mixed_dataset, out_dir, cfg):
    """
    Preprocess both original IRMAS data and mixed multi-label data.

    Args:
        irmas_root: Path to original IRMAS dataset
        mixed_dataset: List of (audio_tensor, label_vector) tuples
        out_dir: Output directory for processed data
        cfg: Configuration dictionary
    """
    out_dir = pathlib.Path(out_dir)
    train_dir = out_dir / 'train'
    val_dir = out_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # First, process original IRMAS data (single-label)
    print("Processing original IRMAS data...")
    irmas_path = pathlib.Path(irmas_root)
    wav_files = list(irmas_path.rglob("*.wav"))

    if wav_files:
        # Split original data into train and validation (90/10 split)
        np.random.shuffle(wav_files)
        split_idx = int(len(wav_files) * 0.9)
        train_files = wav_files[:split_idx]
        val_files = wav_files[split_idx:]

        # Process training files
        print(f"Processing {len(train_files)} original training files...")
        for wav in tqdm.tqdm(train_files):
            specs_dict = process_file(wav, cfg)

            # Create a directory for this audio file
            rel_dir = wav.relative_to(irmas_path).with_suffix("")
            file_out_dir = train_dir / rel_dir
            file_out_dir.mkdir(parents=True, exist_ok=True)

            # Save each spectrogram
            for (band_label, n_fft), spec in specs_dict.items():
                spec_filename = f"{band_label}_fft{n_fft}.npy"
                np.save(file_out_dir / spec_filename, spec)

        # Process validation files
        print(f"Processing {len(val_files)} original validation files...")
        for wav in tqdm.tqdm(val_files):
            specs_dict = process_file(wav, cfg)

            # Create a directory for this audio file
            rel_dir = wav.relative_to(irmas_path).with_suffix("")
            file_out_dir = val_dir / rel_dir
            file_out_dir.mkdir(parents=True, exist_ok=True)

            # Save each spectrogram
            for (band_label, n_fft), spec in specs_dict.items():
                spec_filename = f"{band_label}_fft{n_fft}.npy"
                np.save(file_out_dir / spec_filename, spec)

    # Now, process mixed data (multi-label)
    print(f"\nProcessing {len(mixed_dataset)} mixed multi-label samples...")

    # Split mixed data into train and validation (90/10 split)
    np.random.shuffle(mixed_dataset)
    split_idx = int(len(mixed_dataset) * 0.9)
    train_mixed = mixed_dataset[:split_idx]
    val_mixed = mixed_dataset[split_idx:]

    # Process mixed training data
    print(f"Processing {len(train_mixed)} mixed training samples...")
    for i, (audio_tensor, labels) in enumerate(tqdm.tqdm(train_mixed)):
        specs_dict, labels_dict = process_mixed_audio(audio_tensor, labels, cfg, f"mixed_train_{i}")

        # Create directory for this mixed sample
        active_labels_str = "_".join(labels_dict['active_labels'])
        file_out_dir = train_dir / f"mixed_{i}_{active_labels_str}"
        file_out_dir.mkdir(parents=True, exist_ok=True)

        # Save spectrograms
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

        # Save labels information
        np.save(file_out_dir / "labels.npy", labels_dict['multi_label_vector'])

    # Process mixed validation data
    print(f"Processing {len(val_mixed)} mixed validation samples...")
    for i, (audio_tensor, labels) in enumerate(tqdm.tqdm(val_mixed)):
        specs_dict, labels_dict = process_mixed_audio(audio_tensor, labels, cfg, f"mixed_val_{i}")

        # Create directory for this mixed sample
        active_labels_str = "_".join(labels_dict['active_labels'])
        file_out_dir = val_dir / f"mixed_{i}_{active_labels_str}"
        file_out_dir.mkdir(parents=True, exist_ok=True)

        # Save spectrograms
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

        # Save labels information
        np.save(file_out_dir / "labels.npy", labels_dict['multi_label_vector'])

    print(f"✅ Preprocessing complete!")
    print(
        f"   Original data: {len(train_files) if 'train_files' in locals() else 0} train + {len(val_files) if 'val_files' in locals() else 0} val")
    print(f"   Mixed data: {len(train_mixed)} train + {len(val_mixed)} val")
    print(
        f"   Total: {(len(train_files) if 'train_files' in locals() else 0) + len(train_mixed)} train + {(len(val_files) if 'val_files' in locals() else 0) + len(val_mixed)} val")


def preprocess_data(in_dir, out_dir, cfg):
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create train and validation directories
    train_dir = out_dir / 'train'
    val_dir = out_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Get all WAV files
    wav_files = list(in_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    # Split into train and validation sets (90/10 split)
    np.random.shuffle(wav_files)
    split_idx = int(len(wav_files) * 0.9)
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]

    # Process training files
    print("Processing training files...")
    for wav in tqdm.tqdm(train_files):
        specs_dict = process_file(wav, cfg)

        # Create a directory for this audio file
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = train_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)

        # Save each spectrogram with band and FFT size information in the filename
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

    # Process validation files
    print("Processing validation files...")
    for wav in tqdm.tqdm(val_files):
        specs_dict = process_file(wav, cfg)

        # Create a directory for this audio file
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = val_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)

        # Save each spectrogram with band and FFT size information in the filename
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)

    print(f"Processed {len(train_files)} training files and {len(val_files)} validation files")


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