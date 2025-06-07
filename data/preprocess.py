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
    Preprocess both original IRMAS data and mixed multi-label data with 80/10/10 split.

    Args:
        irmas_root: Path to original IRMAS dataset
        mixed_dataset: List of (audio_tensor, label_vector) tuples
        out_dir: Output directory for processed data
        cfg: Configuration dictionary
        original_data_percentage: Percentage of original data to use (0.0 to 1.0)
    """
    out_dir = pathlib.Path(out_dir)
    train_dir = out_dir / 'train'
    val_dir = out_dir / 'val'
    test_dir = out_dir / 'test'  # Add test directory

    # Create all directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # First, process original IRMAS data (single-label)
    print("Processing original IRMAS data...")
    irmas_path = pathlib.Path(irmas_root)
    wav_files = list(irmas_path.rglob("*.wav"))

    if wav_files:
        # Apply percentage filter to original data
        if 0 < original_data_percentage < 1.0:
            original_count = int(len(wav_files) * original_data_percentage)
            np.random.shuffle(wav_files)  # Shuffle before taking subset
            wav_files = wav_files[:original_count]
            print(f"Using {original_data_percentage * 100}% of original data: {len(wav_files)} files")
        elif original_data_percentage == 0:
            wav_files = []
            print("Skipping original data (percentage = 0)")
        else:
            print(f"Using all original data: {len(wav_files)} files")

        if wav_files:
            # Split original data into train/val/test (80/10/10 split)
            np.random.shuffle(wav_files)
            train_split = int(len(wav_files) * 0.8)
            val_split = int(len(wav_files) * 0.9)

            train_files = wav_files[:train_split]
            val_files = wav_files[train_split:val_split]
            test_files = wav_files[val_split:]

            print(f"Original data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

            # Process training files
            print(f"Processing {len(train_files)} original training files...")
            for wav in tqdm.tqdm(train_files):
                specs_dict = process_file(wav, cfg)
                rel_dir = wav.relative_to(irmas_path).with_suffix("")
                file_out_dir = train_dir / rel_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                for (band_label, n_fft), spec in specs_dict.items():
                    spec_filename = f"{band_label}_fft{n_fft}.npy"
                    np.save(file_out_dir / spec_filename, spec)

            # Process validation files
            print(f"Processing {len(val_files)} original validation files...")
            for wav in tqdm.tqdm(val_files):
                specs_dict = process_file(wav, cfg)
                rel_dir = wav.relative_to(irmas_path).with_suffix("")
                file_out_dir = val_dir / rel_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                for (band_label, n_fft), spec in specs_dict.items():
                    spec_filename = f"{band_label}_fft{n_fft}.npy"
                    np.save(file_out_dir / spec_filename, spec)

            # Process test files
            print(f"Processing {len(test_files)} original test files...")
            for wav in tqdm.tqdm(test_files):
                specs_dict = process_file(wav, cfg)
                rel_dir = wav.relative_to(irmas_path).with_suffix("")
                file_out_dir = test_dir / rel_dir
                file_out_dir.mkdir(parents=True, exist_ok=True)
                for (band_label, n_fft), spec in specs_dict.items():
                    spec_filename = f"{band_label}_fft{n_fft}.npy"
                    np.save(file_out_dir / spec_filename, spec)
        else:
            # If no original files, create empty variables for summary
            train_files, val_files, test_files = [], [], []

    # Now, process mixed data (multi-label) with same 80/10/10 split
    print(f"\nProcessing {len(mixed_dataset)} mixed multi-label samples...")
    np.random.shuffle(mixed_dataset)

    train_mixed_split = int(len(mixed_dataset) * 0.8)
    val_mixed_split = int(len(mixed_dataset) * 0.9)

    train_mixed = mixed_dataset[:train_mixed_split]
    val_mixed = mixed_dataset[train_mixed_split:val_mixed_split]
    test_mixed = mixed_dataset[val_mixed_split:]

    print(f"Mixed data split: {len(train_mixed)} train, {len(val_mixed)} val, {len(test_mixed)} test")

    # Process mixed training data
    for i, (audio_tensor, labels) in enumerate(tqdm.tqdm(train_mixed)):
        specs_dict, labels_dict = process_mixed_audio(audio_tensor, labels, cfg, f"mixed_train_{i}")
        active_labels_str = "_".join(labels_dict['active_labels'])
        file_out_dir = train_dir / f"mixed_{i}_{active_labels_str}"
        file_out_dir.mkdir(parents=True, exist_ok=True)
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)
        np.save(file_out_dir / "labels.npy", labels_dict['multi_label_vector'])

    # Process mixed validation data
    for i, (audio_tensor, labels) in enumerate(tqdm.tqdm(val_mixed)):
        specs_dict, labels_dict = process_mixed_audio(audio_tensor, labels, cfg, f"mixed_val_{i}")
        active_labels_str = "_".join(labels_dict['active_labels'])
        file_out_dir = val_dir / f"mixed_{i}_{active_labels_str}"
        file_out_dir.mkdir(parents=True, exist_ok=True)
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)
        np.save(file_out_dir / "labels.npy", labels_dict['multi_label_vector'])

    # Process mixed test data
    for i, (audio_tensor, labels) in enumerate(tqdm.tqdm(test_mixed)):
        specs_dict, labels_dict = process_mixed_audio(audio_tensor, labels, cfg, f"mixed_test_{i}")
        active_labels_str = "_".join(labels_dict['active_labels'])
        file_out_dir = test_dir / f"mixed_{i}_{active_labels_str}"
        file_out_dir.mkdir(parents=True, exist_ok=True)
        for (band_label, n_fft), spec in specs_dict.items():
            spec_filename = f"{band_label}_fft{n_fft}.npy"
            np.save(file_out_dir / spec_filename, spec)
        np.save(file_out_dir / "labels.npy", labels_dict['multi_label_vector'])

    print(f"✅ Preprocessing complete with 80/10/10 split!")
    print(
        f"   Original data: {len(train_files) if 'train_files' in locals() else 0} train + {len(val_files) if 'val_files' in locals() else 0} val + {len(test_files) if 'test_files' in locals() else 0} test")
    print(f"   Mixed data: {len(train_mixed)} train + {len(val_mixed)} val + {len(test_mixed)} test")
    total_train = (len(train_files) if 'train_files' in locals() else 0) + len(train_mixed)
    total_val = (len(val_files) if 'val_files' in locals() else 0) + len(val_mixed)
    total_test = (len(test_files) if 'test_files' in locals() else 0) + len(test_mixed)
    print(f"   Total: {total_train} train + {total_val} val + {total_test} test")

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