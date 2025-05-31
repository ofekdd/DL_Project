
#!/usr/bin/env python3
"""Convert wav clips to multi-band STFT spectrograms saved as .npy.

Usage:
    python data/preprocess.py --in_dir data/raw/IRMAS --out_dir data/processed
"""
import argparse, librosa, numpy as np, pathlib, tqdm, yaml, os

def generate_multi_stft(
    y: np.ndarray,
    sr: int,
    n_ffts=(256, 512, 1024),
    band_ranges=((0, 1000), (1000, 4000), (4000, 11025))
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
