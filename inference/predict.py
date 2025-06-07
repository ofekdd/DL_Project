#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib

from data.preprocess import generate_multi_stft
from models.multi_stft_cnn import MultiSTFTCNN
from var import LABELS, n_ffts, band_ranges_as_tuples


def extract_features(path, cfg):
    """
    Extract 9 spectrograms (3 window sizes × 3 frequency bands) from audio file.

    Args:
        path: Path to audio file
        cfg: Configuration dictionary

    Returns:
        List of 9 tensors, each of shape [1, 1, F, T]
    """
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    specs_dict = generate_multi_stft(y, sr)

    # For MultiSTFTCNN, we need all 9 spectrograms in the correct order
    specs_list = []
    for band_range in band_ranges_as_tuples:
        band_label = f"{band_range[0]}-{band_range[1]}Hz"
        for n_fft in n_ffts:
            key = (band_label, n_fft)
            if key in specs_dict:
                spec = specs_dict[key]
                spec_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
                specs_list.append(spec_tensor)
            else:
                # If a specific spectrogram is missing, use a zero tensor of appropriate shape
                print(f"Warning: Missing spectrogram for {key}")
                # Use a small dummy tensor as fallback
                specs_list.append(torch.zeros(1, 1, 10, 10))

    return specs_list


def predict(model, wav_path, cfg):
    """
    Run inference on a single audio file.

    Args:
        model: Trained MultiSTFTCNN model
        wav_path: Path to audio file
        cfg: Configuration dictionary

    Returns:
        Dictionary mapping label names to prediction scores
    """
    model.eval()

    # Extract features as list of 9 spectrograms
    specs_list = extract_features(wav_path, cfg)

    with torch.no_grad():
        # Model expects list of tensors as input
        preds = model(specs_list).squeeze().numpy()

    return {label: float(preds[i]) for i, label in enumerate(LABELS)}


def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    # Using MultiSTFTCNN model directly as specified
    model = MultiSTFTCNN(n_classes=len(LABELS))
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    results = predict(model, wav, cfg)
    print(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)