#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib

from data.preprocess import generate_multi_stft
from models.multi_stft_cnn_with_stft import MultiSTFTCNN
from var import LABELS, n_ffts, band_ranges


def predict(model, wav_path, cfg):
    model.eval()

    # Load audio as waveform (like the model expects)
    y, sr = librosa.load(wav_path, sr=cfg['sample_rate'], mono=True)
    waveform = torch.tensor(y).unsqueeze(0)  # Add batch dimension [1, N]

    with torch.no_grad():
        preds = model(waveform).squeeze().numpy()

    return {label: float(preds[i]) for i, label in enumerate(LABELS)}


def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    # Using MultiSTFTCNN model directly as specified
    model = MultiSTFTCNN(n_classes=len(LABELS))
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    results = predict(model, wav, cfg)
    print(results)


# def extract_features(path, cfg):
#     y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
#     specs_dict = generate_multi_stft(y, sr)
#
#     # For MultiSTFTCNN, we need all 9 spectrograms (3 window sizes × 3 frequency bands)
#     # First, collect all spectrograms
#     raw_specs = []
#     for n_fft in n_ffts:
#         for band_range in band_ranges:
#             key = (band_range, n_fft)
#             if key in specs_dict:
#                 spec = specs_dict[key]
#                 spec_tensor = torch.tensor(spec).unsqueeze(0)  # [1, F, T]
#                 raw_specs.append(spec_tensor)
#             else:
#                 # If a specific spectrogram is missing, use a zero tensor of appropriate shape
#                 # This is a fallback and should be rare
#                 print(f"Warning: Missing spectrogram for {key}")
#                 # Use a small dummy tensor as fallback
#                 raw_specs.append(torch.zeros(1, 10, 10))
#
#     # Ensure each spectrogram has reasonable dimensions
#     # For inference with a single sample, we don't need to pad to match other samples
#     # But we should ensure each spectrogram has appropriate dimensions
#     specs_list = []
#     for spec in raw_specs:
#         # Add batch dimension (batch, channel, freq, time)
#         spec_tensor = spec.unsqueeze(0)  # [1, 1, F, T]
#         specs_list.append(spec_tensor)
#
#     return specs_list

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)