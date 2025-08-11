#!/usr/bin/env python3
"""CLI inference (Wavelet-first, STFT fallback)"""
import argparse, yaml, torch, librosa, numpy as np, pathlib

from utils.model_loader import load_model_from_checkpoint
from var import LABELS

# ---------------------------
# Waveform loader (WaveletCNN path)
# ---------------------------
def load_waveform_tensor(path, cfg, device):
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    x = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N]
    return x

# ---------------------------
# STFT fallback (old models)
# ---------------------------
def extract_features_stft(path, cfg):
    """
    Fallback for legacy MultiSTFT models:
    returns list of 3 tensors [1,1,F,T]
    """
    import librosa
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)

    optimized_stfts = [
        ((0, 1000), 1024),
        ((1000, 4000), 512),
        ((4000, 11025), 256),
    ]

    specs_list = []
    for (f_low, f_high), n_fft in optimized_stfts:
        hop_length = n_fft // 4
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(S)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        mask = (freqs >= f_low) & (freqs < f_high)
        band = mag[mask, :]
        log_band = librosa.power_to_db(band, ref=np.max).astype(np.float32)
        specs_list.append(torch.tensor(log_band).unsqueeze(0).unsqueeze(0))  # [1,1,F,T]

    return specs_list

# ---------------------------
# Core predict
# ---------------------------
def predict(model, wav_path, cfg):
    """
    Run inference on a single audio file.
    Uses WaveletCNN (waveform) when possible; falls back to 3×STFT for old checkpoints.
    Returns dict[label] = probability (softmax over classes).
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        try:
            # Wavelet path: waveform -> logits [1, C]
            x = load_waveform_tensor(wav_path, cfg, device)     # [1, N]
            logits = model(x).squeeze(0)                        # [C]
        except Exception as e:
            # Fallback: STFT path (legacy MultiSTFT models expecting list input)
            print(f"ℹ️ Wavelet path failed ({e}). Falling back to STFT inference…")
            specs_list = extract_features_stft(wav_path, cfg)
            specs_list = [t.to(device) for t in specs_list]
            logits =
