#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib

from data.preprocess import generate_multi_stft
from models.multi_stft_cnn import MultiSTFTCNN
from var import LABELS, n_ffts, band_ranges_as_tuples, IRMAS_TO_LABEL_MAP

def safe_sigmoid(logits):
    """If logits already look like probabilities (0-1), return as-is."""
    if logits.min() >= 0 and logits.max() <= 1:
        return logits
    return torch.sigmoid(logits)

def extract_features(path, cfg):
    """
    Extract 3 spectrograms (3 window sizes × 3 frequency bands) from audio file.

    Args:
        path: Path to audio file
        cfg: Configuration dictionary

    Returns:
        List of 9 tensors, each of shape [1, 1, F, T]
    """
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    specs_dict = generate_multi_stft(y, sr)

    # For MultiSTFTCNN, we need all 3 spectrograms in the correct order
    specs_list = []
    optimized_stfts = [
        ("0-1000Hz", 1024),
        ("1000-4000Hz", 512),
        ("4000-11025Hz", 256),
    ]

    for band_label, n_fft in optimized_stfts:
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


def load_model_from_checkpoint(ckpt_path, n_classes):
    """
    Load model from PyTorch Lightning checkpoint, handling key prefix issues.

    Args:
        ckpt_path: Path to checkpoint file
        n_classes: Number of classes

    Returns:
        Loaded model
    """
    # Use the unified model loader
    from utils.model_loader import load_model
    return load_model(ckpt_path, n_classes=n_classes)

def predict(model, wav_path, cfg):
    model.eval()
    specs = extract_features(wav_path, cfg)
    with torch.no_grad():
        logits = model(specs).squeeze()
        probs  = safe_sigmoid(logits).cpu().numpy()
    return {lab: float(probs[i]) for i, lab in enumerate(LABELS)}

def apply_thresholds(preds, default=0.5, table=None):
    """Return binary vector + active label list."""
    bin_vec = {
        lab: 1 if score >= (table.get(lab, default) if table else default) else 0
        for lab, score in preds.items()
    }
    active = [lab for lab, v in bin_vec.items() if v]
    return bin_vec, active

def predict_with_ground_truth(model, wav_path, cfg,
                              threshold=0.5, thresholds=None, show_gt=True):
    preds = predict(model, wav_path, cfg)
    bin_vec, active = apply_thresholds(preds, threshold, thresholds)
    result = dict(predictions=preds,
                  binary_predictions=bin_vec,
                  active_instruments=active,
                  threshold=threshold)

    if show_gt:
        # same filename-parsing logic as before
        fname = pathlib.Path(wav_path).name
        gt = [IRMAS_TO_LABEL_MAP[x] for x in
              __import__("re").findall(r"\[([a-z]{3})\]", fname)
              if x in IRMAS_TO_LABEL_MAP]
        # after parsing brackets into `ground_truth` …
        if not gt:
            txt = pathlib.Path(wav_path).with_suffix(".txt")
            if txt.exists():
                with open(txt) as fh:
                    for line in fh:
                        lab = line.strip().lower()
                        mapped = IRMAS_TO_LABEL_MAP.get(lab, lab)
                        if mapped in LABELS:
                            gt.append(mapped)

        result["ground_truth"] = gt
        if gt:
            correct = max(preds, key=preds.get) in gt
            result["correct"] = correct
            correct_bits = sum(
                1 for lab in LABELS if bin_vec[lab] == (lab in gt)
            )
            result["accuracy"] = correct_bits / len(LABELS)

    return result


def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))

    # Load model with improved checkpoint handling
    model = load_model_from_checkpoint(ckpt, n_classes=len(LABELS))

    # Get enhanced predictions
    result = predict_with_ground_truth(model, wav, cfg, show_ground_truth=True)

    # Print results
    print("\n" + "=" * 50)
    print(f"🎵 File: {pathlib.Path(wav).name}")

    if "ground_truth" in result:
        print(f"🎯 Ground truth: {result['ground_truth']}")

    print(f"📊 Predictions:")
    predictions = result["predictions"]
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    for label, score in sorted_preds:
        icon = "🔥" if score > 0.5 else "🔸"
        print(f"  {icon} {label:<15} {score:.4f}")

    if "correct" in result:
        status = "✅ CORRECT" if result["correct"] else "❌ INCORRECT"
        print(f"🎯 Top prediction: {status}")

    print("=" * 50)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)