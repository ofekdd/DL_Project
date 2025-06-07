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


def load_model_from_checkpoint(ckpt_path, n_classes):
    """
    Load model from PyTorch Lightning checkpoint, handling key prefix issues.

    Args:
        ckpt_path: Path to checkpoint file
        n_classes: Number of classes

    Returns:
        Loaded model
    """
    # Create model
    model = MultiSTFTCNN(n_classes=n_classes)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Extract state dict - handle both direct state_dict and PyTorch Lightning format
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove "model." prefix if present (PyTorch Lightning adds this)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]  # Remove "model." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Load the corrected state dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed, trying with strict=False: {e}")
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Model loaded with strict=False (some weights may be missing)")

    return model


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
        preds = model(specs_list).squeeze()

        # Apply sigmoid to get probabilities (since we're using BCELoss in training)
        if len(preds.shape) > 0:  # Handle single sample case
            preds = torch.sigmoid(preds).numpy()
        else:
            preds = torch.sigmoid(preds.unsqueeze(0)).numpy()

    return {label: float(preds[i]) for i, label in enumerate(LABELS)}


def predict_with_ground_truth(model, wav_path, cfg, show_ground_truth=True):
    """
    Enhanced prediction function that can show ground truth labels.

    Args:
        model: Trained model
        wav_path: Path to wav file
        cfg: Configuration
        show_ground_truth: Whether to parse and show ground truth from filename

    Returns:
        Dictionary with predictions and optionally ground truth
    """
    # Get predictions
    predictions = predict(model, wav_path, cfg)

    result = {"predictions": predictions}

    if show_ground_truth:
        # Parse ground truth from filename
        filename = pathlib.Path(wav_path).name

        # IRMAS label mapping
        irmas_to_label = {
            'cel': 'cello', 'cla': 'clarinet', 'flu': 'flute',
            'gac': 'acoustic_guitar', 'gel': 'acoustic_guitar',
            'org': 'organ', 'pia': 'piano', 'sax': 'saxophone',
            'tru': 'trumpet', 'vio': 'violin', 'voi': 'voice'
        }

        # Extract labels from filename like [sax][jaz_blu]1737__1.wav
        import re
        irmas_pattern = r'\[([a-z]{3})\]'
        irmas_matches = re.findall(irmas_pattern, filename)

        ground_truth = []
        for irmas_label in irmas_matches:
            if irmas_label in irmas_to_label:
                ground_truth.append(irmas_to_label[irmas_label])

        result["ground_truth"] = ground_truth

        # Calculate if prediction is correct (top prediction matches any ground truth)
        if ground_truth:
            top_prediction = max(predictions.items(), key=lambda x: x[1])[0]
            result["correct"] = top_prediction in ground_truth

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