#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib

from data.preprocess import generate_multi_stft
from models.multi_stft_cnn import MultiSTFTCNN
from utils.model_loader import load_model_from_checkpoint
from var import LABELS, n_ffts, band_ranges_as_tuples

#TODO: maybe adapt to  single label

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

def predict(model, wav_path, cfg):
    """
    Run inference on a single audio file.

    Args:
        model: Trained model (either MultiSTFTCNN or MultiSTFTCNN_WithPANNs)
        wav_path: Path to audio file
        cfg: Configuration dictionary

    Returns:
        Dictionary mapping label names to prediction scores
    """
    model.eval()

    # Extract features as list of 3 spectrograms
    specs_list = extract_features(wav_path, cfg)

    with torch.no_grad():
        # Model expects list of tensors as input
        logits = model(specs_list).squeeze()

        # For single-label classification, apply softmax to get probabilities
        if len(logits.shape) > 0:  # Handle single sample case
            probs = torch.nn.functional.softmax(logits, dim=0).numpy()
        else:
            probs = torch.nn.functional.softmax(logits.unsqueeze(0), dim=1).squeeze(0).numpy()

    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


def predict_with_ground_truth(model, wav_path, cfg, show_ground_truth=True):
    """
    Enhanced prediction function that can show ground truth labels.
    For single-label classification, we pick the most dominant instrument.

    Args:
        model: Trained model
        wav_path: Path to wav file
        cfg: Configuration
        show_ground_truth: Whether to parse and show ground truth from filename

    Returns:
        Dictionary with predictions and optionally ground truth
    """
    # Get predictions (softmax probabilities)
    predictions = predict(model, wav_path, cfg)

    # Get the top prediction (most dominant instrument)
    top_prediction = max(predictions.items(), key=lambda x: x[1])[0]
    top_score = predictions[top_prediction]

    result = {
        "predictions": predictions,
        "top_prediction": top_prediction,
        "top_score": top_score
    }

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

        # For testing dataset, ground truth could have multiple labels
        # Consider prediction correct if it matches any of the ground truth labels
        if ground_truth:
            result["correct"] = top_prediction in ground_truth
            result["accuracy"] = 1.0 if result["correct"] else 0.0

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

    print(f"🎺 Top prediction: {result['top_prediction']} ({result['top_score']:.4f})")

    print(f"📊 All predictions (sorted):")
    predictions = result["predictions"]
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    for label, score in sorted_preds:
        # Highlight the top prediction
        icon = "🔥" if label == result['top_prediction'] else "🔸"
        print(f"  {icon} {label:<15} {score:.4f}")

    if "correct" in result:
        status = "✅ CORRECT" if result["correct"] else "❌ INCORRECT"
        print(f"🎯 Evaluation: {status}")
        if result["correct"]:
            print(f"   (Top prediction matches one of the ground truth labels)")

    print("=" * 50)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)
