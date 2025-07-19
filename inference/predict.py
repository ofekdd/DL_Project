#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib

from data.preprocess import generate_multi_stft
from utils.model_loader import load_model_from_checkpoint
from var import LABELS, n_ffts, band_ranges_as_tuples

def extract_features(path, cfg, use_log_mel=False):
    """
    Extract 3 spectrograms (3 window sizes × 3 frequency bands) from audio file.

    Args:
        path: Path to audio file
        cfg: Configuration dictionary
        use_log_mel: Whether to use log-mel spectrogram instead of multi-STFT

    Returns:
        If use_log_mel=False: List of 3 tensors, each of shape [1, 1, F, T]
        If use_log_mel=True: Single tensor of shape [1, 1, F, T]
    """
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    
    if use_log_mel:
        # Get mel parameters from config or use defaults
        mel_params = cfg.get('mel_params', {})
        n_fft = mel_params.get('n_fft', 2048)
        hop_length = mel_params.get('hop_length', 512)
        n_mels = mel_params.get('n_mels', 128)
        fmin = mel_params.get('fmin', 20)
        fmax = mel_params.get('fmax', 11025)
        
        # Import the function here to avoid circular imports
        from data.preprocess import generate_log_mel_spectrogram
        
        # Generate log-mel spectrogram
        log_mel_spec = generate_log_mel_spectrogram(
            y, sr, n_fft=n_fft, hop_length=hop_length, 
            n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        
        # Convert to tensor
        spec_tensor = torch.tensor(log_mel_spec).unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]
        return spec_tensor
    else:
        # Original multi-STFT approach
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


def predict(model, wav_path, cfg, use_log_mel=False):
    """
    Run inference on a single audio file.

    Args:
        model: Trained model (MultiSTFTCNN, MultiSTFTCNN_WithPANNs, LogMelCNN, or LogMelCNN_WithPANNs)
        wav_path: Path to audio file
        cfg: Configuration dictionary
        use_log_mel: Whether to use log-mel spectrogram instead of multi-STFT

    Returns:
        Dictionary mapping label names to prediction scores
    """
    model.eval()

    # Extract features based on the model type
    features = extract_features(wav_path, cfg, use_log_mel=use_log_mel)

    with torch.no_grad():
        # Process features based on model type
        if use_log_mel:
            # LogMelCNN models expect a single tensor
            logits = model(features).squeeze()
        else:
            # MultiSTFTCNN models expect a list of tensors
            logits = model(features).squeeze()

        # For single-label classification, apply softmax to get probabilities
        if len(logits.shape) > 0:  # Handle single sample case
            probs = torch.nn.functional.softmax(logits, dim=0).numpy()
        else:
            probs = torch.nn.functional.softmax(logits.unsqueeze(0), dim=1).squeeze(0).numpy()

    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


def predict_with_ground_truth(model, wav_path, cfg, show_ground_truth=True, use_log_mel=False):
    """
    Enhanced prediction function that can show ground truth labels.
    Uses softmax to pick the most likely instrument.

    Args:
        model: Trained model
        wav_path: Path to wav file
        cfg: Configuration
        show_ground_truth: Whether to parse and show ground truth from filename
        use_log_mel: Whether to use log-mel spectrogram instead of multi-STFT

    Returns:
        Dictionary with predictions and optionally ground truth
    """
    # Get predictions (softmax probabilities)
    predictions = predict(model, wav_path, cfg, use_log_mel=use_log_mel)

    # Get the top prediction (most likely instrument)
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

        # Try to extract labels from different filename patterns
        import re

        # 1. Standard IRMAS pattern like [sax][jaz_blu]1737__1.wav
        irmas_pattern = r'\[([a-z]{3})\]'
        irmas_matches = re.findall(irmas_pattern, filename)

        # 2. Check for annotations file nearby
        annotation_path = pathlib.Path(wav_path).with_suffix('.txt')
        annotation_matches = []
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    content = f.read().strip().split('\n')
                    for line in content:
                        code = line.strip().lower()[:3]  # get first 3 chars
                        if code in irmas_to_label:
                            annotation_matches.append(code)
            except Exception as e:
                print(f"Warning: Could not read annotation file: {e}")

        # Combine all matches
        all_matches = irmas_matches + annotation_matches

        ground_truth = []
        for irmas_label in all_matches:
            if irmas_label in irmas_to_label:
                ground_truth.append(irmas_to_label[irmas_label])

        result["ground_truth"] = ground_truth

        # For testing dataset, ground truth could have multiple labels
        # Consider prediction correct if it matches any of the ground truth labels
        if ground_truth:
            result["correct"] = top_prediction in ground_truth
            result["accuracy"] = 1.0 if result["correct"] else 0.0

    return result


def predict_batch_with_accuracy(model, wav_files, cfg, show_details=True):
    """
    Run inference on multiple audio files and calculate overall accuracy.

    Args:
        model: Trained model
        wav_files: List of wav file paths
        cfg: Configuration dictionary
        show_details: Whether to show per-file details

    Returns:
        Dictionary with overall accuracy and per-file results
    """
    model.eval()

    total_correct = 0
    total_files = 0
    file_results = []

    print(f"🎵 Running inference on {len(wav_files)} files...")

    for idx, wav_path in enumerate(wav_files, 1):
        result = predict_with_ground_truth(model, wav_path, cfg, show_ground_truth=True)

        # Track accuracy if ground truth is available
        if "correct" in result:
            total_files += 1
            if result["correct"]:
                total_correct += 1

        file_results.append(result)

        if show_details:
            print(f"\n🎵 {idx}/{len(wav_files)} {pathlib.Path(wav_path).name}")

            # Ground truth
            gt = result.get("ground_truth", [])
            if gt:
                print(f"   🎯 Ground truth: {', '.join(gt)}")
            else:
                print("   ⚠️ Ground truth: Not available")

            # Prediction
            print(f"   🎺 Top prediction: {result['top_prediction']} ({result['top_score']:.4f})")

            # Accuracy
            if "correct" in result:
                status = "✅ CORRECT" if result["correct"] else "❌ INCORRECT"
                print(f"   📊 Evaluation: {status}")

                # Show running accuracy
                running_accuracy = total_correct / total_files if total_files > 0 else 0
                print(f"   📈 Running accuracy: {total_correct}/{total_files} = {running_accuracy:.1%}")
            else:
                print("   ⚠️ Evaluation: Not possible (no ground truth)")

            print("-" * 50)

    # Calculate final accuracy
    final_accuracy = total_correct / total_files if total_files > 0 else 0

    print(f"\n✅ Inference completed!")
    print(f"📊 Final accuracy: {total_correct}/{total_files} = {final_accuracy:.1%}")

    return {
        "total_correct": total_correct,
        "total_files": total_files,
        "accuracy": final_accuracy,
        "file_results": file_results
    }


def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))

    # Load model with improved checkpoint handling
    model = load_model_from_checkpoint(ckpt, n_classes=len(LABELS))

    # Get enhanced predictions
    result = predict_with_ground_truth(model, wav, cfg, show_ground_truth=True)

    # Print results
    print("\n" + "=" * 50)
    print(f"🎵 File: {pathlib.Path(wav).name}")

    if "ground_truth" in result and result["ground_truth"]:
        print(f"🎯 Ground truth: {', '.join(result['ground_truth'])}")
    else:
        print("⚠️ Ground truth: Not available (filename doesn't match IRMAS pattern)")

    print(f"🎺 Top prediction: {result['top_prediction']} ({result['top_score']:.4f})")

    print(f"📊 All predictions (sorted by softmax probability):")
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
    else:
        print("⚠️ Evaluation not possible (no ground truth available)")

    print("=" * 50)

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/panns_enhanced.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)