#!/usr/bin/env python3
"""Enhanced prediction with adaptive thresholds"""

import argparse
import yaml
import torch
import pathlib
import numpy as np

from inference.predict import extract_features, predict, load_model_from_checkpoint
from var import LABELS


def load_thresholds(thresholds_path, default_threshold=0.5):
    """Load adaptive thresholds from YAML file.

    Args:
        thresholds_path: Path to thresholds YAML file
        default_threshold: Default threshold to use if file not found

    Returns:
        Dictionary mapping instrument names to thresholds
    """
    # Default thresholds
    thresholds = {label: default_threshold for label in LABELS}

    # Load adaptive thresholds if provided
    if thresholds_path:
        try:
            with open(thresholds_path, 'r') as f:
                thresholds_data = yaml.safe_load(f)
                if 'thresholds' in thresholds_data:
                    # Update with adaptive thresholds
                    thresholds.update(thresholds_data['thresholds'])
                    print(f"✅ Loaded adaptive thresholds from {thresholds_path}")
                    # Show some example thresholds
                    examples = list(thresholds_data['thresholds'].items())[:3]
                    examples_str = ", ".join([f"{k}: {v:.2f}" for k, v in examples])
                    print(f"   Sample thresholds: {examples_str}...")
        except Exception as e:
            print(f"⚠️ Error loading thresholds: {e}")
            print(f"   Using default threshold: {default_threshold}")
    else:
        print(f"ℹ️ No thresholds file provided, using fixed threshold: {default_threshold}")

    return thresholds


def predict_with_adaptive_thresholds(model, wav_path, cfg, thresholds=None, show_ground_truth=True):
    """Run inference with adaptive thresholds.

    Args:
        model: Trained model
        wav_path: Path to audio file
        cfg: Configuration dictionary
        thresholds: Dictionary mapping instrument names to thresholds, or None for default
        show_ground_truth: Whether to parse and show ground truth from filename

    Returns:
        Dictionary with predictions and analysis
    """
    # Default fixed threshold
    default_threshold = 0.5

    # Use provided thresholds or create default
    if thresholds is None:
        thresholds = {label: default_threshold for label in LABELS}

    # Get raw predictions
    predictions = predict(model, wav_path, cfg)

    # Apply thresholds to get binary predictions
    binary_predictions = {}
    for label, score in predictions.items():
        threshold = thresholds.get(label, default_threshold)
        binary_predictions[label] = 1 if score >= threshold else 0

    # Get active instruments
    active_instruments = [label for label, is_active in binary_predictions.items() if is_active == 1]

    # Create result dictionary
    result = {
        "predictions": predictions,
        "binary_predictions": binary_predictions,
        "active_instruments": active_instruments,
        "thresholds": thresholds
    }

    # Try to parse ground truth if requested
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

        # Calculate per-instrument accuracy
        if ground_truth:
            # Create ground truth binary vector
            gt_binary = {label: 1 if label in ground_truth else 0 for label in LABELS}

            # Calculate per-instrument correctness
            correct_predictions = sum(1 for label in LABELS
                                    if binary_predictions[label] == gt_binary[label])

            # Calculate overall sample accuracy
            result["correct_count"] = correct_predictions
            result["total_count"] = len(LABELS)
            result["accuracy"] = correct_predictions / len(LABELS)

            # Legacy 'correct' field (top prediction matches any ground truth)
            top_prediction = max(predictions.items(), key=lambda x: x[1])[0]
            result["correct"] = top_prediction in ground_truth

    return result


def display_prediction_results(result, show_all_instruments=False, highlight_threshold=0.3):
    """Display prediction results in a user-friendly format.

    Args:
        result: Result dictionary from predict_with_adaptive_thresholds
        show_all_instruments: Whether to show all instruments or just active ones
        highlight_threshold: Score threshold for highlighting in output
    """
    print("\n" + "=" * 60)
    print(f"🎵 File: {pathlib.Path(result.get('wav_path', 'unknown')).name}")

    # Show ground truth if available
    if "ground_truth" in result:
        print(f"🎯 Ground truth: {result['ground_truth']}")

    # Show active instruments
    active = result.get("active_instruments", [])
    print(f"🔊 Detected instruments: {', '.join(active) if active else 'None'}")

    # Show accuracy if available
    if "accuracy" in result:
        print(f"📊 Overall accuracy: {result['correct_count']}/{result['total_count']} = {result['accuracy']:.1%}")

    # Show detailed predictions
    print("\n📋 Detailed predictions:")
    print(f"{'Instrument':<15} {'Score':<10} {'Threshold':<10} {'Decision'}")
    print("-" * 50)

    # Get predictions and sort by score
    predictions = result["predictions"]
    thresholds = result.get("thresholds", {label: 0.5 for label in LABELS})
    binary = result["binary_predictions"]

    # Sort instruments by score
    sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    # Filter to show only active instruments if requested
    if not show_all_instruments:
        # Always show top 3 plus any active ones
        top_3 = [label for label, _ in sorted_items[:3]]
        to_show = set(active).union(set(top_3))
        sorted_items = [(label, score) for label, score in sorted_items if label in to_show]

    # Display each instrument
    for label, score in sorted_items:
        threshold = thresholds.get(label, 0.5)
        is_active = binary.get(label, 0) == 1

        # Choose icon based on activation and score
        if is_active:
            icon = "✅" if score >= highlight_threshold else "✓"
        else:
            icon = "❌" if score >= highlight_threshold else "✗"

        # Print with formatting
        print(f"{label:<15} {score:.4f}    {threshold:.4f}    {icon}")

    print("=" * 60)


def main():
    """Command-line interface for adaptive threshold prediction."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to checkpoint file")
    p.add_argument("wav", help="Path to audio file")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml", help="Config file path")
    p.add_argument("--thresholds", default=None, help="Path to thresholds YAML file")
    p.add_argument("--show-all", action="store_true", help="Show all instruments, not just active ones")
    args = p.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load model
    model = load_model_from_checkpoint(args.ckpt, n_classes=len(LABELS))

    # Load thresholds
    thresholds = load_thresholds(args.thresholds)

    # Make prediction
    result = predict_with_adaptive_thresholds(model, args.wav, cfg, thresholds)
    result['wav_path'] = args.wav  # Add path for display

    # Display results
    display_prediction_results(result, show_all_instruments=args.show_all)

    return result


if __name__ == "__main__":
    main()
