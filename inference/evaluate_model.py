#!/usr/bin/env python3
"""Evaluate model performance with threshold-based evaluation."""
import argparse
import yaml
import pathlib
import torch
import numpy as np
from pathlib import Path

#from models.multi_stft_cnn import MultiSTFTCNN
from inference.predict import extract_features, predict_with_ground_truth
from var import LABELS


def evaluate_with_threshold_clarity(model, wav_path, cfg, threshold=0.6, show_details=True):
    """Enhanced evaluation with clear binary success/failure per instrument."""
    from inference.predict import predict
    from var import LABELS

    # Extract features
    features = extract_features(wav_path, cfg)

    with torch.no_grad():
        preds = model(features).squeeze()
        probs = torch.sigmoid(preds).numpy()

    # Get predictions and ground truth
    result = predict_with_ground_truth(model, wav_path, cfg, show_ground_truth=True)
    predictions = result["predictions"]
    ground_truth = result.get("ground_truth", [])

    # Convert to binary vectors
    pred_binary = {label: 1 if predictions[label] >= threshold else 0 for label in LABELS}
    true_binary = {label: 1 if label in ground_truth else 0 for label in LABELS}

    # Calculate per-instrument results
    instrument_results = {}
    correct_count = 0

    for label in LABELS:
        pred_val = pred_binary[label]
        true_val = true_binary[label]
        is_correct = (pred_val == true_val)

        if is_correct:
            correct_count += 1

        # Determine the type of result for clearer feedback
        if true_val == 1 and pred_val == 1:
            result_type = "TRUE_POSITIVE"
            emoji = "✅"
        elif true_val == 0 and pred_val == 0:
            result_type = "TRUE_NEGATIVE"
            emoji = "✅"
        elif true_val == 1 and pred_val == 0:
            result_type = "FALSE_NEGATIVE"  # Missed detection
            emoji = "❌"
        else:  # true_val == 0 and pred_val == 1
            result_type = "FALSE_POSITIVE"  # False alarm
            emoji = "❌"

        instrument_results[label] = {
            'predicted': pred_val,
            'actual': true_val,
            'correct': is_correct,
            'type': result_type,
            'confidence': predictions[label],
            'emoji': emoji
        }

    # Calculate overall accuracy
    overall_accuracy = correct_count / len(LABELS)

    if show_details:
        print(f"\n File: {Path(wav_path).name}")
        print(f" Ground Truth: {ground_truth}")
        print(f" Predicted (≥{threshold}): {[label for label, val in pred_binary.items() if val == 1]}")
        print(f" Overall Accuracy: {correct_count}/{len(LABELS)} = {overall_accuracy:.1%}")
        print("\n" + "="*70)
        print("PER-INSTRUMENT BREAKDOWN:")
        print("="*70)

        # Group results for clearer display
        correct_instruments = []
        errors = []

        for label in LABELS:
            res = instrument_results[label]
            if res['correct']:
                correct_instruments.append(f"{res['emoji']} {label}")
            else:
                error_desc = f"{res['emoji']} {label}: {res['type']} (conf: {res['confidence']:.3f})"
                errors.append(error_desc)

        print("✅ CORRECT PREDICTIONS:")
        for item in correct_instruments:
            print(f"  {item}")

        if errors:
            print("\n❌ ERRORS:")
            for error in errors:
                print(f"  {error}")

        print("="*70)

    return {
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'total_count': len(LABELS),
        'instrument_results': instrument_results,
        'threshold': threshold,
        'sample_score': f"{correct_count}/{len(LABELS)}"
    }


def run_comprehensive_threshold_evaluation(model, test_files, cfg, threshold=0.6):
    """Run evaluation on multiple files with clear aggregated results."""
    all_results = []
    total_correct = 0
    total_predictions = 0

    # Per-instrument statistics
    instrument_stats = {label: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for label in LABELS}

    print(f"\n RUNNING COMPREHENSIVE EVALUATION (threshold={threshold})")
    print("="*80)

    for i, wav_file in enumerate(test_files):
        print(f"\n Sample {i+1}/{len(test_files)}")

        result = evaluate_with_threshold_clarity(model, str(wav_file), cfg, threshold, show_details=True)
        all_results.append(result)

        total_correct += result['correct_count']
        total_predictions += result['total_count']

        # Update per-instrument stats
        for label, res in result['instrument_results'].items():
            if res['type'] == 'TRUE_POSITIVE':
                instrument_stats[label]['tp'] += 1
            elif res['type'] == 'TRUE_NEGATIVE':
                instrument_stats[label]['tn'] += 1
            elif res['type'] == 'FALSE_POSITIVE':
                instrument_stats[label]['fp'] += 1
            elif res['type'] == 'FALSE_NEGATIVE':
                instrument_stats[label]['fn'] += 1

    # Calculate final metrics
    overall_accuracy = total_correct / total_predictions

    print(f"\n FINAL RESULTS (threshold={threshold})")
    print("="*80)
    print(f" Overall Accuracy: {total_correct}/{total_predictions} = {overall_accuracy:.1%}")
    print(f" Average per-sample: {overall_accuracy:.1%}")

    # Per-instrument performance
    print(f"\n PER-INSTRUMENT PERFORMANCE:")
    print("-"*80)

    for label in LABELS:
        stats = instrument_stats[label]

        # Calculate metrics
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Accuracy for this specific instrument
        inst_accuracy = (stats['tp'] + stats['tn']) / len(test_files)

        print(f"{label:<15} | Acc: {inst_accuracy:.1%} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")

    return {
        'overall_accuracy': overall_accuracy,
        'threshold': threshold,
        'per_instrument_stats': instrument_stats,
        'all_sample_results': all_results
    }


from utils.model_loader import load_model

def load_model_from_checkpoint(ckpt_path, n_classes, cfg=None):
    model = load_model(ckpt_path, n_classes=n_classes)
    model.eval()
    return model



def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to checkpoint file")
    p.add_argument("test_dir", help="Directory containing test WAV files")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml", help="Config file path")
    p.add_argument("--threshold", type=float, default=0.6, help="Classification threshold")
    p.add_argument("--limit", type=int, default=5, help="Number of test files to evaluate")
    args = p.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load model
    model = load_model_from_checkpoint(args.ckpt, n_classes=len(LABELS), cfg=cfg)

    # Get test files
    test_dir = pathlib.Path(args.test_dir)
    wav_files = list(test_dir.rglob("*.wav"))[:args.limit]

    if not wav_files:
        print(f"No WAV files found in {test_dir}")
        return

    print(f"Found {len(wav_files)} test files")

    # Run evaluation
    run_comprehensive_threshold_evaluation(model, wav_files, cfg, threshold=args.threshold)


if __name__ == "__main__":
    main()
