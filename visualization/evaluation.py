#!/usr/bin/env python3
"""Comprehensive model evaluation with statistics and plots."""

import torch
import numpy as np
import pathlib
import yaml
import re
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

from data.dataset import MultiSTFTNpyDataset
from utils.model_loader import load_model_from_checkpoint
from var import LABELS



def extract_ground_truth_from_filename(filename):
    """Extract ground truth labels from IRMAS filename format."""
    irmas_to_label = {
        'cel': 'cello', 'cla': 'clarinet', 'flu': 'flute',
        'gac': 'acoustic_guitar', 'gel': 'acoustic_guitar',
        'org': 'organ', 'pia': 'piano', 'sax': 'saxophone',
        'tru': 'trumpet', 'vio': 'violin', 'voi': 'voice'
    }

    # Extract labels from filename like [sax][jaz_blu]1737__1.wav
    irmas_pattern = r'\[([a-z]{3})\]'
    irmas_matches = re.findall(irmas_pattern, filename)

    ground_truth = []
    for irmas_label in irmas_matches:
        if irmas_label in irmas_to_label:
            ground_truth.append(irmas_to_label[irmas_label])

    return ground_truth


def evaluate_model(model, test_dir, cfg, threshold=0.5):
    """
    Comprehensive model evaluation.

    Returns:
        Dictionary with all evaluation metrics and predictions
    """
    print(f"🔍 Loading test dataset from: {test_dir}")

    # Create test dataset
    test_dataset = MultiSTFTNpyDataset(test_dir, max_samples=None)

    if len(test_dataset) == 0:
        print("❌ No test data found!")
        return None

    print(f"📊 Found {len(test_dataset)} test samples")

    model.eval()
    all_predictions = []
    all_ground_truth = []
    all_probabilities = []
    file_results = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            try:
                specs, labels = test_dataset[i]

                # Get directory name for filename extraction
                dir_name = test_dataset.dirs[i].name

                # IMPORTANT FIX: Add batch dimension to each spectrogram
                # Convert from [1, H, W] to [1, 1, H, W]
                specs_batched = []
                for spec in specs:
                    if len(spec.shape) == 3:  # [1, H, W]
                        spec_batched = spec.unsqueeze(0)  # [1, 1, H, W]
                    else:  # Already has batch dimension
                        spec_batched = spec
                    specs_batched.append(spec_batched)

                # Forward pass
                preds = model(specs_batched).squeeze()

                # Apply sigmoid to get probabilities
                if len(preds.shape) == 0:  # Single sample case
                    preds = torch.sigmoid(preds.unsqueeze(0))
                else:
                    preds = torch.sigmoid(preds)

                probs = preds.numpy()
                binary_preds = (probs > threshold).astype(int)

                # Store results
                all_predictions.append(binary_preds)
                all_ground_truth.append(labels.numpy())
                all_probabilities.append(probs)

                # Extract ground truth from filename if available
                gt_from_filename = extract_ground_truth_from_filename(dir_name)

                file_results.append({
                    'filename': dir_name,
                    'predictions': probs,
                    'binary_predictions': binary_preds,
                    'ground_truth': labels.numpy(),
                    'ground_truth_from_filename': gt_from_filename
                })

                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(test_dataset)} samples...")

            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                # Print more debug info for the first few errors
                if i < 5:
                    import traceback
                    traceback.print_exc()
                continue

    if not all_predictions:
        print("❌ No successful predictions!")
        return None

    # Convert to numpy arrays
    y_true = np.array(all_ground_truth)
    y_pred = np.array(all_predictions)
    y_probs = np.array(all_probabilities)

    print(f"✅ Successfully evaluated {len(all_predictions)} samples")

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs)

    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'file_results': file_results,
        'threshold': threshold
    }

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}

    # Overall metrics
    metrics['exact_match_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = np.mean(y_true != y_pred)

    # Per-class metrics
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # Mean Average Precision
    try:
        metrics['map'] = average_precision_score(y_true, y_probs, average='macro')
    except:
        metrics['map'] = 0.0

    # Per-class detailed metrics
    per_class_metrics = {}
    for i, label in enumerate(LABELS):
        per_class_metrics[label] = {
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'support': y_true[:, i].sum()
        }

    metrics['per_class'] = per_class_metrics

    return metrics


def plot_evaluation_results(eval_results, save_dir="evaluation_plots"):
    """Create comprehensive evaluation plots."""
    pathlib.Path(save_dir).mkdir(exist_ok=True)

    y_true = eval_results['y_true']
    y_pred = eval_results['y_pred']
    y_probs = eval_results['y_probs']
    metrics = eval_results['metrics']

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Per-class Performance Bar Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Precision, Recall, F1 per class
    labels = LABELS
    precisions = [metrics['per_class'][label]['precision'] for label in labels]
    recalls = [metrics['per_class'][label]['recall'] for label in labels]
    f1s = [metrics['per_class'][label]['f1'] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    ax1.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax1.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)

    ax1.set_xlabel('Instrument Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Support (number of samples) per class
    supports = [metrics['per_class'][label]['support'] for label in labels]
    ax2.bar(x, supports, alpha=0.7, color='coral')
    ax2.set_xlabel('Instrument Classes')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Test Set Support per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Confusion Matrix Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # For multilabel, we'll create a confusion matrix for each class
    for idx, label in enumerate(labels[:4]):  # Show first 4 classes
        if idx < 4:
            row = idx // 2
            col = idx % 2
            cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[row, col],
                        xticklabels=['Not ' + label, label],
                        yticklabels=['Not ' + label, label])
            axes[row, col].set_title(f'Confusion Matrix: {label}')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Probability Distribution Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot probability distributions for each class
    for i, label in enumerate(labels):
        positive_probs = y_probs[y_true[:, i] == 1, i]
        negative_probs = y_probs[y_true[:, i] == 0, i]

        if len(positive_probs) > 0:
            ax.hist(positive_probs, bins=20, alpha=0.5, label=f'{label} (positive)', density=True)
        if len(negative_probs) > 0:
            ax.hist(negative_probs, bins=20, alpha=0.3, label=f'{label} (negative)', density=True)

    ax.axvline(x=eval_results['threshold'], color='red', linestyle='--',
               label=f'Threshold ({eval_results["threshold"]})')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distributions by Class and True Label')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/probability_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Overall Metrics Summary
    fig, ax = plt.subplots(figsize=(10, 6))

    overall_metrics = {
        'Exact Match\nAccuracy': metrics['exact_match_accuracy'],
        'Hamming\nLoss': metrics['hamming_loss'],
        'Precision\n(Macro)': metrics['precision_macro'],
        'Recall\n(Macro)': metrics['recall_macro'],
        'F1-Score\n(Macro)': metrics['f1_macro'],
        'Mean Average\nPrecision': metrics['map']
    }

    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())

    bars = ax.bar(metric_names, metric_values, alpha=0.7, color='skyblue', edgecolor='navy')
    ax.set_ylabel('Score')
    ax.set_title('Overall Model Performance Metrics')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/overall_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Plots saved to {save_dir}/")


def print_evaluation_summary(eval_results):
    """Print a comprehensive evaluation summary."""
    metrics = eval_results['metrics']

    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"📊 Overall Performance:")
    print(f"   • Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"   • Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"   • Mean Average Precision: {metrics['map']:.4f}")

    print(f"\n📈 Macro-Averaged Metrics:")
    print(f"   • Precision: {metrics['precision_macro']:.4f}")
    print(f"   • Recall: {metrics['recall_macro']:.4f}")
    print(f"   • F1-Score: {metrics['f1_macro']:.4f}")

    print(f"\n📈 Micro-Averaged Metrics:")
    print(f"   • Precision: {metrics['precision_micro']:.4f}")
    print(f"   • Recall: {metrics['recall_micro']:.4f}")
    print(f"   • F1-Score: {metrics['f1_micro']:.4f}")

    print(f"\n🎼 Per-Class Performance:")
    print(f"{'Instrument':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)

    for label in LABELS:
        class_metrics = metrics['per_class'][label]
        print(f"{label:<15} {class_metrics['precision']:<10.4f} "
              f"{class_metrics['recall']:<10.4f} {class_metrics['f1']:<10.4f} "
              f"{class_metrics['support']:<8.0f}")

    # Find best and worst performing classes
    f1_scores = [(label, metrics['per_class'][label]['f1']) for label in LABELS]
    f1_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Best Performing Classes:")
    for label, f1 in f1_scores[:3]:
        print(f"   • {label}: F1 = {f1:.4f}")

    print(f"\n⚠️  Worst Performing Classes:")
    for label, f1 in f1_scores[-3:]:
        print(f"   • {label}: F1 = {f1:.4f}")

    print("=" * 60)


def run_comprehensive_evaluation(checkpoint_path, test_dir, config_path, threshold=0.5):
    """
    Run complete evaluation pipeline.
    """
    print("🚀 Starting comprehensive model evaluation...")

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load model
    print(f"📥 Loading model from {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, len(LABELS), cfg)

    # Run evaluation
    eval_results = evaluate_model(model, test_dir, cfg, threshold)

    if eval_results is None:
        print("❌ Evaluation failed!")
        return None

    # Print summary
    print_evaluation_summary(eval_results)

    # Create plots
    print("📊 Creating evaluation plots...")
    plot_evaluation_results(eval_results)

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("test_dir", help="Path to test data directory")
    parser.add_argument("--config", default="configs/panns_enhanced.yaml", help="Config file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")

    args = parser.parse_args()

    run_comprehensive_evaluation(args.checkpoint, args.test_dir, args.config, args.threshold)