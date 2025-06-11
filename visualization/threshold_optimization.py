#!/usr/bin/env python3
"""Threshold optimization for instrument classification.

This module helps find optimal classification thresholds for each instrument
to maximize different metrics (F1 score or balanced accuracy).
"""

import argparse
import yaml
import torch
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import f1_score, balanced_accuracy_score

# Import project modules
from inference.predict import load_model_from_checkpoint
from var import LABELS
from data.dataset import create_dataloaders


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate classification metrics for binary predictions.

    Args:
        y_true: Binary ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate metrics
    metrics = {}

    # Overall metrics
    metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro')
    metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro')  
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true.ravel(), y_pred_binary.ravel())

    # Per-class metrics
    for i, label in enumerate(LABELS):
        metrics[f'f1_{label}'] = f1_score(y_true[:, i], y_pred_binary[:, i])

    return metrics


def optimize_threshold_for_instrument(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    instrument_idx: int,
    metric: str = 'f1',
    thresholds: np.ndarray = np.arange(0.1, 0.9, 0.05)
) -> Tuple[float, float]:
    """Find optimal threshold for a specific instrument.

    Args:
        y_true: Binary ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        instrument_idx: Index of the instrument to optimize
        metric: Metric to optimize ('f1' or 'balanced')
        thresholds: Array of thresholds to try

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    best_score = 0
    best_threshold = 0.5

    for threshold in thresholds:
        # Apply threshold only to this instrument
        y_pred_binary = (y_pred[:, instrument_idx] >= threshold).astype(int)

        # Calculate metric
        if metric == 'f1':
            score = f1_score(y_true[:, instrument_idx], y_pred_binary)
        elif metric == 'balanced':
            score = balanced_accuracy_score(y_true[:, instrument_idx], y_pred_binary)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Update if better
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def find_optimal_thresholds(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    metric: str = 'f1',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """Find optimal thresholds for all instruments.

    Args:
        model: Trained model
        val_loader: Validation data loader
        metric: Metric to optimize ('f1' or 'balanced')
        device: Device to run inference on

    Returns:
        Dictionary mapping instrument names to optimal thresholds
    """
    model.to(device)
    model.eval()

    # Collect all predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Handle multiple input formats
            if isinstance(batch[0], list):  # Multi-STFT format
                x_list = batch[0]
                x_list = [x.to(device) for x in x_list]
                y = batch[1].to(device)
                preds = model(x_list)
            else:  # Standard format
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                preds = model(x)

            # Collect batch results
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Concatenate batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    # Optimize threshold for each instrument
    thresholds = {}
    for i, label in enumerate(LABELS):
        optimal_threshold, best_score = optimize_threshold_for_instrument(
            y_true, y_pred, i, metric)
        thresholds[label] = float(optimal_threshold)  # Convert to Python float for YAML
        print(f"Optimal {metric} threshold for {label}: {optimal_threshold:.3f} (score: {best_score:.3f})")

    return thresholds


def save_thresholds(thresholds: Dict[str, float], output_path: str, metric: str):
    """Save thresholds to YAML file.

    Args:
        thresholds: Dictionary mapping instrument names to thresholds
        output_path: Path to save YAML file
        metric: Metric used for optimization ('f1' or 'balanced')
    """
    # Create output directory if it doesn't exist
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create YAML data
    data = {
        'thresholds': thresholds,
        'metric': metric,
        'description': f'Optimal thresholds to maximize {metric} score'
    }

    # Save to file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Saved optimal thresholds to {output_path}")


def plot_threshold_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Dict[str, float],
    default_threshold: float = 0.5,
    output_path: Optional[str] = None
):
    """Plot comparison of default vs. optimized thresholds.

    Args:
        y_true: Binary ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        thresholds: Dictionary mapping instrument names to thresholds
        default_threshold: Default threshold to compare against
        output_path: Path to save plot, or None to display
    """
    # Calculate metrics with default threshold
    default_metrics = calculate_metrics(y_true, y_pred, default_threshold)

    # Apply optimized thresholds
    y_pred_optimized = np.zeros_like(y_pred, dtype=int)
    for i, label in enumerate(LABELS):
        threshold = thresholds.get(label, default_threshold)
        y_pred_optimized[:, i] = (y_pred[:, i] >= threshold).astype(int)

    # Calculate per-class F1 with optimized thresholds
    optimized_f1 = []
    default_f1 = []
    for i, label in enumerate(LABELS):
        opt_f1 = f1_score(y_true[:, i], y_pred_optimized[:, i])
        def_f1 = default_metrics[f'f1_{label}']
        optimized_f1.append(opt_f1)
        default_f1.append(def_f1)

    # Plot comparison
    plt.figure(figsize=(15, 8))
    x = np.arange(len(LABELS))
    width = 0.35

    plt.bar(x - width/2, default_f1, width, label=f'Default ({default_threshold})')
    plt.bar(x + width/2, optimized_f1, width, label='Optimized')

    plt.xlabel('Instrument')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison: Default vs. Optimized Thresholds')
    plt.xticks(x, LABELS, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Saved comparison plot to {output_path}")
    else:
        plt.show()


def main():
    """Command-line interface for threshold optimization."""
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to checkpoint file")
    p.add_argument("--val-dir", default="data/processed/val", help="Validation data directory")
    p.add_argument("--config", default="configs/default.yaml", help="Config file path")
    p.add_argument("--metric", choices=['f1', 'balanced'], default='f1', 
                   help="Metric to optimize (f1 or balanced accuracy)")
    p.add_argument("--output", default=None, 
                   help="Output YAML file path (default: configs/optimal_thresholds_{metric}.yaml)")
    p.add_argument("--plot", action="store_true", help="Generate comparison plot")
    args = p.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = f"configs/optimal_thresholds_{args.metric}.yaml"

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load model
    model = load_model_from_checkpoint(args.ckpt, n_classes=len(LABELS))

    # Create validation dataloader
    _, val_loader = create_dataloaders(
        train_dir="data/processed/train",  # Not used but required
        val_dir=args.val_dir,
        batch_size=cfg.get('batch_size', 32),
        num_workers=cfg.get('num_workers', 4),
        use_multi_stft=True  # Assuming Multi-STFT model
    )

    print(f"Finding optimal thresholds to maximize {args.metric} score...")
    thresholds = find_optimal_thresholds(model, val_loader, args.metric)

    # Save thresholds
    save_thresholds(thresholds, args.output, args.metric)

    # Generate comparison plot if requested
    if args.plot:
        # Get validation data again
        model.eval()
        device = next(model.parameters()).device

        # Collect predictions and labels
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch[0], list):  # Multi-STFT format
                    x_list = batch[0]
                    x_list = [x.to(device) for x in x_list]
                    y = batch[1].to(device)
                    preds = model(x_list)
                else:  # Standard format
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    preds = model(x)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)

        # Generate plot
        plot_path = args.output.replace('.yaml', '.png')
        plot_threshold_comparison(y_true, y_pred, thresholds, output_path=plot_path)

    print("✅ Threshold optimization complete!")


if __name__ == "__main__":
    main()
