import torch
import yaml
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from models.panns_enhanced import MultiSTFTCNN_WithPANNs
from var import LABELS
from data.download_pnn import download_panns_checkpoint


def load_panns_model_from_checkpoint(checkpoint_path, n_classes, cfg=None):
    """Load a trained PANNs-enhanced model from checkpoint."""
    # Use the unified model loader
    from utils.model_loader import load_model
    return load_model(checkpoint_path, cfg=cfg, n_classes=n_classes, force_architecture='panns')


def compare_model_performance(test_files, original_model, panns_model, cfg, threshold=0.6):
    """Compare performance between original and PANNs-enhanced models."""
    from inference.predict import extract_features, predict_with_ground_truth

    results = {
        "original": {"correct": 0, "total": 0, "per_sample": []},
        "panns": {"correct": 0, "total": 0, "per_sample": []}
    }

    for i, wav_file in enumerate(test_files):
        print(f"\nTesting file {i+1}/{len(test_files)}: {pathlib.Path(wav_file).name}")

        # Extract features once
        wav_path = str(wav_file)
        specs_list = extract_features(wav_path, cfg)

        # Get ground truth
        gt_result = predict_with_ground_truth(original_model, wav_path, cfg, show_ground_truth=True)
        ground_truth = gt_result.get("ground_truth", [])

        # Test with original model
        with torch.no_grad():
            orig_preds = original_model(specs_list).squeeze()
            orig_probs = torch.sigmoid(orig_preds).numpy()

        # Test with PANNs model
        with torch.no_grad():
            panns_preds = panns_model(specs_list).squeeze()

        # Convert to binary predictions using threshold
        orig_binary = (orig_probs >= threshold).astype(int)
        panns_binary = (panns_preds.numpy() >= threshold).astype(int)

        # Calculate accuracy for each model
        orig_correct = 0
        panns_correct = 0

        for j, label in enumerate(LABELS):
            # Check if prediction matches ground truth
            label_in_gt = label in ground_truth
            orig_pred = orig_binary[j] == 1
            panns_pred = panns_binary[j] == 1

            if (orig_pred == label_in_gt):
                orig_correct += 1

            if (panns_pred == label_in_gt):
                panns_correct += 1

        # Update results
        results["original"]["correct"] += orig_correct
        results["original"]["total"] += len(LABELS)
        results["original"]["per_sample"].append(orig_correct / len(LABELS))

        results["panns"]["correct"] += panns_correct
        results["panns"]["total"] += len(LABELS)
        results["panns"]["per_sample"].append(panns_correct / len(LABELS))

        # Display comparison
        print(f"Ground truth: {ground_truth}")
        print(f"Original model score: {orig_correct}/{len(LABELS)} = {orig_correct/len(LABELS):.1%}")
        print(f"PANNs model score: {panns_correct}/{len(LABELS)} = {panns_correct/len(LABELS):.1%}")

        if panns_correct > orig_correct:
            print(f"✅ PANNs model performed BETTER on this sample!")
        elif panns_correct < orig_correct:
            print(f"⚠️ PANNs model performed WORSE on this sample.")
        else:
            print(f"🟰 Both models performed the same on this sample.")

    # Calculate overall results
    results["original"]["accuracy"] = results["original"]["correct"] / results["original"]["total"]
    results["panns"]["accuracy"] = results["panns"]["correct"] / results["panns"]["total"]

    # Display final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON:")
    print(f"Original model accuracy: {results['original']['correct']}/{results['original']['total']} = {results['original']['accuracy']:.1%}")
    print(f"PANNs model accuracy: {results['panns']['correct']}/{results['panns']['total']} = {results['panns']['accuracy']:.1%}")

    improvement = (results['panns']['accuracy'] - results['original']['accuracy']) * 100
    print(f"Improvement: {improvement:.1f} percentage points")
    print("="*80)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(["Original Model", "PANNs-Enhanced Model"], 
            [results["original"]["accuracy"], results["panns"]["accuracy"]], 
            color=["blue", "green"])
    plt.ylabel("Accuracy")
    plt.title(f"Model Performance Comparison (threshold={threshold})")
    plt.ylim(0, 1)

    for i, model in enumerate(["original", "panns"]):
        plt.text(i, results[model]["accuracy"] + 0.02, 
                 f"{results[model]['accuracy']:.1%}", 
                 ha='center')

    plt.tight_layout()
    plt.show()

    return results
