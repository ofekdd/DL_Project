import torch
import numpy as np
from var import LABELS

def create_weighted_target(label_vector, alpha=0.80):
    """
    Creates a weighted target probability vector for multi-label classification.

    The function distributes probability mass between positive and negative classes:
    - Positive classes (1s) share alpha (default 80%) of probability mass equally
    - Negative classes (0s) share the remaining (1-alpha) probability mass equally

    Args:
        label_vector: A binary tensor of shape (n_classes,) indicating presence (1) or absence (0)
                     of each instrument class
        alpha: Proportion of probability mass to assign to positive classes (default: 0.80)

    Returns:
        A tensor of shape (n_classes,) with weighted probability values that sum to 1.0
    """
    # Ensure label_vector is a tensor
    if not isinstance(label_vector, torch.Tensor):
        label_vector = torch.tensor(label_vector)

    # Calculate beta (complement of alpha)
    beta = 1.0 - alpha

    # Get total number of classes (L)
    L = len(label_vector)

    # Count positive classes (k)
    k = label_vector.sum().item()

    # Handle edge cases
    if k == 0:  # No positive classes
        # Distribute evenly across all classes
        return torch.ones(L) / L

    if k == L:  # All classes are positive
        # Distribute evenly across all classes
        return torch.ones(L) / L

    # Create the target probability vector
    weight_vector = torch.zeros_like(label_vector, dtype=torch.float32)

    # Assign weights to positive classes: α/k for each
    pos_weight = alpha / k
    weight_vector[label_vector == 1] = pos_weight

    # Assign weights to negative classes: β/(L-k) for each
    neg_weight = beta / (L - k)
    weight_vector[label_vector == 0] = neg_weight

    # Ensure the weights sum to 1.0 (handling potential floating-point errors)
    weight_vector = weight_vector / weight_vector.sum()

    return weight_vector

def create_weighted_batch_targets(label_batch, alpha=0.80):
    """
    Apply the weighted target function to a batch of label vectors.

    Args:
        label_batch: Tensor of shape (batch_size, n_classes) with binary labels
        alpha: Proportion of probability mass to assign to positive classes

    Returns:
        Tensor of shape (batch_size, n_classes) with weighted probability targets
    """
    batch_size = label_batch.shape[0]
    weighted_batch = torch.zeros_like(label_batch, dtype=torch.float32)

    for i in range(batch_size):
        weighted_batch[i] = create_weighted_target(label_batch[i], alpha)

    return weighted_batch


def example_target_weighting():
    """
    Demonstrate the target weighting scheme with a concrete example.
    """
    print("Example of weighted target probability vector:")
    print(f"Total classes (L): {len(LABELS)} = {LABELS}")

    # Create an example with 2 positive labels (as in your description)
    example_label = torch.zeros(len(LABELS))
    example_label[0] = 1  # cello
    example_label[5] = 1  # piano

    # Get the positive label names
    pos_indices = (example_label == 1).nonzero().flatten().tolist()
    pos_labels = [LABELS[i] for i in pos_indices]

    print(f"Positive labels (k=2): {pos_labels}")

    # Default 80/20 split
    weighted_target = create_weighted_target(example_label, alpha=0.80)

    print("\nWith alpha=0.80 (80/20 split):")
    for i, (label, weight) in enumerate(zip(LABELS, weighted_target)):
        status = "TRUE" if example_label[i] == 1 else "false"
        print(f"  {label:15s} ({status:5s}): {weight:.6f}")

    print(f"Sum of weights: {weighted_target.sum().item():.6f}")

    # Alternative 70/30 split
    alt_weighted_target = create_weighted_target(example_label, alpha=0.70)

    print("\nWith alpha=0.70 (70/30 split):")
    for i, (label, weight) in enumerate(zip(LABELS, alt_weighted_target)):
        status = "TRUE" if example_label[i] == 1 else "false"
        print(f"  {label:15s} ({status:5s}): {weight:.6f}")

    print(f"Sum of weights: {alt_weighted_target.sum().item():.6f}")


if __name__ == "__main__":
    example_target_weighting()
