#!/usr/bin/env python3
"""Unified model loading utilities for both regular and PANNs-enhanced models."""

import torch
from pathlib import Path
import yaml

from models.panns_enhanced import MultiSTFTCNN_WithPANNs
from data.download_pnn import download_panns_checkpoint
from var import LABELS


def detect_model_architecture(checkpoint_path):
    """Detect model architecture by examining checkpoint state dict keys.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        str: 'panns' or 'regular'
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Check for architecture-specific keys
    has_feature_extractors = any('feature_extractors' in key for key in state_dict.keys())
    has_fusion = any('fusion' in key for key in state_dict.keys())
    has_branches = any('branches' in key for key in state_dict.keys())

    print(f"🔍 Checkpoint analysis:")
    print(f"  - feature_extractors keys: {has_feature_extractors}")
    print(f"  - fusion keys: {has_fusion}")
    print(f"  - branches keys: {has_branches}")

    if has_feature_extractors and has_fusion and not has_branches:
        return 'panns'
    elif has_branches and not has_feature_extractors:
        return 'regular'
    else:
        # Fallback to filename detection
        filename = Path(checkpoint_path).name.lower()
        if 'panns' in filename:
            print("  - Using filename-based detection: PANNs")
            return 'panns'
        else:
            print("  - Using filename-based detection: Regular")
            return 'regular'


def clean_state_dict(state_dict):
    """Clean checkpoint state dictionary by removing prefix and handling special keys.

    Args:
        state_dict: Original state dictionary

    Returns:
        dict: Cleaned state dictionary
    """
    cleaned_dict = {}

    for key, value in state_dict.items():
        # Remove "model." prefix if present (PyTorch Lightning adds this)
        if key.startswith("model."):
            new_key = key[6:]  # Remove "model." prefix
        else:
            new_key = key

        cleaned_dict[new_key] = value

    return cleaned_dict


def load_model(checkpoint_path, cfg=None, n_classes=None, force_architecture=None):
    """Universal model loading function that handles both model types.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: Optional configuration dictionary
        n_classes: Number of output classes (default: uses len(LABELS))
        force_architecture: Override detection with 'panns' or 'regular'

    Returns:
        Loaded model
    """
    # Use default number of classes if not specified
    if n_classes is None:
        n_classes = len(LABELS)

    # Detect architecture if not forced
    if force_architecture:
        architecture = force_architecture
        print(f"🔧 Forcing architecture: {architecture}")
    else:
        architecture = detect_model_architecture(checkpoint_path)
        print(f"🔍 Detected architecture: {architecture}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean state dict
    cleaned_state_dict = clean_state_dict(state_dict)

    print(f"🏗️ Creating PANNs-enhanced model")
    panns_path = download_panns_checkpoint()
    model = MultiSTFTCNN_WithPANNs(
        n_classes=n_classes,
        pretrained_path=panns_path,
        freeze_backbone=False
    )
    # Load with error handling
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("✅ Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed, trying non-strict: {str(e)[:150]}...")
        try:
            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
            print(f"✅ Model loaded with strict=False")
            if missing:
                print(f"   Missing keys: {len(missing)} (first few): {missing[:3]}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)} (first few): {unexpected[:3]}")
        except Exception as e2:
            print(f"❌ Loading failed completely: {e2}")
            raise e2

    model.eval()
    return model


# Backwards compatibility aliases
def load_model_from_checkpoint(ckpt_path, n_classes, cfg=None):
    """Backwards compatibility function for existing code."""
    return load_model(ckpt_path, cfg=cfg, n_classes=n_classes)


def load_panns_model_from_checkpoint(checkpoint_path, n_classes, cfg=None):
    """Backwards compatibility function for existing code."""
    return load_model(checkpoint_path, cfg=cfg, n_classes=n_classes, force_architecture='panns')
