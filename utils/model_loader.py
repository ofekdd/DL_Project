#!/usr/bin/env python3
"""Unified model loading utilities for both regular and PANNs-enhanced models."""

import torch
from pathlib import Path
import yaml

from models.multi_stft_cnn import MultiSTFTCNN
from models.panns_enhanced import MultiSTFTCNN_WithPANNs
from models.log_mel_cnn import LogMelCNN, LogMelCNN_WithPANNs
from data.download_pnn import download_panns_checkpoint
from var import LABELS


def detect_model_architecture(checkpoint_path):
    """Detect model architecture by examining checkpoint state dict keys.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        tuple: (model_type, use_log_mel)
        model_type: 'panns' or 'regular'
        use_log_mel: True if log-mel model, False if multi-STFT model
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
    has_features = any('features.' in key for key in state_dict.keys())  # LogMelCNN specific

    print(f"🔍 Checkpoint analysis:")
    print(f"  - feature_extractors keys: {has_feature_extractors}")
    print(f"  - fusion keys: {has_fusion}")
    print(f"  - branches keys: {has_branches}")
    print(f"  - features keys: {has_features}")

    # Determine model type (panns or regular)
    model_type = None
    use_log_mel = False
    
    # Check for log-mel models first
    if has_features and has_fusion and not has_branches and not has_feature_extractors:
        # LogMelCNN_WithPANNs has a single feature_extractor (not plural) and fusion
        model_type = 'panns'
        use_log_mel = True
    elif has_features and not has_fusion and not has_branches and not has_feature_extractors:
        # LogMelCNN has features but no fusion, branches, or feature_extractors
        model_type = 'regular'
        use_log_mel = True
    # Then check for multi-STFT models
    elif has_feature_extractors and has_fusion and not has_branches:
        # MultiSTFTCNN_WithPANNs has feature_extractors (plural) and fusion
        model_type = 'panns'
        use_log_mel = False
    elif has_branches and not has_feature_extractors:
        # MultiSTFTCNN has branches but no feature_extractors
        model_type = 'regular'
        use_log_mel = False
    
    # If we couldn't determine the model type, fall back to filename detection
    if model_type is None:
        filename = Path(checkpoint_path).name.lower()
        if 'logmel' in filename:
            use_log_mel = True
            if 'panns' in filename:
                model_type = 'panns'
            else:
                model_type = 'regular'
        else:
            use_log_mel = False
            if 'panns' in filename:
                model_type = 'panns'
            else:
                model_type = 'regular'
        print(f"  - Using filename-based detection: {model_type}, log-mel: {use_log_mel}")
    
    return model_type, use_log_mel


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


def load_model(checkpoint_path, cfg=None, n_classes=None, force_architecture=None, force_log_mel=None):
    """Universal model loading function that handles all model types.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: Optional configuration dictionary
        n_classes: Number of output classes (default: uses len(LABELS))
        force_architecture: Override detection with 'panns' or 'regular'
        force_log_mel: Override detection with True (log-mel) or False (multi-STFT)

    Returns:
        Loaded model
    """
    # Use default number of classes if not specified
    if n_classes is None:
        n_classes = len(LABELS)

    # Detect architecture if not forced
    if force_architecture:
        architecture = force_architecture
        use_log_mel = force_log_mel if force_log_mel is not None else False
        print(f"🔧 Forcing architecture: {architecture}, log-mel: {use_log_mel}")
    else:
        architecture, use_log_mel = detect_model_architecture(checkpoint_path)
        if force_log_mel is not None:
            use_log_mel = force_log_mel
            print(f"🔧 Forcing log-mel: {use_log_mel}")
        print(f"🔍 Detected architecture: {architecture}, log-mel: {use_log_mel}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean state dict
    cleaned_state_dict = clean_state_dict(state_dict)

    # Create appropriate model based on architecture and log-mel flag
    if architecture == 'panns':
        panns_path = download_panns_checkpoint()
        if use_log_mel:
            print(f"🏗️ Creating LogMelCNN_WithPANNs model")
            model = LogMelCNN_WithPANNs(
                n_classes=n_classes,
                pretrained_path=panns_path,
                freeze_backbone=False
            )
        else:
            print(f"🏗️ Creating MultiSTFTCNN_WithPANNs model")
            model = MultiSTFTCNN_WithPANNs(
                n_classes=n_classes,
                pretrained_path=panns_path,
                freeze_backbone=False
            )
    else:
        if use_log_mel:
            print(f"🏗️ Creating LogMelCNN model")
            model = LogMelCNN(
                n_classes=n_classes,
                in_channels=1
            )
        else:
            print(f"🏗️ Creating MultiSTFTCNN model")
            # Use configuration if provided
            if cfg:
                n_branches = cfg.get('n_branches', 3)
                branch_output_dim = cfg.get('branch_output_dim', 128)
            else:
                n_branches = 3
                branch_output_dim = 128

            model = MultiSTFTCNN(
                n_classes=n_classes, 
                n_branches=n_branches, 
                branch_output_dim=branch_output_dim
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
def load_model_from_checkpoint(ckpt_path, n_classes, cfg=None, use_log_mel=False):
    """Backwards compatibility function for existing code."""
    return load_model(ckpt_path, cfg=cfg, n_classes=n_classes, force_log_mel=use_log_mel)


def load_panns_model_from_checkpoint(checkpoint_path, n_classes, cfg=None, use_log_mel=False):
    """Backwards compatibility function for existing code."""
    return load_model(checkpoint_path, cfg=cfg, n_classes=n_classes, force_architecture='panns', force_log_mel=use_log_mel)


# New functions for log-mel models
def load_log_mel_model_from_checkpoint(ckpt_path, n_classes, cfg=None):
    """Load a LogMelCNN model from checkpoint."""
    return load_model(ckpt_path, cfg=cfg, n_classes=n_classes, force_log_mel=True)


def load_log_mel_panns_model_from_checkpoint(checkpoint_path, n_classes, cfg=None):
    """Load a LogMelCNN_WithPANNs model from checkpoint."""
    return load_model(checkpoint_path, cfg=cfg, n_classes=n_classes, force_architecture='panns', force_log_mel=True)
