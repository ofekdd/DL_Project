# Multi-Instrument Recognition System
# Instrument Recognition in Musical Audio Signals (IRMAS)

## Overview
This project implements a deep learning model for recognizing musical instruments in audio recordings, using the IRMAS dataset. The model combines multi-scale STFT spectrograms with transfer learning from PANNs (Pre-trained Audio Neural Networks).

## Key Components

### Data Processing
- Converts audio files to multi-band spectrograms optimized for different frequency ranges:
  - Low frequencies (0-1000Hz): Long window (1024)
  - Mid frequencies (1000-4000Hz): Medium window (512)
  - High frequencies (4000-11025Hz): Short window (256)

### Models
1. **Basic Model**: Multi-branch CNN processing different frequency bands
2. **PANNs-Enhanced Model**: Transfer learning from AudioSet pretrained models

## Training Process

### Two-Phase Training for PANNs Model
1. **Phase 1**: Train only the fusion and classifier layers with frozen PANNs backbone
   - Higher learning rate (0.001)
   - Preserves pretrained knowledge
2. **Phase 2**: Fine-tune the entire model including PANNs backbone
   - Lower learning rate (0.0001)
   - Adapts pretrained features to our task

### Dataset Sizes
- Train: 10,728 samples
- Validation: 2,682 samples
- Test: 2,874 samples

## Performance Notes

### Optimization Tips
- **Multi-label Classification**: The model now properly handles multiple instruments in one sample using BCE loss and multilabel metrics
- **Batch Size**: Using 16 provides a good balance between speed and stability
- **Training Phases**: The two-phase approach helps preserve pretrained knowledge while adapting to our task

### Interpretation
- The 671 batches seen in training logs = 10,728 samples ÷ 16 batch size (rounded up)
- F1 and Accuracy now use multilabel metrics for proper evaluation
- Expect gradual improvements over multiple epochs during fine-tuning

## Usage
```bash
# Preprocess data
python data/preprocess.py --in_dir path/to/IRMAS --out_dir data/processed

# Train PANNs-enhanced model
python training/panns_train.py --config configs/panns_enhanced.yaml
```
A deep learning system for recognizing multiple instruments in audio recordings.

## Features

- Multi-label classification of 11 instrument classes
- Multi-band, multi-resolution spectrogram analysis
- PANNs-enhanced model using AudioSet pretrained features
- Threshold-based evaluation metrics

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- PyTorch Lightning

### Installation

```bash
pip install -r requirements.txt
```

## Training

### PANNs-Enhanced Model Training

Use pretrained AudioSet CNN14 model for a significant performance boost:

```bash
python -m training.panns_train --config configs/panns_enhanced.yaml
```

This uses a two-phase training approach:
1. First phase: Frozen backbone with higher learning rate
2. Second phase: Full model fine-tuning with lower learning rate

## Inference

### Threshold-Based Evaluation

```bash
python -m inference.evaluate_model <checkpoint_path> <test_directory> --threshold 0.6
```

## Model Architecture

### PANNs-Enhanced Model

- Uses AudioSet pretrained CNN14 backbone
- 3 feature extractors with pretrained weights
- Feature fusion and classification layers
- Typically achieves 80%+ mAP with less training

## Evaluation

Our threshold-based evaluation provides a clear binary assessment of model performance:

- Each instrument is predicted as present (1) or absent (0) using a threshold (default: 0.6)
- Sample accuracy is calculated as correct predictions / total labels (e.g., 10/11 = 90.9%)
- Detailed per-instrument metrics show exactly what the model gets right and wrong

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Instrument Classifier

A PyTorch Lightning project for multi‑label musical instrument recognition from audio clips.
Clone, install dependencies, preprocess data, and train:

```bash
git clone <repo-url>
cd instrument_classifier
pip install -r requirements.txt
python data/download_irmas.py  --out_dir data/raw
python data/preprocess.py      --in_dir data/raw/IRMAS --out_dir data/processed

```

See `configs/default.yaml` for full hyper‑parameters.
