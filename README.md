# Multi-Instrument Recognition System

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

### Standard Model Training

```bash
python -m training.train --config configs/multi_stft_cnn.yaml
```

### PANNs-Enhanced Model Training

Use pretrained AudioSet CNN14 model for a significant performance boost:

```bash
python -m training.panns_train --config configs/panns_enhanced.yaml
```

This uses a two-phase training approach:
1. First phase: Frozen backbone with higher learning rate
2. Second phase: Full model fine-tuning with lower learning rate

## Inference

### Single File Inference

```bash
python -m inference.predict <checkpoint_path> <wav_file> --config configs/multi_stft_cnn.yaml
```

### Threshold-Based Evaluation

```bash
python -m inference.evaluate_model <checkpoint_path> <test_directory> --threshold 0.6
```

## Model Architecture

### Standard Multi-STFT CNN

- 9 CNN branches (3 window sizes × 3 frequency bands)
- Each branch processes a different spectrogram representation
- Feature fusion and classifier layers

### PANNs-Enhanced Model

- Uses AudioSet pretrained CNN14 backbone
- 9 feature extractors with pretrained weights
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
python training/train.py       --config configs/model_resnet.yaml

# Optimize detection thresholds for better accuracy
python visualization/optimize_thresholds.py --metric f1 CHECKPOINT_PATH
```

See `configs/default.yaml` for full hyper‑parameters.
