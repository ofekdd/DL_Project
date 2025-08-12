# Instrument Recognition in Musical Audio (IRMAS)

A PyTorch Lightning project for recognizing musical instruments in audio clips using multi-band STFT features and a PANNs (AudioSet CNN14) warm-start.

## Overview
- Dataset: IRMAS (training is single-label per file; testing may include multiple instruments per file).
- Features: Three STFT spectrograms per clip, each optimized for a frequency band:
  - 0–1000 Hz with n_fft=1024
  - 1000–4000 Hz with n_fft=512
  - 4000–11025 Hz with n_fft=256
- Model: MultiSTFTCNN enhanced with PANNs CNN14 feature extractors (one per band), fused via MLP and trained with a two-phase schedule (freeze then fine-tune).

## Repository structure
- configs/panns_enhanced.yaml – training and preprocessing hyperparameters
- data/download_irmas.py – download IRMAS training and testing zips
- data/preprocess.py – generate and save spectrograms (.npy) for 3 bands per audio
- data/dataset.py – dataset and dataloaders for the saved spectrograms
- models/panns_enhanced.py – PANNs-enhanced MultiSTFT model
- training/panns_train.py – Lightning training script (two-phase training)
- inference/predict.py – CLI for single-file predictions and simple batch accuracy
- utils/model_loader.py – robust checkpoint loader (auto-downloads PANNs weights)
- var.py – class labels and STFT band definitions

## Installation
1) Create environment and install deps:
```bash
pip install -r requirements.txt
```

## Data
1) Download IRMAS (training + testing parts):
```bash
python data/download_irmas.py --out_dir data/raw
```
This will create data/raw/IRMAS-TrainingData and IRMAS-TestingData-Part*/.

2) Preprocess audio into 3-band spectrograms (.npy):
```bash
python data/preprocess.py --in_dir data/raw/IRMAS-TrainingData --out_dir data/processed --config configs/panns_enhanced.yaml
```
Notes:
- The CLI above performs a simple “mirror” conversion of any wavs under --in_dir to .npy spectrogram folders under --out_dir.
- The file also contains a preprocess_data() function that can build train/val/test splits directly from IRMAS root. For most use cases, the CLI form is sufficient; training defaults expect data/processed/train and data/processed/val.

## Training (PANNs-enhanced)
Two-phase schedule:
- Phase 1: PANNs backbone frozen; only fusion + classifier train at a higher LR.
- Phase 2: Unfreeze all and fine-tune at a lower LR.

Run:
```bash
python training/panns_train.py --config configs/panns_enhanced.yaml
```
The script will automatically download the PANNs CNN14 checkpoint (AudioSet) on first use.

Default data locations used by the trainer (override via YAML):
- train_dir: data/processed/train
- val_dir:   data/processed/val

## Inference
Single file (top-1 softmax over 11 classes):
```bash
python inference/predict.py <checkpoint.ckpt> <path/to/file.wav> --config configs/panns_enhanced.yaml
```
Batch prediction with running accuracy (if ground truth can be parsed from filenames/annotations) is available via predict_batch_with_accuracy in inference/predict.py.

## Hyperparameters (configs/panns_enhanced.yaml)
Training/data:
- sample_rate: 22050 – resample target for librosa.load
- n_mels: 64 – not used by current STFT pipeline but kept for compatibility
- hop_length: 512 – STFT hop for some utilities
- batch_size: 32 – effective batch size for DataLoader
- num_workers: 2 – DataLoader workers
- num_epochs: 100 – total epochs for two-phase training
- freeze_epochs: 4 – epochs with PANNs backbone frozen
- limit_val_batches: 1.0 – fraction of val set per epoch (use <1.0 for quick runs)
- num_sanity_val_steps: 2 – sanity steps before training
- original_data_percentage: 1 – fraction of original IRMAS data to keep (for advanced preprocess path)
- max_original_samples: null – optional cap on original samples
- max_samples: null – cap training samples loaded by dataset (useful for quick runs)
- max_test_samples: null – cap test samples during preprocess

Optimization:
- frozen_learning_rate: 0.003 – LR while backbone is frozen
- finetune_learning_rate: 0.0002 – LR after unfreezing
- Optimizer: Adam with weight_decay=2e-4 (see training/panns_train.py)
- Metric: torchmetrics Accuracy(task="multiclass")

Model specifics:
- Backbone: PANNs CNN14 feature extractors (auto-downloaded to pretrained/Cnn14_mAP=0.431.pth)
- Fusion MLP: 3×512 -> 1536 -> 768 -> 512 with BatchNorm, ReLU, Dropout(0.4/0.3)
- Classifier: Linear(512 -> 11)

Labels (var.LABELS):
["cello","clarinet","flute","acoustic_guitar","organ","piano","saxophone","trumpet","violin","voice","other"]

## Tips
- For quick experiments, reduce batch_size and set max_samples, limit_val_batches < 1.0, and num_epochs small in the YAML.
- Ensure each processed sample directory contains three files:
  - 0-1000Hz_fft1024.npy
  - 1000-4000Hz_fft512.npy
  - 4000-11025Hz_fft256.npy

## License
MIT License (see LICENSE if present).
