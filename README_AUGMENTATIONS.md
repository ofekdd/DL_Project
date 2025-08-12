# Data Augmentation for Audio Classification

## Overview

This project now includes data augmentation techniques to improve the robustness and performance of the audio classification model. Augmentation is applied during preprocessing to create diverse training samples.

## Implemented Augmentation Techniques

### Audio-Level Augmentations (Time Domain)

These augmentations are applied to raw audio waveforms before STFT processing:

1. **Time Stretching**: Speeds up or slows down audio without changing pitch (range: 0.9-1.1x)
2. **Pitch Shifting**: Shifts pitch up or down without changing tempo (range: ±2 semitones)
3. **Noise Addition**: Adds Gaussian noise to simulate recording conditions (factor: 0.01)
4. **Volume Changes**: Varies amplitude to handle different recording levels (range: 0.8-1.2x)

### Spectrogram-Level Augmentations (Frequency Domain)

These augmentations are applied after STFT processing:

1. **Frequency Masking**: Masks random frequency bands (SpecAugment technique)
2. **Time Masking**: Masks random time segments (SpecAugment technique)

## Configuration

Augmentation settings can be customized in the YAML configuration file:

```yaml
augmentation:
  enabled: true  # Master switch for all augmentations
  audio_augmentation_prob: 0.5  # Probability of applying audio-level augmentations
  spec_augmentation_prob: 0.5  # Probability of applying spectrogram-level augmentations
  # Audio augmentation parameters
  time_stretch_range: [0.9, 1.1]
  pitch_shift_range: [-2, 2]
  noise_factor: 0.01
  volume_range: [0.8, 1.2]
  # Spectrogram augmentation parameters
  freq_mask_param: 8
  time_mask_param: 15
```

## Benefits for Multi-Instrument Classification

- **Time/Pitch shifts**: Help model generalize across different tempos and tunings
- **Noise addition**: Improves robustness to recording conditions
- **SpecAugment**: Prevents overfitting to specific frequency patterns
- **Volume changes**: Handles different recording levels

## Implementation Details

- Augmentation is only applied to training data, not validation or test data
- Original samples are always preserved; augmented versions are added as additional samples
- Each original training sample generates one augmented version
- The augmentation process doubles the size of the training dataset

## Usage

Augmentation is automatically applied during preprocessing when enabled in the config:

```bash
python data/preprocess.py --in_dir data/raw/IRMAS --out_dir data/processed --config configs/panns_enhanced.yaml
```

To disable augmentation, set `augmentation.enabled: false` in your config file.
