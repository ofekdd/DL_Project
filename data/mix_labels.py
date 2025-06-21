import torch
import random
import torch.nn.functional as F
import librosa
import pathlib
from var import LABELS


def create_multilabel_dataset(irmas_root, cfg, max_original_samples=None, num_mixtures=None,
                              min_instruments=None, max_instruments=None, use_weighted_targets=False, alpha=0.80):
    """
    Create a multi-label dataset by loading IRMAS data and creating synthetic mixtures.

    Args:
        irmas_root: Path to IRMAS dataset root directory
        cfg: Configuration dictionary
        max_original_samples: Maximum original samples to load (None to use config)
        num_mixtures: Number of synthetic mixtures to create (None to use config)
        min_instruments: Minimum instruments per mixture (None to use config)
        max_instruments: Maximum instruments per mixture (None to use config)
        use_weighted_targets: If True, generates weighted probability targets instead of binary labels
        alpha: Proportion of probability mass to assign to positive classes (if using weighted targets)

    Returns:
        Tuple of (original_dataset, mixed_dataset)
    """
    # Use config values if parameters not explicitly provided
    if max_original_samples is None:
        max_original_samples = cfg.get('max_original_samples', 8000)
    if num_mixtures is None:
        num_mixtures = cfg.get('num_mixtures', 1000)
    if min_instruments is None:
        min_instruments = cfg.get('min_instruments', 1)
    if max_instruments is None:
        max_instruments = cfg.get('max_instruments', 2)

    # Handle string "None" values from YAML config
    def convert_config_value(value, default):
        if isinstance(value, str) and value.lower() == 'none':
            return None
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return value

    max_original_samples = convert_config_value(max_original_samples, 8000)
    num_mixtures = convert_config_value(num_mixtures, 1000)
    min_instruments = convert_config_value(min_instruments, 1)
    max_instruments = convert_config_value(max_instruments, 2)

    print(f"Creating multilabel dataset with config:")
    print(f"  max_original_samples: {max_original_samples}")
    print(f"  num_mixtures: {num_mixtures}")
    print(f"  min_instruments: {min_instruments}")
    print(f"  max_instruments: {max_instruments}")

    # Load the original single-instrument dataset
    original_dataset = load_irmas_audio_dataset(irmas_root, cfg, max_original_samples)

    if not original_dataset:
        print("Failed to load original dataset. Check the IRMAS data path.")
        return [], []

    print(f"Original dataset size: {len(original_dataset)}")

    # Create synthetic multi-label mixtures
    print("Creating synthetic multi-label mixtures...")
    mixed_dataset = create_synthetic_mixtures(
        original_dataset,
        num_new_samples=num_mixtures,
        min_instruments=min_instruments,
        max_instruments=max_instruments,
        silence_ratio=0.02,
        max_shift=1000,
        gain_range=(0.8, 1.2),
        target_peak=0.9
    )

    print(f"Created {len(mixed_dataset)} synthetic multi-label samples")

    # Show some examples
    print("\nExample synthetic mixtures:")
    for i in range(min(3, len(mixed_dataset))):
        audio, labels = mixed_dataset[i]

        if use_weighted_targets:
            # For weighted targets, show probabilities
            top_indices = torch.topk(labels, min(3, len(labels))).indices
            top_labels = [(LABELS[j], labels[j].item()) for j in top_indices]
            print(f"  Mix {i + 1}: Audio shape {audio.shape}, Top weighted labels: {top_labels}")
        else:
            # For binary labels, show present instruments
            active_labels = [LABELS[j] for j, val in enumerate(labels) if val == 1]
            print(f"  Mix {i + 1}: Audio shape {audio.shape}, Labels: {active_labels}")

    return original_dataset, mixed_dataset


def create_synthetic_mixtures(
    dataset,
    num_new_samples=10000,
    min_instruments=1,
    max_instruments=3,
    silence_ratio=0.02,
    max_shift=1000,
    gain_range=(0.8, 1.2),
    target_peak=0.9
):
    """
    Create synthetic audio mixtures with multiple instruments and realism.

    Args:
        dataset: List of (audio_tensor, label_tensor)
        num_new_samples: Total number of synthetic samples to generate
        min_instruments: Minimum instruments per mix
        max_instruments: Maximum instruments per mix
        silence_ratio: Proportion of silent/no-instrument samples (e.g. 0.02)
        max_shift: Maximum samples for random time shift
        gain_range: Range for random gain after mixing
        target_peak: Max amplitude after mix to prevent clipping

    Returns:
        List of (mixed_audio_tensor, multi_label_tensor)
    """
    mixed_dataset = []

    for _ in range(num_new_samples):
        # 🔇 1. Insert silent samples randomly (~2%)
        if random.random() < silence_ratio:
            dummy_audio = torch.zeros_like(dataset[0][0])
            dummy_label = torch.zeros_like(dataset[0][1])
            mixed_dataset.append((dummy_audio, dummy_label))
            continue

        # 🎵 2. Randomly select 1–4 instruments
        k = random.randint(min_instruments, max_instruments)
        samples = random.sample(dataset, k)
        audios, labels = zip(*samples)

        # ✂️ 3. Truncate all to shortest length
        audios = [a.squeeze(0) if a.dim() == 2 else a for a in audios]
        min_len = min(a.shape[0] for a in audios)

        # 🕒 4. Random time shift for each instrument
        def random_shift(x, max_shift):
            shift = random.randint(0, max_shift)
            return F.pad(x, (shift, 0))[:x.shape[0]]

        shifted_audios = [random_shift(a[:min_len], max_shift) for a in audios]

        # 🎚️ 5. Mix with natural amplitude + apply random gain per mix
        mixed_audio = torch.stack(shifted_audios).sum(dim=0) / k

        gain = random.uniform(*gain_range)
        mixed_audio = mixed_audio * gain

        # 🔉 6. Normalize to prevent clipping
        peak = mixed_audio.abs().max()
        if peak > target_peak:
            mixed_audio = mixed_audio * (target_peak / peak)

        # 🧠 7. Combine labels (multi-label vector)
        combined_label = torch.stack(labels).max(dim=0).values

        mixed_dataset.append((mixed_audio, combined_label))

    return mixed_dataset
