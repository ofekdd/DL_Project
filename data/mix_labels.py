import torch
import random
import librosa
import pathlib
from var import LABELS


def load_irmas_audio_dataset(irmas_root, cfg, max_samples=100):
    """
    Load raw audio files from IRMAS dataset into a list of (audio_tensor, label_vector) tuples

    Args:
        irmas_root: Path to IRMAS dataset root directory
        cfg: Configuration dictionary containing sample_rate
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of (audio_tensor, label_vector) tuples
    """
    irmas_path = pathlib.Path(irmas_root) / "IRMAS-TrainingData"

    if not irmas_path.exists():
        print(f"Warning: Training data path {irmas_path} does not exist")
        return []

    dataset = []
    label_map = {label: i for i, label in enumerate(LABELS)}

    # Get all WAV files from the training data
    wav_files = list(irmas_path.rglob("*.wav"))

    # Limit samples if specified
    if max_samples and len(wav_files) > max_samples:
        wav_files = random.sample(wav_files, max_samples)

    print(f"Loading {len(wav_files)} audio files...")

    for wav_file in wav_files:
        try:
            # Load audio
            y, sr = librosa.load(wav_file, sr=cfg['sample_rate'], mono=True)
            audio_tensor = torch.tensor(y, dtype=torch.float32)

            # Extract label from filename or parent directory
            # IRMAS files are typically named like "guitar_001.wav" or in folders by instrument
            if wav_file.parent.name in label_map:
                label_str = wav_file.parent.name
            else:
                # Try to extract from filename
                label_str = wav_file.stem.split('_')[0]

            # Create one-hot label vector
            label_vector = torch.zeros(len(LABELS), dtype=torch.long)
            if label_str in label_map:
                label_vector[label_map[label_str]] = 1
            else:
                print(f"Warning: Unknown label '{label_str}' for file {wav_file}")
                continue

            dataset.append((audio_tensor, label_vector))

        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue

    print(f"Successfully loaded {len(dataset)} audio samples")
    return dataset


def create_multilabel_dataset(irmas_root, cfg, max_original_samples=8000, num_mixtures=1000,
                              min_instruments=1, max_instruments=2):
    """
    Create a multi-label dataset by loading IRMAS data and creating synthetic mixtures.

    Args:
        irmas_root: Path to IRMAS dataset root directory
        cfg: Configuration dictionary
        max_original_samples: Maximum original samples to load
        num_mixtures: Number of synthetic mixtures to create
        min_instruments: Minimum instruments per mixture
        max_instruments: Maximum instruments per mixture

    Returns:
        Tuple of (original_dataset, mixed_dataset)
    """
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
        max_instruments=max_instruments
    )

    print(f"Created {len(mixed_dataset)} synthetic multi-label samples")

    # Show some examples
    print("\nExample synthetic mixtures:")
    for i in range(min(3, len(mixed_dataset))):
        audio, labels = mixed_dataset[i]
        active_labels = [LABELS[j] for j, val in enumerate(labels) if val == 1]
        print(f"  Mix {i + 1}: Audio shape {audio.shape}, Labels: {active_labels}")

    return original_dataset, mixed_dataset


def create_synthetic_mixtures(dataset, num_new_samples=1000, min_instruments=1, max_instruments=2):
    """
    Creates synthetic audio mixtures from a dataset of single-instrument samples.

    Args:
        dataset: List of tuples (audio_tensor, label_vector)
        num_new_samples: How many new synthetic samples to create
        min_instruments: Minimum instruments per mix
        max_instruments: Maximum instruments per mix

    Returns:
        List of (mixed_audio_tensor, multi_label_vector)
    """
    mixed_dataset = []

    for _ in range(num_new_samples):
        # 1. Choose how many instruments to mix
        k = random.randint(min_instruments, max_instruments)

        # 2. Randomly sample k audio-label pairs
        samples = random.sample(dataset, k)
        audios, labels = zip(*samples)

        # 3. Stack and mix audio (assumes same length)
        audios = [a.squeeze(0) if a.dim() == 2 else a for a in audios]
        min_len = min(a.shape[0] for a in audios)
        trimmed = torch.stack([a[:min_len] for a in audios])  # Shape: (k, n)
        mixed_audio = trimmed.sum(dim=0) / k  # Normalize

        # 4. Combine labels (element-wise OR)
        combined_label = torch.stack(labels).max(dim=0).values  # multi-label

        # 5. Add to result
        mixed_dataset.append((mixed_audio, combined_label))

    return mixed_dataset