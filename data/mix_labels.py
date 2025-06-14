import torch
import random
import torch.nn.functional as F


def load_irmas_audio_dataset(irmas_root, cfg, max_samples=None):
    """
    Load raw audio files from IRMAS dataset into a list of (audio_tensor, label_vector) tuples

    Args:
        irmas_root: Path to IRMAS dataset root directory
        cfg: Configuration dictionary containing sample_rate
        max_samples: Maximum number of samples to load (None to use config value)

    Returns:
        List of (audio_tensor, label_vector) tuples
    """
    # Use config value if max_samples not explicitly provided
    if max_samples is None:
        max_samples = cfg.get('max_original_samples', 100)

    irmas_path = pathlib.Path(irmas_root) / "IRMAS-TrainingData"

    if not irmas_path.exists():
        print(f"Warning: Training data path {irmas_path} does not exist")
        return []

    # IRMAS directory name to our label mapping
    irmas_to_label_map = {
        'cel': 'cello',
        'cla': 'clarinet',
        'flu': 'flute',
        'gac': 'acoustic_guitar',  # Guitar acoustic
        'gel': 'acoustic_guitar',  # Guitar electric -> acoustic_guitar
        'org': 'organ',
        'pia': 'piano',
        'sax': 'saxophone',
        'tru': 'trumpet',
        'vio': 'violin',
        'voi': 'voice'
    }

    dataset = []
    label_map = {label: i for i, label in enumerate(LABELS)}

    # Get all WAV files from the training data
    wav_files = list(irmas_path.rglob("*.wav"))

    # Limit samples if specified
    if max_samples and len(wav_files) > max_samples:
        wav_files = random.sample(wav_files, max_samples)

    print(f"Loading {len(wav_files)} audio files...")

    successful_loads = 0
    for wav_file in wav_files:
        try:
            # Extract label from parent directory name
            irmas_label = wav_file.parent.name

            # Map IRMAS label to our label
            if irmas_label in irmas_to_label_map:
                our_label = irmas_to_label_map[irmas_label]

                if our_label in label_map:
                    # Load audio
                    y, sr = librosa.load(wav_file, sr=cfg['sample_rate'], mono=True)
                    audio_tensor = torch.tensor(y, dtype=torch.float32)

                    # Create one-hot label vector
                    label_vector = torch.zeros(len(LABELS), dtype=torch.long)
                    label_vector[label_map[our_label]] = 1

                    dataset.append((audio_tensor, label_vector))
                    successful_loads += 1
                else:
                    print(f"Warning: Our label '{our_label}' not in LABELS")
            else:
                print(f"Warning: Unknown IRMAS label '{irmas_label}' for file {wav_file.name}")

        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue

    print(f"Successfully loaded {successful_loads} audio samples")
    return dataset


def create_multilabel_dataset(irmas_root, cfg, max_original_samples=None, num_mixtures=None,
                              min_instruments=None, max_instruments=None):
    """
    Create a multi-label dataset by loading IRMAS data and creating synthetic mixtures.

    Args:
        irmas_root: Path to IRMAS dataset root directory
        cfg: Configuration dictionary
        max_original_samples: Maximum original samples to load (None to use config)
        num_mixtures: Number of synthetic mixtures to create (None to use config)
        min_instruments: Minimum instruments per mixture (None to use config)
        max_instruments: Maximum instruments per mixture (None to use config)

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
