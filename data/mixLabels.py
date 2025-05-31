import torch
import random


def create_synthetic_mixtures(dataset, num_new_samples=1000, min_instruments=2, max_instruments=4):
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
