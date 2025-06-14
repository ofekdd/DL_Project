import torch
import random
import torch.nn.functional as F

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
