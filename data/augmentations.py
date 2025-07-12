import torch
import numpy as np
import random

class SpecAugment:
    """SpecAugment for audio spectrograms.

    Implements time warping, frequency masking, and time masking as described in:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """

    def __init__(self, freq_mask_param=10, time_mask_param=24, n_freq_masks=2, n_time_masks=2):
        """Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum frequency masking width
            time_mask_param: Maximum time masking width
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spec):
        """Apply SpecAugment to the spectrogram.

        Args:
            spec: Spectrogram tensor of shape [batch_size, channels, n_mels, time]

        Returns:
            Augmented spectrogram
        """
        aug_spec = spec.clone()
        batch_size, _, n_mels, n_steps = aug_spec.shape

        # Apply frequency masking
        for i in range(self.n_freq_masks):
            for b in range(batch_size):
                f = np.random.randint(0, self.freq_mask_param)
                f0 = np.random.randint(0, n_mels - f)
                aug_spec[b, :, f0:f0+f, :] = 0

        # Apply time masking
        for i in range(self.n_time_masks):
            for b in range(batch_size):
                t = np.random.randint(0, self.time_mask_param)
                t0 = np.random.randint(0, n_steps - t)
                aug_spec[b, :, :, t0:t0+t] = 0

        return aug_spec

def apply_spec_augment(x, p=0.5):
    """Apply SpecAugment with probability p.

    Args:
        x: List of spectrogram tensors [batch_size, channels, n_mels, time]
        p: Probability of applying augmentation

    Returns:
        Augmented spectrograms
    """
    if random.random() < p:
        aug = SpecAugment()
        return [aug(spec) for spec in x]
    return x
