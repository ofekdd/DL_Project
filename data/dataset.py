import torch
import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader

from var import band_ranges, n_ffts, LABELS


def pad_collate(batch):
    xs, ys = zip(*batch)
    # find max mel bins in batch
    H = max(x.shape[1] for x in xs)
    W = max(x.shape[2] for x in xs)      # optional time padding
    padded = []
    for x in xs:
        pad_h = H - x.shape[1]
        pad_w = W - x.shape[2]
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        padded.append(x_padded)
    return torch.stack(padded), torch.stack(ys)

def multi_stft_pad_collate(batch):
    """
    Collate function for MultiSTFTNpyDataset.
    Each item in batch is a list of 9 spectrograms and a label.
    """
    # Unzip the batch into lists of spectrograms and labels
    specs_list, ys = zip(*batch)

    # specs_list is now a tuple of lists, where each list contains 9 spectrograms
    # We need to transpose this to get 9 lists, each containing a spectrogram from each item
    # This way we can pad each spectrogram type separately
    transposed_specs = list(zip(*specs_list))

    # Pad each spectrogram type separately
    padded_specs = []
    for spec_group in transposed_specs:
        # Find max dimensions for this spectrogram type
        H = max(spec.shape[1] for spec in spec_group)
        W = max(spec.shape[2] for spec in spec_group)

        # Pad each spectrogram to the max dimensions
        padded_group = []
        for spec in spec_group:
            pad_h = H - spec.shape[1]
            pad_w = W - spec.shape[2]
            padded = torch.nn.functional.pad(spec, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
            padded_group.append(padded)

        # Stack the padded spectrograms
        padded_specs.append(torch.stack(padded_group))

    # Return the list of padded spectrograms and the stacked labels
    return padded_specs, torch.stack(ys)

class MultiSTFTNpyDataset(Dataset):
    """
    Dataset for loading all 9 spectrograms (3 window sizes × 3 frequency bands) for each audio file.
    """
    def __init__(self, root, max_samples=None):
        # Get all directories (each directory corresponds to one audio file)
        self.dirs = list(set(file.parent for file in pathlib.Path(root).rglob("*.npy")))

        # Limit the number of samples if max_samples is specified
        if max_samples is not None and max_samples > 0 and max_samples < len(self.dirs):
            self.dirs = self.dirs[:max_samples]

        self.label_map = {label: i for i, label in enumerate(LABELS)}

        # Define the expected spectrogram files for each audio
        self.band_ranges = band_ranges
        self.n_ffts = n_ffts

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        # Get the directory for this audio file
        audio_dir = self.dirs[idx]

        # Parse label from folder name
        label_str = audio_dir.name.split("_")[0]
        y = torch.zeros(len(LABELS))

        # Map label string to index
        if label_str in self.label_map:
            y[self.label_map[label_str]] = 1.0

        # Load all 9 spectrograms
        specs = []
        for band_range in self.band_ranges:
            for n_fft in self.n_ffts:
                spec_path = audio_dir / f"{band_range}_fft{n_fft}.npy"

                if spec_path.exists():
                    spec = np.load(spec_path)
                    spec_tensor = torch.tensor(spec).unsqueeze(0)  # [1,H,W]
                else:
                    # If a specific spectrogram is missing, use a zero tensor of appropriate shape
                    # This is a fallback and should be rare
                    print(f"Warning: Missing spectrogram for {spec_path}")
                    # Use a small dummy tensor as fallback
                    spec_tensor = torch.zeros(1, 10, 10)

                specs.append(spec_tensor)

        return specs, y

def create_dataloaders(train_dir, val_dir, batch_size, num_workers, use_multi_stft=True, max_samples=None):
    """
    Create train and validation dataloaders

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_multi_stft: Whether to use MultiSTFTNpyDataset (for MultiSTFTCNN model)
        max_samples: Maximum number of samples to use for training (None for all samples)

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    train_ds = MultiSTFTNpyDataset(train_dir, max_samples=max_samples)
    val_ds = MultiSTFTNpyDataset(val_dir)
    collate_fn = multi_stft_pad_collate

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader
