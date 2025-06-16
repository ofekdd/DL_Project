import torch
import numpy as np
import pathlib
import re
from torch.utils.data import Dataset, DataLoader

from var import band_ranges, n_ffts, LABELS


def pad_collate(batch):
    xs, ys = zip(*batch)
    # find max mel bins in batch
    H = max(x.shape[1] for x in xs)
    W = max(x.shape[2] for x in xs)  # optional time padding
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
        # Handle string "None" from YAML config or convert string numbers to int
        if isinstance(max_samples, str):
            if max_samples.lower() == 'none':
                max_samples = None
            else:
                try:
                    max_samples = int(max_samples)
                except ValueError:
                    print(f"Warning: Could not convert max_samples '{max_samples}' to int, using None")
                    max_samples = None

        # Get all directories (each directory corresponds to one audio file)
        self.dirs = list(set(file.parent for file in pathlib.Path(root).rglob("*.npy")))

        print(f"Found {len(self.dirs)} total sample directories in {root}")

        # Limit the number of samples if max_samples is specified
        if max_samples is not None and isinstance(max_samples, int) and max_samples > 0 and max_samples < len(self.dirs):
            self.dirs = self.dirs[:max_samples]
            print(f"Limited to {max_samples} samples")

        self.label_map = {label: i for i, label in enumerate(LABELS)}

        # IRMAS directory name to our label mapping
        self.irmas_to_label_map = {
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

        # Define the expected spectrogram files for each audio
        self.band_ranges = band_ranges
        self.n_ffts = n_ffts

        # Debug: Check a few sample directories and their labels
        print(f"Dataset initialization complete. Final dataset size: {len(self.dirs)}")
        if len(self.dirs) > 0:
            print("Sample directories:")
            for i, dir_path in enumerate(self.dirs[:3]):
                print(f"  {i}: {dir_path.name}")
                # Test label parsing
                try:
                    labels = self._parse_labels_from_folder_name(dir_path.name)
                    if labels:
                        print(f"    -> Parsed labels: {labels}")
                    else:
                        print(f"    -> No valid labels found")
                except Exception as e:
                    print(f"    -> Error parsing label: {e}")

    def _parse_labels_from_folder_name(self, folder_name):
        """
        Parse instrument labels from folder name.

        Handles various formats:
        - Mixed samples: "mixed_3_piano", "mixed_68_trumpet_voice"
        - Original IRMAS: "238__[org][dru][jaz_blu]1125__1", "[pia][jaz_blu]1471__1"
        - Simple format: "piano_123", "trumpet_456"
        """
        labels = []

        if folder_name.startswith("mixed_"):
            # Handle mixed samples: e.g., "mixed_3_piano" or "mixed_68_trumpet_voice"
            parts = folder_name.split("_")
            if len(parts) >= 3:
                # Extract instrument labels (everything after "mixed_X_")
                instrument_parts = parts[2:]  # Skip "mixed" and the ID number

                # Join the parts back and split by common separators
                instrument_string = "_".join(instrument_parts)

                # Check each label in our mapping
                for our_label in self.label_map.keys():
                    if our_label in instrument_string:
                        labels.append(our_label)

        else:
            # Handle original IRMAS samples with complex naming
            # Look for patterns like [cel], [pia], [org], etc.
            irmas_pattern = r'\[([a-z]{3})\]'
            irmas_matches = re.findall(irmas_pattern, folder_name)

            for irmas_label in irmas_matches:
                if irmas_label in self.irmas_to_label_map:
                    our_label = self.irmas_to_label_map[irmas_label]
                    if our_label not in labels:  # Avoid duplicates
                        labels.append(our_label)

            # If no IRMAS pattern found, try simple format
            if not labels:
                # Try simple format: "piano_123", "trumpet_456"
                label_str = folder_name.split("_")[0]
                if label_str in self.label_map:
                    labels.append(label_str)
                elif label_str in self.irmas_to_label_map:
                    labels.append(self.irmas_to_label_map[label_str])

        return labels

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        # Get the directory for this audio file
        audio_dir = self.dirs[idx]

        # Parse labels from folder name
        y = torch.zeros(len(LABELS), dtype=torch.long)
        folder_name = audio_dir.name

        # Parse labels using the improved function
        parsed_labels = self._parse_labels_from_folder_name(folder_name)

        # Set the corresponding indices in the label vector
        for label in parsed_labels:
            if label in self.label_map:
                y[self.label_map[label]] = 1

        # Load all 9 spectrograms
        specs = []
        missing_files = 0
        optimized_stfts = [
            ("0-1000Hz", 1024),
            ("1000-4000Hz", 512),
            ("4000-11025Hz", 256),
        ]

        for band_label, n_fft in optimized_stfts:
            spec_path = audio_dir / f"{band_label}_fft{n_fft}.npy"

            if spec_path.exists():  #
                spec = np.load(spec_path)
                spec_tensor = torch.tensor(spec).unsqueeze(0)  # [1,H,W]
            else:
                print(f"Warning: Missing spectrogram for {spec_path}")
                missing_files += 1
                spec_tensor = torch.zeros(1, 10, 10)

            specs.append(spec_tensor)

        # Debug: Check if we have any valid labels
        if y.sum() == 0:
            print(f"Warning: No valid labels found for {folder_name}")

        if missing_files > 0:
            print(f"Warning: {missing_files}/9 spectrograms missing for {folder_name}")

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
    print(f"Creating training dataset with max_samples={max_samples}")
    train_ds = MultiSTFTNpyDataset(train_dir, max_samples=max_samples)
    print(f"Training dataset size: {len(train_ds)}")

    print(f"Creating validation dataset...")
    val_ds = MultiSTFTNpyDataset(val_dir)
    print(f"Validation dataset size: {len(val_ds)}")

    collate_fn = multi_stft_pad_collate

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty! Check your data preprocessing and label parsing.")

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