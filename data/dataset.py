import torch
import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader

LABELS = ["cello", "clarinet", "flute", "acoustic_guitar", "organ", "piano", "saxophone", "trumpet", "violin", "voice", "other"]

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

class NpyDataset(Dataset):
    def __init__(self, root):
        self.files = list(pathlib.Path(root).rglob("*.npy"))
        self.label_map = {label: i for i, label in enumerate(LABELS)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])
        x = torch.tensor(spec).unsqueeze(0)  # [1,H,W]

        # Parse label from folder name
        label_str = self.files[idx].parent.name.split("_")[0]
        y = torch.zeros(len(LABELS))

        # Map label string to index
        if label_str in self.label_map:
            y[self.label_map[label_str]] = 1.0

        return x, y

def create_dataloaders(train_dir, val_dir, batch_size, num_workers):
    """
    Create train and validation dataloaders
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    train_ds = NpyDataset(train_dir)
    val_ds = NpyDataset(val_dir)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=pad_collate
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=pad_collate
    )
    
    return train_loader, val_loader