# /data/dataset.py
import torch
import numpy as np
import pathlib
import re
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from var import LABELS


def scalogram_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pads 2D scalograms to max S and max T in batch."""
    xs, ys = zip(*batch)
    max_S = max(x.shape[0] for x in xs)
    max_T = max(x.shape[1] for x in xs)
    padded = []
    for x in xs:
        pad_S = max_S - x.shape[0]
        pad_T = max_T - x.shape[1]
        if pad_S or pad_T:
            x = torch.nn.functional.pad(x, (0, pad_T, 0, pad_S))
        padded.append(x.unsqueeze(0))  # add channel -> [1,S,T]
    return torch.stack(padded, 0), torch.stack(ys, 0)   # [B,1,S,T], [B,C]


class ScalogramDataset(Dataset):
    """
    Each sample directory must contain: scalogram.npy  (shape [S,T], float32)
    """
    def __init__(self, root: str | pathlib.Path, max_samples: Optional[int] = None):
        root = pathlib.Path(root)
        self.dirs = sorted({p.parent for p in root.rglob("scalogram.npy")})
        print(f"Found {len(self.dirs)} sample directories in {root}")

        if isinstance(max_samples, str):
            try:
                max_samples = None if max_samples.lower() == "none" else int(max_samples)
            except ValueError:
                max_samples = None
        if isinstance(max_samples, int) and 0 < max_samples < len(self.dirs):
            self.dirs = self.dirs[:max_samples]
            print(f"Limiting to {max_samples} samples")

        self.label_map = {label: i for i, label in enumerate(LABELS)}
        self.irmas_to_label_map = {
            'cel': 'cello','cla': 'clarinet','flu': 'flute',
            'gac': 'acoustic_guitar','gel': 'acoustic_guitar',
            'org': 'organ','pia': 'piano','sax': 'saxophone',
            'tru': 'trumpet','vio': 'violin','voi': 'voice'
        }

        print(f"Dataset ready. Final size: {len(self.dirs)}")
        for i, d in enumerate(self.dirs[:3]):
            print(f"  ex[{i}]: {d.name} -> {self._parse_labels_from_folder_name(d.name)}")

    def _parse_labels_from_folder_name(self, folder_name: str) -> list[str]:
        labels: list[str] = []
        matches = re.findall(r'\[([a-z]{3})\]', folder_name)
        for code in matches:
            if code in self.irmas_to_label_map:
                lbl = self.irmas_to_label_map[code]
                if lbl not in labels:
                    labels.append(lbl)
        if not labels:
            token = folder_name.split("_")[0]
            if token in self.label_map:
                labels.append(token)
            elif token in self.irmas_to_label_map:
                labels.append(self.irmas_to_label_map[token])
        return labels

    def __len__(self): return len(self.dirs)

    def __getitem__(self, idx: int):
        d = self.dirs[idx]
        f = d / "scalogram.npy"
        try:
            S = np.load(f).astype(np.float32)  # [S,T]
        except Exception as e:
            print(f"⚠️  Failed loading scalogram {f}: {e}")
            S = np.zeros((32, 32), dtype=np.float32)
        x = torch.from_numpy(S)  # [S,T]

        y = torch.zeros(len(LABELS), dtype=torch.long)
        for lbl in self._parse_labels_from_folder_name(d.name):
            if lbl in self.label_map:
                y[self.label_map[lbl]] = 1
        return x, y


def create_dataloaders(train_dir: str | pathlib.Path,
                       val_dir: str | pathlib.Path,
                       batch_size: int,
                       num_workers: int,
                       max_samples: Optional[int] = None,
                       pin_memory: bool = True,
                       persistent_workers: Optional[bool] = None):
    print("Creating scalogram datasets…")
    train_ds = ScalogramDataset(train_dir, max_samples=max_samples)
    val_ds   = ScalogramDataset(val_dir,  max_samples=None)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check preprocessing paths.")

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=scalogram_pad_collate,
        pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=scalogram_pad_collate,
        pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False,
    )
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    return train_loader, val_loader
