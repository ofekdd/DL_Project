

# /data/dataset.py
import torch
import numpy as np
import pathlib
import re
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

from var import LABELS

# -----------------------------
# simple 1D pad-collate for waveforms
# -----------------------------
def waveform_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Batch is a list of (waveform_1d, label_vec).
    Pads each waveform to the max length in the batch (right-pad with zeros).
    Returns:
        x : FloatTensor [B, T_max]
        y : LongTensor  [B, num_classes]
    """
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    padded = []
    for x in xs:
        if x.shape[0] < max_len:
            pad = max_len - x.shape[0]
            x = torch.nn.functional.pad(x, (0, pad))  # right-pad
        padded.append(x)
    x_batch = torch.stack(padded, dim=0)          # [B, T_max]
    y_batch = torch.stack(ys, dim=0)              # [B, C]
    return x_batch, y_batch


class WaveformNpyDataset(Dataset):
    """
    Loads one waveform per sample from <sample_dir>/waveform.npy and
    builds a multi-hot label vector from the directory name.

    Expects your preprocessor to create structure like:
      processed/train/<sample_dir>/waveform.npy
      processed/val/<sample_dir>/waveform.npy
      processed/test/<sample_dir>/waveform.npy
    """

    def __init__(self, root: str | pathlib.Path, max_samples: Optional[int] = None):
        root = pathlib.Path(root)

        # collect sample dirs that contain waveform.npy
        self.dirs = sorted({p.parent for p in root.rglob("waveform.npy")})
        print(f"Found {len(self.dirs)} sample directories in {root}")

        # optional cap
        if isinstance(max_samples, str):
            if max_samples.lower() == "none":
                max_samples = None
            else:
                try:
                    max_samples = int(max_samples)
                except ValueError:
                    print(f"Warning: cannot parse max_samples='{max_samples}', using all")
                    max_samples = None

        if isinstance(max_samples, int) and 0 < max_samples < len(self.dirs):
            self.dirs = self.dirs[:max_samples]
            print(f"Limiting to {max_samples} samples")

        # label maps
        self.label_map = {label: i for i, label in enumerate(LABELS)}
        self.irmas_to_label_map = {
            'cel': 'cello',
            'cla': 'clarinet',
            'flu': 'flute',
            'gac': 'acoustic_guitar',
            'gel': 'acoustic_guitar',
            'org': 'organ',
            'pia': 'piano',
            'sax': 'saxophone',
            'tru': 'trumpet',
            'vio': 'violin',
            'voi': 'voice'
        }

        print(f"Dataset ready. Final size: {len(self.dirs)}")
        for i, d in enumerate(self.dirs[:3]):
            try:
                print(f"  ex[{i}]: {d.name} -> {self._parse_labels_from_folder_name(d.name)}")
            except Exception as e:
                print(f"  ex[{i}]: {d.name} (label parse error: {e})")

    # -----------------------------
    # label parsing from folder name
    # -----------------------------
    def _parse_labels_from_folder_name(self, folder_name: str) -> list[str]:
        """
        Handles:
        - IRMAS test-style names: "...[cel][pia]..." -> multi-label
        - simple: "piano_123"
        """
        labels: list[str] = []

        # IRMAS pattern: [cel] [pia] ...
        irmas_pattern = r'\[([a-z]{3})\]'
        irmas_matches = re.findall(irmas_pattern, folder_name)
        for irmas_lbl in irmas_matches:
            if irmas_lbl in self.irmas_to_label_map:
                mapped = self.irmas_to_label_map[irmas_lbl]
                if mapped not in labels:
                    labels.append(mapped)

        # fallback simple first token if none found
        if not labels:
            token = folder_name.split("_")[0]
            if token in self.label_map:
                labels.append(token)
            elif token in self.irmas_to_label_map:
                labels.append(self.irmas_to_label_map[token])

        return labels

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx: int):
        d = self.dirs[idx]
        wav_path = d / "waveform.npy"
        if not wav_path.exists():
            # fallback (shouldn't happen if preprocessing succeeded)
            x = np.zeros((1,), dtype=np.float32)
        else:
            try:
                x = np.load(wav_path).astype(np.float32)  # [T]
            except Exception as e:
                print(f"⚠️  Corrupt waveform: {wav_path} – {e}")
                x = np.zeros((1,), dtype=np.float32)

        # to tensor 1D
        x_t = torch.from_numpy(x)  # [T]

        # build multi-hot y
        y = torch.zeros(len(LABELS), dtype=torch.long)
        for lbl in self._parse_labels_from_folder_name(d.name):
            if lbl in self.label_map:
                y[self.label_map[lbl]] = 1

        if y.sum() == 0:
            # keep training robust, but let us know
            # print(f"Warning: no labels parsed for '{d.name}'")
            pass

        return x_t, y


def create_dataloaders(
    train_dir: str | pathlib.Path,
    val_dir: str | pathlib.Path,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None
):
    """
    Build train/val loaders that return:
      x: FloatTensor [B, T_max]  (padded)
      y: LongTensor  [B, C]
    """
    print(f"Creating waveform datasets…")
    train_ds = WaveformNpyDataset(train_dir, max_samples=max_samples)
    val_ds   = WaveformNpyDataset(val_dir,  max_samples=None)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check preprocessing paths.")

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=waveform_pad_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=waveform_pad_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    return train_loader, val_loader
