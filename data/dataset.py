# /data/dataset.py
import torch
import numpy as np
import pathlib
import re
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import torchaudio

from var import LABELS

# -----------------------------
# simple 1D pad-collate for waveforms
# -----------------------------
def waveform_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Batch: list of (waveform_1d, label_vec).
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


# -----------------------------
# helpers for audio I/O and shaping
# -----------------------------
def _to_mono(x: torch.Tensor) -> torch.Tensor:
    """x: [C, T] -> [T] mono."""
    if x.dim() == 1:
        return x
    if x.size(0) == 1:
        return x.squeeze(0)
    return x.mean(dim=0)

_resamplers: Dict[tuple, torchaudio.transforms.Resample] = {}

def _resample_if_needed(x: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """x: [T] mono. Returns [T’] at sr_out."""
    if sr_in == sr_out:
        return x
    key = (sr_in, sr_out)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(sr_in, sr_out)
    # Resample expects [1, T]
    retu
