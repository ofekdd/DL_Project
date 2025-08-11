# /data/dataset.py
import torch
import numpy as np
import pathlib
import re
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

# Try torchaudio, but don't require it
try:
    import torchaudio
    TORCHAUDIO_OK = True
except Exception:
    TORCHAUDIO_OK = False
    torchaudio = None

# Try librosa for fallback WAV loading (optional)
try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False
    librosa = None

from var import LABELS

# -----------------------------
# simple 1D pad-collate for waveforms
# -----------------------------
def waveform_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    padded = []
    for x in xs:
        if x.shape[0] < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - x.shape[0]))
        padded.append(x)
    return torch.stack(padded, 0), torch.stack(ys, 0)

# -----------------------------
# helpers
# -----------------------------
def _to_mono(x: torch.Tensor) -> torch.Tensor:
    # x: [C, T] or [T]
    if x.dim() == 1:
        return x
    if x.size(0) == 1:
        return x.squeeze(0)
    return x.mean(dim=0)

_resamplers: Dict[tuple, "torchaudio.transforms.Resample"] = {}

def _resample_if_needed(x: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return x
    if TORCHAUDIO_OK:
        key = (sr_in, sr_out)
        if key not in _resamplers:
            _resamplers[key] = torchaudio.transforms.Resample(sr_in, sr_out)
        return _resamplers[key](x.unsqueeze(0)).squeeze(0)
    # Fallback: librosa (only if available)
    if LIBROSA_OK:
        y = librosa.resample(x.cpu().numpy(), orig_sr=sr_in, target_sr=sr_out)
        return torch.from_numpy(y.astype(np.float32))
    # If no resampler, return as-is (not ideal, but safe)
    return x

def _fixed_length(x: torch.Tensor, target_len: Optional[int], split: str) -> torch.Tensor:
    if target_len is None:
        return x
    n = x.shape[0]
    if n == target_len:
        return x
    if n < target_len:
        return torch.nn.functional.pad(x, (0, target_len - n))
    # crop
    if split == 'train':
        import random
        start = random.randint(0, n - target_len)
    else:
        start = max(0, (n - target_len) // 2)
    return x[start:start + target_len]

class WaveformDataset(Dataset):
    """
    Expects per-sample folder containing either:
      - waveform.npy  (preferred, pre-resampled to cfg['sample_rate'])
      - audio.wav     (optional; loaded with torchaudio if available, else librosa)
    """
    def __init__(
        self,
        root: str | pathlib.Path,
        sample_rate: int,
        seconds: Optional[float] = 4.0,
        max_samples: Optional[int] = None,
        split: str = 'train',
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.sample_rate = int(sample_rate)
        self.target_len = int(seconds * sample_rate) if seconds is not None else None
        self.split = split

        npy_dirs = {p.parent for p in self.root.rglob("waveform.npy")}
        wav_dirs = {p.parent for p in self.root.rglob("audio.wav")}
        # prefer dirs that have an npy (no resampling cost)
        self.dirs = sorted(npy_dirs.union(wav_dirs))

        print(f"Found {len(self.dirs)} sample directories in {self.root}")
        if isinstance(max_samples, str):
            try: max_samples = None if max_samples.lower() == "none" else int(max_samples)
            except: max_samples = None
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
        irmas_matches = re.findall(r'\[([a-z]{3})\]', folder_name)
        for code in irmas_matches:
            if code in self.irmas_to_label_map:
                lbl = self.irmas_to_label_map[code]
                if lbl not in labels: labels.append(lbl)
        if not labels:
            token = folder_name.split("_")[0]
            if token in self.label_map: labels.append(token)
            elif token in self.irmas_to_label_map: labels.append(self.irmas_to_label_map[token])
        return labels

    def __len__(self): return len(self.dirs)

    def _load_waveform(self, d: pathlib.Path) -> torch.Tensor:
        npy_file = d / "waveform.npy"
        if npy_file.exists():
            try:
                x = np.load(npy_file).astype(np.float32)  # [T], already at target sr
                return torch.from_numpy(x)
            except Exception as e:
                print(f"⚠️  Corrupt waveform: {npy_file} – {e}")

        wav_file = d / "audio.wav"
        if wav_file.exists():
            # Use torchaudio if available
            if TORCHAUDIO_OK:
                try:
                    x, sr_in = torchaudio.load(str(wav_file))  # [C, T]
                    x = _to_mono(x).to(torch.float32)
                    x = _resample_if_needed(x, sr_in, self.sample_rate)
                    return x
                except Exception as e:
                    print(f"⚠️  Failed to load with torchaudio: {wav_file} – {e}")
            # Fallback to librosa (if present)
            if LIBROSA_OK:
                try:
                    y, _ = librosa.load(str(wav_file), sr=self.sample_rate, mono=True)
                    return torch.from_numpy(y.astype(np.float32))
                except Exception as e:
                    print(f"⚠️  Failed to load with librosa: {wav_file} – {e}")

        # total fallback
        return torch.zeros(1, dtype=torch.float32)

    def __getitem__(self, idx: int):
        d = self.dirs[idx]
        x = self._load_waveform(d)            # [T]
        if x.numel() > 0:
            peak = x.abs().max().clamp_min(1e-8)
            x = (x / peak).clamp(-1.0, 1.0)
        x = _fixed_length(x, self.target_len, split=self.split)

        y = torch.zeros(len(LABELS), dtype=torch.long)
        for lbl in self._parse_labels_from_folder_name(d.name):
            if lbl in self.label_map:
                y[self.label_map[lbl]] = 1
        return x, y


def create_dataloaders(
    train_dir: str | pathlib.Path,
    val_dir: str | pathlib.Path,
    batch_size: int,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None,
    sample_rate: int = 22050,
    seconds: Optional[float] = 4.0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
):
    print("Creating waveform datasets…")

    use_cuda = torch.cuda.is_available()
    if pin_memory is None:
        pin_memory = bool(use_cuda)
    if num_workers is None:
        num_workers = 2 if use_cuda else 0
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    train_ds = WaveformDataset(train_dir, sample_rate, seconds, max_samples, split='train')
    val_ds   = WaveformDataset(val_dir,   sample_rate, seconds, None,        split='val')

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check preprocessing paths.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=waveform_pad_collate,
        pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=waveform_pad_collate,
        pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False,
    )
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    return train_loader, val_loader
