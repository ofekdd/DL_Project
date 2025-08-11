# /data/preprocess.py
#!/usr/bin/env python3
"""Prepare IRMAS into train/val/test folders storing Morlet scalograms as .npy."""

import argparse
import pathlib
import random
import itertools
import yaml
import numpy as np
import librosa
import tqdm
import math
import torch
import torch.nn.functional as F

import math, torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def _make_scales(num_scales, s_min, s_max, device):
    return torch.logspace(
        math.log10(s_min), math.log10(s_max), num_scales,
        device=device, dtype=torch.float32
    )

@torch.no_grad()
def _morlet_kernel(scale_samples, w0, support, device, dtype=torch.complex64):
    s = float(scale_samples)
    half = int(math.ceil(support * s))
    t = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    gauss = torch.exp(- (t ** 2) / (2.0 * (s ** 2)))
    carrier = torch.exp(1j * (w0 * (t / s)))
    psi = (math.pi ** (-0.25)) * carrier * gauss
    psi = psi.to(dtype)
    psi = psi / torch.linalg.vector_norm(psi)
    return torch.flip(psi, dims=[0])  # conv-as-corr

@torch.no_grad>()
def _prep_kernels_fft(num_scales, s_min, s_max, w0, support, N_target, device):
    scales = _make_scales(num_scales, s_min, s_max, device)
    ks = []
    Ls = []
    for s in scales:
        k = _morlet_kernel(float(s.item()), w0, support, device)
        ks.append(k)
        Ls.append(k.numel())
    Lmax = max(Ls)
    pad_left = [F.pad(k, (Lmax - k.numel(), 0)) for k in ks]  # left-pad to align centers
    bank = torch.stack(pad_left, 0)  # [S, Lmax]

    # FFT length fixed so we can reuse kernel FFTs across files
    fft_len = 1 << int(math.ceil(math.log2(N_target + Lmax - 1)))
    K = torch.fft.rfft(bank, n=fft_len, dim=-1)  # [S, fft_len//2+1], complex
    center = (Lmax - 1) // 2  # for 'same' crop later
    return K, fft_len, center, Lmax

@torch.no_grad()
def cwt_fft_signal(y_1d: np.ndarray, K: torch.Tensor, fft_len: int, center: int, N_target: int, device):
    # y_1d MUST be length N_target (crop or right-pad beforehand)
    x = torch.tensor(y_1d, dtype=torch.float32, device=device)
    X = torch.fft.rfft(x, n=fft_len)              # [F]
    Y = torch.fft.irfft(K * X[None, :], n=fft_len, dim=-1)  # [S, fft_len]
    S = Y[:, center:center + N_target]            # [S, T] 'same' crop
    mag = torch.abs(S).clamp_min(1e-8)
    S_db = 20.0 * torch.log10(mag / mag.amax(dim=-1, keepdim=True).clamp_min(1e-8))
    return S_db  # [S, T] float32

# ----------------------------
# Wavelet helpers (CPU)
# ----------------------------
def _make_scales(num_scales: int = 256,
                 s_min_samples: float = 2.0,
                 s_max_samples: float = 512.0,
                 device="cpu"):
    return torch.logspace(
        math.log10(s_min_samples),
        math.log10(s_max_samples),
        num_scales,
        device=device,
        dtype=torch.float32
    )

def _morlet_kernel(scale_samples: float,
                   w0: float = 6.0,
                   support: float = 6.0,
                   device="cpu",
                   dtype=torch.complex64):
    s = float(scale_samples)
    half_len = int(math.ceil(support * s))
    t = torch.arange(-half_len, half_len + 1, device=device, dtype=torch.float32)
    gauss = torch.exp(- (t ** 2) / (2.0 * (s ** 2)))
    carrier = torch.exp(1j * (w0 * (t / s)))
    psi = (math.pi ** (-0.25)) * carrier * gauss
    psi = psi.to(dtype)
    psi = psi / torch.linalg.vector_norm(psi)
    return torch.flip(psi, dims=[0])  # conv-as-correlation

def _build_morlet_bank(num_scales=256, s_min=2.0, s_max=512.0, w0=6.0, support=6.0, device="cpu"):
    scales = _make_scales(num_scales, s_min, s_max, device=device)
    ker_list = []
    lengths = []
    for s in scales:
        ker = _morlet_kernel(float(s.item()), w0=w0, support=support, device=device)
        ker_list.append(ker)
        lengths.append(ker.numel())
    L_max = max(lengths)
    padded = [F.pad(k, (L_max - k.numel(), 0)) for k in ker_list]
    kernels = torch.stack(padded, dim=0).unsqueeze(1)  # [S,1,L]
    return kernels  # complex64

def wav_to_scalogram_db(y: np.ndarray,
                        num_scales=256,
                        s_min=2.0,
                        s_max=512.0,
                        w0=6.0,
                        support=6.0) -> np.ndarray:
    """
    y: mono float waveform (numpy) in [-1,1]
    Returns: log-magnitude scalogram in dB as float32, shape [S, T]
    """
    device = "cpu"
    x = torch.from_numpy(y.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    bank = _build_morlet_bank(num_scales, s_min, s_max, w0, support, device=device)   # [S,1,L]
    pad = (bank.shape[-1] - 1) // 2
    yR = F.conv1d(x, bank.real, padding=pad)
    yI = F.conv1d(x, bank.imag, padding=pad)
    cwt = torch.complex(yR, yI).squeeze(0)  # [S, T]
    mag = torch.abs(cwt).clamp_min(1e-8)
    mag_db = 20.0 * torch.log10(mag / mag.amax(dim=(-1), keepdim=True).clamp_min(1e-8))
    return mag_db.cpu().numpy().astype(np.float32)


# ----------------------------
# IO helpers
# ----------------------------
def load_waveform(path, target_sr, mono=True):
    y, _ = librosa.load(path, sr=target_sr, mono=mono)
    return y

def standardize_waveform(y: np.ndarray,
                         sr: int,
                         max_duration_sec: float | None = None,
                         peak_normalize: bool = True,
                         eps: float = 1e-8):
    if max_duration_sec is not None and max_duration_sec > 0:
        max_len = int(round(sr * max_duration_sec))
        if y.shape[0] > max_len:
            y = y[:max_len]
        elif y.shape[0] < max_len:
            y = np.pad(y, (0, max_len - y.shape[0]))
    if peak_normalize:
        peak = np.max(np.abs(y)) + eps
        if peak > 0:
            y = y / peak
    return y.astype(np.float32)

def save_scalogram(sample_dir: pathlib.Path, S_db: np.ndarray):
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "scalogram.npy", S_db)   # [S, T]


# ----------------------------
# Main preprocessing
# ----------------------------
def preprocess_data(irmas_root, out_dir, cfg, original_data_percentage=1.0):
    """
    Build processed folders with Morlet scalograms saved as scalogram.npy
    under per-sample directories (train/val/test).
    """
    irmas_root = pathlib.Path(irmas_root)
    out_dir = pathlib.Path(out_dir)
    train_dir, val_dir, test_dir = (out_dir / s for s in ("train", "val", "test"))
    for p in (train_dir, val_dir, test_dir):
        p.mkdir(parents=True, exist_ok=True)

    target_sr = int(cfg.get("sample_rate", 22050))
    max_duration_sec = cfg.get("max_duration_sec", None)
    peak_normalize = bool(cfg.get("peak_normalize", True))

    # wavelet config (keep consistent with predict)
    num_scales   = int(cfg.get("num_scales", 256))
    s_min_samples = float(cfg.get("s_min_samples", 2.0))
    s_max_samples = float(cfg.get("s_max_samples", 512.0))
    w0           = float(cfg.get("morlet_w0", 6.0))
    support      = float(cfg.get("morlet_support", 6.0))

    # 1) TRAIN / VAL
    train_src = irmas_root / "IRMAS-TrainingData"
    if not train_src.exists():
        raise FileNotFoundError(f"Expected training data at {train_src}")

    wav_files = list(train_src.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wavs found under {train_src}")

    print(f"🎻 Found {len(wav_files)} training wavs")
    if 0 < original_data_percentage < 1.0:
        k = int(len(wav_files) * original_data_percentage)
        wav_files = random.sample(wav_files, k)
        print(f"⚖️ Sub-sampling: {k} files ({original_data_percentage*100:.1f}%)")

    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * 0.8)
    tr_files, val_files = wav_files[:split_idx], wav_files[split_idx:]
    print(f"📊 Train/val split → {len(tr_files)} train | {len(val_files)} val")

    def _process_split(files, split_dir, prefix):
        for i, wav in enumerate(tqdm.tqdm(files, desc=f"→ {split_dir.name} ({len(files)})")):
            try:
                y = load_waveform(wav, target_sr, mono=True)
                y = standardize_waveform(y, target_sr, max_duration_sec=max_duration_sec,
                                         peak_normalize=peak_normalize)
                S_db = wav_to_scalogram_db(
                    y,
                    num_scales=num_scales,
                    s_min=s_min_samples, s_max=s_max_samples,
                    w0=w0, support=support
                )
                irmas_label = wav.parent.name  # single-label folders
                base = wav.stem
                sample_dir = split_dir / f"{prefix}_{irmas_label}_{i:04d}_{base}"
                save_scalogram(sample_dir, S_db)
            except Exception as e:
                print(f"❌ {wav}: {e}")

    _process_split(tr_files, train_dir, "original")
    _process_split(val_files, val_dir, "original")

    # 2) TEST
    test_roots = [p for p in irmas_root.glob("IRMAS-TestingData-Part*") if p.is_dir()]
    if not test_roots:
        sf = irmas_root / "IRMAS-TestingData"
        if sf.exists():
            test_roots = [sf]

    test_wavs = list(itertools.chain.from_iterable(tr.rglob("*.wav") for tr in test_roots))
    cap = cfg.get("max_test_samples")
    if cap is not None and len(test_wavs) > cap:
        random.shuffle(test_wavs)
        test_wavs = test_wavs[:cap]
        print(f"⚖️  Capping test set to {cap} samples (quick-run mode)")

    for i, wav in enumerate(tqdm.tqdm(test_wavs, desc="→ test")):
        try:
            y = load_waveform(wav, target_sr, mono=True)
            y = standardize_waveform(y, target_sr, max_duration_sec=max_duration_sec,
                                     peak_normalize=peak_normalize)

            S_db = wav_to_scalogram_db(
                y,
                num_scales=num_scales,
                s_min=s_min_samples, s_max=s_max_samples,
                w0=w0, support=support
            )

            txt = wav.with_suffix(".txt")
            if txt.exists():
                with open(txt) as fh:
                    lbls = [ln.strip() for ln in fh if ln.strip()]
                lbl_tag = "+".join(lbls) if lbls else "unknown"
            else:
                lbl_tag = "unknown"

            base = wav.stem
            sample_dir = test_dir / f"irmasTest_{lbl_tag}_{i:04d}_{base}"
            save_scalogram(sample_dir, S_db)
        except Exception as e:
            print(f"❌ {wav}: {e}")

    # 3) summary
    def _count(split_dir, tag):
        return len([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith(tag)])

    print("\n📋 Final counts")
    for split, p in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if p.exists():
            print(f"   {split:<5}: {len(list(p.iterdir())):>5} samples"
                  f"  (original={_count(p,'original')}, irmasTest={_count(p,'irmasTest')})")
    print(f"\n✅ Preprocessing finished. Scalograms saved to {out_dir}")


def main(in_dir, out_dir, config):
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(config, "r"))

    # direct export mode (process any wavs under in_dir)
    wav_files = list(in_dir.rglob("*.wav"))
    for wav in tqdm.tqdm(wav_files, desc="→ exporting"):
        y = load_waveform(wav, target_sr=cfg.get("sample_rate", 22050), mono=True)
        y = standardize_waveform(
            y, cfg.get("sample_rate", 22050),
            max_duration_sec=cfg.get("max_duration_sec", None),
            peak_normalize=bool(cfg.get("peak_normalize", True))
        )
        S_db = wav_to_scalogram_db(
            y,
            num_scales=int(cfg.get("num_scales", 256)),
            s_min=float(cfg.get("s_min_samples", 2.0)),
            s_max=float(cfg.get("s_max_samples", 512.0)),
            w0=float(cfg.get("morlet_w0", 6.0)),
            support=float(cfg.get("morlet_support", 6.0)),
        )
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = out_dir / rel_dir
        save_scalogram(file_out_dir, S_db)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=False, help="Optional direct export mode")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()

    if args.in_dir:
        main(args.in_dir, args.out_dir, args.config)
    else:
        cfg = yaml.safe_load(open(args.config, "r"))
        irmas_root = cfg.get("irmas_root", ".")
        original_data_percentage = cfg.get("original_data_percentage", 1.0)
        preprocess_data(
            irmas_root=irmas_root,
            out_dir=args.out_dir,
            cfg=cfg,
            original_data_percentage=original_data_percentage
        )
