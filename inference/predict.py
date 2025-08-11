#!/usr/bin/env python3
"""CLI inference (scalogram CNN)."""
import argparse, yaml, torch, librosa, numpy as np, pathlib, math, torch.nn.functional as F
from utils.model_loader import load_model_from_checkpoint
from var import LABELS


# ---- same wavelet helpers as preprocess (kept local for inference) ----
def _make_scales(num_scales=256, s_min_samples=2.0, s_max_samples=512.0, device="cpu"):
    return torch.logspace(math.log10(s_min_samples), math.log10(s_max_samples),
                          num_scales, device=device, dtype=torch.float32)

def _morlet_kernel(scale_samples: float, w0=6.0, support=6.0, device="cpu", dtype=torch.complex64):
    s = float(scale_samples)
    half_len = int(math.ceil(support * s))
    t = torch.arange(-half_len, half_len + 1, device=device, dtype=torch.float32)
    gauss = torch.exp(- (t ** 2) / (2.0 * (s ** 2)))
    carrier = torch.exp(1j * (w0 * (t / s)))
    psi = (math.pi ** (-0.25)) * carrier * gauss
    psi = psi.to(dtype)
    psi = psi / torch.linalg.vector_norm(psi)
    return torch.flip(psi, dims=[0])

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
    return torch.stack(padded, dim=0).unsqueeze(1)  # [S,1,L]

def wav_to_scalogram_tensor(path, cfg, device):
    y, _ = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    # optional trim/pad for speed
    max_sec = cfg.get("max_duration_sec", None)
    if max_sec:
        max_len = int(round(cfg['sample_rate'] * max_sec))
        if y.shape[0] > max_len: y = y[:max_len]
        elif y.shape[0] < max_len: y = np.pad(y, (0, max_len - y.shape[0]))
    if np.max(np.abs(y)) > 0: y = y / (np.max(np.abs(y)) + 1e-8)

    bank = _build_morlet_bank(
        num_scales=int(cfg.get("num_scales", 256)),
        s_min=float(cfg.get("s_min_samples", 2.0)),
        s_max=float(cfg.get("s_max_samples", 512.0)),
        w0=float(cfg.get("morlet_w0", 6.0)),
        support=float(cfg.get("morlet_support", 6.0)),
        device=device
    )  # complex

    x = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    pad = (bank.shape[-1] - 1) // 2
    yR = F.conv1d(x, bank.real, padding=pad)
    yI = F.conv1d(x, bank.imag, padding=pad)
    cwt = torch.complex(yR, yI).squeeze(0)  # [S,T]
    mag = torch.abs(cwt).clamp_min(1e-8)
    S_db = 20.0 * torch.log10(mag / mag.amax(dim=-1, keepdim=True).clamp_min(1e-8))
    return S_db.unsqueeze(0)  # [1,S,T]


def predict(model, wav_path, cfg):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        S = wav_to_scalogram_tensor(wav_path, cfg, device)  # [1,S,T]
        logits = model(S.unsqueeze(0)).squeeze(0)           # -> model expects [B,1,S,T]
        # logits already sigmoid probs in our architecture:
        probs = logits.detach().cpu().numpy()
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    model = load_model_from_checkpoint(ckpt, n_classes=len(LABELS))
    result = predict(model, wav, cfg)

    print("\n" + "=" * 50)
    print(f"🎵 File: {pathlib.Path(wav).name}")
    sorted_preds = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for label, score in sorted_preds:
        icon = "🔥" if (label == sorted_preds[0][0]) else "·"
        print(f"  {icon} {label:<15} {score:.4f}")
    print("=" * 50)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)
