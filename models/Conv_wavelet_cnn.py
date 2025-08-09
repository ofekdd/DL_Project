import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# ----------------------------
# 1) scale grid (your earlier choice)
# ----------------------------
def make_scales(num_scales: int = 256,
                s_min_samples: float = 2.0,
                s_max_samples: float = 512.0,
                device=None):
    """
    Log-spaced scales (in SAMPLES!) between s_min_samples and s_max_samples.
    """
    device = device or 'cpu'
    return torch.logspace(
        math.log10(s_min_samples),
        math.log10(s_max_samples),
        num_scales,
        device=device,
        dtype=torch.float32
    )


def morlet_kernel(scale_samples: float,
                  w0: float = 6.0,
                  support: float = 6.0,
                  device=None,
                  dtype=torch.complex64):
    """
    Complex Morlet kernel at a given scale (scale in SAMPLES).
    t is measured in samples; Gaussian uses (t/scale)^2; carrier uses t/scale.
    """
    device = device or 'cpu'
    s = float(scale_samples)

    half_len = int(math.ceil(support * s))
    t = torch.arange(-half_len, half_len + 1, device=device, dtype=torch.float32)  # samples

    gauss = torch.exp(- (t ** 2) / (2.0 * (s ** 2)))
    carrier = torch.exp(1j * (w0 * (t / s)))  # radians

    psi = (math.pi ** (-0.25)) * carrier * gauss
    psi = psi.to(dtype)

    # L2 normalize
    psi = psi / torch.linalg.vector_norm(psi)

    # flip for conv-as-correlation
    return torch.flip(psi, dims=[0])


def build_morlet_bank(sr: float,
                      num_scales: int = 256,
                      s_min: float = 2.0,
                      s_max: float = 512.0,
                      w0: float = 6.0,
                      support: float = 6.0,
                      device=None,
                      dtype=torch.complex64):
    """
    Bank of complex Morlet kernels, each padded to the max length.
    NOTE: s_min and s_max are in SAMPLES (not seconds).
    """
    device = device or 'cpu'
    scales = make_scales(num_scales, s_min, s_max, device=device)  # samples
    ker_list, lengths = [], []

    for s in scales:
        ker = morlet_kernel(float(s.item()), w0=w0, support=support, device=device, dtype=dtype)
        ker_list.append(ker)
        lengths.append(ker.numel())

    L_max = max(lengths)

    padded = []
    for ker in ker_list:
        pad_left = L_max - ker.numel()
        padded.append(F.pad(ker, (pad_left, 0)))

    kernels = torch.stack(padded, dim=0).unsqueeze(1)  # [S,1,L]
    return kernels, lengths, scales


# ----------------------------
# 4) fast application: conv1d (preferred)
# ----------------------------
def cwt_conv1d(x: torch.Tensor,
               kernels: torch.Tensor,
               padding: str = 'same'):
    """
    x: [B, 1, N] real or complex
    kernels: [S, 1, L] complex
    returns complex CWT: [B, S, N]
    """
    # PyTorch conv1d expects real dtype. Split complex into real+imag:
    k_real = torch.view_as_real(kernels).permute(0,1,2,3)  # [S,1,L,2]
    # Split into two real convs: Re and Im parts
    kR = kernels.real
    kI = kernels.imag
    # padding
    if padding == 'same':
        pad = (kernels.shape[-1] - 1) // 2
    else:
        pad = 0
    # convs
    yR = F.conv1d(x, kR, padding=pad) - F.conv1d(x, kI, padding=pad) * 0  # keep explicit
    yI = F.conv1d(x, kI, padding=pad)
    # pack back to complex
    y = torch.complex(yR, yI)
    return y.squeeze(1)  # [B, S, N]


# ----------------------------
# 5) optional: explicit sparse operator W (huge; for small N only)
#    W @ a  == vec(CWT) with 'same' padding
# ----------------------------
'''
def build_sparse_wavelet_operator(signal_len: int,
                                  sr: float,
                                  num_scales: int = 512,
                                  s_min: float = 1.0,
                                  s_max: float = 512.0,
                                  w0: float = 6.0,
                                  support: float = 8.0,
                                  device=None,
                                  dtype=torch.complex64):
    """
    Builds a sparse COO matrix W of shape [num_scales*signal_len, signal_len]
    such that vec = W @ a  ==  CWT flattened by scale-major.
    NOTE: memory/time grows O(S * N * L). Use only for small N.
    """
    device = device or 'cpu'
    kernels, lengths, scales = build_morlet_bank(sr, num_scales, s_min, s_max, w0, support, device, dtype)
    S, _, L = kernels.shape
    N = signal_len

    # collect COO entries
    rows, cols, vals_real, vals_imag = [], [], [], []
    for s_idx in range(S):
        ker = kernels[s_idx, 0]  # [L], complex
        # center alignment for 'same'
        center = (L - 1) // 2
        for t in range(N):
            # row block offset for this scale
            row_base = s_idx * N + t
            # place kernel centered at t
            left = t - center
            for k in range(L):
                c = left + k
                if 0 <= c < N:
                    rows.append(row_base)
                    cols.append(c)
                    vals_real.append(float(ker[k].real.item()))
                    vals_imag.append(float(ker[k].imag.item()))
    # build complex as two real sparse matrices combined
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values_real = torch.tensor(vals_real, dtype=torch.float32, device=device)
    values_imag = torch.tensor(vals_imag, dtype=torch.float32, device=device)
    W_real = torch.sparse_coo_tensor(indices, values_real, (S * N, N), device=device)
    W_imag = torch.sparse_coo_tensor(indices, values_imag, (S * N, N), device=device)
    return W_real, W_imag  # use as: y = (W_real @ a) + 1j*(W_imag @ a)
'''

class WaveletFrontEnd(torch.nn.Module):
    def __init__(self,
                 sr: int,
                 num_scales: int = 256,
                 s_min: float = 2.0,
                 s_max: float = 512.0,
                 w0: float = 6.0,
                 support: float = 6.0,
                 trainable: bool = False):
        super().__init__()
        ker, _, _ = build_morlet_bank(
            sr=sr,
            num_scales=num_scales,
            s_min=s_min,   # samples
            s_max=s_max,   # samples
            w0=w0,
            support=support,
            device='cpu',
            dtype=torch.complex64
        )
        if trainable:
            self.kR = torch.nn.Parameter(ker.real, requires_grad=True)
            self.kI = torch.nn.Parameter(ker.imag, requires_grad=True)
        else:
            self.register_buffer('kR', ker.real)
            self.register_buffer('kI', ker.imag)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B,1,N]
        pad = (self.kR.shape[-1] - 1) // 2
        yR = F.conv1d(x, self.kR, padding=pad)
        yI = F.conv1d(x, self.kI, padding=pad)
        y = torch.complex(yR, yI)              # [B,S,N]
        mag = torch.abs(y).clamp_min(1e-8)
        mag_db = 20.0 * torch.log10(mag / mag.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8))
        return y, mag_db


class WaveletCNN(nn.Module):
    """
    WaveletFrontEnd -> 9x Conv (GN+ReLU), pooled between stages -> GAP -> 2x FC.
    Input:  waveform [B, N]
    Output: probs [B, n_classes]  (sigmoid for multi-label)
    """
    def __init__(self, n_classes: int, sr: int, num_scales: int = 512):
        super().__init__()
        self.cwt = WaveletFrontEnd(sr=sr, num_scales=num_scales)  # builds once

        def block(in_ch, out_ch, groups):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(inplace=True),
            )

        # 9 convs total:
        self.features = nn.Sequential(
            # Stage 1 (2 convs) -> pool
            block(1,   32, 4),     # conv1
            block(32,  32, 4),     # conv2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2 (2 convs) -> pool
            block(32,  64, 8),     # conv3
            block(64,  64, 8),     # conv4
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3 (2 convs) -> pool
            block(64,  128, 16),   # conv5
            block(128, 128, 16),   # conv6
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4 (2 convs) -> pool
            block(128, 256, 32),   # conv7
            block(256, 256, 32),   # conv8
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 5 (1 conv) -> no pool (9th conv)
            block(256, 512, 32),   # conv9

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # -> [B, 512]
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, N] real audio
        _, scalogram_db = self.cwt(waveform)   # [B, S, T]
        x = scalogram_db.unsqueeze(1)          # [B, 1, S, T]
        x = self.features(x)                   # [B, 512, 1, 1]
        return self.classifier(x)              # [B, n_classes]



if __name__ == "__main__":
    import torch

    # pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # optional speedup when input sizes are fairly consistent
    torch.backends.cudnn.benchmark = True

    n_classes = 11
    sr = 22050

    # build model (kernels are created once) and move to device
    model = WaveletCNN(n_classes=n_classes, sr=sr).to(device)
    model.eval()

    # fake batch of waveforms on the same device
    B = 2
    seconds = 2.0
    N = int(sr * seconds)
    x = torch.randn(B, N, device=device)

    with torch.no_grad():
        y = model(x)

    print("Input waveform:", x.shape)  # [B, N]
    print("Output:", y.shape)          # [B, n_classes]
    print("Output device:", y.device)


