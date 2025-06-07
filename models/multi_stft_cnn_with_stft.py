import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from data.preprocess import generate_multi_stft
from var import n_ffts, band_ranges

class STFTBranch(nn.Module):
    def __init__(self, in_channels=1, out_features=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_features, kernel_size=3, padding=1), nn.BatchNorm2d(out_features), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.cnn(x)

class MultiSTFTCNN(nn.Module):
    def __init__(self, n_classes=11, n_branches=9, branch_output_dim=128, sr=22050, resize_to=(64, 128)):
        super().__init__()
        self.branches = nn.ModuleList([
            STFTBranch(out_features=branch_output_dim) for _ in range(n_branches)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(n_branches * branch_output_dim, n_classes),
            nn.Sigmoid()
        )
        self.sr = sr
        self.resize_to = resize_to

    def _process_spec(self, spec: np.ndarray):
        db = librosa.amplitude_to_db(spec + 1e-6)
        db = (db - db.mean()) / (db.std() + 1e-6)
        tensor = torch.tensor(db, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        tensor = F.interpolate(tensor.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False)
        return tensor.squeeze(0)  # [1, H, W]

    def forward(self, waveforms: torch.Tensor):
        """
        Args:
            waveforms: Tensor of shape (B, N) — raw audio per batch

        Returns:
            Output: Tensor of shape (B, n_classes)
        """
        batch_outputs = []
        for waveform in waveforms:
            waveform_np = waveform.detach().cpu().numpy()
            stft_dict = generate_multi_stft(waveform_np, self.sr)

            spectrograms = []
            for n_fft in n_ffts:
                for band in band_ranges:
                    key = (band, n_fft)
                    if key in stft_dict:
                        spec = self._process_spec(stft_dict[key])
                    else:
                        print(f"[WARN] Missing STFT for key {key}, using fallback zeros.")
                        spec = torch.zeros((1, *self.resize_to))
                    spectrograms.append(spec)

            batch_outputs.append(spectrograms)

        # Transpose: group all 9 views as 9 batches
        # Now we have: List[9 tensors of shape (B, 1, H, W)]
        inputs_per_branch = []
        for i in range(9):
            stacked = torch.stack([batch_outputs[b][i] for b in range(len(batch_outputs))])  # (B, 1, H, W)
            inputs_per_branch.append(stacked.to(waveforms.device))

        # Run each branch
        features = [branch(x) for branch, x in zip(self.branches, inputs_per_branch)]
        x = torch.cat(features, dim=1)  # shape: (B, 9 * 128)
        return self.classifier(x)
