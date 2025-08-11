# models/Conv_wavelet_cnn.py
import torch
import torch.nn as nn

class WaveletCNN(nn.Module):
    """
    Pure CNN over scalograms:
      Input  : [B, 1, S, T]  (or [B, S, T])
      Output : [B, n_classes]  (sigmoid for multi-label)
    9 convs (GN+ReLU), pools between stages, GAP, 2xFC.
    """
    def __init__(self, n_classes: int):
        super().__init__()

        def block(in_ch, out_ch, groups):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(1,   32, 4),   # 1
            block(32,  32, 4),   # 2
            nn.MaxPool2d(2, 2),

            block(32,  64, 8),   # 3
            block(64,  64, 8),   # 4
            nn.MaxPool2d(2, 2),

            block(64,  128, 16), # 5
            block(128, 128, 16), # 6
            nn.MaxPool2d(2, 2),

            block(128, 256, 32), # 7
            block(256, 256, 32), # 8
            nn.MaxPool2d(2, 2),

            block(256, 512, 32), # 9
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

    def forward(self, x):
        # x: [B,1,S,T] or [B,S,T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)
