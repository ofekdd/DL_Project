
import torch, torch.nn as nn

class CNNBaseline(nn.Module):
    """Very small 2‑D CNN for log‑mel inputs (1×64×T)."""
    def __init__(self, n_classes: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
