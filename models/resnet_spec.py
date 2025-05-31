
import torch, torch.nn as nn
from torchvision.models import resnet34

class ResNetSpec(nn.Module):
    """ResNet‑34 backbone adapted for single‑channel spectrogram input."""
    def __init__(self, n_classes: int = 11):
        super().__init__()
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace FC
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)
