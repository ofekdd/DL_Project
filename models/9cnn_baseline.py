import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTBranch(nn.Module):
    """One CNN branch that adapts to input shape."""
    def __init__(self, in_channels=1, out_features=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_features, kernel_size=3, padding=1), nn.BatchNorm2d(out_features), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Shape: (B, out_features, 1, 1)
            nn.Flatten()  # → (B, out_features)
        )

    def forward(self, x):
        return self.cnn(x)

class MultiSTFTCNN(nn.Module):
    def __init__(self, n_classes=11, n_branches=9, branch_output_dim=128):
        super().__init__()
        self.branches = nn.ModuleList([
            STFTBranch(out_features=branch_output_dim) for _ in range(n_branches)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(n_branches * branch_output_dim, n_classes),
            nn.Sigmoid()  # multi-label
        )

    def forward(self, x_list):
        """
        x_list: List[Tensor] of shape (B, 1, F, T), length = 9
        """
        features = [branch(x) for branch, x in zip(self.branches, x_list)]
        combined = torch.cat(features, dim=1)  # shape: (B, 9 * 128)
        return self.classifier(combined)



#model = MultiSTFTCNN(n_classes=11)
#outputs = model([
#    torch.randn(8, 1, 22, 500),  # example STFT 1
#    torch.randn(8, 1, 40, 400),  # example STFT 2
#    # ... more STFTs ...
#    torch.randn(8, 1, 50, 300),  # STFT 9
#])
