import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multi_stft_cnn import STFTBranch

class WaveletCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            nn.Sigmoid()  # multi-label
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class MultiSTFTCNN_WithPANNs(nn.Module):
    """Enhanced MultiSTFTCNN using PANNs pretrained features."""

    def __init__(self, n_classes, pretrained_path, freeze_backbone=True):
        super().__init__()

        # Create 3 PANNs feature extractors (one per spectrogram)
        self.feature_extractors = nn.ModuleList([
            WaveletCNN(pretrained_path) for _ in range(3)
        ])

        # Fusion layer to combine features from 3 spectrograms
        self.fusion = nn.Sequential(
            nn.Linear(3 * 512, 1024),  # 3 spectrograms × 512 features each
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes),
            nn.Sigmoid()  # multi-label classification
        )

        # Initialize with backbone frozen
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze PANNs feature extractors."""
        for extractor in self.feature_extractors:
            for param in extractor.parameters():
                param.requires_grad = False
        print("📝 PANNs backbone layers frozen for initial training")

    def unfreeze_backbone(self):
        """Unfreeze PANNs feature extractors for fine-tuning."""
        for extractor in self.feature_extractors:
            for param in extractor.parameters():
                param.requires_grad = True
        print("🔓 PANNs backbone layers unfrozen for fine-tuning")

    def forward(self, spectrograms_list):
        """
        Forward pass with list of 3 spectrograms.

        Args:
            spectrograms_list: List of 3 tensors, each [batch, 1, freq, time]
        """
        features = []

        # Extract features from each spectrogram
        for i, spec in enumerate(spectrograms_list):
            feat = self.feature_extractors[i](spec)  # [batch, 512]
            features.append(feat)

        # Concatenate all features
        combined_features = torch.cat(features, dim=1)  # [batch, 3*512]

        # Fusion and classification
        fused = self.fusion(combined_features)  # [batch, 512]
        output = self.classifier(fused)  # [batch, n_classes]

        return output
