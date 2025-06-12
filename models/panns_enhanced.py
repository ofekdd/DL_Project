import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multi_stft_cnn import STFTBranch

class PANNsFeatureExtractor(nn.Module):
    """Extract the convolutional layers from PANNs CNN14 for feature extraction."""

    def __init__(self, pretrained_path):
        super().__init__()

        # Load the pretrained PANNs model
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # CNN14 architecture (simplified - key conv blocks)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Load pretrained weights from PANNs
        self.load_pretrained_weights(checkpoint)

    def load_pretrained_weights(self, checkpoint):
        """Load weights from PANNs checkpoint."""
        model_dict = self.state_dict()
        pretrained_dict = checkpoint['model']

        # Debug: Print available PANNs keys to understand the structure
        panns_conv_keys = [k for k in pretrained_dict.keys() if 'conv_block' in k]
        print(f"🔍 Found {len(panns_conv_keys)} PANNs conv keys")
        if panns_conv_keys:
            print(f"   Sample keys: {panns_conv_keys[:5]}")  # Show first 5 keys

        # CORRECTED mapping based on actual PANNs CNN14 structure
        key_mapping = {
            # Our key -> PANNs key (note the different naming convention)
            'conv_block1.0.weight': 'conv_block1.conv1.weight',
            'conv_block1.1.weight': 'conv_block1.bn1.weight',
            'conv_block1.1.bias': 'conv_block1.bn1.bias',
            'conv_block1.1.running_mean': 'conv_block1.bn1.running_mean',
            'conv_block1.1.running_var': 'conv_block1.bn1.running_var',
            'conv_block1.1.num_batches_tracked': 'conv_block1.bn1.num_batches_tracked',

            'conv_block2.0.weight': 'conv_block2.conv1.weight',
            'conv_block2.1.weight': 'conv_block2.bn1.weight',
            'conv_block2.1.bias': 'conv_block2.bn1.bias',
            'conv_block2.1.running_mean': 'conv_block2.bn1.running_mean',
            'conv_block2.1.running_var': 'conv_block2.bn1.running_var',
            'conv_block2.1.num_batches_tracked': 'conv_block2.bn1.num_batches_tracked',

            'conv_block3.0.weight': 'conv_block3.conv1.weight',
            'conv_block3.1.weight': 'conv_block3.bn1.weight',
            'conv_block3.1.bias': 'conv_block3.bn1.bias',
            'conv_block3.1.running_mean': 'conv_block3.bn1.running_mean',
            'conv_block3.1.running_var': 'conv_block3.bn1.running_var',
            'conv_block3.1.num_batches_tracked': 'conv_block3.bn1.num_batches_tracked',

            'conv_block4.0.weight': 'conv_block4.conv1.weight',
            'conv_block4.1.weight': 'conv_block4.bn1.weight',
            'conv_block4.1.bias': 'conv_block4.bn1.bias',
            'conv_block4.1.running_mean': 'conv_block4.bn1.running_mean',
            'conv_block4.1.running_var': 'conv_block4.bn1.running_var',
            'conv_block4.1.num_batches_tracked': 'conv_block4.bn1.num_batches_tracked',
        }

        # Filter and load weights
        filtered_dict = {}
        loaded_keys = 0

        for our_key, panns_key in key_mapping.items():
            if panns_key in pretrained_dict and our_key in model_dict:
                try:
                    our_shape = model_dict[our_key].shape
                    panns_shape = pretrained_dict[panns_key].shape

                    # Handle first conv layer input channel mismatch
                    if our_key == 'conv_block1.0.weight':
                        if our_shape[1] == 1 and panns_shape[1] == 1:
                            # Both single channel - direct copy
                            filtered_dict[our_key] = pretrained_dict[panns_key]
                            loaded_keys += 1
                        elif our_shape[1] == 1 and panns_shape[1] > 1:
                            # PANNs has more channels, take average
                            adapted_weight = pretrained_dict[panns_key].mean(dim=1, keepdim=True)
                            filtered_dict[our_key] = adapted_weight
                            loaded_keys += 1
                            print(f"   ✅ Adapted first conv layer: {panns_shape} -> {our_shape}")
                        else:
                            print(f"   ⚠️ Cannot adapt first conv layer: {panns_shape} vs {our_shape}")
                    elif our_shape == panns_shape:
                        # Direct copy for matching shapes
                        filtered_dict[our_key] = pretrained_dict[panns_key]
                        loaded_keys += 1
                    else:
                        print(f"   ⚠️ Shape mismatch for {our_key}: {our_shape} vs {panns_shape}")

                except Exception as e:
                    print(f"   ❌ Error loading {our_key}: {e}")

        print(f"✅ Successfully loaded {loaded_keys}/{len(key_mapping)} layers from PANNs")

        # Update model weights
        if filtered_dict:
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
            print(f"✅ PANNs pretrained weights integrated successfully!")
        else:
            print("⚠️ No pretrained weights were loaded. Using random initialization.")

    def forward(self, x):
        """Extract features from input spectrogram."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Flatten the output
        x = torch.flatten(x, 1)

        return x

class MultiSTFTCNN_WithPANNs(nn.Module):
    """Enhanced MultiSTFTCNN using PANNs pretrained features."""

    def __init__(self, n_classes, pretrained_path, freeze_backbone=True):
        super().__init__()

        # Create 9 PANNs feature extractors (one per spectrogram)
        self.feature_extractors = nn.ModuleList([
            PANNsFeatureExtractor(pretrained_path) for _ in range(9)
        ])

        # Fusion layer to combine features from 9 spectrograms
        self.fusion = nn.Sequential(
            nn.Linear(9 * 512, 1024),  # 9 spectrograms × 512 features each
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
        Forward pass with list of 9 spectrograms.

        Args:
            spectrograms_list: List of 9 tensors, each [batch, 1, freq, time]
        """
        features = []

        # Extract features from each spectrogram
        for i, spec in enumerate(spectrograms_list):
            feat = self.feature_extractors[i](spec)  # [batch, 512]
            features.append(feat)

        # Concatenate all features
        combined_features = torch.cat(features, dim=1)  # [batch, 9*512]

        # Fusion and classification
        fused = self.fusion(combined_features)  # [batch, 512]
        output = self.classifier(fused)  # [batch, n_classes]

        return output
