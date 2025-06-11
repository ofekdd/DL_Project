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

        # Filter and adapt keys to match our architecture
        filtered_dict = {}

        # Mapping between PANNs keys and our keys
        key_mapping = {
            'conv_block1.0.weight': 'conv1.weight',
            'conv_block1.1.weight': 'bn1.weight',
            'conv_block1.1.bias': 'bn1.bias',
            'conv_block1.1.running_mean': 'bn1.running_mean',
            'conv_block1.1.running_var': 'bn1.running_var',

            'conv_block2.0.weight': 'conv2.weight',
            'conv_block2.1.weight': 'bn2.weight',
            'conv_block2.1.bias': 'bn2.bias',
            'conv_block2.1.running_mean': 'bn2.running_mean',
            'conv_block2.1.running_var': 'bn2.running_var',

            'conv_block3.0.weight': 'conv3.weight',
            'conv_block3.1.weight': 'bn3.weight',
            'conv_block3.1.bias': 'bn3.bias',
            'conv_block3.1.running_mean': 'bn3.running_mean',
            'conv_block3.1.running_var': 'bn3.running_var',

            'conv_block4.0.weight': 'conv4.weight',
            'conv_block4.1.weight': 'bn4.weight',
            'conv_block4.1.bias': 'bn4.bias',
            'conv_block4.1.running_mean': 'bn4.running_mean',
            'conv_block4.1.running_var': 'bn4.running_var',
        }

        # Attempt to load weights, but don't fail if they don't match
        loaded_keys = 0
        for our_key, panns_key in key_mapping.items():
            if panns_key in pretrained_dict:
                try:
                    our_shape = model_dict[our_key].shape
                    panns_shape = pretrained_dict[panns_key].shape

                    # Handle first conv layer - PANNs might have different input channels
                    if our_key == 'conv_block1.0.weight' and our_shape != panns_shape:
                        # Adapt the first layer if needed (e.g., different input channels)
                        if our_shape[1] != panns_shape[1]:
                            # If we have 1 channel and PANNs has 3, average over the 3 channels
                            adapted_weight = pretrained_dict[panns_key].mean(dim=1, keepdim=True)
                            if adapted_weight.shape == our_shape:
                                filtered_dict[our_key] = adapted_weight
                                loaded_keys += 1
                    elif our_shape == panns_shape:
                        filtered_dict[our_key] = pretrained_dict[panns_key]
                        loaded_keys += 1
                except Exception as e:
                    print(f"Couldn't load {our_key} from {panns_key}: {e}")

        print(f"✅ Loaded {loaded_keys} layers from PANNs pretrained model")

        # Update model weights with pretrained weights
        if filtered_dict:
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
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
