import torch
import torch.nn as nn

class PANNsFeatureExtractor(nn.Module):
    """Extract the convolutional layers from PANNs CNN14 for feature extraction."""

    def __init__(self, pretrained_path):
        super().__init__()

        # Load the pretrained PANNs model
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # CNN14 architecture (enhanced with additional conv blocks and increased filters)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # Increased filters from 64 to 96
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # Increased filters from 128 to 192
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # Increased filters from 256 to 384
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # Added additional conv layer
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # Increased filters from 512 to 768
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # Increased filters from 512 to 768
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Load pretrained weights from PANNs
        self.load_pretrained_weights(checkpoint)

    def load_pretrained_weights(self, checkpoint):
        """Load weights from PANNs checkpoint with intelligent upscaling."""
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

                    # ENHANCED: Handle first conv layer with 1 channel input
                    if our_key == 'conv_block1.0.weight':
                        # Handle first layer, which has 1 input channel in both models
                        if our_shape[1] == panns_shape[1] == 1:
                            # Intelligently expand the filters: tile or interpolate to get from 64 to 96 filters
                            source_weight = pretrained_dict[panns_key]  # Shape: [64, 1, 3, 3]
                            target_weight = torch.zeros(our_shape)  # Shape: [96, 1, 3, 3]

                            # Copy the original 64 filters
                            target_weight[:64] = source_weight

                            # For the additional 32 filters, create variations of existing ones
                            for i in range(64, 96):
                                # Use existing filters with small random variations
                                source_idx = i % 64  # Cycle through existing filters
                                target_weight[i] = source_weight[source_idx] * (0.9 + 0.2 * torch.rand(1))

                            filtered_dict[our_key] = target_weight
                            loaded_keys += 1
                            print(f"   ✅ Expanded first conv layer: {panns_shape} → {our_shape}")

                    # ENHANCED: Handle conv layer weight expansions (both input and output channels)
                    elif 'weight' in our_key and len(our_shape) == 4:  # Conv2d weights
                        if our_shape[0] > panns_shape[0] and our_shape[1] > panns_shape[1]:
                            # Both input and output channels need expanding
                            # Copy the core, then tile with variations for additional filters
                            source_weight = pretrained_dict[panns_key]
                            target_weight = torch.zeros(our_shape)

                            # Copy core weights
                            target_weight[:panns_shape[0], :panns_shape[1]] = source_weight

                            # Fill additional rows and columns with variations of existing filters
                            for i in range(our_shape[0]):
                                for j in range(our_shape[1]):
                                    if i >= panns_shape[0] or j >= panns_shape[1]:
                                        # Use existing weights with variations
                                        source_i = i % panns_shape[0]
                                        source_j = j % panns_shape[1]
                                        variation = 0.8 + 0.4 * torch.rand(1)
                                        target_weight[i, j] = source_weight[source_i, source_j] * variation

                            filtered_dict[our_key] = target_weight
                            loaded_keys += 1
                            print(f"   ✅ Expanded conv weights: {panns_shape} → {our_shape}")

                        elif our_shape[0] > panns_shape[0]:  # Just output channels expanded
                            # Copy original filters, then add variations for additional filters
                            source_weight = pretrained_dict[panns_key]
                            target_weight = torch.zeros(our_shape)

                            # Copy original filters
                            target_weight[:panns_shape[0]] = source_weight

                            # Create variations for additional filters
                            for i in range(panns_shape[0], our_shape[0]):
                                source_idx = i % panns_shape[0]
                                target_weight[i] = source_weight[source_idx] * (0.9 + 0.2 * torch.rand(1))

                            filtered_dict[our_key] = target_weight
                            loaded_keys += 1
                            print(f"   ✅ Expanded output channels: {panns_shape} → {our_shape}")

                    # ENHANCED: Handle BatchNorm parameter expansions
                    elif any(param in our_key for param in ['weight', 'bias', 'running_mean', 'running_var']) and len(our_shape) == 1:
                        # BatchNorm parameters need expansion
                        if our_shape[0] > panns_shape[0]:
                            source_param = pretrained_dict[panns_key]
                            target_param = torch.zeros(our_shape)

                            # Copy original parameters
                            target_param[:panns_shape[0]] = source_param

                            # For additional parameters, use mean values with small variations
                            if 'weight' in our_key or 'bias' in our_key:
                                mean_value = source_param.mean()
                                target_param[panns_shape[0]:] = mean_value * (0.95 + 0.1 * torch.rand(our_shape[0] - panns_shape[0]))
                            elif 'running_mean' in our_key:
                                mean_value = source_param.mean()
                                target_param[panns_shape[0]:] = mean_value * (0.98 + 0.04 * torch.rand(our_shape[0] - panns_shape[0]))
                            elif 'running_var' in our_key:
                                mean_value = source_param.mean()
                                target_param[panns_shape[0]:] = mean_value * (0.9 + 0.2 * torch.rand(our_shape[0] - panns_shape[0]))

                            filtered_dict[our_key] = target_param
                            loaded_keys += 1
                            print(f"   ✅ Expanded BN parameters: {panns_shape} → {our_shape}")

                    elif our_shape == panns_shape:  # Direct copy for matching shapes
                        filtered_dict[our_key] = pretrained_dict[panns_key]
                        loaded_keys += 1
                    else:
                        print(f"   ⚠️ Shape mismatch for {our_key}: {our_shape} vs {panns_shape}")

                except Exception as e:
                    print(f"   ❌ Error loading {our_key}: {e}")

        print(f"✅ Successfully loaded and expanded {loaded_keys}/{len(key_mapping)} layers from PANNs")

        # Update model weights
        if filtered_dict:
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"✅ PANNs pretrained weights integrated and expanded successfully!")
        else:
            print("⚠️ No pretrained weights were loaded. Using random initialization.")

    def forward(self, x):
        """Extract features from input spectrogram."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Flatten the output
        x = torch.flatten(x, 1)  # Output is now 768 features

        return x


class MultiSTFTCNN_WithPANNs(nn.Module):
    """Enhanced MultiSTFTCNN using PANNs pretrained features."""

    def __init__(self, n_classes, pretrained_path, freeze_backbone=True):
        super().__init__()

        # Create 3 PANNs feature extractors (one per spectrogram)
        self.feature_extractors = nn.ModuleList([
            PANNsFeatureExtractor(pretrained_path) for _ in range(3)
        ])

        # Further enhanced fusion layer to combine features from 3 spectrograms with increased capacity
        self.fusion = nn.Sequential(
            nn.Linear(3 * 768, 2048),  # 3 spectrograms × 768 features each
            nn.BatchNorm1d(2048),      # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.4),           # Increased dropout for better regularization
            nn.Linear(2048, 1536),     # First intermediate layer
            nn.BatchNorm1d(1536),      # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1536, 1024),     # Second intermediate layer
            nn.BatchNorm1d(1024),      # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),      # Final projection to match feature extractor size
            nn.BatchNorm1d(768),       # Added batch normalization
            nn.ReLU()
        )

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
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
            feat = self.feature_extractors[i](spec)  # [batch, 768]
            features.append(feat)

        # Concatenate all features
        combined_features = torch.cat(features, dim=1)  # [batch, 3*768]

        # Fusion and classification
        fused = self.fusion(combined_features)  # [batch, 768]
        output = self.classifier(fused)  # [batch, n_classes]

        return output