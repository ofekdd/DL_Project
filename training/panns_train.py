#!/usr/bin/env python3
"""Entry point for training with PANNs warm-start."""
import pytorch_lightning as pl
import torch
import yaml
import argparse
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from models.panns_enhanced import MultiSTFTCNN_WithPANNs
from training.callbacks import default_callbacks
from training.metrics import MetricCollection
from data.dataset import create_dataloaders
from data.download_pnn import download_panns_checkpoint
from data.augmentations import apply_spec_augment
from var import LABELS


class PANNsLitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Get PANNs-specific config
        self.freeze_epochs = int(cfg.get('freeze_epochs', 3))
        self.frozen_lr = float(cfg.get('frozen_learning_rate', 1e-3))
        self.finetune_lr = float(cfg.get('finetune_learning_rate', 1e-4))
        self.mixup_alpha = float(cfg.get('mixup_alpha', 0.2))  # Mixup interpolation strength

        # Get PANNs checkpoint path
        panns_path = download_panns_checkpoint()

        # Create PANNs-enhanced model
        n_classes = len(LABELS)
        self.model = MultiSTFTCNN_WithPANNs(
            n_classes=n_classes,
            pretrained_path=panns_path,
            freeze_backbone=True
        )

        # Setup metrics
        self.metrics = MetricCollection(n_classes)

        # Save hyperparameters
        self.save_hyperparameters(cfg)

        print(f"✅ PANNs-enhanced model initialized!")
        print(f"   Initial phase: {self.freeze_epochs} epochs with frozen backbone (LR={self.frozen_lr})")
        print(f"   Fine-tuning phase: remaining epochs with full model (LR={self.finetune_lr})")

    def forward(self, x):
        return self.model(x)

    def mixup_data(self, x, y, alpha=0.2):
        '''Applies mixup augmentation to the batch'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x[0].size(0)
        index = torch.randperm(batch_size).to(x[0].device)

        mixed_x = []
        for spec in x:
            mixed_x.append(lam * spec + (1 - lam) * spec[index, :])

        # For one-hot encoded targets
        if y.dim() > 1:
            mixed_y = lam * y + (1 - lam) * y[index]
            return mixed_x, mixed_y, lam
        else:  # For class index targets
            y_a, y_b = y, y[index]
            return mixed_x, (y_a, y_b, lam)

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def common_step(self, batch, stage):
        x, y = batch

        # Apply mixup during training only
        if stage == 'train' and self.current_epoch >= self.freeze_epochs and np.random.random() < 0.5:
            # Use mixup data augmentation
            if y.dim() > 1 and y.size(1) > 1:  # Multi-hot encoded
                target_classes = torch.argmax(y, dim=1)  # Get the dominant instrument
                x, (targets_a, targets_b, lam), _ = self.mixup_data(x, target_classes)
                logits = self(x)
                loss = self.mixup_criterion(torch.nn.functional.cross_entropy, logits, targets_a, targets_b, lam)

                # For metrics, we'll use the primary target
                probs = torch.nn.functional.softmax(logits, dim=1)
                metrics = self.metrics(probs, targets_a)  # Use primary targets for metrics
            else:  # Already single-label format
                x, (targets_a, targets_b, lam), _ = self.mixup_data(x, y)
                logits = self(x)
                loss = self.mixup_criterion(torch.nn.functional.cross_entropy, logits, targets_a, targets_b, lam)

                # For metrics, we'll use the primary target
                probs = torch.nn.functional.softmax(logits, dim=1)
                metrics = self.metrics(probs, targets_a)  # Use primary targets for metrics
        else:
            # Regular forward pass without mixup
            # Apply SpecAugment during training
            if stage == 'train':
                x = apply_spec_augment(x, p=0.7)
            logits = self(x)

            # For single-label classification, use cross-entropy loss
            # Convert multi-hot targets to class indices if needed
            if y.dim() > 1 and y.size(1) > 1:  # Multi-hot encoded
                target_classes = torch.argmax(y, dim=1)  # Get the dominant instrument
            else:  # Already single-label format
                target_classes = y

            loss = torch.nn.functional.cross_entropy(logits, target_classes)

            # Apply softmax to get probabilities for metrics
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Calculate metrics
            metrics = self.metrics(probs, target_classes)

        self.log_dict({f"{stage}/loss": loss, **{f"{stage}/{k}": v for k, v in metrics.items()}},
                      prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self.common_step(batch, "train")

    def validation_step(self, batch, _):
        return self.common_step(batch, "val")

    def on_train_epoch_start(self):
        # Unfreeze backbone when reaching the specified epoch
        if self.current_epoch == self.freeze_epochs:
            self.model.unfreeze_backbone()
            print(f"\n{'='*80}")
            print(f"EPOCH {self.current_epoch}: UNFREEZING BACKBONE FOR FINE-TUNING")
            print(f"{'='*80}\n")

    def configure_optimizers(self):
        # Use different learning rates based on current training phase
        if self.current_epoch < self.freeze_epochs:
            # Only train fusion and classifier when backbone is frozen
            trainable_params = list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())
            lr = self.frozen_lr
            print(f"Optimizer: Using higher learning rate ({lr}) for fusion+classifier only")
        else:
            # Train everything after unfreezing
            trainable_params = self.parameters()
            lr = self.finetune_lr
            print(f"Optimizer: Using lower learning rate ({lr}) for full model fine-tuning")

        # Apply higher weight decay from config or default
        weight_decay = float(self.hparams.get('weight_decay', 1e-4))
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        # Implement learning rate scheduler if enabled
        if self.hparams.get('use_lr_scheduler', False):
            patience = int(self.hparams.get('lr_scheduler_patience', 3))
            factor = float(self.hparams.get('lr_scheduler_factor', 0.5))
            print(f"📉 Using LR scheduler: patience={patience}, factor={factor}")

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',
                factor=factor, 
                patience=patience, 
                verbose=True,
                min_lr=1e-6
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/Accuracy',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        return optimizer


def main(config):
    # Handle both file path and dictionary inputs
    if isinstance(config, dict):
        cfg = config
    else:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)

    # Get train and val directories from config
    train_dir = cfg.get('train_dir', "data/processed/train")
    val_dir = cfg.get('val_dir', "data/processed/val")

    print(f"Using train_dir: {train_dir}")
    print(f"Using val_dir: {val_dir}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=int(cfg['batch_size']),
        num_workers=int(cfg['num_workers']),
        use_multi_stft=True,  # Use MultiSTFTNpyDataset for spectrograms
        max_samples=cfg.get('max_samples', None)
    )

    # Create PANNs-enhanced model
    model = PANNsLitModel(cfg)

    # Setup callbacks with extended patience for the two-phase training
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/Accuracy', mode='max', patience=12),  # Increased patience to avoid early termination
        ModelCheckpoint(
            monitor='val/Accuracy',
            mode='max',
            save_top_k=3,  # Save top 3 models
            filename='panns-{epoch:02d}-{val_Accuracy:.3f}'
        )
    ]

    # Configure trainer with extended training epochs
    trainer_kwargs = {
        'max_epochs': int(cfg.get('num_epochs', 15)),  # Default to more epochs for the two-phase approach
        'callbacks': callbacks,
        'accelerator': "auto",
        'gradient_clip_val': float(cfg.get('gradiend_clip_val', 1.0)),  # Add gradient clipping
    }

    # Add validation efficiency settings if specified
    if 'limit_val_batches' in cfg:
        # Ensure limit_val_batches is a float
        limit_val_batches = float(cfg['limit_val_batches'])
        trainer_kwargs['limit_val_batches'] = limit_val_batches
        print(f"Limiting validation to {limit_val_batches * 100 if limit_val_batches <= 1 else limit_val_batches}{'%' if limit_val_batches <= 1 else ' batches'}")

    if 'num_sanity_val_steps' in cfg:
        # Ensure num_sanity_val_steps is an integer
        num_sanity_val_steps = int(cfg['num_sanity_val_steps'])
        trainer_kwargs['num_sanity_val_steps'] = num_sanity_val_steps
        print(f"Using {num_sanity_val_steps} sanity validation steps")

    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Print training plan
    print("\n" + "="*80)
    print("TRAINING PLAN:")
    print(f"- Phase 1 (epochs 0-{model.freeze_epochs-1}): Train only fusion & classifier layers")
    print(f"  Learning rate: {model.frozen_lr}")
    print(f"- Phase 2 (epochs {model.freeze_epochs}+): Fine-tune entire model")
    print(f"  Learning rate: {model.finetune_lr}")
    print("="*80 + "\n")

    # Start training
    trainer.fit(model, train_loader, val_loader)

    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/panns_enhanced.yaml")
    args = p.parse_args()
    main(args.config)
