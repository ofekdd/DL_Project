#!/usr/bin/env python3
"""Entry point for training with PANNs warm-start."""
import pytorch_lightning as pl
import torch
import yaml
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from torchmetrics.classification import Accuracy

from models.panns_enhanced import MultiSTFTCNN_WithPANNs
from data.dataset import create_dataloaders
from data.download_pnn import download_panns_checkpoint
from var import LABELS


class PANNsLitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # Get PANNs-specific config
        self.freeze_epochs = cfg.get('freeze_epochs', 3)
        self.frozen_lr = cfg.get('frozen_learning_rate', 1e-3)
        self.finetune_lr = cfg.get('finetune_learning_rate', 1e-4)
        self.label_smoothing = cfg.get('label_smoothing', 0.0)

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
        self.metrics = Accuracy(task="multiclass", num_classes=n_classes)

        # Save hyperparameters
        self.save_hyperparameters(cfg)

        print(f"✅ PANNs-enhanced model initialized!")
        print(f"   Initial phase: {self.freeze_epochs} epochs with frozen backbone (LR={self.frozen_lr})")
        print(f"   Fine-tuning phase: remaining epochs with full model (LR={self.finetune_lr})")
        print(f"   Label smoothing: {self.label_smoothing}")

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, stage):
        x, y = batch
        logits = self(x)

        # For single-label classification, use cross-entropy loss
        # Convert multi-hot targets to class indices if needed
        if y.dim() > 1 and y.size(1) > 1:  # Multi-hot encoded
            target_classes = torch.argmax(y, dim=1)  # Get the dominant instrument
        else:  # Already single-label format
            target_classes = y

        loss = torch.nn.functional.cross_entropy(logits, target_classes, label_smoothing=self.label_smoothing)

        # Apply softmax to get probabilities for metrics
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Calculate metrics
        accuracy = self.metrics(probs, target_classes)

        # Metrics is now a scalar tensor, not a dict
        self.log_dict({f"{stage}/loss": loss, f"{stage}/Accuracy": accuracy},
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
        # Setup two param groups from the start. Backbone is frozen via requires_grad,
        # and will be unfrozen at epoch == freeze_epochs.
        backbone_params = []
        for extractor in self.model.feature_extractors:
            backbone_params += list(extractor.parameters())
        fusion_cls_params = list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())

        optimizer = torch.optim.Adam(
            [
                {"params": fusion_cls_params, "lr": self.frozen_lr},
                {"params": backbone_params, "lr": self.finetune_lr},
            ],
            weight_decay=2e-4,
        )
        # Mild cosine decay over total epochs (keeps LR scheduling simple & robust)
        t_max = int(self.hparams.get("num_epochs", 50)) if hasattr(self, "hparams") else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


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
    # SpecAugment and sampler settings
    spec_aug_cfg = cfg.get('spec_augment', {})
    augment = (isinstance(spec_aug_cfg, dict) and spec_aug_cfg.get('enabled', False)) or (isinstance(spec_aug_cfg, bool) and spec_aug_cfg)
    spec_aug_params = {k: v for k, v in spec_aug_cfg.items() if k != 'enabled'} if isinstance(spec_aug_cfg, dict) else None
    use_weighted_sampler = cfg.get('use_weighted_sampler', True)
    print(f"SpecAugment: {'ON' if augment else 'OFF'}; Weighted sampler: {'ON' if use_weighted_sampler else 'OFF'}")

    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        use_multi_stft=True,  # Use MultiSTFTNpyDataset for spectrograms
        max_samples=cfg.get('max_samples', None),
        augment=augment,
        spec_aug=spec_aug_params,
        use_weighted_sampler=use_weighted_sampler
    )

    # Create PANNs-enhanced model
    model = PANNsLitModel(cfg)

    # Setup callbacks with extended patience for the two-phase training
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/Accuracy', mode='max', patience=12),  # Increased patience for deeper model
        ModelCheckpoint(
            monitor='val/Accuracy',
            mode='max',
            save_top_k=2,  # Save top 2 models
            filename='panns-{epoch:02d}-{val_Accuracy:.3f}'
        )
    ]

    # Configure trainer with extended training epochs
    trainer_kwargs = {
        'max_epochs': cfg.get('num_epochs', 15),  # Default to more epochs for the two-phase approach
        'callbacks': callbacks,
        'accelerator': "auto",
    }

    # Mixed precision for VRAM savings (safe on Colab GPUs)
    if 'precision' in cfg:
        trainer_kwargs['precision'] = cfg['precision']
        print(f"Using precision={cfg['precision']}")

    # Add validation efficiency settings if specified
    if 'limit_val_batches' in cfg:
        trainer_kwargs['limit_val_batches'] = cfg['limit_val_batches']
        print(f"Limiting validation to {cfg['limit_val_batches'] * 100 if cfg['limit_val_batches'] <= 1 else cfg['limit_val_batches']}{'%' if cfg['limit_val_batches'] <= 1 else ' batches'}")

    if 'num_sanity_val_steps' in cfg:
        trainer_kwargs['num_sanity_val_steps'] = cfg['num_sanity_val_steps']
        print(f"Using {cfg['num_sanity_val_steps']} sanity validation steps")

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
