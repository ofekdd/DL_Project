#!/usr/bin/env python3
"""Entry point for training with PANNs warm-start."""
import pytorch_lightning as pl
import torch
import yaml
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from losses.asymmetric_loss import AsymmetricLoss
from models.panns_enhanced import MultiSTFTCNN_WithPANNs
from training.callbacks import default_callbacks
from training.metrics import MetricCollection
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

        # Get PANNs checkpoint path
        panns_path = download_panns_checkpoint()

        # Create PANNs-enhanced model
        n_classes = len(LABELS)
        self.model = MultiSTFTCNN_WithPANNs(
            n_classes=n_classes,
            pretrained_path=panns_path,
            freeze_backbone=True
        )

        self.criterion = AsymmetricLoss(
            gamma_pos=cfg.get("gamma_pos", 0),
            gamma_neg=cfg.get("gamma_neg", 4),
            clip=cfg.get("neg_clip", 0.05),
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

    def common_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())

        # ── existing macro metrics ─────────────────────────────
        probs = torch.sigmoid(logits)
        metrics = self.metrics(probs, y)

        # ── NEW: class-wise Average Precision ─────────────────
        # 1. `metrics` (MetricCollection) keeps an internal buffer
        #    and exposes AP per class at any moment via .ap_per_class().
        # 2. We convert it to a dict {class_name: value}.
        ap_dict = {f"AP_{c}": ap for c, ap in metrics.ap_per_class().items()}

        # ── single combined logging call ───────────────────────
        self.log_dict(
            {f"{stage}/loss": loss,
             **{f"{stage}/{k}": v for k, v in metrics.items()},  # macro mAP, F1 …
             **{f"{stage}/{k}": v for k, v in ap_dict.items()}  # NEW per-class AP
             },
            prog_bar=True
        )

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

        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)


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
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        use_multi_stft=True,  # Use MultiSTFTNpyDataset for spectrograms
        max_samples=cfg.get('max_samples', None)
    )

    # Create PANNs-enhanced model
    model = PANNsLitModel(cfg)

    # Setup callbacks with extended patience for the two-phase training
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/mAP', mode='max', patience=8),  # More patience for two-phase training
        ModelCheckpoint(
            monitor='val/mAP',
            mode='max',
            save_top_k=3,
            save_last=True,
            filename='panns-{epoch:02d}-{val_mAP:.3f}'
        )
    ]

    # Configure trainer with extended training epochs
    trainer_kwargs = {
        'max_epochs': cfg.get('num_epochs', 15),  # Default to more epochs for the two-phase approach
        'callbacks': callbacks,
        'accelerator': "auto",
    }

    # Add validation efficiency settings if specified
    if 'limit_val_batches' in cfg:
        trainer_kwargs['limit_val_batches'] = cfg['limit_val_batches']
        print(f"Limiting validation to {cfg['limit_val_batches'] * 100 if cfg['limit_val_batches'] <= 1 else cfg['limit_val_batches']}{'%' if cfg['limit_val_batches'] <= 1 else ' batches'}")

    if 'num_sanity_val_steps' in cfg:
        trainer_kwargs['num_sanity_val_steps'] = cfg['num_sanity_val_steps']
        print(f"Using {cfg['num_sanity_val_steps']} sanity validation steps")

    if 'check_val_every_n_epoch' in cfg:
        trainer_kwargs['check_val_every_n_epoch'] = cfg['check_val_every_n_epoch']
        print(f"Running full validation every {cfg['check_val_every_n_epoch']} epoch(s)")

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
