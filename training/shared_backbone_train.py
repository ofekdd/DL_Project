#!/usr/bin/env python3

import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import create_dataloaders
from models.shared_backbone_panns import SharedBackbonePANNs
from data.download_pnn import download_panns_checkpoint
from var import LABELS
from utils.callbacks import default_callbacks


class InstrumentClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Define metrics
        from torchmetrics import MetricCollection
        from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

        metrics = MetricCollection({
            'mAP': MultilabelAveragePrecision(num_labels=len(LABELS)),
            'F1': MultilabelF1Score(num_labels=len(LABELS), threshold=0.5)
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

        # For loss logging
        self.train_loss = 0.0
        self.train_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

        # Update and log metrics
        self.train_metrics.update(y_hat, y.int())

        # Simple loss averaging
        self.train_loss += loss.item()
        self.train_count += 1

        # Log to progress bar only (batch metrics are noisy)
        self.log('train/loss', loss, prog_bar=True, logger=False)

        return loss

    def on_train_epoch_end(self):
        # Log metrics
        self.log_dict(self.train_metrics.compute())

        # Log average loss for the epoch
        if self.train_count > 0:
            avg_loss = self.train_loss / self.train_count
            self.log('train/loss', avg_loss)
            self.train_loss = 0.0
            self.train_count = 0

        # Reset metrics for next epoch
        self.train_metrics.reset()

        # Unfreeze backbone after 3 epochs
        if self.current_epoch == 2:
            print("\n" + "=" * 80)
            print(f"EPOCH {self.current_epoch+1}: UNFREEZING BACKBONE FOR FINE-TUNING")
            print("=" * 80 + "\n")
            self.model.unfreeze_backbone()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

        # Update metrics
        self.val_metrics.update(y_hat, y.int())

        # Log validation loss
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        # Log metrics
        self.log_dict(self.val_metrics.compute())

        # Reset metrics for next epoch
        self.val_metrics.reset()

    def configure_optimizers(self):
        # Define optimizer with different learning rates
        backbone_params = list(self.model.shared_backbone.parameters())
        adapter_params = list(self.model.spec_adapters.parameters())
        fusion_params = list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())

        # Use parameter groups with different learning rates
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower rate for backbone
            {'params': adapter_params, 'lr': self.learning_rate},
            {'params': fusion_params, 'lr': self.learning_rate}
        ])

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6,
            threshold=0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mAP",
                "interval": "epoch",
                "frequency": 1
            }
        }


def main(cfg):
    # Ensure PANNs weights are available
    panns_path = download_panns_checkpoint()

    # Create the shared backbone model
    model = SharedBackbonePANNs(
        n_classes=len(LABELS),
        pretrained_path=panns_path,
        freeze_backbone=True  # Start with frozen backbone
    )

    # Create lightning module
    classifier = InstrumentClassifier(
        model=model,
        learning_rate=cfg.get('learning_rate', 1e-4)
    )

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dir=cfg.get('train_dir', 'data/processed/train'),
        val_dir=cfg.get('val_dir', 'data/processed/val'),
        batch_size=cfg.get('batch_size', 32),
        num_workers=cfg.get('num_workers', 4),
        max_samples=cfg.get('max_samples', None),
        use_multi_stft=True  # Use 9 spectrograms
    )

    # Setup callbacks
    callbacks = default_callbacks()

    # Add custom callback to unfreeze backbone after 3 epochs
    logger = TensorBoardLogger(save_dir="lightning_logs", name="shared_backbone")

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=cfg.get('num_epochs', 50),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_checkpointing=True,
        val_check_interval=1.0,  # Check validation every epoch
        limit_val_batches=cfg.get('limit_val_batches', 1.0)
    )

    # Start training
    trainer.fit(classifier, train_loader, val_loader)

    print("\n✅ Training complete!")

    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/shared_backbone.yaml")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    main(cfg)
