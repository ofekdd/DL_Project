#!/usr/bin/env python3
"""Entry point for training with PANNs warm-start."""
import pytorch_lightning as pl
import torch
import yaml
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, stage):
        try:
            # Debug info for batch loading
            x, y = batch
            print(f"[DEBUG] Batch loaded: x type: {type(x)}, len: {len(x) if isinstance(x, list) else 'N/A'}, y shape: {y.shape}")

            # Check spectrograms shapes
            if isinstance(x, list):
                for i, spec in enumerate(x):
                    print(f"[DEBUG] Spectrogram {i} shape: {spec.shape}, device: {spec.device}, dtype: {spec.dtype}")

            # Forward pass with timing
            import time
            start_time = time.time()
            logits = self(x)
            forward_time = time.time() - start_time
            print(f"[DEBUG] Forward pass completed in {forward_time:.3f}s, logits shape: {logits.shape}")

            # For single-label classification, use cross-entropy loss
            # Convert multi-hot targets to class indices if needed
            if y.dim() > 1 and y.size(1) > 1:  # Multi-hot encoded
                target_classes = torch.argmax(y, dim=1)  # Get the dominant instrument
            else:  # Already single-label format
                target_classes = y

            print(f"[DEBUG] Target classes shape: {target_classes.shape}, device: {target_classes.device}")

            loss = torch.nn.functional.cross_entropy(logits, target_classes)
            print(f"[DEBUG] Loss calculated: {loss.item():.4f}")

        except Exception as e:
            import traceback
            print(f"[ERROR] Exception in common_step: {e}")
            print(traceback.format_exc())
            raise e

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

        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=2e-4)  # Increased weight decay for better regularization

        # Check if learning rate scheduler is enabled
        use_scheduler = self.hparams.get('use_lr_scheduler', False)
        if not use_scheduler:
            return optimizer

        # Configure learning rate scheduler
        scheduler_type = self.hparams.get('lr_scheduler_type', 'cosine')
        warmup_epochs = self.hparams.get('lr_warmup_epochs', 0)
        min_lr_factor = self.hparams.get('lr_min_factor', 0.01)
        max_epochs = self.trainer.max_epochs

        # Adjust max epochs to account for the different phases
        if self.current_epoch < self.freeze_epochs:
            # When in freezing phase, scheduler is only for this phase
            effective_max_epochs = self.freeze_epochs
        else:
            # When in fine-tuning phase, scheduler is for remaining epochs
            effective_max_epochs = max_epochs - self.freeze_epochs
            warmup_epochs = min(warmup_epochs, effective_max_epochs // 5)  # Adjust warmup for second phase

        print(f"Scheduler: {scheduler_type} with {warmup_epochs} warmup epochs, min LR factor {min_lr_factor}")

        if scheduler_type.lower() == 'cosine':
            # Create the scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=effective_max_epochs - warmup_epochs,
                eta_min=lr * min_lr_factor
            )

            # If warmup is requested, use a chain of schedulers
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )

                # Sequential scheduler will handle the transition automatically
                sequential_scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_epochs]
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": sequential_scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                        "name": "sequential_lr"
                    }
                }

            # Just return the cosine scheduler if no warmup
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "cosine_annealing_lr"
                }
            }

        # Default return if no specific scheduler is matched
        return optimizer


def main(config):
    # Check CUDA availability and memory status
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"\n[DEBUG] CUDA is available: {device_count} device(s)")
        print(f"[DEBUG] Current device: {current_device}, name: {device_name}")

        # Memory info
        try:
            total_mem = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(current_device) / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(current_device) / (1024**3)  # GB
            free = total_mem - reserved
            print(f"[DEBUG] GPU memory: total={total_mem:.2f}GB, reserved={reserved:.2f}GB, allocated={allocated:.2f}GB, free={free:.2f}GB")
        except Exception as e:
            print(f"[DEBUG] Error getting CUDA memory info: {e}")
    else:
        print("[WARNING] CUDA is not available. Training will be slow on CPU.")

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

    # Debug info about dataloaders
    print(f"\n[DEBUG] Train dataloader: {len(train_loader)} batches, batch size: {cfg['batch_size']}")
    print(f"[DEBUG] Val dataloader: {len(val_loader)} batches, batch size: {cfg['batch_size']}")

    # Check first batch
    try:
        print("[DEBUG] Attempting to load first batch from train loader...")
        for i, (specs, labels) in enumerate(train_loader):
            if i == 0:
                print(f"[DEBUG] First batch loaded successfully")
                if isinstance(specs, list):
                    print(f"[DEBUG] Number of spectrograms: {len(specs)}")
                    for j, spec in enumerate(specs):
                        print(f"[DEBUG] Spectrogram {j} shape: {spec.shape}, dtype: {spec.dtype}")
                else:
                    print(f"[DEBUG] Spectrograms shape: {specs.shape}")
                print(f"[DEBUG] Labels shape: {labels.shape}, dtype: {labels.dtype}")
                print(f"[DEBUG] Labels summary: {labels.sum(dim=0)}")
                break
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception loading first batch: {e}")
        print(traceback.format_exc())

    # Create PANNs-enhanced model
    model = PANNsLitModel(cfg)

    # Setup callbacks with extended patience for the two-phase training
    callbacks = [
        LearningRateMonitor(logging_interval='step'),  # More detailed LR logging for scheduler visualization
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
        'profiler': "simple",  # Add profiler to see where time is being spent
    }

    # Temporarily reduce batch size for first few iterations if GPU memory might be an issue
    if cfg.get('debug_first_iteration', False):
        small_batch_size = max(1, cfg['batch_size'] // 4)
        print(f"[DEBUG] Using reduced batch size for debugging: {small_batch_size}")
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=small_batch_size,
            shuffle=True,
            num_workers=0
        )

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
