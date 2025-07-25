# !/usr/bin/env python3
"""Entry point for training."""
import pytorch_lightning as pl, torch, yaml, argparse
from torchmetrics import MetricCollection
from models.panns_enhanced import WaveletCNN
from training.callbacks import default_callbacks
from training.metrics import MetricCollection
from data.dataset import create_dataloaders
from var import LABELS


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        n_classes = len(LABELS)
        # Using MultiSTFTCNN model directly as specified
        self.model = WaveletCNN(
            n_classes=n_classes,
        )
        self.metrics = MetricCollection(n_classes)
        self.lr = float(cfg["learning_rate"])  # Ensure learning_rate is a float
        self.save_hyperparameters(cfg)

    def forward(self, x): return self.model(x)

    def common_step(self, batch, stage):
        x, y = batch
        preds = self(x)
        # Convert y to float for binary_cross_entropy
        y_float = y.float()
        loss = torch.nn.functional.binary_cross_entropy(preds, y_float)
        # Keep y as long for metrics
        metrics = self.metrics(preds, y)
        self.log_dict({f"{stage}/loss": loss, **{f"{stage}/{k}": v for k, v in metrics.items()}},
                      prog_bar=True)
        return loss

    def training_step(self, batch, _):  return self.common_step(batch, "train")

    def validation_step(self, batch, _): return self.common_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main(config):
    # Handle both file path and dictionary inputs
    if isinstance(config, dict):
        cfg = config
    else:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)

    # Get train and val directories from config, with fallback to default paths
    train_dir = cfg.get('train_dir', "data/processed/train")
    val_dir = cfg.get('val_dir', "data/processed/val")

    print(f"Using train_dir: {train_dir}")
    print(f"Using val_dir: {val_dir}")

    # Use the create_dataloaders function with use_multi_stft=True
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        use_multi_stft=True,  # Use MultiSTFTNpyDataset for MultiSTFTCNN model
        max_samples=cfg.get('max_samples', None)  # Limit number of samples if specified
    )

    model = LitModel(cfg)

    # Configure trainer with validation efficiency settings from config
    trainer_kwargs = {
        'max_epochs': cfg['num_epochs'],
        'callbacks': default_callbacks(),
        'accelerator': "auto",
    }

    # Add validation efficiency settings if specified in config
    if 'limit_val_batches' in cfg:
        trainer_kwargs['limit_val_batches'] = cfg['limit_val_batches']
        print(
            f"🎛️  Limiting validation to {cfg['limit_val_batches'] * 100 if cfg['limit_val_batches'] <= 1 else cfg['limit_val_batches']}{'%' if cfg['limit_val_batches'] <= 1 else ' batches'}")

    if 'num_sanity_val_steps' in cfg:
        trainer_kwargs['num_sanity_val_steps'] = cfg['num_sanity_val_steps']
        print(f"🎛️  Using {cfg['num_sanity_val_steps']} sanity validation steps")

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)