
#!/usr/bin/env python3
"""Entry point for training."""
import pytorch_lightning as pl, torch, yaml, argparse
from torchmetrics import MetricCollection
from models.multi_stft_cnn import MultiSTFTCNN
from training.callbacks import default_callbacks
from training.metrics import MetricCollection
from data.dataset import create_dataloaders
from var import LABELS

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        n_classes = len(LABELS)
        # Using MultiSTFTCNN model directly as specified
        self.model = MultiSTFTCNN(
            n_classes=n_classes,
            n_branches=cfg.get('n_branches', 9),
            branch_output_dim=cfg.get('branch_output_dim', 128)
        )
        self.metrics = MetricCollection(n_classes)
        self.lr = cfg["learning_rate"]
        self.save_hyperparameters(cfg)

    def forward(self, x): return self.model(x)

    def common_step(self, batch, stage):
        x, y = batch
        preds = self(x)
        loss = torch.nn.functional.binary_cross_entropy(preds, y)
        metrics = self.metrics(preds, y)
        self.log_dict({f"{stage}/loss": loss, **{f"{stage}/{k}":v for k,v in metrics.items()}},
                       prog_bar=True)
        return loss

    def training_step(self, batch, _):  return self.common_step(batch, "train")
    def validation_step(self, batch, _):return self.common_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main(config):
    # Handle both file path and dictionary inputs
    if isinstance(config, dict):
        cfg = config
    else:
        cfg = yaml.safe_load(open(config))
    # Use the create_dataloaders function with use_multi_stft=True
    train_loader, val_loader = create_dataloaders(
        train_dir="data/processed/train",
        val_dir="data/processed/val",
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        use_multi_stft=True,  # Use MultiSTFTNpyDataset for MultiSTFTCNN model
        max_samples=cfg.get('max_samples', None)  # Limit number of samples if specified
    )
    model = LitModel(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg['num_epochs'],
        callbacks=default_callbacks(),
        accelerator="auto"
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
