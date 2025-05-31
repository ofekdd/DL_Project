
#!/usr/bin/env python3
"""Entry point for training."""
# TODO: this is all weird, do it again
import pytorch_lightning as pl, torch, yaml, argparse, pathlib
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchmetrics import MetricCollection
import numpy as np
from models import CNNBaseline, ResNetSpec
from training.callbacks import default_callbacks
from training.metrics import MetricCollection
from data.dataset import NpyDataset, pad_collate

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        n_classes = 11
        if cfg.get("model_name","cnn") == "resnet34":
            self.model = ResNetSpec(n_classes)
        else:
            self.model = CNNBaseline(n_classes)
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
    cfg = yaml.safe_load(open(config))
    train_ds = NpyDataset("data/processed/train")
    val_ds   = NpyDataset("data/processed/val")
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=pad_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], collate_fn=pad_collate)
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
