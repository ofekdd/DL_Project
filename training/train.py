# training/train.py
# !/usr/bin/env python3
"""Entry point for training the scalogram CNN."""
import argparse
import yaml
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from models.Conv_wavelet_cnn import WaveletCNN
from training.callbacks import default_callbacks
from training.metrics import MetricCollection
from data.dataset import create_dataloaders
from var import LABELS


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        n_classes = len(LABELS)
        self.model = WaveletCNN(n_classes=n_classes)
        self.metrics = MetricCollection(n_classes)
        self.lr = float(cfg["learning_rate"])

    def forward(self, x):
        # x: [B,1,S,T]
        return self.model(x)  # probs in [0,1]

    def common_step(self, batch, stage: str):
        x, y = batch                      # x: [B,1,S,T], y: [B,C] (multi-hot)
        preds = self(x)                   # [B,C] in [0,1]
        loss = F.binary_cross_entropy(preds, y.float())
        metrics = self.metrics(preds, y)
        self.log_dict(
            {f"{stage}/loss": loss, **{f"{stage}/{k}": v for k, v in metrics.items()}},
            prog_bar=True, on_step=False, on_epoch=True,
            sync_dist=torch.cuda.is_available()
        )
        return loss

    def training_step(self, batch, _):  return self.common_step(batch, "train")
    def validation_step(self, batch, _): return self.common_step(batch, "val")
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


def main(config):
    cfg = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))

    train_dir = cfg.get('train_dir', "data/processed/train")
    val_dir   = cfg.get('val_dir',   "data/processed/val")

    print(f"Using train_dir: {train_dir}")
    print(f"Using val_dir  : {val_dir}")

    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        max_samples=cfg.get('max_samples', None),
    )

    model = LitModel(cfg)

    trainer_kwargs = {
        'max_epochs': cfg['num_epochs'],
        'callbacks': default_callbacks(),
        'accelerator': "auto",
        'devices': "auto",
    }
    if 'limit_val_batches' in cfg:
        trainer_kwargs['limit_val_batches'] = cfg['limit_val_batches']
        print(f"limit_val_batches: {cfg['limit_val_batches']}")
    if 'num_sanity_val_steps' in cfg:
        trainer_kwargs['num_sanity_val_steps'] = cfg['num_sanity_val_steps']
        print(f"num_sanity_val_steps: {cfg['num_sanity_val_steps']}")

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.config)
