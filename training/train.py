# !/usr/bin/env python3
"""Entry point for training the WaveletCNN on raw waveforms."""
import argparse
import yaml
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from models.Conv_wavelet_cnn import WaveletCNN
from training.callbacks import default_callbacks
from training.metrics import MetricCollection  # your wrapper: mAP + F1
from data.dataset import create_dataloaders    # <-- waveform version
from var import LABELS


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        n_classes = len(LABELS)
        sr = int(cfg["sample_rate"])
        self.model = WaveletCNN(n_classes=n_classes, sr=sr)

        # Your metrics wrapper returns {"mAP": ..., "F1": ...}
        self.metrics = MetricCollection(n_classes)
        self.lr = float(cfg["learning_rate"])

        print(f"[WaveletCNN] sr={sr}, n_classes={n_classes}")

    def forward(self, x):
        # x: [B, T] waveform
        return self.model(x)  # probs in [0,1] (model ends with Sigmoid)

    def common_step(self, batch, stage: str):
        x, y = batch                      # x: [B, T], y: [B, C] multi-hot (0/1)
        preds = self(x)                   # preds: [B, C] in [0,1]
        loss = F.binary_cross_entropy(preds, y.float())

        # metrics expect probabilities + binary targets
        metrics = self.metrics(preds, y)

        self.log_dict(
            {f"{stage}/loss": loss, **{f"{stage}/{k}": v for k, v in metrics.items()}},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True if torch.cuda.is_available() else False,
        )
        return loss

    def training_step(self, batch, _):
        return self.common_step(batch, "train")

    def validation_step(self, batch, _):
        return self.common_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


def main(config):
    # Accept a dict or a path to YAML
    if isinstance(config, dict):
        cfg = config
    else:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)

    # Data dirs (produced by waveform preprocess)
    train_dir = cfg.get('train_dir', "data/processed/train")
    val_dir   = cfg.get('val_dir',   "data/processed/val")

    print(f"Using train_dir: {train_dir}")
    print(f"Using val_dir  : {val_dir}")

    # Waveform dataloaders (expects waveform.npy per folder)
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        max_samples=cfg.get('max_samples', None)  # keep a small cap for fast runs if desired
    )

    model = LitModel(cfg)

    trainer_kwargs = {
        'max_epochs': cfg['num_epochs'],
        'callbacks': default_callbacks(),
        'accelerator': "auto",
        'devices': "auto",
        # optional: mixed precision if you like
        # 'precision': "16-mixed",
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
    # feel free to switch default config path
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.config)
