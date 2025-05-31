
# Instrument Classifier

A PyTorch Lightning project for multi‑label musical instrument recognition from audio clips.
Clone, install dependencies, preprocess data, and train:

```bash
git clone <repo-url>
cd instrument_classifier
pip install -r requirements.txt
python data/download_irmas.py  --out_dir data/raw
python data/preprocess.py      --in_dir data/raw/IRMAS --out_dir data/processed
python training/train.py       --config configs/model_resnet.yaml
```

See `configs/default.yaml` for full hyper‑parameters.
