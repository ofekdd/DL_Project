
#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib
from models import CNNBaseline, ResNetSpec

LABELS = ["cello","clarinet","flute","acoustic_guitar","organ","piano","saxophone","trumpet","violin","voice","other"]

def extract_features(path, cfg):
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=cfg['n_mels'], hop_length=cfg['hop_length'])
    logmel = librosa.power_to_db(mels, ref=np.max).astype(np.float32)
    return torch.tensor(logmel).unsqueeze(0).unsqueeze(0)

def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    if cfg.get("model_name") == "resnet34":
        model = ResNetSpec(len(LABELS))
    else:
        model = CNNBaseline(len(LABELS))
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.eval()
    x = extract_features(wav, cfg)
    with torch.no_grad():
        preds = model(x).squeeze().numpy()
    res = {label: float(preds[i]) for i, label in enumerate(LABELS)}
    print(res)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)
