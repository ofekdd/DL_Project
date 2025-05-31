
#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib
from models import CNNBaseline, ResNetSpec
from data.preprocess import generate_multi_stft

LABELS = ["cello","clarinet","flute","acoustic_guitar","organ","piano","saxophone","trumpet","violin","voice","other"]

def extract_features(path, cfg):
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    specs_dict = generate_multi_stft(y, sr)

    # For inference, we'll use the middle frequency band (1000-4000Hz) with the middle FFT size (512)
    # This is a simplification - in a real application, you might want to use all bands and FFT sizes
    key = ("1000-4000Hz", 512)
    if key in specs_dict:
        spec = specs_dict[key]
        return torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    else:
        # Fallback to the first available spectrogram if the preferred one is not available
        first_key = list(specs_dict.keys())[0]
        spec = specs_dict[first_key]
        return torch.tensor(spec).unsqueeze(0).unsqueeze(0)

def predict(model, wav_path, cfg):
    model.eval()
    x = extract_features(wav_path, cfg)
    with torch.no_grad():
        preds = model(x).squeeze().numpy()
    return {label: float(preds[i]) for i, label in enumerate(LABELS)}

def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    if cfg.get("model_name") == "resnet34":
        model = ResNetSpec(len(LABELS))
    else:
        model = CNNBaseline(len(LABELS))
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    results = predict(model, wav, cfg)
    print(results)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)
