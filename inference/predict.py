
#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib
from models.multi_stft_cnn import MultiSTFTCNN
from data.preprocess import generate_multi_stft
from data.dataset import LABELS
from var import n_ffts, band_ranges


def extract_features(path, cfg):
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    specs_dict = generate_multi_stft(y, sr)

    # For MultiSTFTCNN, we need all 9 spectrograms (3 window sizes × 3 frequency bands)
    # Convert each spectrogram to a tensor with shape (1, 1, F, T)
    specs_list = []
    for n_fft in n_ffts:
        for band_range in band_ranges:
            key = (band_range, n_fft)
            if key in specs_dict:
                spec = specs_dict[key]
                specs_list.append(torch.tensor(spec).unsqueeze(0).unsqueeze(0))
            else:
                # If a specific spectrogram is missing, use a zero tensor of appropriate shape
                # This is a fallback and should be rare
                print(f"Warning: Missing spectrogram for {key}")
                # Use a small dummy tensor as fallback
                specs_list.append(torch.zeros(1, 1, 10, 10))

    return specs_list

def predict(model, wav_path, cfg):
    model.eval()
    x = extract_features(wav_path, cfg)
    with torch.no_grad():
        preds = model(x).squeeze().numpy()
    return {label: float(preds[i]) for i, label in enumerate(LABELS)}

def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    # Using MultiSTFTCNN model directly as specified
    model = MultiSTFTCNN(n_classes=len(LABELS))
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    results = predict(model, wav, cfg)
    print(results)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("wav")
    p.add_argument("--config", default="configs/multi_stft_cnn.yaml")
    args = p.parse_args()
    main(args.ckpt, args.wav, args.config)
