
#!/usr/bin/env python3
"""CLI inference """
import argparse, yaml, torch, librosa, numpy as np, pathlib
from models import CNNBaseline, ResNetSpec, MultiSTFTCNN
from data.preprocess import generate_multi_stft

LABELS = ["cello","clarinet","flute","acoustic_guitar","organ","piano","saxophone","trumpet","violin","voice","other"]

def extract_features(path, cfg, model_type="cnn"):
    y, sr = librosa.load(path, sr=cfg['sample_rate'], mono=True)
    specs_dict = generate_multi_stft(y, sr)

    if model_type == "9cnn":
        # For 9CNN, we need all 9 spectrograms (3 window sizes × 3 frequency bands)
        n_ffts = (256, 512, 1024)
        band_ranges = ((0, 1000), (1000, 4000), (4000, 11025))

        # Create a list of tensors for each spectrogram
        spec_list = []
        for n_fft in n_ffts:
            for (f_low, f_high) in band_ranges:
                band_label = f"{f_low}-{f_high}Hz"
                key = (band_label, n_fft)

                if key in specs_dict:
                    spec = specs_dict[key]
                    spec_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    spec_list.append(spec_tensor)
                else:
                    # If a spectrogram is missing, use a dummy tensor
                    print(f"Warning: Spectrogram for {key} not found")
                    spec_list.append(torch.zeros(1, 1, 10, 10))  # Dummy tensor

        return spec_list
    else:
        # For CNN and ResNet, we'll use the middle frequency band (1000-4000Hz) with the middle FFT size (512)
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
    model_type = cfg.get("model_name", "cnn")
    x = extract_features(wav_path, cfg, model_type)
    with torch.no_grad():
        preds = model(x).squeeze().numpy()
    return {label: float(preds[i]) for i, label in enumerate(LABELS)}

def main(ckpt, wav, config):
    cfg = yaml.safe_load(open(config))
    model_name = cfg.get("model_name", "cnn")

    if model_name == "resnet34":
        model = ResNetSpec(len(LABELS))
    elif model_name == "9cnn":
        model = MultiSTFTCNN(len(LABELS))
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
