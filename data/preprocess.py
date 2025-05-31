
#!/usr/bin/env python3
"""Convert wav clips to log‑mel spectrogram tensors saved as .npy.

Usage:
    python data/preprocess.py --in_dir data/raw/IRMAS --out_dir data/processed
"""
import argparse, librosa, numpy as np, pathlib, tqdm, yaml

def process_file(wav_path, cfg):
    y, sr = librosa.load(wav_path, sr=cfg['sample_rate'], mono=True)
    mels = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg['n_mels'], hop_length=cfg['hop_length'], fmin=30
    )
    logmel = librosa.power_to_db(mels, ref=np.max).astype(np.float32)
    return logmel

def main(in_dir, out_dir, config):
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(config))

    wav_files = list(in_dir.rglob("*.wav"))
    for wav in tqdm.tqdm(wav_files):
        spec = process_file(wav, cfg)
        rel = wav.relative_to(in_dir).with_suffix(".npy")
        np.save(out_dir / rel, spec)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.in_dir, args.out_dir, args.config)
