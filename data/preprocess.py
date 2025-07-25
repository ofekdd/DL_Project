#!/usr/bin/env python3
"""Convert wav clips to wavelet scalograms saved as .npy."""
import argparse, librosa, numpy as np, pathlib, tqdm, yaml, os, random, itertools
from scipy.signal import cwt, morlet2
from var import LABELS

def generate_wavelet_scalogram(y: np.ndarray, sr: int, num_scales: int = 512):
    """
    Compute complex Morlet wavelet transform and return magnitude scalogram.
    """
    widths = np.logspace(np.log10(1), np.log10(num_scales), num_scales)
    scalogram = cwt(y, lambda M, s: morlet2(M, s, w=6), widths)
    mag = np.abs(scalogram)
    mag_db = librosa.amplitude_to_db(mag, ref=np.max).astype(np.float32)
    return mag_db

def process_file(wav_path, cfg):
    y, sr = librosa.load(wav_path, sr=cfg["sample_rate"], mono=True)
    return generate_wavelet_scalogram(y, sr)

def preprocess_data(irmas_root, out_dir, cfg, original_data_percentage=1.0):
    def generate_specs(y, sr):
        return generate_wavelet_scalogram(y, sr)

    out_dir = pathlib.Path(out_dir)
    train_dir, val_dir, test_dir = (out_dir / s for s in ("train", "val", "test"))
    for p in (train_dir, val_dir, test_dir):
        p.mkdir(parents=True, exist_ok=True)

    irmas_root = pathlib.Path(irmas_root)
    train_src = irmas_root / "IRMAS-TrainingData"
    wav_files = list(train_src.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wavs found under {train_src}")

    if 0 < original_data_percentage < 1.0:
        k = int(len(wav_files) * original_data_percentage)
        wav_files = random.sample(wav_files, k)

    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * 0.8)
    tr_files, val_files = wav_files[:split_idx], wav_files[split_idx:]

    def _process_split(files, split_dir, prefix):
        for i, wav in enumerate(tqdm.tqdm(files, desc=f"→ {split_dir.name}")):
            try:
                y, sr = librosa.load(wav, sr=cfg["sample_rate"], mono=True)
                irmas_label = wav.parent.name
                base = wav.stem
                sample_dir = split_dir / f"{prefix}_{irmas_label}_{i:04d}_{base}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                scalogram = generate_specs(y, sr)
                np.save(sample_dir / "wavelet.npy", scalogram)
            except Exception as e:
                print(f"❌ {wav}: {e}")

    _process_split(tr_files, train_dir, "original")
    _process_split(val_files, val_dir, "original")

    test_roots = [p for p in irmas_root.glob("IRMAS-TestingData-Part*") if p.is_dir()]
    if not test_roots:
        sf = irmas_root / "IRMAS-TestingData"
        if sf.exists():
            test_roots = [sf]

    test_wavs = list(itertools.chain.from_iterable(tr.rglob("*.wav") for tr in test_roots))
    cap = cfg.get("max_test_samples")
    if cap is not None and len(test_wavs) > cap:
        random.shuffle(test_wavs)
        test_wavs = test_wavs[:cap]

    for i, wav in enumerate(tqdm.tqdm(test_wavs, desc="→ test")):
        try:
            y, sr = librosa.load(wav, sr=cfg["sample_rate"], mono=True)
            txt = wav.with_suffix(".txt")
            if txt.exists():
                with open(txt) as fh:
                    lbls = [ln.strip() for ln in fh if ln.strip()]
                lbl_tag = "+".join(lbls) if lbls else "unknown"
            else:
                lbl_tag = "unknown"

            base = wav.stem
            sample_dir = test_dir / f"irmasTest_{lbl_tag}_{i:04d}_{base}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            scalogram = generate_specs(y, sr)
            np.save(sample_dir / "wavelet.npy", scalogram)
        except Exception as e:
            print(f"❌ {wav}: {e}")

    print("\n📋 Final counts:")
    for split, p in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if p.exists():
            print(f"   {split:<5}: {len(list(p.iterdir())):>5} samples")

    print(f"\n✅ Preprocessing finished. Features saved to {out_dir}")

def main(in_dir, out_dir, config):
    in_dir, out_dir = pathlib.Path(in_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(config))

    wav_files = list(in_dir.rglob("*.wav"))
    for wav in tqdm.tqdm(wav_files):
        specs = process_file(wav, cfg)
        rel_dir = wav.relative_to(in_dir).with_suffix("")
        file_out_dir = out_dir / rel_dir
        file_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(file_out_dir / "wavelet.npy", specs)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.in_dir, args.out_dir, args.config)
