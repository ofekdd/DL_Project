
#!/usr/bin/env python3
"""Download the IRMAS dataset (≈2 GB) and extract it.

Example:
    python data/download_irmas.py --out_dir data/raw
"""
import argparse, hashlib, os, tarfile, urllib.request, sys, pathlib

IRMAS_URL = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1"
MD5       = "4fd9f5ed5a18d8e2687e6360b5f60afe"  # expected archive checksum

def md5(fname, chunk=2**20):
    m = hashlib.md5()
    with open(fname, 'rb') as fh:
        while True:
            data = fh.read(chunk)
            if not data: break
            m.update(data)
    return m.hexdigest()

def main(out_dir: str):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "IRMAS.zip"

    if not archive_path.exists():
        print("Downloading IRMAS ...")
        urllib.request.urlretrieve(IRMAS_URL, archive_path)
    else:
        print("Archive already exists, skipping download")

    print("Verifying checksum ...")
    if md5(archive_path) != MD5:
        print("Checksum mismatch!", file=sys.stderr)
        sys.exit(1)

    print("Extracting ...")
    import zipfile
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(out_dir)
    print("Done. Data at", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw", help="Destination directory")
    args = p.parse_args()
    main(args.out_dir)
