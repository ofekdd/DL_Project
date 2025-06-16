# Download and setup PANNs
import torch
import urllib.request
import os
from pathlib import Path


def download_panns_checkpoint():
    """Download PANNs CNN14 checkpoint pretrained on AudioSet."""

    checkpoint_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
    # Alternative URL if the above fails
    alternative_url = "https://github.com/qiuqiangkong/audioset_tagging_cnn/releases/download/v0.1/Cnn14_mAP=0.431.pth"
    checkpoint_path = "pretrained/Cnn14_mAP=0.431.pth"

    # Create directory
    os.makedirs("pretrained", exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("📥 Downloading PANNs CNN14 checkpoint (this may take a few minutes)...")
        try:
            # Try primary URL first
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
            print(f"✅ Downloaded to {checkpoint_path} from primary URL")
        except Exception as e:
            print(f"⚠️ Primary download failed: {e}")
            print("📥 Trying alternative download URL...")
            try:
                # Try alternative URL
                urllib.request.urlretrieve(alternative_url, checkpoint_path)
                print(f"✅ Downloaded to {checkpoint_path} from alternative URL")
            except Exception as e2:
                print(f"❌ Alternative download also failed: {e2}")
                print("Please download the PANNs checkpoint manually from:")
                print(alternative_url)
                print("And place it in: pretrained/Cnn14_mAP=0.431.pth")
                raise RuntimeError("Failed to download PANNs checkpoint")
    else:
        print(f"✅ Checkpoint already exists at {checkpoint_path}")

    return checkpoint_path


# Only download when this file is run directly
if __name__ == "__main__":
    panns_checkpoint_path = download_panns_checkpoint()
    print(f"PANNs checkpoint ready at: {panns_checkpoint_path}")