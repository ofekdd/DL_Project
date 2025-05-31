
from .cnn_baseline import CNNBaseline
from .resnet_spec import ResNetSpec
import importlib.util
import sys
import os

# Import the 9CNN model dynamically
spec = importlib.util.spec_from_file_location("nine_cnn", os.path.join(os.path.dirname(__file__), "9cnn_baseline.py"))
nine_cnn = importlib.util.module_from_spec(spec)
sys.modules["nine_cnn"] = nine_cnn
spec.loader.exec_module(nine_cnn)
MultiSTFTCNN = nine_cnn.MultiSTFTCNN

__all__ = ["CNNBaseline", "ResNetSpec", "MultiSTFTCNN"]
