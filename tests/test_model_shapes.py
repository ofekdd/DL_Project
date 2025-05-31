
import torch
from models import CNNBaseline, ResNetSpec

def test_cnn_baseline_shape():
    model = CNNBaseline(11)
    x = torch.randn(2,1,64,300)
    out = model(x)
    assert out.shape == (2,11)

def test_resnet_shape():
    model = ResNetSpec(11)
    x = torch.randn(2,1,224,224)
    out = model(x)
    assert out.shape == (2,11)
