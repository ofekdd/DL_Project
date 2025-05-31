
import numpy as np, tempfile, shutil, os
from data.preprocess import process_file
import yaml

def test_process_file(tmp_path):
    # synthetic 0.5 s sine wave
    import soundfile as sf, numpy as np
    sr, f = 22050, 440
    t = np.linspace(0,0.5,int(sr*0.5),False)
    y = 0.5*np.sin(2*np.pi*f*t)
    wav = tmp_path/"test.wav"
    sf.write(wav, y, sr)
    cfg = yaml.safe_load(open("configs/default.yaml"))
    spec = process_file(wav, cfg)
    assert spec.shape[0] == cfg['n_mels']
