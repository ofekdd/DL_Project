
import numpy as np, tempfile, shutil, os, librosa
from data.preprocess import process_file, generate_multi_stft
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
    specs_dict = process_file(wav, cfg)

    # Check that we have 9 spectrograms (3 window sizes × 3 frequency bands)
    assert len(specs_dict) == 9

    # Check that each spectrogram has the expected format
    n_ffts = (256, 512, 1024)
    band_ranges = ((0, 1000), (1000, 4000), (4000, 11025))

    for n_fft in n_ffts:
        for (f_low, f_high) in band_ranges:
            band_label = f"{f_low}-{f_high}Hz"
            key = (band_label, n_fft)

            # Check that this spectrogram exists
            assert key in specs_dict

            # Get frequency indices within this band
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            band_mask = (freqs >= f_low) & (freqs < f_high)
            expected_freq_bins = np.sum(band_mask)

            # Check that the spectrogram has the expected number of frequency bins
            assert specs_dict[key].shape[0] == expected_freq_bins
