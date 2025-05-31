import matplotlib.pyplot as plt
import librosa
import librosa.display
from data.preprocess import generate_multi_stft
from var import n_ffts, band_ranges

def visualize_audio(wav_path, cfg):
    """
    Visualize audio waveform and spectrograms.

    Parameters:
        wav_path (str): Path to the audio file
        cfg (dict): Configuration dictionary with sample_rate

    Returns:
        None (displays plots)
    """
    # Load audio
    y, sr = librosa.load(wav_path, sr=cfg['sample_rate'], mono=True)

    # Compute multi-band STFT spectrograms
    specs_dict = generate_multi_stft(y, sr)

    # Plot waveform and selected spectrograms
    plt.figure(figsize=(15, 12))

    # Plot waveform
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # use keys to plot from var.py
    # Create a list of keys to plot by combining band_ranges and n_ffts
    keys_to_plot = []
    for band_range in band_ranges:
        for n_fft in n_ffts:
            keys_to_plot.append((band_range, n_fft))

    # Select only 3 keys to plot (one for each frequency band with the middle FFT size)
    middle_fft = n_ffts[1]  # 512
    keys_to_plot = [(band, middle_fft) for band in band_ranges]

    for i, key in enumerate(keys_to_plot):
        if key in specs_dict:
            plt.subplot(4, 1, i+2)
            spec = specs_dict[key]
            hop_length = 512 // 4  # hop_length for FFT size 512
            librosa.display.specshow(spec, sr=sr, x_axis='time', hop_length=hop_length)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {key[0]}, FFT size: {key[1]}')
        else:
            print(f"Spectrogram for {key} not found")

    plt.tight_layout()
    plt.show()
