

LABELS = ["cello", "clarinet", "flute", "acoustic_guitar", "organ", "piano", "saxophone", "trumpet", "violin", "voice", "other"]

band_ranges = ["0-1000Hz", "1000-4000Hz", "4000-11025Hz"]
band_ranges_as_tuples = [(0, 1000), (1000, 4000), (4000, 11025)]
n_ffts = [256, 512, 1024]

optimized_stfts = [
    ("0-1000Hz", 1024),
    ("1000-4000Hz", 512),
    ("4000-11025Hz", 256),
]