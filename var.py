

LABELS = ['accordion', 'banjo', 'bass', 'celesta', 'clarinet', 'cymbals', 'drums',
 'electric_bass', 'flute', 'guitar', 'mallet_percussion', 'organ', 'piano',
 'reed_instrument', 'saxophone', 'synthesizer', 'trombone', 'trumpet',
 'violin', 'voice']

band_ranges = ["0-1000Hz", "1000-4000Hz", "4000-11025Hz"]
band_ranges_as_tuples = [(0, 1000), (1000, 4000), (4000, 11025)]
n_ffts = [256, 512, 1024]

optimized_stfts = [
    ("0-1000Hz", 1024),
    ("1000-4000Hz", 512),
    ("4000-11025Hz", 256),
]

# IRMAS label mappings
IRMAS_TO_LABEL_MAP = {
    'cel': 'cello',
    'cla': 'clarinet',
    'flu': 'flute',
    'gac': 'acoustic_guitar',
    'gel': 'acoustic_guitar',
    'org': 'organ',
    'pia': 'piano',
    'sax': 'saxophone',
    'tru': 'trumpet',
    'vio': 'violin',
    'voi': 'voice'
}