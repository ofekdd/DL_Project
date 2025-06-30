
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

def default_callbacks():
    return [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/F1', mode='max', patience=5),
        ModelCheckpoint(monitor='val/F1', mode='max', save_top_k=1, filename='{epoch}-{val_F1:.3f}')
    ]
