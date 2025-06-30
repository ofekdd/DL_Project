
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

def default_callbacks():
    return [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/Accuracy', mode='max', patience=5),
        ModelCheckpoint(monitor='val/Accuracy', mode='max', save_top_k=1, filename='{epoch}-{val_Accuracy:.3f}')
    ]
