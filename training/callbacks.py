from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

def default_callbacks():
    return [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/loss', mode='min', patience=5),
        ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            filename='{epoch}-{val_loss:.3f}'
        ),
    ]
