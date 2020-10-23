import pytorch_lightning as pl
from adni_model import ADNI_MODEL
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':

    
    log_dir = "lightning_logs"
    os.makedirs(log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, 'checkpoints'), verbose=True)
    # save_best_only=False,
    #stop_callback = EarlyStopping( monitor='val_loss', mode='auto', patience=5, verbose=True)

    #CSV_FILE = 'All_Data.csv '
    #ROOT_DIR = r'C:\StFX\Project\ADNI_DATASET\ADNI1_Complete_1Yr_1.5T\All_Files_Classified\All_Data'
    #adni_model = ADNI_MODEL(root_dir = ROOT_DIR, csv_file = CSV_FILE)

    adni_model = ADNI_MODEL()
    trainer = pl.Trainer(gpus=1, min_epochs=2, max_epochs=2, checkpoint_callback=checkpoint_callback)
    trainer.fit(adni_model)
    trainer.test(adni_model)