import pytorch_lightning as pl
from adni_model import ADNI_MODEL
import os
import torch
from torchvision import transforms
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from adni_data_loader import ADNIDataloader
from nitorch.transforms import  ToTensor, SagittalTranslate, SagittalFlip, \
                                AxialTranslate, normalization_factors, Normalize, \
                                IntensityRescale

if __name__ == '__main__':

    log_dir = "lightning_logs_1"

    os.makedirs(log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, 'checkpoints'), verbose=True)
    # save_best_only=False,
    stop_callback = EarlyStopping( monitor='val_loss', mode='auto', patience=5, verbose=True)

    csv_file = 'All_Data.csv'

    root_dir = r'C:\StFX\Project\ADNI_DATASET\ADNI1_Complete_1Yr_1.5T\All_Files_Classified\All_Data'

    adni_model = ADNI_MODEL(root_dir = root_dir, csv_file = csv_file)

    trainer = pl.Trainer(gpus=1, min_epochs=1, max_epochs=1, checkpoint_callback=checkpoint_callback)
    trainer.fit(adni_model)
    trainer.test(adni_model)