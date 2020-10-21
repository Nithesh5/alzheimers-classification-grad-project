from adni_model  import  ADNI_MODEL
import pytorch_lightning as pl

if __name__ == '__main__':

    model_test = ADNI_MODEL()

    trainer = pl.Trainer()

    trainer.test(model_test)