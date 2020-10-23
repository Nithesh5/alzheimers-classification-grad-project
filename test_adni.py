from adni_model  import  ADNI_MODEL
import pytorch_lightning as pl

if __name__ == '__main__':

    #ref https://github.com/PyTorchLightning/pytorch-lightning/issues/924

    model_test = ADNI_MODEL.load_from_checkpoint('version_0/checkpoints.ckpt')
    trainer = pl.Trainer(gpus=1)
    trainer.test(model_test)