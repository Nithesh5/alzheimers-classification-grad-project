import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from adni_data_loader import ADNIDataloader
import numpy as np
import pandas as pd
from pytorch_lightning.metrics.functional import accuracy
from torchvision import transforms
from nitorch.transforms import  ToTensor, SagittalTranslate, SagittalFlip, \
                                AxialTranslate, normalization_factors, Normalize, \
                                IntensityRescale
from pytorch_lightning.metrics import Accuracy


class ADNI_MODEL(pl.LightningModule):
    def __init__(self, root_dir, csv_file):
        super().__init__()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.batch_size = 1
        self.root_dir = root_dir
        self.csv_file = csv_file
        """
        #old start
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]
        labels_df_AD = labels_df_AD.head(40)
        labels_df_CN = labels_df_CN.head(40)
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        labels_df = labels_df.sample(n=80, random_state=3)
        print(labels_df.head(80))

        #        print(labels_df['image_name'])
        #print(labels_df)

        test_percent = 20
        random_seed = 33
        test_size = int(np.round((len(labels_df) * test_percent) / 100))
        #print("test_size", test_size)
        trainandValidation = len(labels_df) - test_size
        #print("trainandValidation", trainandValidation)
        torch.manual_seed(random_seed)
        trainandValidation_ds, test_ds = random_split(labels_df, [trainandValidation, test_size])

        validation_percent = 20
        random_seed = 43
        val_size = int(np.round((len(trainandValidation_ds) * validation_percent) / 100))
        train_size = len(trainandValidation_ds) - val_size
        torch.manual_seed(random_seed)
        train_ds, val_ds = random_split(trainandValidation_ds, [train_size, val_size])

        print("len of train and val and test")
        print(len(train_ds), len(val_ds), len(test_ds))



        train_indices = train_ds.indices
        #print("train_indices")
        #print(train_indices)
        train_labels_df = labels_df
        train_ds = train_labels_df.reset_index(drop=True)
        train_ds = train_ds.loc[train_indices, :].copy().reset_index()

        val_indices = val_ds.indices
        val_labels_df = labels_df
        #print("val_indices")
        #print(val_indices)
        val_ds = val_labels_df.reset_index(drop=True)
        val_ds = val_ds.loc[val_indices, :].copy().reset_index()

        test_indices = test_ds.indices
        #print("test_indices")
        #print(test_indices)
        test_labels_df = labels_df
        test_ds = test_labels_df.reset_index(drop=True)
        test_ds = test_ds.loc[test_indices, :].copy().reset_index()

        #old end
        """
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]


        labels_df_AD = labels_df_AD.iloc[0:320, :] #labels_df_AD.head(40)
        labels_df_CN = labels_df_CN.iloc[0:320, :] #.head(40)
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        train_labels_df = labels_df.sample(frac=1, random_state=3)
        print(labels_df.head(80))

        train_ds = train_labels_df.reset_index(drop=True)
        print("train_ds")
        print(train_ds)


        #######################################################################################
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]

        labels_df_AD = labels_df_AD.iloc[320:420, :]
        labels_df_CN = labels_df_CN.iloc[320:420, :]
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        test_labels_df = labels_df.sample(frac=1, random_state=3)

        val_ds = test_labels_df.reset_index(drop=True)
        print("val_ds")
        print(val_ds)


        #######################################################################################
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]
        labels_df_AD = labels_df_AD.iloc[420:471, :]
        labels_df_CN = labels_df_CN.iloc[420:471, :]
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        valid_labels_df = labels_df.sample(frac=1, random_state=3)

        test_ds = valid_labels_df.reset_index(drop=True)
        print("test_ds")
        print(test_ds)
        #######################################################################################





        augmentations = [SagittalFlip(), SagittalTranslate(dist=(-2, 3))]
        transform = transforms.Compose(augmentations + [ToTensor()])
        self.train_dataset = ADNIDataloader(df=train_ds,
                                 root_dir=root_dir,
                                 transform=transform)

        transform = transforms.Compose([ToTensor()])
        self.val_dataset = ADNIDataloader(df=val_ds,
                                     root_dir=root_dir,
                                     transform=transform)

        transform = transforms.Compose([ToTensor()])
        self.test_dataset = ADNIDataloader(df=test_ds,
                                      root_dir=root_dir,
                                      transform=transform)


















        self.Conv_1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.dense_1 = nn.Linear(4800, 128)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dense_2 = nn.Linear(128, 2)
        #self.softmax = nn.Softmax(dim=1) # bz applying cross entropy , it has inbuilt softmax

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        #x = self.softmax(x)
        return x

    def configure_optimizers(self):
        lr = 1e-4
        wd = 1e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        x = x.type(torch.FloatTensor)
        x = x.cuda()
        y_hat = self(x)

        #print("train Labels")
        #print(y)

        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        loss = F.cross_entropy(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        acc = self.train_acc(preds, y)

        self.log('train_acc_step', self.train_acc(preds, y), prog_bar=True)

#        logs = {'train_loss': loss}

#        batch_dictionary = {
#            "loss": loss,
#            "log": logs,
#            "correct": correct,
#            "total": total
#        }

#        self.train_acc(preds, y)
#        self.log('train_acc_New', self.train_acc, prog_bar=True)

#        self.log('train_loss', loss, prog_bar=True)
#        self.log('train_acc', acc, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#        correct = sum([x["correct"] for x in outputs])
#        total = sum([x["total"] for x in outputs])
#        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct / total}
#        epoch_dictionary = {
#            'loss': avg_loss,
#            'log': tensorboard_logs
#        }
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log('avg_train_loss', avg_loss, prog_bar=True)
#        return avg_loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x=x.type(torch.FloatTensor)

        x = x.cuda()
        y_hat = self(x)

        #print("train Labels")
        #print(y)

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)

        a = self.valid_acc(preds, y)
        #print("ddddddddddd type(a",type(a))
        self.log('valid_acc', a, prog_bar=True)

        self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss}

#    def validation_end(self, outputs):
#        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#        tensorboard_logs = {'val_loss': avg_loss}
#        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        #return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        x=x.type(torch.FloatTensor)
        x = x.cuda()
        y_hat = self(x)

        #print("train Labels")
        #print(y)

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        correct = preds.eq(y).sum().item()
        total = len(y)

        acc = accuracy(preds, y)

        c_matric =pl.metrics.functional.classification.confusion_matrix(pred=preds, target=y, num_classes=2)
        #print("test_step c_matric")
        #print(c_matric)

        # Calling self.log will surface up scalars for you in TensorBoard
        #self.log('test_loss_nithesh', loss, prog_bar=True)
#        self.log('test_acc', acc, prog_bar=True)


        logs = {'test_loss': loss}
        batch_dictionary = {
            "test_loss": loss,
            "correct": correct,
            "logs": logs,
            "total": total,
            "c_matric" : c_matric
        }

        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        c_matric = sum([x["c_matric"] for x in outputs])

        print("===============================================================================================================================")
        print("test_epoch_end c_matric")
        print(c_matric)
        print("Avg Test Accuracy")
        avg_test_accuracy =  correct / total
        print(avg_test_accuracy)
        #tensorboard_logs = {'loss':  avg_loss, "Accuracy": correct / total}
#        epoch_dictionary = {
#            'loss': avg_loss,
#            'log': tensorboard_logs
#        }
        self.log('avg_test_loss', avg_test_accuracy, prog_bar=True)
#        self.log('c_matric', c_matric, prog_bar=True)

    #        return avg_loss

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)