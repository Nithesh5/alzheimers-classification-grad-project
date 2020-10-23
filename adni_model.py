import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adni_data_loader import ADNIDataloader
import pandas as pd
from pytorch_lightning.metrics.functional import accuracy
from torchvision import transforms
from nitorch.transforms import  ToTensor, SagittalTranslate, SagittalFlip
from pytorch_lightning.metrics import Accuracy

class ADNI_MODEL(pl.LightningModule):
    def __init__(self):
        csv_file = 'All_Data.csv'
        root_dir = r'C:\StFX\Project\ADNI_DATASET\ADNI1_Complete_1Yr_1.5T\All_Files_Classified\All_Data'
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
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        labels_df = labels_df.sample(n=80, random_state=3)
        
        test_percent = 20
        random_seed = 33
        test_size = int(np.round((len(labels_df) * test_percent) / 100))
        trainandValidation = len(labels_df) - test_size
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
        train_labels_df = labels_df
        train_ds = train_labels_df.reset_index(drop=True)
        train_ds = train_ds.loc[train_indices, :].copy().reset_index()

        val_indices = val_ds.indices
        val_labels_df = labels_df
        val_ds = val_labels_df.reset_index(drop=True)
        val_ds = val_ds.loc[val_indices, :].copy().reset_index()

        test_indices = test_ds.indices
        test_labels_df = labels_df
        test_ds = test_labels_df.reset_index(drop=True)
        test_ds = test_ds.loc[test_indices, :].copy().reset_index()

        #old end
        """
        #Train dataset
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]


        labels_df_AD = labels_df_AD.iloc[0:320, :]
        labels_df_CN = labels_df_CN.iloc[0:320, :]
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        train_labels_df = labels_df.sample(frac=1, random_state=3)
        train_ds = train_labels_df.reset_index(drop=True)

        #######################################################################################
        #Validation dataset
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]

        labels_df_AD = labels_df_AD.iloc[320:420, :]
        labels_df_CN = labels_df_CN.iloc[320:420, :]
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        test_labels_df = labels_df.sample(frac=1, random_state=3)
        val_ds = test_labels_df.reset_index(drop=True)

        #######################################################################################
        # Test dataset
        labels_df = pd.read_csv(csv_file)
        labels_df_AD = labels_df[labels_df.label == "AD"]
        labels_df_CN = labels_df[labels_df.label == "CN"]
        labels_df_AD = labels_df_AD.iloc[420:471, :]
        labels_df_CN = labels_df_CN.iloc[420:471, :]
        labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
        valid_labels_df = labels_df.sample(frac=1, random_state=3)
        test_ds = valid_labels_df.reset_index(drop=True)
        #######################################################################################

        print("len of train and val and test")
        print(len(train_ds), len(val_ds), len(test_ds))

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
        x, y = batch
        x = x.type(torch.FloatTensor)
        x = x.cuda()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log('train_acc_step', self.train_acc(preds, y), prog_bar=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log('avg_train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x=x.type(torch.FloatTensor)
        x = x.cuda()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log('valid_acc', self.valid_acc(preds, y), prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)

    def test_step(self, batch, batch_nb):
        x, y = batch
        x=x.type(torch.FloatTensor)
        x = x.cuda()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        correct = preds.eq(y).sum().item()
        total = len(y)
        #acc = accuracy(preds, y)
        c_matric =pl.metrics.functional.classification.confusion_matrix(pred=preds, target=y, num_classes=2)
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
        print(c_matric)
        print("Avg Test Accuracy")
        avg_test_accuracy =  correct / total
        print(avg_test_accuracy)
        self.log('avg_test_loss', avg_test_accuracy, prog_bar=True)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)