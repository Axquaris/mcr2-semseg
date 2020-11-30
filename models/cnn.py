import torch
import torch.nn as nn
import wandb
from mcr2_loss import MaximalCodingRateReduction
import pytorch_lightning as pl
from models.classifiers import *


def to_wandb_im(x):
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0)
    return x.cpu().numpy()


def get_mnist_semseg(in_c, feat_dim):
    n_channels = [in_c, 16, 32, 64, feat_dim]
    layers = []

    for idx in range(len(n_channels)-2):
        i, o = n_channels[idx], n_channels[idx+1]
        layers.append(nn.Conv2d(i, o, kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(o))
    i, o = n_channels[-2], n_channels[-1]
    layers.append(nn.Conv2d(i, o, kernel_size=5, padding=2))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm2d(o))

    return nn.Sequential(*layers)


class CNN(pl.LightningModule):
    def __init__(self, encoder, num_classes, feat_dim, loss, task, lr, **unused_kwargs):
        super(CNN, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss = loss
        self.task = task
        self.lr = lr

        if self.loss == 'mcr2':
            self.criterion = MaximalCodingRateReduction(num_classes)
            self.classifier = None
            self.reset_agg()
        elif self.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            self.classifier = nn.Conv2d(feat_dim, num_classes, kernel_size=1, padding=0)
        self.accuracy = pl.metrics.Accuracy()

    def reset_agg(self):
        self.__ZtPiZ = torch.zeros(self.num_classes, self.feat_dim, self.feat_dim).cuda()
        self.__Z_mean = torch.zeros(self.feat_dim).cuda()
        self.__num_batches = 0.

    @property
    def ZtPiZ(self):
        return self.__ZtPiZ / self.__num_batches

    @property
    def Z_mean(self):
        return self.__Z_mean / self.__num_batches

    def forward(self, x):
        z = self.encoder(x)
        if self.loss == 'mcr2':
            # Normalize to unit length
            z = z / torch.norm(z, dim=1, keepdim=True)
        return z

    def training_step(self, batch, batch_idx):
        # x      shape (batch_size, C,     H, W)
        # feats  shape (batch_size, feat_dim, H, W)
        # labels shape (batch_size, H, W)
        x, labels = batch
        labels = torch.squeeze(labels)
        feats = self(x)
        if self.task == 'classify':
            labels, _ = labels.view(labels.shape[0], -1).max(-1)
            feats = feats.view(*feats.shape[:-2], -1).mean(-1)

        if self.loss == 'mcr2':
            Z = feats.transpose(0, 1).view(feats.shape[1], -1).T
            Y = labels.view(-1)
            mcr_ret = self.criterion(Z, Y)
            self.__ZtPiZ += mcr_ret.ZtPiZ
            self.__Z_mean += mcr_ret.Z_mean
            self.__num_batches += 1

            loss = mcr_ret.loss
            preds = self.classifier(Z).view(labels.shape) if self.classifier else None

            self.log('train_discrim_loss', mcr_ret.discrim_loss)
            self.log('train_compress_loss', mcr_ret.compress_loss)
            self.log('train_ZtPiZ_mean', torch.mean(mcr_ret.ZtPiZ))
            self.log('train_Z_mean', torch.mean(mcr_ret.Z_mean))
        else:
            logits = self.classifier(feats)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        if preds is not None:
            self.log('train_acc', self.accuracy(preds, labels), on_step=True, prog_bar=True)
        if batch_idx % 10 == 0:
            if self.task == 'semseg':
                class_labels = {0: 'background'}
                class_labels.update({v+1: str(v) for v in range(10)})
                masks = {"ground_truth": {"mask_data": to_wandb_im(labels[0]), "class_labels": class_labels}}
                if preds is not None:
                    masks["predictions"] = {"mask_data": to_wandb_im(preds[0]), "class_labels": class_labels}
                img = wandb.Image(to_wandb_im(x[0]), masks=masks)
                self.logger.experiment.log({'train_img': [img]}, commit=False)
        return loss

    def training_epoch_end(self, outputs):
        if self.loss == 'mcr2':
            self.classifier = FastNearestSubspace(self.ZtPiZ, self.Z_mean,
                                                  num_classes=self.num_classes,
                                                  n_components=self.feat_dim // self.num_classes)
            self.reset_agg()

        # self.log('train_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)  # TODO LR param
