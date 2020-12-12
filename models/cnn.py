import torch
import torch.nn as nn
import wandb

from mcr2_loss import MaximalCodingRateReduction
import pytorch_lightning as pl
from models.classifiers import *
from models.resnet import ResNet18, ResNet10MNIST
from models.unet import UNet


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
        layers.append(nn.BatchNorm2d(o))
        layers.append(nn.ReLU())
    i, o = n_channels[-2], n_channels[-1]
    layers.append(nn.Conv2d(i, o, kernel_size=5, padding=2))
    layers.append(nn.BatchNorm2d(o))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)

def get_mnist_resnet(in_c, feat_dim, depth="10"):
    if depth == "10":
        return ResNet10MNIST(feature_dim=feat_dim, in_c=in_c)
    else:
        return ResNet18(feature_dim=feat_dim, in_c=in_c)

def get_mnist_unet(in_c, feat_dim):
    return UNet(in_c, feat_dim)


class CNN(pl.LightningModule):
    def __init__(self, encoder, num_classes, dim_z, loss, task, encode_arch="cnn"):
        super(CNN, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.dim_z = dim_z
        self.loss = loss
        self.task = task
        self.encode_arch = encode_arch

        if self.loss == 'mcr2':
            self.criterion = MaximalCodingRateReduction(num_classes)
            self.classifier = None
            self.reset_agg()

        elif self.loss == 'ce' and self.task == 'classify':
            self.criterion = nn.CrossEntropyLoss()
            self.classifier = nn.Linear(dim_z, num_classes)
        elif self.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            self.classifier = nn.Conv2d(dim_z, num_classes, kernel_size=1, padding=0)
        self.accuracy = pl.metrics.Accuracy()



    def reset_agg(self):
        self.__ZtPiZ = torch.zeros(self.num_classes, self.dim_z, self.dim_z).cuda()
        self.__Z_mean = torch.zeros(self.dim_z).cuda()
        self.__num_batches = 0.

    @property
    def ZtPiZ(self):
        return self.__ZtPiZ / self.__num_batches

    @property
    def Z_mean(self):
        return self.__Z_mean / self.__num_batches

    def forward(self, x):
        z = self.encoder(x)
        return z

    def training_step(self, batch, batch_idx):
        # x      shape (batch_size, C,     H, W)
        # feats  shape (batch_size, dim_z, H, W)
        # labels shape (batch_size, H, W)
        x, labels = batch
        labels = torch.squeeze(labels)
        feats = self(x)

        if self.task == 'classify' and self.encode_arch == 'cnn':
            labels, _ = labels.view(labels.shape[0], -1).max(-1)
            feats = feats.view(*feats.shape[:-2], -1).mean(-1)
        elif self.task == 'classify':
            labels, _ = labels.view(labels.shape[0], -1).max(-1)

        if self.loss == 'mcr2':
            # Normalize to unit length
            feats = feats / torch.norm(feats, dim=1, keepdim=True)

            Z = feats.transpose(0, 1).view(feats.shape[1], -1).T
            Y = labels.view(-1)
            mcr_ret = self.criterion(Z, Y)
            loss = mcr_ret.loss
            self.__ZtPiZ += mcr_ret.ZtPiZ
            self.__Z_mean += mcr_ret.Z_mean
            self.__num_batches += 1

            self.log('train_discrim_loss', mcr_ret.discrim_loss)
            self.log('train_compress_loss', mcr_ret.compress_loss)
            self.log('train_ZtPiZ_mean', torch.mean(mcr_ret.ZtPiZ))
            self.log('train_Z_mean', torch.mean(mcr_ret.Z_mean))

            if self.classifier:
                classif_out = self.classifier(Z)
                preds = classif_out.view(labels.shape)
                self.log('train_acc', self.accuracy(preds, labels), on_step=True, prog_bar=True)
            else:
                preds = None
        else:
            logits = self.classifier(feats)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)
            self.log('train_acc', self.accuracy(preds, labels), on_step=True, prog_bar=True)

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        if batch_idx % 10 == 0:
            if self.task == 'semseg':
                class_labels = {0: 'background'}
                class_labels.update({v+1: str(v) for v in range(10)})
                masks = {"ground_truth": {"mask_data": to_wandb_im(labels[0]), "class_labels": class_labels}}
                if preds is not None:
                    masks["predictions"] = {"mask_data": to_wandb_im(preds[0]), "class_labels": class_labels}
                img = wandb.Image(to_wandb_im(x[0]), masks=masks)
                self.logger.experiment.log({'train_img': [img]})
        return loss

    def training_epoch_end(self, outputs):
        if self.loss == 'mcr2':
            self.classifier = FastNearestSubspace(self.ZtPiZ, self.Z_mean,
                                                  num_classes=self.num_classes,
                                                  n_components=self.dim_z // self.num_classes)
            self.reset_agg()

        # self.log('train_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)
