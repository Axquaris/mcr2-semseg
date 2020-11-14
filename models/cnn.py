import torch
import torch.nn as nn
from mcr2_loss import MaximalCodingRateReduction
import pytorch_lightning as pl


def get_mnist_semseg():
    n_channels = [1, 16, 32, 64]

    layers = []
    for idx in range(len(n_channels)-2):
        i, o = n_channels[idx], n_channels[idx+1]
        layers.append(nn.Conv2d(i, o, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(o))
        layers.append(nn.ReLU())
    i, o = n_channels[-2], n_channels[-1]
    layers.append(nn.Conv2d(i, o, kernel_size=3, padding=1))
    layers.append(nn.BatchNorm2d(o))
    layers.append(nn.ReLU())

    encoder = nn.Sequential(*layers)
    return CNN(encoder, 11, 64)


class CNN(pl.LightningModule):
    def __init__(self, encoder, num_classes, dim_z):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.dim_z = dim_z

        self.criterion = MaximalCodingRateReduction(num_classes)
        self.ZtPiZ = torch.zeros(num_classes, dim_z, dim_z).cuda()

    def forward(self, x):
        z = self.encoder(x)
        return z

    def training_step(self, batch, batch_idx):
        # x shape (batch_size, C,     H, W)
        # z shape (batch_size, dim_z, H, W)
        # y shape (batch_size, H, W)
        x, y = batch
        feats = self(x)

        Z = feats.permute(1, 0, 2, 3).reshape(feats.shape[1], -1).T
        Y = y.reshape(-1)
        loss, ZtPiZ = self.criterion(Z, Y)
        self.ZtPiZ += ZtPiZ

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)
