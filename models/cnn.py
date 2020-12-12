import torch.nn as nn


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