#  Adapted from https://github.com/RobRomijnders/segm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join, exists
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch


class DigitSS(torch.utils.data.Dataset):
    """
    Object to sample the data that we can segment. The sample function combines data
    from MNIST and CIFAR and overlaps them
    """
    norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    norm_1c = transforms.Normalize([0.449], [0.226])

    def __init__(self, root='~/data', train=True, cifar_bg=False):
        self.train = train
        self.cifar_bg = cifar_bg

        mnist_root = join(root, 'mnist')
        self.mnist = MNIST(mnist_root, train=train, download=True)

        usps_root = join(root, 'mnist')
        self.mnist = MNIST(mnist_root, train=train, download=True)

        if cifar_bg:
            cifar10_root = join(root, 'cifar10')
            self.cifar10 = CIFAR10(cifar10_root, train=train, download=True)

        self.class_labels = {0: 'background'}
        self.class_labels.update({v + 1: str(v) for v in range(10)})

    def __getitem__(self, item):
        """
        Randomly inserts the MNIST images into cifar images
        :param item:
        :return:
        """
        mnist_im, label = self.mnist[item]
        mnist_im = transforms.Resize(size=(28, 28))(mnist_im)
        mnist_im = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(mnist_im)
        mnist_im = transforms.ToTensor()(mnist_im)

        mask = mnist_im.clone()
        mask[mask < .3] = -1  # background label
        mask[mask >= .3] = label

        if self.cifar_bg:
            cifar10, _ = self.cifar10[item]
            cifar10 = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(cifar10)
            cifar10 = transforms.ToTensor()(cifar10)

            cifar10[mask.repeat(3, 1, 1) != -1] = 0
            im = MnistSS.norm_3c(mnist_im.repeat(3, 1, 1) + cifar10)
        else:
            im = MnistSS.norm_1c(mnist_im)

        return im, (mask + 1).type(torch.long)

    def __len__(self):
        if self.cifar_bg:
            return min(len(self.mnist), len(self.cifar10))
        return len(self.mnist)

