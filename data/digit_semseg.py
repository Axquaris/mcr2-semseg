#  Adapted from https://github.com/RobRomijnders/segm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join, exists
from torchvision.datasets import CIFAR10, MNIST, QMNIST, USPS
from torchvision import transforms
import torch


class DigitSS(torch.utils.data.Dataset):
    """
    Object to sample the data that we can segment. The sample function combines data
    from MNIST and CIFAR and overlaps them
    """
    norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    norm_1c = transforms.Normalize([0.449], [0.226])

    def __init__(self, root='~/data', train=True, cifar_bg=False, digit_sources=['mnist']):
        self.train = train
        self.cifar_bg = cifar_bg

        digit_sets = []
        if 'mnist' in digit_sources:
            digit_sets.append(MNIST(join(root, 'mnist'), train=train, download=True))
        if 'qmnist' in digit_sources:
            # Dataset details: https://github.com/facebookresearch/qmnist
            digit_sets.append(QMNIST(join(root, 'qmnist'), train=train, download=True, compat=False))
        if 'usps' in digit_sources:
            digit_sets.append(USPS(join(root, 'usps'), train=train, download=True))
        self.digit_sets = digit_sets

        if cifar_bg:
            self.bg = CIFAR10(join(root, 'cifar10'), train=train, download=True)

        self.class_labels = {0: 'background'}
        self.class_labels.update({v + 1: str(v) for v in range(10)})

    def _get_digit(self, item):
        subset = self.digit_sets[item % len(self.digit_sets)]
        if isinstance(subset, QMNIST):
            return subset[item // 3][0]  # take only class label for now
        return subset[item // 3]

    def __getitem__(self, item):
        """
        Randomly inserts the MNIST images into cifar images
        :param item:
        :return:
        """
        digit_im, label = self._get_digit(item)
        # digit_im = transforms.Resize(size=(28, 28))(digit_im)
        # digit_im = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(digit_im)
        digit_im = transforms.ToTensor()(digit_im)

        mask = digit_im.clone()
        mask[mask < .3] = -1  # background label
        mask[mask >= .3] = label

        if self.cifar_bg:
            bg_im, _ = self.bg[item % len(self.bg)]
            bg_im = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(bg_im)
            bg_im = transforms.ToTensor()(bg_im)

            bg_im[mask.repeat(3, 1, 1) != -1] = 0
            im = DigitSS.norm_3c(digit_im.repeat(3, 1, 1) + bg_im)
        else:
            im = DigitSS.norm_1c(digit_im)

        return im, (mask + 1).type(torch.long)

    def __len__(self):
        return min([len(subset) for subset in self.digit_sets])

