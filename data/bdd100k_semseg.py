#  Adapted from https://github.com/RobRomijnders/segm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as pth
import json

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch


class BddSS(torch.utils.data.Dataset):
    norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    norm_1c = transforms.Normalize([0.449], [0.226])
    data = None

    def __init__(self, root='/datasets/bdd100k', train=True):
        """

        :param root:
        :param train: [train, val, or test]
        """
        self.train = train
        self.root = pth.Path(root)

        if BddSS.data is None:
            BddSS.init_data(self.root, self.train)

        mid_idx = len(BddSS.data) * 4 // 5
        if self.train:
            self.data = BddSS.data[:mid_idx]
        else:
            self.data = BddSS.data[mid_idx:]

        # TODO: class name lookup dict for semseg and attrs

    @staticmethod
    def init_data(root, train):
        BddSS.data = []
        labels_json = []
        for part in 'train', 'val':
            labels_file = root.joinpath(f'labels/detection20/det_v2_{part}_release.json')
            with open(labels_file) as f:
                labels_json += json.load(f)

        img_pth = root.joinpath(f'seg/images/{train}')
        for d in labels_json:
            img_files = [f for f in img_pth.iterdir()]
            img_names = [f.name for f in img_files]
            if d['name'] in img_names:
                img_pth = root.joinpath(f'seg/images/{train}/{d["name"]}')
                label_pth = root.joinpath(f'seg/labels/{train}/{d["name"]}')
                attr = d['attributes']  # TODO map to class number vec
                BddSS.data.append((img_pth, label_pth, attr))

    def __getitem__(self, item):
        img_pth, label_pth, attr = self.data[item]

        with open(img_pth, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')

        with open(label_pth, 'rb') as f:
            label = Image.open(f)

        # mnist_im, label = self.mnist[item]
        # mnist_im = transforms.Resize(size=(28, 28))(mnist_im)
        # mnist_im = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(mnist_im)
        # mnist_im = transforms.ToTensor()(mnist_im)
        #
        # mask = mnist_im.clone()
        # mask[mask < .3] = -1  # background label
        # mask[mask >= .3] = label
        #
        # if self.cifar_bg:
        #     cifar10, _ = self.cifar10[item]
        #     cifar10 = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(cifar10)
        #     cifar10 = transforms.ToTensor()(cifar10)
        #
        #     cifar10[mask.repeat(3, 1, 1) != -1] = 0
        #     im = MnistSS.norm_3c(mnist_im.repeat(3, 1, 1) + cifar10)
        # else:
        #     im = MnistSS.norm_1c(mnist_im)
        #

        # TODO: process img, map label/attr to class idxs

        return img, label, attr

    def __len__(self):
        return len(self.data)
