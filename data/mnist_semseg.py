#  Adapted from https://github.com/RobRomijnders/segm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join, exists
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch

def plot_data(X):
    """
    Generic function to plot the images in a grid
    of num_plot x num_plot
    :param X:
    :return:
    """
    plt.figure()
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for i in range(num_plot):
        for j in range(num_plot):
            idx = np.random.randint(0, X.shape[0])
            ax[i,j].imshow(X[idx])
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)  # No horizontal space between subplots
    f.subplots_adjust(wspace=0)

class MnistSS(torch.utils.data.Dataset):
    """
    Object to sample the data that we can segment. The sample function combines data
    from MNIST and CIFAR and overlaps them
    """
    norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    norm_1c = transforms.Normalize([0.449], [0.226])

    def __init__(self, root='~/data', train=True, norm=True, cifar_bg=False):
        self.train = train
        self.norm = norm
        self.cifar_bg = cifar_bg

        mnist_root = join(root, 'mnist')
        self.mnist = MNIST(mnist_root, train=train, download=True)

        if cifar_bg:
            cifar10_root = join(root, 'cifar10')
            self.cifar10 = CIFAR10(cifar10_root, train=train, download=True)

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
        # cifar_im = self.cifar10[item]

        # width_start = np.random.randint(0, 32 - size_mnist, size=(batch_size))
        # height_start = np.random.randint(0, 32 - size_mnist, size=(batch_size))
        # color_range = 200
        #
        # mnist_batch = np.repeat(np.expand_dims(im_mnist * color_range, 3), 3, 3)
        #
        # segm_maps = np.zeros((batch_size, 32, 32))
        #
        # for i in range(batch_size):
        #     im_cifar[i, width_start[i]:width_start[i] + size_mnist, height_start[i]:height_start[i] + size_mnist] += \
        #     mnist_batch[i]
        #     segm_maps[i, width_start[i]:width_start[i] + size_mnist, height_start[i]:height_start[i] + size_mnist] += \
        #     mnist_mask[i]
        # im_cifar = np.clip(im_cifar, 0, 255)
        #
        # if norm:
        #     im_cifar = (im_cifar - 130.) / 70.
        # return im_cifar, segm_maps

    def __len__(self):
        if self.cifar_bg:
            return min(len(self.mnist), len(self.cifar10))
        return len(self.mnist)

