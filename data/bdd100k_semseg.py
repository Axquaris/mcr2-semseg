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

from collections import namedtuple

# Code based on https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py
Label = namedtuple("Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... . We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate).
        "trainId",
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored during evaluations or not
        "color",  # The color of this label
    ],
)


# Our extended list of label types. Our train id is compatible with Cityscapes
labels = [
    # name id trainId category catId hasInstances ignoreInEval color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    Label("guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    Label("traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    Label("traffic sign frame", 27, 255, "object", 3, False, True, (250, 170, 250)),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]


class BddSS(torch.utils.data.Dataset):
    norm_3c = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

        attr_labels = {
            'weather': ['rainy', 'snowy', 'clear', 'overcast', 'undefined', 'partly cloudy', 'foggy'],
            'scene': ['tunnel', 'residential', 'parking lot', 'undefined', 'city street', 'gas stations', 'highway'],
            'timeofday': ['daytime', 'night', 'dawn/dusk', 'undefined']
        }
        # number attribute labels
        self._attr_to_labels = {k: {l: i for i, l in enumerate(attr_labels[k])} for k in attr_labels}
        self.attr_labels =     {k: {i: l for i, l in enumerate(attr_labels[k])} for k in attr_labels}

        classes = ["bicycle", "bus", "car", "motorcycle", "person", "rider",
                   "traffic light", "traffic sign", "train", "truck"]
        # number class labels
        label_names = [l.name for l in labels]
        self._class_to_colors = {c: torch.tensor(labels[label_names.index(c)].color) for c in classes}
        self.class_labels =     {i+1: c for i, c in enumerate(classes)}  # 0 is for background

        self._classes_encountered = {l.name: 0 for l in labels}

    @staticmethod
    def init_data(root, train):
        train = 'train' if train else 'val'

        BddSS.data = []

        img_pth = root.joinpath(f'seg/images/{train}')
        img_files = [f for f in img_pth.iterdir()]
        img_names = [f.name for f in img_files]

        for part in 'train', 'val':
            labels_file = root.joinpath(f'labels/detection20/det_v2_{part}_release.json')
            with open(labels_file) as f:
                labels_json = json.load(f)

            for d in labels_json:
                if d['name'] in img_names:
                    img_pth = root.joinpath(f'seg/images/train/{d["name"]}')
                    # label_pth = root.joinpath(f'seg/labels/train/{d["name"][:-4]}_train_id.png')
                    label_pth = root.joinpath(f'seg/color_labels/train/{d["name"][:-4]}_train_color.png')
                    attr = d['attributes']  # TODO map to class number vec
                    BddSS.data.append((img_pth, label_pth, attr))

    def __getitem__(self, item):
        img_pth, label_pth, attr = self.data[item]

        with open(img_pth, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
            img = transforms.ToTensor()(img)
            img = BddSS.norm_3c(img)

        with open(label_pth, 'rb') as f:
            color_label = Image.open(f)
            color_label = transforms.ToTensor()(color_label) * 255  # Saved data normalized to [0, 1]
            color_label = color_label.type(torch.long).permute(1, 2, 0)
            label = torch.zeros_like(color_label)

            for i, name in enumerate(self._class_to_colors):
                color = self._class_to_colors[name]
                idx = i + 1  # 0 is for background

                self._classes_encountered[name] += 1

                mask = (color_label == color.unsqueeze(0).unsqueeze(0)).all(-1)
                label[mask] = idx

        attr = torch.tensor([self._attr_to_labels[category][attr[category]] for category in attr])

        # img = transforms.Resize(size=(28, 28))(img)
        # img = transforms.RandomCrop(size=(28, 28), pad_if_needed=True)(img)

        # TODO: process img, map label/attr to class idxs

        return img, label, attr

    def __len__(self):
        return len(self.data)
