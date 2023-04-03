import torchvision.datasets.voc as voc
import os
# import math
# from tqdm import tqdm
import torch
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import average_precision_score, accuracy_score
# import pandas as pd

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def get_categories(labels_dir):
    """
    Get the object categories

    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError

    else:
        categories = []

        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])

        return categories


def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding

    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """

    ls = target['annotation']['object']

    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k).argmax(dim=0) # .long()

class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super().__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)

    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)



