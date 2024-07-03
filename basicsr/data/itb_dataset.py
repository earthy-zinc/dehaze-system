import os
import random

import math
import cv2
import h5py
import numpy as np
import pyiqa
import torch
import torchvision
from PIL import Image
from natsort import natsort
from torch.utils import data as data
from torchvision.transforms import functional as FF, ToTensor, Resize

from basicsr.utils.registry import DATASET_REGISTRY


def cropping_ohaze(hazy, index):

    if hazy.shape[2] == 4846:
        hazy1 = hazy[:, :2560, :2560]
        hazy2 = hazy[:, :2560, -2560:]
        hazy3 = hazy[:, -2560:, :2560]
        hazy4 = hazy[:, -2560:, -2560:]

    else:
        assert max(hazy.shape[1], hazy.shape[2]) <= 4096

        hazy1 = hazy[:, :2048, :2048]
        hazy2 = hazy[:, :2048, -2048:]
        hazy3 = hazy[:, -2048:, :2048]
        hazy4 = hazy[:, -2048:, -2048:]

    hazy = [hazy1, hazy2, hazy3, hazy4]

    return hazy

@DATASET_REGISTRY.register()
class ITBDataset(data.Dataset):
    def __init__(self, opt):
        super(ITBDataset, self).__init__()
        if os.path.exists("/mnt/e/DeepLearningCopies/2023/RIDCP"):
            base_path = "/mnt/d/DeepLearning/dataset/"
        elif os.path.exists("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP"):
            base_path = "/quzhong_fix/wpx/dataset/"
        elif os.path.exists("/mnt/workspace/ridcp"):
            base_path = "/mnt/data/"
        elif os.path.exists("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP"):
            base_path = "/var/lib/docker/user1/wpx/dataset/"
        elif os.path.exists("/Crack_detection/wpx"):
            base_path = "/Crack_detection/wpx/dataset/"
        else:
            base_path = "D:/DeepLearning/dataset/"
        self.opt = opt
        if base_path.find("O-HAZE") or base_path.find("I-HAZE"):
            self.type = "IO-HAZE"
        else:
            self.type = "OTHER"
        haze_image_path = [os.path.join(base_path, opt['haze_path'], x) for x in
                           natsort.natsorted(os.listdir(
                               os.path.join(base_path, opt['haze_path'])))]
        clear_image_path = [os.path.join(base_path, opt['clear_path'], x) for x in
                            natsort.natsorted(os.listdir(
                                os.path.join(base_path, opt['clear_path'])))]
        if self.opt['mode'] == 'val':
            self.haze_image_path = haze_image_path[-5:]
            self.clear_image_path = clear_image_path[-5:]
        else:
            self.haze_image_path = haze_image_path
            self.clear_image_path = clear_image_path

    def __getitem__(self, index):
        haze = Image.open(self.haze_image_path[index]).convert("RGB")
        clear = Image.open(self.clear_image_path[index]).convert("RGB")
        transform = torchvision.transforms.Compose([
            ToTensor()
        ])
        haze = transform(haze)
        clear = transform(clear)
        haze_shape = haze.shape

        if haze.shape[0] == 5:
            assert torch.equal(haze[-1:, :, :], torch.ones(1, haze.shape[1], haze.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = haze[:3, :, :]

        haze = cropping_ohaze(hazy, index)
        return {
            "gt": transform(clear),
            "lq": transform(haze),
            "gt_path": self.clear_image_path[index],
            "lq_path": self.haze_image_path[index],
        }

    def __len__(self):
        return len(self.haze_image_path)


