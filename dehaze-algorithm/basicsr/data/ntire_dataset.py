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


@DATASET_REGISTRY.register()
class NtireH5Dataset(data.Dataset):
    def __init__(self, opt):
        super(NtireH5Dataset, self).__init__()
        self.opt = opt
        self.data_path = opt["data_path"]
        h5f = h5py.File(self.data_path, 'r')
        self.keys = list(h5f.keys())
        if opt["shuffle"]:
            random.shuffle(self.keys)
        h5f.close()

    def __getitem__(self, index):
        h5f = h5py.File(self.data_path, 'r')
        key = self.keys[index]
        img_data = np.array(h5f[key])
        h5f.close()
        return {
            "gt": torch.Tensor(img_data[0]).squeeze(),
            "lq": torch.Tensor(img_data[1]).squeeze(),
            "gt_path": self.data_path + "/" + key + ".jpg",
            "lq_path": self.data_path + "/" + key + ".jpg",
        }

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class NtireDataset(data.Dataset):
    def __init__(self, opt):
        super(NtireDataset, self).__init__()
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
        self.max_size = 2000 * 2000

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
        return {
            "gt": transform(clear),
            "lq": transform(haze),
            "gt_path": self.clear_image_path[index],
            "lq_path": self.haze_image_path[index],
        }

    def __len__(self):
        return len(self.haze_image_path)


