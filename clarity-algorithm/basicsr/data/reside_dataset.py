import os
import random

import torchvision
from PIL import Image
from natsort import natsort
from torch.utils import data as data
from torchvision.transforms import functional as FF

from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ResideDataset(data.Dataset):
    def __init__(self, opt):
        super(ResideDataset, self).__init__()
        self.opt = opt
        self.crop_size = None
        self.normalize = None
        self.rotation = False
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
        haze_path = base_path + opt['haze_path']
        clear_path = base_path + opt['clear_path']
        self.haze_image_path = [os.path.join(haze_path, x) for x in
                                natsort.natsorted(os.listdir(haze_path))]
        self.clear_image_path = [os.path.join(clear_path, x) for x in
                                 natsort.natsorted(os.listdir(clear_path))]

    def __getitem__(self, index):
        haze = Image.open(self.haze_image_path[index]).convert("RGB")
        clear = Image.open(self.clear_image_path[index]).convert("RGB")
        clear, haze = self.aug_data(clear, haze)
        return {
            "gt": clear,
            "lq": haze,
            "gt_path": self.clear_image_path[index],
            "lq_path": self.haze_image_path[index],
        }

    def aug_data(self, clear, haze):
        if self.rotation:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze = torchvision.transforms.RandomHorizontalFlip(rand_hor)(haze)
            clear = torchvision.transforms.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                haze = FF.rotate(haze, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        haze = torchvision.transforms.ToTensor()(haze)
        if self.normalize is not None:
            haze = torchvision.transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(haze)
        clear = torchvision.transforms.ToTensor()(clear)
        # 随机裁剪
        if self.crop_size is not None:
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(haze,
                                                                      output_size=(self.crop_size, self.crop_size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        return clear, haze

    def __len__(self):
        return len(self.haze_image_path)


@DATASET_REGISTRY.register()
class ResideClearDataset(data.Dataset):
    def __init__(self, opt):
        super(ResideClearDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt['size']
        self.normalize = opt['normalize']
        self.clear_image_path = [os.path.join(opt['clear_path'], x)
                                 for x in natsort.natsorted(os.listdir(opt['clear_path']))
                                 ]

    def __getitem__(self, index):
        clear = Image.open(self.clear_image_path[index]).convert("RGB")
        clear = self.aug_data(clear)
        return {
            'gt': clear,
            'gt_path': self.clear_image_path[index]
        }

    def aug_data(self, clear):
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        clear = torchvision.transforms.RandomHorizontalFlip(rand_hor)(clear)
        if rand_rot:
            clear = FF.rotate(clear, 90 * rand_rot)
        clear = torchvision.transforms.ToTensor()(clear)
        # 随机裁剪
        if self.crop_size is not None:
            i, j, h, w = (torchvision.transforms.RandomCrop
                          .get_params(clear, output_size=self.crop_size))
            clear = FF.crop(clear, i, j, h, w)
        return clear

    def __len__(self):
        return len(self.clear_image_path)
