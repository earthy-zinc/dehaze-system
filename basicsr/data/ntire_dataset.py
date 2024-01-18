import os
import random

import cv2
import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from natsort import natsort
from torch.utils import data as data
from torchvision.transforms import functional as FF, ToTensor

from basicsr.utils.registry import DATASET_REGISTRY


def data_augmentation(clear, haze, mode):
    r"""Performs data augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    clear = np.transpose(clear, (1, 2, 0))
    haze = np.transpose(haze, (1, 2, 0))
    if mode == 0:
        # original
        clear = clear
        haze = haze
    elif mode == 1:
        # flip up and down
        clear = np.flipud(clear)
        haze = np.flipud(haze)
    elif mode == 2:
        # rotate counterwise 90 degree
        clear = np.rot90(clear)
        haze = np.rot90(haze)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        clear = np.rot90(clear)
        clear = np.flipud(clear)
        haze = np.rot90(haze)
        haze = np.flipud(haze)
    elif mode == 4:
        # rotate 180 degree
        clear = np.rot90(clear, k=2)
        haze = np.rot90(haze, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        clear = np.rot90(clear, k=2)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=2)
        haze = np.flipud(haze)
    elif mode == 6:
        # rotate 270 degree
        clear = np.rot90(clear, k=3)
        haze = np.rot90(haze, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        clear = np.rot90(clear, k=3)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=3)
        haze = np.flipud(haze)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(clear, (2, 0, 1)), np.transpose(haze, (2, 0, 1))


def img_to_patches(img, win, stride, Syn=True):
    chl, raw, col = img.shape
    chl = int(chl)
    num_raw = np.ceil((raw - win) / stride + 1).astype(np.uint8)
    num_col = np.ceil((col - win) / stride + 1).astype(np.uint8)
    count = 0
    total_process = int(num_col) * int(num_raw)
    img_patches = np.zeros([chl, win, win, total_process])
    if Syn:
        for i in range(num_raw):
            for j in range(num_col):
                if stride * i + win <= raw and stride * j + win <= col:
                    img_patches[:, :, :, count] = img[:, stride * i: stride * i + win, stride * j: stride * j + win]
                elif stride * i + win > raw and stride * j + win <= col:
                    img_patches[:, :, :, count] = img[:, raw - win: raw, stride * j: stride * j + win]
                elif stride * i + win <= raw and stride * j + win > col:
                    img_patches[:, :, :, count] = img[:, stride * i: stride * i + win, col - win: col]
                else:
                    img_patches[:, :, :, count] = img[:, raw - win: raw, col - win: col]
                count += 1
    return img_patches


def train_data(dataset_name, size, stride, path):
    """synthetic Haze images"""
    files2_clear = os.listdir(path + 'clean/')
    files2_haze = os.listdir(path + 'hazy/')
    with h5py.File(dataset_name, 'w') as h5f:
        count = 0
        scales = [0.2]
        for i in range(len(files2_clear)):
            hazy_0 = np.array(Image.open(path + 'hazy/' + files2_haze[i])) / 255
            clear_0 = np.array(Image.open(path + 'clean/' + files2_clear[i])) / 255
            print(files2_clear[i])
            print(files2_haze[i])
            for sca in scales:
                print(sca)
                if sca == 0:
                    hazy = cv2.resize(hazy_0, (size, size))
                    clear = cv2.resize(clear_0, (size, size))
                else:
                    hazy = cv2.resize(hazy_0, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)
                    clear = cv2.resize(clear_0, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)

                hazy = img_to_patches(hazy.transpose(2, 0, 1), size, stride)
                clear = img_to_patches(clear.transpose(2, 0, 1), size, stride)
                for nx in range(clear.shape[3]):
                    clear_out, hazy_out = data_augmentation(clear[:, :, :, nx].copy(), hazy[:, :, :, nx].copy(),
                                                            np.random.randint(0, 7))
                    dataset = np.stack((clear_out, hazy_out))
                    h5f.create_dataset(str(count), data=dataset)
                    count += 1
                    print(count, dataset.shape)
    print('Data Num: %d \nData Patch: %d' % (count, size))

    h5f.close()


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
        if self.opt["mode"] == "validation":
            return 500
        return len(self.keys)


@DATASET_REGISTRY.register()
class NtireDataset(data.Dataset):
    def __init__(self, opt):
        super(NtireDataset, self).__init__()
        self.opt = opt
        self.data_path = opt["data_path"]
        self.haze_image_path = [os.path.join(opt['haze_path'], x) for x in
                                natsort.natsorted(os.listdir(opt['haze_path']))]
        self.clear_image_path = [os.path.join(opt['clear_path'], x) for x in
                                 natsort.natsorted(os.listdir(opt['clear_path']))]

    def __getitem__(self, index):
        haze = Image.open(self.haze_image_path[index]).convert("RGB")
        clear = Image.open(self.clear_image_path[index]).convert("RGB")
        return {
            "gt": ToTensor()(clear),
            "lq": ToTensor()(haze),
            "gt_path": self.clear_image_path[index],
            "lq_path": self.haze_image_path[index],
        }

    def __len__(self):
        if self.opt["mode"] == "validation":
            return 500
        return len(self.haze_image_path)


if __name__ == "__main__":
    pass
    # train_data("I-HAZE.h5", 256, 250, "D://DeepLearning//dataset//I-HAZE//")
    #train_data("O-HAZE.h5", 256, 250, "D://DeepLearning//dataset//O-HAZE//")
