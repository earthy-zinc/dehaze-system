import os

import cv2
import h5py
import numpy as np
import pyiqa
import torch
from PIL import Image


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
    if raw < win or col < win:
        return np.zeros((chl, win, win, 0))
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


def train_data(h5_file, size, stride, dataset_path):
    """

    synthetic Haze images
    """
    psnr = pyiqa.create_metric("psnr")
    ssim = pyiqa.create_metric("ssim")
    files2_clear = os.listdir(dataset_path + 'clean/')
    files2_haze = os.listdir(dataset_path + 'hazy/')
    with h5py.File(h5_file, 'w') as h5f:
        count = 0
        count_bad = 0
        scales = [0, 0.2]
        for i in range(len(files2_clear)):
            hazy_0 = np.array(Image.open(dataset_path + 'hazy/' + files2_haze[i]).convert("RGB")) / 255
            clear_0 = np.array(Image.open(dataset_path + 'clean/' + files2_clear[i]).convert("RGB")) / 255

            print("分割清晰图片{}".format(str(files2_clear[i])))
            print("分割有雾图片{}".format(str(files2_haze[i])))

            for scale in scales:
                if scale == 0:
                    hazy = cv2.resize(hazy_0, (size, size), interpolation=cv2.INTER_AREA)
                    clear = cv2.resize(clear_0, (size, size), interpolation=cv2.INTER_AREA)
                else:
                    hazy = cv2.resize(hazy_0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    clear = cv2.resize(clear_0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                hazy = img_to_patches(hazy.transpose(2, 0, 1), size, stride)
                clear = img_to_patches(clear.transpose(2, 0, 1), size, stride)

                print("缩放率{}，分割后的图片形状{}".format(scale, str(hazy.shape)))

                for nx in range(clear.shape[3]):
                    clear_out, hazy_out = data_augmentation(
                        clear[:, :, :, nx].copy(),
                        hazy[:, :, :, nx].copy(),
                        np.random.randint(0, 7)
                    )

                    clear_out_tensor = torch.Tensor(clear_out.copy()).squeeze()[None, ::]
                    hazy_out_tensor = torch.Tensor(hazy_out.copy()).squeeze()[None, ::]

                    psnr_hl = psnr(clear_out_tensor, hazy_out_tensor).item()
                    ssim_hl = ssim(clear_out_tensor, hazy_out_tensor).item()
                    if 0.93 > clear_out_tensor.mean() > 0.05 and psnr_hl < 35 and ssim_hl < 0.92:
                        dataset = np.stack((clear_out, hazy_out))
                        h5f.create_dataset(str(count), data=dataset)
                        count += 1
                        print("第{}对图像，分割后的图片形状{}".format(count, str(dataset.shape)))
                    else:
                        count_bad += 1
    print("共计{}对分割后的图片".format(count))
    print("去除了{}张不符合要求过白或过暗的图片".format(count_bad))
    h5f.close()


if __name__ == "__main__":
    train_data("E:/DeepLearningCopies/2023/RIDCP/datasets/NH-HAZE-20-21-23.h5", 256, 250, "D:/DeepLearning/dataset/NTIRE/")
    train_data("E:/DeepLearningCopies/2023/RIDCP/datasets/O-HAZE.h5", 256, 250, "D:/DeepLearning/dataset/O-HAZE/")
    train_data("E:/DeepLearningCopies/2023/RIDCP/datasets/I-HAZE.h5", 256, 250, "D:/DeepLearning/dataset/I-HAZE/")
    train_data("E:/DeepLearningCopies/2023/RIDCP/datasets/DENSE-HAZE.h5", 256, 250, "D:/DeepLearning/dataset/Dense-Haze/")
    # pass
