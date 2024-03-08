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


# gamma correction
def gamma_correction(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def do_gamma_correction(img, gamma_b: float, gamma_g: float, gamma_r: float):
    # do gamma correction
    gamma_img_b = gamma_correction(img[:, :, 0], gamma_b)
    gamma_img_g = gamma_correction(img[:, :, 1], gamma_g)
    gamma_img_r = gamma_correction(img[:, :, 2], gamma_r)
    gamma_img = np.dstack((gamma_img_b, gamma_img_g, gamma_img_r))
    return gamma_img


def ntire_gamma_correction(filepath: str):
    if filepath.find('NH-HAZE-2020/clean') != -1:
        gamma_b = 1.05
        gamma_g = 1.17
        gamma_r = 1.07
    elif filepath.find('NH-HAZE-2020/hazy') != -1:
        gamma_b = 1.9
        gamma_g = 1.6
        gamma_r = 1.24
    elif filepath.find('NH-HAZE-2021/clean') != -1:
        gamma_b = 0.92
        gamma_g = 0.79
        gamma_r = 0.65
    elif filepath.find('NH-HAZE-2021/hazy') != -1:
        gamma_b = 1
        gamma_g = 0.85
        gamma_r = 0.72
    else:
        return np.array(Image.open(filepath).convert("RGB")) / 255
    img = cv2.imread(filepath)
    img = do_gamma_correction(img, gamma_b=gamma_b, gamma_g=gamma_g, gamma_r=gamma_r)
    return np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) / 255


def create_dataset(dataset_name, size, stride, dataset_path):
    """
    synthetic Haze images
    """
    train_dataset = dataset_name + ".h5"
    val_dataset = dataset_name + "-test.h5"

    psnr = pyiqa.create_metric("psnr")
    ssim = pyiqa.create_metric("ssim")

    if dataset_name == "NH-HAZE-20-21-23" and isinstance(dataset_path, list):
        train_haze_list = []
        test_haze_list = []
        train_clean_list = []
        test_clean_list = []
        for path in dataset_path:
            files2_clean = os.listdir(path + 'clean/')
            files2_hazy = os.listdir(path + 'hazy/')
            assert len(files2_clean) > 5 and len(files2_hazy) > 5
            # [:-5] 获取倒数第5个元素之前的所有元素
            # [-5:] 获取数组中后5个元素
            train_haze_list.extend(files2_hazy[:-5])
            test_haze_list.extend(files2_hazy[-5:])
            train_clean_list.extend(files2_clean[:-5])
            test_clean_list.extend(files2_clean[-5:])
    else:
        assert isinstance(dataset_path, str)
        files2_clean = os.listdir(dataset_path + 'clean/')
        files2_hazy = os.listdir(dataset_path + 'hazy/')
        assert len(files2_clean) > 5 and len(files2_hazy) > 5
        train_haze_list = files2_hazy[:-5]
        test_haze_list = files2_hazy[-5:]
        train_clean_list = files2_clean[:-5]
        test_clean_list = files2_clean[-5:]

    with h5py.File(train_dataset, 'w') as h5f:
        count = 0
        count_bad = 0
        scales = [0, 0.3]
        for i in range(len(train_haze_list)):
            hazy_0 = np.array(Image.open(dataset_path + 'hazy/' + train_haze_list[i]).convert("RGB")) / 255
            clear_0 = np.array(Image.open(dataset_path + 'clean/' + train_clean_list[i]).convert("RGB")) / 255

            print("分割清晰图片{}".format(str(train_clean_list[i])))
            print("分割有雾图片{}".format(str(train_haze_list[i])))

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
                    if 0.94 > clear_out_tensor.mean() > 0.05 and psnr_hl < 38 and ssim_hl < 0.93:
                        dataset = np.stack((clear_out, hazy_out))
                        h5f.create_dataset(str(count), data=dataset)
                        count += 1
                        print("第{}对图像，分割后的图片形状{}".format(count, str(dataset.shape)))
                    else:
                        count_bad += 1
    print("共计{}对分割后的图片".format(count))
    print("去除了{}张不符合要求过白或过暗的图片".format(count_bad))
    h5f.close()

    scale = 0.3
    with h5py.File(val_dataset, 'w') as h5f:
        count = 0
        for i in range(len(test_haze_list)):
            hazy_0 = np.array(Image.open(dataset_path + 'hazy/' + test_haze_list[i]).convert("RGB")) / 255
            clear_0 = np.array(Image.open(dataset_path + 'clean/' + test_clean_list[i]).convert("RGB")) / 255
            hazy = cv2.resize(hazy_0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            clear = cv2.resize(clear_0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            clear = np.transpose(clear, (2, 0, 1))
            hazy = np.transpose(hazy, (2, 0, 1))
            print("处理测试图片{}，形状{}".format(test_clean_list[i], clear.shape))
            dataset = np.stack((clear, hazy))
            h5f.create_dataset(str(count), data=dataset)
            count += 1
    print("共计{}对测试的图片".format(count))
    h5f.close()


if __name__ == "__main__":
    if os.path.exists("/mnt/e/DeepLearningCopies/2023/RIDCP"):
        base_path = "/mnt/d/DeepLearning/dataset/"
    elif os.path.exists("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP"):
        base_path = "/quzhong_fix/wpx/dataset/"
    elif os.path.exists("/mnt/workspace/ridcp"):
        base_path = "/mnt/data/"
    elif os.path.exists("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP"):
        base_path = "/var/lib/docker/user1/wpx/dataset/"
    else:
        base_path = "D:/DeepLearning/dataset/"

    create_dataset("O-HAZE", 256, 200, base_path + "O-HAZE/")
    create_dataset("I-HAZE", 256, 200, base_path + "I-HAZE/")
    create_dataset("DENSE-HAZE", 256, 200, base_path + "Dense-Haze/")
    create_dataset("NH-HAZE-20", 256, 200, base_path + "NH-HAZE-2020/")
    create_dataset("NH-HAZE-21", 256, 200, base_path + "NH-HAZE-2021/")
    create_dataset("NH-HAZE-23", 256, 200, base_path + "NH-HAZE-2023/")
    create_dataset("NH-HAZE-20-21-23", 256, 200,  [
        base_path + "NH-HAZE-2020/",
        base_path + "NH-HAZE-2021/",
        base_path + "NH-HAZE-2023/"
    ])

