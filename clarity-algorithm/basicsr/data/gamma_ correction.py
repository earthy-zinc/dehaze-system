import os
import cv2
import numpy as np


# gamma correction
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def do_gamma_correction(filepath, gamma_B, gamma_G, gamma_R):
    img = cv2.imread(filepath)
    # do gamma correction
    gammaImg_B = gammaCorrection(img[:, :, 0], gamma_B)
    gammaImg_G = gammaCorrection(img[:, :, 1], gamma_G)
    gammaImg_R = gammaCorrection(img[:, :, 2], gamma_R)
    gammaImg = np.dstack((gammaImg_B,gammaImg_G, gammaImg_R))
    return gammaImg


readpath = "./dehaze_dataset/train/NTIRE2020/hazy"
savepath = "./data_hazy_gc/20_hazyRGB_gamma"
#do_gamma_correction(readpath, gamma_B=1.9, gamma_G=1.6, gamma_R=1.24)
# for 2020 GT images, gamma values should be: R(1.07), G(1.17), B(1.05), please verify the adjusted mean and variance

readpath = "./dehaze_dataset/train/NTIRE2021/hazy"
savepath = "./data_hazy_gc/21_hazyRGB_gamma"
do_gamma_correction(readpath, gamma_B=1, gamma_G=0.85, gamma_R=0.72)
# for 2021 GT images, gamma values should be: R(0.65), G(0.79), B(0.92), please verify the adjusted mean and variance
