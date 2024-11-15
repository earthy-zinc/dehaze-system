import cv2
import numpy as np
import torch
from PIL import Image

from ..metrics import get_niqe, get_brisque, get_nima, get_ssim, get_psnr, get_lpips


def ntire_gamma_correction(filepath: str) -> np.ndarray:
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

def do_gamma_correction(img, gamma_b: float, gamma_g: float, gamma_r: float):
    # do gamma correction
    gamma_img_b = gamma_correction(img[:, :, 0], gamma_b)
    gamma_img_g = gamma_correction(img[:, :, 1], gamma_g)
    gamma_img_r = gamma_correction(img[:, :, 2], gamma_r)
    gamma_img = np.dstack((gamma_img_b, gamma_img_g, gamma_img_r))
    return gamma_img

def gamma_correction(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def imgOperation(filepath):
    haze = ntire_gamma_correction(filepath)
    haze = cv2.resize(haze, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    haze = img2tensor(haze)
    return haze

def calculate(haze_image_path: str, clear_image_path: str = None):
    haze = imgOperation(haze_image_path)
    if clear_image_path is not None:
        clear = imgOperation(clear_image_path)
    else:
        clear = haze

    result = [
        get_niqe(haze, clear),
        get_nima(haze, clear),
        get_brisque(haze, clear),
        get_psnr(haze, clear),
        get_ssim(haze, clear),
        get_lpips(haze, clear)
    ]
    return result
