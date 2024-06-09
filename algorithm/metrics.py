import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate(haze_image_path: str, clear_image_path: str):
    haze = Image.open(haze_image_path).convert('RGB')
    clear = Image.open(clear_image_path).convert('RGB')
    haze = np.array(haze)
    clear = np.array(clear)
    current_psnr = peak_signal_noise_ratio(haze, clear)
    current_ssim = structural_similarity(haze, clear, channel_axis=2)
    return current_psnr, current_ssim
