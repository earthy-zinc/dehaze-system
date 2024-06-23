import pyiqa
from PIL import Image
from torchvision.transforms import ToTensor

niqe = pyiqa.create_metric("niqe")
nima = pyiqa.create_metric("nima")
brisque = pyiqa.create_metric("brisque")
psnr = pyiqa.create_metric("psnr")
ssim = pyiqa.create_metric("ssim")
lpips = pyiqa.create_metric("lpips")


def calculate(haze_image_path: str, clear_image_path: str = None):
    haze = ToTensor()(Image.open(haze_image_path).convert('RGB'))[None, ::]
    if clear_image_path is not None:
        clear = ToTensor()(Image.open(clear_image_path).convert('RGB'))[None, ::]
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


def get_niqe(haze, clear=None):
    niqe_haze = niqe(haze).item()
    if clear is not None:
        niqe_clear = niqe(clear).item()
        value = {"haze": niqe_haze, "clear": niqe_clear}
    else:
        value = niqe_haze

    niqe_result = {
        "id": 1,
        "label": "NIQE",
        "value": value,
        "better": "lower",
        "description": "NIQE是一种无参考图像空间质量评估指标，用于评估图像的失真程度。"
                       "该指标能够估计图像的自然感知质量，无需参考图像或主观评分。NIQE"
                       "指标的计算基于图像的局部特征，通过分析图像的统计特征来估计图像的质量。"
    }
    return niqe_result


def get_nima(haze, clear=None):
    nima_haze = nima(haze).item()
    if clear is not None:
        nima_clear = nima(clear).item()
        value = {"haze": nima_haze, "clear": nima_clear}
    else:
        value = nima_haze

    nima_result = {
        "id": 2,
        "label": "NIMA",
        "value": value,
        "better": "higher",
        "description": "NIMA是一种无参考技术，它可以预测图像的质量，而不依赖于通常不可用的原始参考图像。"
                       "NIMA使用CNN来预测每个图像的质量分数分布。"
                       "在1到10的范围内，NIMA会将这张图的得分可能性分配给这10个分数。"
    }
    return nima_result


def get_brisque(haze, clear=None):
    brisque_haze = brisque(haze).item()
    if clear is not None:
        brisque_clear = brisque(clear).item()
        value = {"haze": brisque_haze, "clear": brisque_clear}
    else:
        value = brisque_haze

    brisque_result = {
        "id": 3,
        "label": "BRISQUE",
        "value": value,
        "better": "lower",
        "description": "BRISQUE是一种无参考图像空间质量评估指标，用于评估图像的失真程度。"
                       "BRISQUE指标的计算基于图像的局部特征，通过分析图像的统计特征来估计图像的质量。"
                       "BRISQUE的评价依据是自然图像的亮度归一化后是趋向于高斯分布的，而失真会破坏这种分布，所以通过测量这种统计特征的改变可以达到衡量失真程度的目的。"
    }
    return brisque_result


def get_psnr(haze, clear=None):
    if clear is None:
        return
    psnr_value = psnr(haze, clear).item()
    psnr_result = {
        "id": 4,
        "label": "PSNR",
        "value": psnr_value,
        "better": "higher",
        "description": "PSNR全称为“Peak Signal-to-Noise Ratio”，中文意思即为峰值信噪比，是衡量图像质量的指标之一。"
                       "PSNR是基于MSE (均方误差)定义。"
    }
    return psnr_result


def get_ssim(haze, clear=None):
    if clear is None:
        return
    ssim_value = ssim(haze, clear).item()
    ssim_result = {
        "id": 5,
        "label": "SSIM",
        "value": ssim_value,
        "better": "higher",
        "description": "SSIM是一种用于量化两幅图像间的结构相似性的指标。"
                       "与L2损失函数不同，SSIM仿照人类的视觉系统（Human Visual System,HVS）实现了结构相似性的有关理论，"
                       "对图像的局部结构变化的感知敏感。SSIM从亮度、对比度以及结构量化图像的属性，"
                       "用均值估计亮度，方差估计对比度，协方差估计结构相似程度。S"
                       "SIM值的范围为0至1，越大代表图像越相似。如果两张图片完全一样时，SSIM值为1。"
    }
    return ssim_result


def get_lpips(haze, clear=None):
    if clear is None:
        return
    lpips_value = lpips(haze, clear).item()
    lpips_result = {
        "id": 6,
        "label": "LPIPS",
        "value": lpips_value,
        "better": "lower",
        "description": "LPIPS是一种评估图像质量的指标，它是基于人类感知的视觉系统而设计的。"
                       "LPIPS使用了深度学习技术为每个图像提取特征，并基于这些特征计算它们之间的相似度。"
                       "LPIPS的值越低表示两张图像越相似，反之，则差异越大。"
    }
    return lpips_result
