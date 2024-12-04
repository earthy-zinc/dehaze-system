from io import BytesIO

import pyiqa
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# 指标模型初始化（整合描述信息）
METRICS = {
    "niqe": {
        "model": pyiqa.create_metric("niqe"),
        "id": 1,
        "label": "NIQE",
        "better": "lower",
        "requires_clear": False,
        "description": "NIQE是一种无参考图像空间质量评估指标，用于评估图像的失真程度。",
    },
    "nima": {
        "model": pyiqa.create_metric("nima"),
        "id": 2,
        "label": "NIMA",
        "better": "higher",
        "requires_clear": False,
        "description": "NIMA是一种无参考技术，它可以预测图像的质量，而不依赖于通常不可用的原始参考图像。",
    },
    "brisque": {
        "model": pyiqa.create_metric("brisque"),
        "id": 3,
        "label": "BRISQUE",
        "better": "lower",
        "requires_clear": False,
        "description": "BRISQUE是一种无参考图像质量评估指标，通过分析图像的统计特征来估计质量。",
    },
    "psnr": {
        "model": pyiqa.create_metric("psnr"),
        "id": 4,
        "label": "PSNR",
        "better": "higher",
        "requires_clear": True,
        "description": "PSNR全称为“Peak Signal-to-Noise Ratio”，是衡量图像质量的指标之一，基于MSE定义。",
    },
    "ssim": {
        "model": pyiqa.create_metric("ssim"),
        "id": 5,
        "label": "SSIM",
        "better": "higher",
        "requires_clear": True,
        "description": "SSIM是一种用于量化两幅图像间结构相似性的指标，范围为0至1，越大越相似。",
    },
    "lpips": {
        "model": pyiqa.create_metric("lpips"),
        "id": 6,
        "label": "LPIPS",
        "better": "lower",
        "requires_clear": True,
        "description": "LPIPS是一种基于深度学习技术的图像质量指标，值越低表示图像越相似。",
    },
}


def calculate(haze_image: BytesIO, clear_image: BytesIO = None):
    """
    计算图像质量指标
    遍历所有指标，根据指标是否需要清晰图像，以及清晰图像是否已提供，决定是否调用 calculate_metric 进行计算。
    :param haze_image: 雾化图像路径
    :param clear_image: 清晰图像路径（可选）
    :return: 计算结果列表
    """
    haze = _to_tensor(haze_image)
    clear = _to_tensor(clear_image) if clear_image else None

    # 动态计算所有指标
    result = [
        calculate_metric(name, haze, clear) for name, metric in METRICS.items()
        if metric["requires_clear"] and clear is not None or not metric["requires_clear"]
    ]
    return result


def calculate_metric(metric_name: str, haze, clear=None):
    """
    动态计算指定指标
    :param metric_name: 指标名称
    :param haze: 雾化图像
    :param clear: 清晰图像（可选）
    :return: 指标计算结果字典
    """
    metric = METRICS[metric_name]
    model = metric["model"]

    # 计算指标
    if metric["requires_clear"] and clear is not None:
        value = model(haze, clear).item()
    else:
        value = model(haze).item()

    # 组织返回结果
    return {
        "id": metric["id"],
        "label": metric["label"],
        "value": value,
        "better": metric["better"],
        "description": metric["description"],
    }


def _to_tensor(image_bytes: BytesIO) -> torch.Tensor:
    return ToTensor()(Image.open(image_bytes).convert("RGB"))[None, ::]
