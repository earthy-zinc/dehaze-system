import logging
from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)

# 指标配置
METRICS_CONFIG = {
    "psnr": {
        "id": 1,
        "label": "PSNR",
        "better": "higher",
        "requires_clear": True,
        "description": "PSNR全称为“Peak Signal-to-Noise Ratio”，是衡量图像质量的指标之一，基于MSE定义。",
    },
    "ssim": {
        "id": 2,
        "label": "SSIM",
        "better": "higher",
        "requires_clear": True,
        "description": "SSIM是一种用于量化两幅图像间结构相似性的指标，范围为0至1，越大越相似。",
    },
    "lpips": {
        "id": 3,
        "label": "LPIPS",
        "better": "lower",
        "requires_clear": True,
        "description": "LPIPS是一种基于深度学习技术的图像质量指标，值越低表示图像越相似。",
    },
    "niqe": {
        "id": 4,
        "label": "NIQE",
        "better": "lower",
        "requires_clear": False,
        "description": "NIQE是一种无参考图像空间质量评估指标，用于评估图像的失真程度。",
    },
    "nima": {
        "id": 5,
        "label": "NIMA",
        "better": "higher",
        "requires_clear": False,
        "description": "NIMA是一种无参考技术，它可以预测图像的质量，而不依赖于通常不可用的原始参考图像。",
    },
    "brisque": {
        "id": 6,
        "label": "BRISQUE",
        "better": "lower",
        "requires_clear": False,
        "description": "BRISQUE是一种无参考图像质量评估指标，通过分析图像的统计特征来估计质量。",
    }
}

# 存储已初始化的模型
_initialized_models = {}

def _get_metric_model(metric_name: str):
    """
    延迟初始化并获取指标模型
    :param metric_name: 指标名称
    :return: 指标模型
    """
    global _initialized_models

    # 如果模型已经初始化，直接返回
    if metric_name in _initialized_models:
        return _initialized_models[metric_name]

    # 延迟导入 pyiqa，避免在应用启动时加载
    try:
        import pyiqa
    except ImportError as e:
        logger.error(f"无法导入 pyiqa 模块: {e}")
        raise

    # 初始化模型
    logger.info(f"初始化指标模型: {metric_name}")
    try:
        model = pyiqa.create_metric(metric_name)
        _initialized_models[metric_name] = model
        logger.info(f"指标模型 {metric_name} 初始化成功")
        return model
    except Exception as e:
        logger.error(f"初始化指标模型 {metric_name} 失败: {e}")
        raise

def get_available_metrics():
    """
    获取所有可用的指标列表
    :return: 指标配置列表
    """
    return [
        {
            "name": name,
            "id": config["id"],
            "label": config["label"],
            "better": config["better"],
            "requires_clear": config["requires_clear"],
            "description": config["description"],
        }
        for name, config in METRICS_CONFIG.items()
    ]

def is_metric_initialized(metric_name: str) -> bool:
    """
    检查指标模型是否已初始化
    :param metric_name: 指标名称
    :return: 是否已初始化
    """
    return metric_name in _initialized_models

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
        calculate_metric(name, haze, clear) for name, metric in METRICS_CONFIG.items()
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
    # 确保模型已初始化
    model = _get_metric_model(metric_name)

    # 获取指标配置
    metric_config = METRICS_CONFIG[metric_name]

    # 计算指标
    try:
        if metric_config["requires_clear"] and clear is not None:
            value = model(haze, clear).item()
        else:
            value = model(haze).item()
    except Exception as e:
        logger.error(f"计算指标 {metric_name} 时出错: {e}")
        raise

    # 组织返回结果
    return {
        "id": metric_config["id"],
        "label": metric_config["label"],
        "value": value,
        "better": metric_config["better"],
        "description": metric_config["description"],
    }


def _to_tensor(image_bytes: BytesIO) -> torch.Tensor:
    return ToTensor()(Image.open(image_bytes).convert("RGB"))[None, ::]
