"""图像质量指标计算（pyiqa）。

METRICS_CONFIG 是对外能力的一部分（能力发现端点据此声明指标集），计算按请求的
指标名逐项隔离失败：单项异常只进 failed，不影响已完成指标。
"""

import logging
import math
from typing import Any

import PIL.Image

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
    },
}

# 已初始化的 pyiqa 指标模型（按指标名复用，避免每次计算都重建网络）
_initialized_models: dict[str, Any] = {}


def calculate_metrics(
    image: PIL.Image.Image,
    reference: PIL.Image.Image | None,
    metric_names: list[str],
) -> tuple[dict[str, float], list[str]]:
    """按指标名逐项计算，返回 (成功指标 name→value, 失败指标名列表)。"""
    haze = _to_tensor(image)
    clear = _to_tensor(reference) if reference is not None else None

    values: dict[str, float] = {}
    failed: list[str] = []
    for name in metric_names:
        try:
            values[name] = _calculate_metric(name, haze, clear)
        except Exception as e:
            logger.error("指标计算失败: metric=%s, error=%s", name, e)
            failed.append(name)
    return values, failed


def _calculate_metric(metric_name: str, haze, clear=None) -> float:
    config = METRICS_CONFIG[metric_name]
    model = _get_metric_model(metric_name)

    if config["requires_clear"]:
        if clear is None:
            raise ValueError(f"指标 {config['label']} 需要参考图")
        value = model(haze, clear).item()
    else:
        value = model(haze).item()

    # JSON 无法承载非有限值：全同图 MSE=0 → PSNR=inf，钳制到 100 dB 上界
    # （与前端雷达图固定上限口径一致）；其余非有限值视为该指标计算异常
    if not math.isfinite(value):
        if metric_name == "psnr" and math.isinf(value):
            return 100.0
        raise ValueError(f"指标 {config['label']} 计算结果非有限值: {value}")
    return value


def _get_metric_model(metric_name: str):
    if metric_name in _initialized_models:
        return _initialized_models[metric_name]

    # 延迟导入 pyiqa：能力发现与权重探测路径不加载指标模型
    import pyiqa

    logger.info("初始化指标模型: %s", metric_name)
    model = pyiqa.create_metric(metric_name)
    _initialized_models[metric_name] = model
    logger.info("指标模型 %s 初始化成功", metric_name)
    return model


def _to_tensor(image: PIL.Image.Image):
    from torchvision.transforms import ToTensor

    return ToTensor()(image.convert("RGB"))[None, ::]
