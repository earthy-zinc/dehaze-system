import os
from collections import OrderedDict
from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config


def load_model(model_path: str) -> torch.nn.Module:
    """
    加载去雾模型并设置为评估模式。
    根据模型文件名动态选择模型结构并加载权重。
    :param model_path: 模型权重文件路径
    :return: 加载完成的模型
    """
    from .model import (
        dehazeformer_t, dehazeformer_s, dehazeformer_b,
        dehazeformer_d, dehazeformer_w, dehazeformer_m, dehazeformer_l
    )
    # 动态解析模型名称并实例化
    model_name = os.path.basename(model_path).replace('-', '_').replace('.pth', '')
    try:
        net = eval(model_name)().to(Config.DEVICE)
    except NameError:
        raise ValueError(f"Invalid model name extracted from '{model_path}': {model_name}")

    # 加载权重文件并处理可能的多 GPU 权重格式
    state_dict = torch.load(model_path, map_location=Config.DEVICE)['state_dict']
    # 去掉多 GPU 模式下的前缀 'module.'
    new_state_dict = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k, v in state_dict.items())
    net.load_state_dict(new_state_dict)
    net.eval()  # 设置模型为评估模式
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    """
    对输入的雾化图像进行去雾处理。
    :param haze_image: 雾化图像（BytesIO 格式）
    :param model_path: 模型权重文件路径
    :return: 去雾后的图像（BytesIO 格式）
    """
    # 加载模型
    net = load_model(model_path)

    # 预处理输入图像
    haze_tensor = preprocess_image(haze_image)

    # 禁用梯度计算并推理
    with torch.no_grad():
        pred = net(haze_tensor)

    # 后处理输出张量并返回
    return postprocess_image(pred)
