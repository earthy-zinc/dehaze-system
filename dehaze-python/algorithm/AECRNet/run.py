from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .models.AECRNet import Dehaze


def get_model(model_path: str) -> torch.nn.Module:
    """
    加载预训练的去雾模型
    :param model_path: 模型路径
    :return: 加载并设置为评估模式的模型
    """
    net = Dehaze(3, 3).to(Config.DEVICE)  # 初始化网络
    net = torch.nn.DataParallel(net, device_ids=Config.DEVICE_ID)  # 支持多GPU
    checkpoint = torch.load(model_path)  # 加载模型权重
    net.load_state_dict(checkpoint['model'])  # 恢复模型状态
    net.eval()  # 设置为评估模式
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    """
    对雾化图像进行去雾处理
    :param haze_image: 雾化图像数据
    :param model_path: 模型路径
    :return: 去雾后的图像数据
    """
    net = get_model(model_path)  # 加载模型
    haze_tensor = preprocess_image(haze_image)  # 预处理图像
    with torch.no_grad():  # 禁用梯度计算
        pred, *_ = net(haze_tensor)  # 模型推理
    return postprocess_image(pred)  # 后处理并返回结果
