from io import BytesIO

import torch
import torchvision
from PIL import Image
from torch import Tensor

from config import Config


def preprocess_image(image_bytes: BytesIO) -> torch.Tensor:
    """
    预处理图像：从BytesIO加载并转换为张量
    :param image_bytes: 输入的图像数据
    :return: 预处理后的图像张量
    """
    image = Image.open(image_bytes).convert('RGB')  # 确保是RGB格式
    return torchvision.transforms.ToTensor()(image).unsqueeze(0).to(Config.DEVICE)  # 转换为Tensor并添加批次维度


def postprocess_image(tensor: torch.Tensor) -> BytesIO:
    """
    后处理图像：将张量裁剪到合法范围并转换为BytesIO
    :param tensor: 预测输出的图像张量
    :return: 转换后的BytesIO对象
    """
    tensor = tensor.clamp(0, 1).cpu().squeeze(0)  # 限制范围并去掉批次维度
    return tensor_to_bytesio(tensor)

def tensor_to_bytesio(tensor: Tensor, image_format="PNG") -> BytesIO:
    """
    将 PyTorch Tensor 转换为 BytesIO 对象。
    :param tensor: PyTorch Tensor (C, H, W)，像素值范围 [0, 1] 或 [0, 255]
    :param image_format: 输出的图片格式，默认 "PNG"
    :return: BytesIO 对象
    """
    # Step 1: 确保张量在 [0, 255] 范围内，并转换为整型
    if tensor.max() <= 1:
        tensor = tensor * 255  # 将 [0, 1] 范围的像素值转换为 [0, 255]
    tensor = tensor.byte()

    # Step 2: 调整维度到 (H, W, C)
    tensor = tensor.permute(1, 2, 0).numpy()

    # Step 3: 转换为 PIL 图像
    image = Image.fromarray(tensor)

    # Step 4: 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)  # 重置流位置，确保后续读取从头开始

    return buffer
