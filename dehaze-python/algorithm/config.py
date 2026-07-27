"""
算法模块配置

仅包含算法运行所需的配置，Web API 配置见 app/config.py
模型权重文件路径解析见 algorithm/model_loader.py
"""

import torch


class Config:
    """算法基础配置"""

    # 设备配置
    DEVICE_ID: list[int] = [0]
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# 算法模块使用此配置
algorithm_config = Config()
