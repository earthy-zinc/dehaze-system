"""
算法模块配置

仅包含算法运行所需的配置，Web API 配置见 app/config.py
"""

import os.path as path

import torch


class Config:
    """算法基础配置"""

    # 设备配置
    DEVICE_ID: list[int] = [0]
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 路径配置（指向项目根目录，而非 algorithm/ 目录）
    PROJECT_PATH: str = path.dirname(path.dirname(path.abspath(__file__)))
    MODEL_PATH: str = path.join(PROJECT_PATH, "trained_model")


# 算法模块使用此配置
algorithm_config = Config()
