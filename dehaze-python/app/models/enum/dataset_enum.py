"""
数据集相关枚举
"""

from enum import Enum


class ImageType(str, Enum):
    """图片类型枚举"""
    CLEAR = 'clear'
    HAZY = 'hazy'


class HazeLevel(str, Enum):
    """雾霾等级枚举"""
    LIGHT = 'light'
    MEDIUM = 'medium'
    HEAVY = 'heavy'
