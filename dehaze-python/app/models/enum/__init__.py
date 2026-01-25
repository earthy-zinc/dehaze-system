"""
枚举类型
"""

from app.models.enum.task_enum import TaskStatus, TaskType
from app.models.enum.dataset_enum import ImageType, HazeLevel

__all__ = [
    'TaskStatus',
    'TaskType',
    'ImageType',
    'HazeLevel',
]
