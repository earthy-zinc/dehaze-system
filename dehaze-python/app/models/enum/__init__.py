"""
枚举类型
"""

from app.models.enum.dataset_enum import ImageType
from app.models.enum.task_enum import TaskStatus, TaskType

__all__ = [
    'TaskStatus',
    'TaskType',
    'ImageType',
]
