"""
任务相关枚举
"""

from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举（与 Java TaskConstants 保持一致，使用大写值）"""
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class TaskType(str, Enum):
    """任务类型枚举（与文档定义一致）"""
    DATASET_EXPORT = 'dataset_export'
    ITEM_DOWNLOAD = 'item_download'
    BATCH_DOWNLOAD = 'batch_download'
    CUSTOM_EXPORT = 'custom_export'
