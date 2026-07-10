"""
任务相关枚举
"""

from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class TaskType(str, Enum):
    """任务类型枚举（与文档定义一致）"""
    DATASET_EXPORT = 'dataset_export'
    ITEM_DOWNLOAD = 'item_download'
    BATCH_DOWNLOAD = 'batch_download'
    CUSTOM_EXPORT = 'custom_export'
