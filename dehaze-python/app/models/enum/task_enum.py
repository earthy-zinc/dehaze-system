"""
任务相关枚举
"""


class TaskStatus:
    """任务状态枚举"""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class TaskType:
    """任务类型枚举"""
    DATASET_EXPORT = 'dataset_export'
    BATCH_PROCESSING = 'batch_processing'
