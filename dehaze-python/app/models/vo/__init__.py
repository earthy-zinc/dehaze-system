"""
视图对象（VO）
"""

from app.models.vo.dataset_vo import (
    DatasetStatistics,
    DatasetVO,
    DatasetItemVO,
    ItemFileVO,
    ImageUrlVO,
)
from app.models.vo.task_vo import TaskVO
from app.models.vo.batch_vo import (
    BatchDeleteResult,
    BatchDeleteResultItem,
    BatchDeleteResultVO,
    BatchActionFailureDetailVO,
    BatchOperationResultVO,
    BatchUploadResultVO,
    BatchUploadSuccessItemVO,
    BatchUploadFailedItemVO,
)

__all__ = [
    # 数据集相关
    'DatasetStatistics',
    'DatasetVO',
    'DatasetItemVO',
    'ItemFileVO',
    'ImageUrlVO',
    # 任务
    'TaskVO',
    # 批量操作
    'BatchDeleteResult',
    'BatchDeleteResultItem',
    'BatchDeleteResultVO',
    'BatchActionFailureDetailVO',
    'BatchOperationResultVO',
    'BatchUploadResultVO',
    'BatchUploadSuccessItemVO',
    'BatchUploadFailedItemVO',
]
