"""
模型层统一导出

目录结构：
- entity/   数据库实体模型（ORM）
- form/     表单/请求对象
- vo/       视图对象
- enum/     枚举类型
- base.py   基础模型类
"""

# 数据库实体模型
from app.models.entity import (
    SysFile,
    SysWpxFile,
    SysUser,
    SysRole,
    SysUserRole,
    SysDept,
    SysMenu,
    SysRoleMenu,
    SysDict,
    SysDictType,
    SysAlgorithm,
    SysDataset,
    SysDatasetItem,
    SysItemFile,
    SysPredLog,
    SysEvalLog,
    SysOperationLog,
    SysTask,
)

# 视图对象
from app.models.vo import (
    DatasetStatistics,
    DatasetVO,
    DatasetItemVO,
    ItemFileVO,
    ImageUrlVO,
    TaskVO,
    BatchDeleteResult,
    BatchDeleteResultItem,
    BatchDeleteResultVO,
    BatchActionFailureDetailVO,
    BatchOperationResultVO,
    BatchUploadResultVO,
    BatchUploadSuccessItemVO,
    BatchUploadFailedItemVO,
)

# 枚举类型
from app.models.enum import (
    TaskStatus,
    TaskType,
    ImageType,
    HazeLevel,
)

# 基础模型
from app.models.base import BaseModel

# 表单/请求对象
from app.models.form import (
    DatasetQuery,
    DatasetAddForm,
    DatasetUpdateForm,
    DatasetItemCreateForm,
    DatasetItemUpdateForm,
    DatasetItemUploadForm,
    BatchDatasetItemUploadForm,
    ItemFileUpdateForm,
    ExportTaskCreateForm,
)

__all__ = [
    # 数据库实体
    'SysFile',
    'SysWpxFile',
    'SysUser',
    'SysRole',
    'SysUserRole',
    'SysDept',
    'SysMenu',
    'SysRoleMenu',
    'SysDict',
    'SysDictType',
    'SysAlgorithm',
    'SysDataset',
    'SysDatasetItem',
    'SysItemFile',
    'SysPredLog',
    'SysEvalLog',
    'SysOperationLog',
    'SysTask',
    # 视图对象
    'DatasetStatistics',
    'DatasetVO',
    'DatasetItemVO',
    'ItemFileVO',
    'ImageUrlVO',
    'TaskVO',
    'BatchDeleteResult',
    'BatchDeleteResultItem',
    'BatchDeleteResultVO',
    'BatchActionFailureDetailVO',
    'BatchOperationResultVO',
    'BatchUploadResultVO',
    'BatchUploadSuccessItemVO',
    'BatchUploadFailedItemVO',
    # 枚举
    'TaskStatus',
    'TaskType',
    'ImageType',
    'HazeLevel',
    # 基础模型
    'BaseModel',
    # 表单/请求对象
    'DatasetQuery',
    'DatasetAddForm',
    'DatasetUpdateForm',
    'DatasetItemCreateForm',
    'DatasetItemUpdateForm',
    'DatasetItemUploadForm',
    'BatchDatasetItemUploadForm',
    'ItemFileUpdateForm',
    'ExportTaskCreateForm',
]
