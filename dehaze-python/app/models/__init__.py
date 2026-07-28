"""
模型层统一导出

目录结构：
- entity/   数据库实体模型（ORM）
- schema/   Pydantic 模型（请求/响应）
- enum/     枚举类型
- base.py   基础模型类
"""

# 基础模型
from app.models.base import BaseModel
# 数据库实体模型
from app.models.entity import (
    SysFile,
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
    SysTask,
    SysWpxFile,
)
# 枚举类型
from app.models.enum import (
    TaskStatus,
    TaskType,
    ImageType,
)

__all__ = [
    # 数据库实体
    'SysFile',
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
    'SysTask',
    'SysWpxFile',
    # 枚举
    'TaskStatus',
    'TaskType',
    'ImageType',
    # 基础模型
    'BaseModel',
]
