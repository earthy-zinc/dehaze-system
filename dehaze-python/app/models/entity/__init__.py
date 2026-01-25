"""
数据库实体模型（ORM）
"""

from app.models.entity.sys_file import SysFile, SysWpxFile
from app.models.entity.sys_user import SysUser, SysRole, SysUserRole
from app.models.entity.sys_dept import SysDept
from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.models.entity.sys_dict import SysDict, SysDictType
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_dataset import SysDataset, SysDatasetItem, SysItemFile
from app.models.entity.sys_log import SysPredLog, SysEvalLog, SysOperationLog
from app.models.entity.sys_task import SysTask

__all__ = [
    # 文件相关
    'SysFile',
    'SysWpxFile',
    # 用户相关
    'SysUser',
    'SysRole',
    'SysUserRole',
    # 部门
    'SysDept',
    # 菜单
    'SysMenu',
    'SysRoleMenu',
    # 字典
    'SysDict',
    'SysDictType',
    # 算法
    'SysAlgorithm',
    # 数据集
    'SysDataset',
    'SysDatasetItem',
    'SysItemFile',
    # 日志
    'SysPredLog',
    'SysEvalLog',
    'SysOperationLog',
    # 任务
    'SysTask',
]
