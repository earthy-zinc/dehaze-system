"""
数据库实体模型（ORM）
"""

from app.models.entity.api_key import SysApiKey
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_dataset import SysDataset, SysDatasetItem, SysItemFile
from app.models.entity.sys_dept import SysDept
from app.models.entity.sys_dict import SysDict, SysDictType
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysPredLog, SysEvalLog, SysOperationLog, SysLoginLog
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.models.entity.sys_member_growth_log import SysMemberGrowthLog
from app.models.entity.sys_member_quota import SysMemberQuota
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.models.entity.sys_task import SysTask
from app.models.entity.sys_user import SysUser, SysRole, SysUserRole
from app.models.entity.sys_wpx_file import SysWpxFile

__all__ = [
    # 文件相关
    'SysFile',
    # 用户相关
    'SysUser',
    'SysRole',
    'SysUserRole',
    'SysApiKey',
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
    'SysLoginLog',
    # 任务
    'SysTask',
    # 会员
    'SysMember',
    'SysMemberBenefit',
    'SysMemberGrowthLog',
    'SysMemberQuota',
    'SysMemberSignIn',
    # WPX 文件映射
    'SysWpxFile',
]
