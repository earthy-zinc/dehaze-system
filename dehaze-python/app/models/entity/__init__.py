"""
数据库实体模型（ORM）
"""

from app.models.entity.api_key import SysApiKey
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_dataset import SysDataset, SysDatasetItem, SysItemFile
from app.models.entity.sys_dept import SysDept
from app.models.entity.sys_dict import SysDict, SysDictType
from app.models.entity.sys_feedback import SysFeedback
from app.models.entity.sys_feedback_reply import SysFeedbackReply
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysPredLog, SysEvalLog
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.models.entity.sys_member_growth_log import SysMemberGrowthLog
from app.models.entity.sys_member_quota import SysMemberQuota
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_payment_record import SysPaymentRecord
from app.models.entity.sys_refund_record import SysRefundRecord
from app.models.entity.sys_auto_renew import SysAutoRenew
from app.models.entity.sys_package import SysPackage
from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_user_coupon import SysUserCoupon
from app.models.entity.sys_promotion import SysPromotion, SysPromotionPackage
from app.models.entity.sys_rating import SysRating
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
    # 订单管理
    'SysOrder',
    'SysPaymentRecord',
    'SysRefundRecord',
    'SysAutoRenew',
    # 套餐管理
    'SysPackage',
    'SysCoupon',
    'SysUserCoupon',
    'SysPromotion',
    'SysPromotionPackage',
    # 反馈评价
    'SysRating',
    'SysFeedback',
    'SysFeedbackReply',
]
