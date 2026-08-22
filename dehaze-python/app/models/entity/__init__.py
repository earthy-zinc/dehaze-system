"""
数据库实体模型（ORM）

⚠ 本文件是 alembic autogenerate 的实体注册清单：migrations/env.py 经
`import app.models` 触发本聚合，将全部实体注册进 Base.metadata。
新增/删除实体必须同步此清单，否则 autogenerate 无法感知对应表。
"""

from app.models.entity.api_key import SysApiKey
from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_endpoint import SysAiAgentEndpoint
from app.models.entity.sys_ai_agent_eval_dataset import SysAiAgentEvalDataset
from app.models.entity.sys_ai_agent_eval_run import SysAiAgentEvalRun
from app.models.entity.sys_ai_agent_eval_sample import SysAiAgentEvalSample
from app.models.entity.sys_ai_agent_mcp import SysAiAgentMcp
from app.models.entity.sys_ai_agent_skill import SysAiAgentSkill
from app.models.entity.sys_ai_agent_subagent import SysAiAgentSubagent
from app.models.entity.sys_ai_agent_thought import SysAiAgentThought
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.models.entity.sys_ai_artifact import SysAiArtifact
from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.models.entity.sys_ai_memory import SysAiMemory
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.entity.sys_ai_message_feedback import SysAiMessageFeedback
from app.models.entity.sys_ai_model import SysAiModel
from app.models.entity.sys_ai_provider import SysAiProvider
from app.models.entity.sys_ai_provider_key import SysAiProviderKey
from app.models.entity.sys_ai_refund import SysAiRefund
from app.models.entity.sys_ai_schedule import SysAiSchedule
from app.models.entity.sys_ai_schedule_run import SysAiScheduleRun
from app.models.entity.sys_ai_skill import SysAiSkill
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_auto_renew import SysAutoRenew
from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_dataset import SysDataset, SysDatasetItem, SysItemFile
from app.models.entity.sys_dept import SysDept
from app.models.entity.sys_dict import SysDict, SysDictType
from app.models.entity.sys_feedback import SysFeedback
from app.models.entity.sys_feedback_reply import SysFeedbackReply
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysEvalLog, SysPredLog
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.models.entity.sys_member_growth_log import SysMemberGrowthLog
from app.models.entity.sys_member_quota import SysMemberQuota
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_package import SysPackage
from app.models.entity.sys_payment_record import SysPaymentRecord
from app.models.entity.sys_promotion import SysPromotion, SysPromotionPackage
from app.models.entity.sys_rating import SysRating
from app.models.entity.sys_refund_record import SysRefundRecord
from app.models.entity.sys_task import SysTask
from app.models.entity.sys_user import SysRole, SysUser, SysUserRole
from app.models.entity.sys_user_coupon import SysUserCoupon
from app.models.entity.sys_wpx_file import SysWpxFile

__all__ = [
    # 文件相关
    "SysFile",
    # 用户相关
    "SysUser",
    "SysRole",
    "SysUserRole",
    "SysApiKey",
    # AI对话
    "SysAiModel",
    "SysAiConversation",
    "SysAiMessage",
    "SysAiBilling",
    "SysAiCreditLog",
    "SysAiAgentThought",
    "SysAiArtifact",
    "SysAiMemory",
    "SysAiMessageFeedback",
    "SysAiProvider",
    "SysAiProviderKey",
    "SysAiRefund",
    "SysAiSchedule",
    "SysAiScheduleRun",
    "SysAiSkill",
    # 智能体管理
    "SysAiAgent",
    "SysAiAgentSkill",
    "SysAiAgentMcp",
    "SysAiAgentSubagent",
    "SysAiAgentVersion",
    "SysAiAgentEvalDataset",
    "SysAiAgentEvalSample",
    "SysAiAgentEvalRun",
    "SysAiAgentEndpoint",
    # 部门
    "SysDept",
    # 菜单
    "SysMenu",
    "SysRoleMenu",
    # 字典
    "SysDict",
    "SysDictType",
    # 算法
    "SysAlgorithm",
    # 数据集
    "SysDataset",
    "SysDatasetItem",
    "SysItemFile",
    # 日志
    "SysPredLog",
    "SysEvalLog",
    # 任务
    "SysTask",
    # 会员
    "SysMember",
    "SysMemberBenefit",
    "SysMemberGrowthLog",
    "SysMemberQuota",
    "SysMemberSignIn",
    # WPX 文件映射
    "SysWpxFile",
    # 订单管理
    "SysOrder",
    "SysPaymentRecord",
    "SysRefundRecord",
    "SysAutoRenew",
    # 套餐管理
    "SysPackage",
    "SysCoupon",
    "SysUserCoupon",
    "SysPromotion",
    "SysPromotionPackage",
    # 反馈评价
    "SysRating",
    "SysFeedback",
    "SysFeedbackReply",
]
