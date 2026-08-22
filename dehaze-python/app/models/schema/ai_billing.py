"""
AI 计费管理模块 Schema 模型
"""

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from app.models.schema.common import OrmResult

# ── 计费记录 ──────────────────────────────────────────


class BillingRecordResult(OrmResult):
    id: int = Field(description="主键")
    user_id: int = Field(description="用户ID")
    conversation_id: int | None = Field(default=None, description="会话ID")
    message_id: int | None = Field(default=None, description="消息ID")
    model: str = Field(description="实际使用模型标识")
    actual_model: str | None = Field(default=None, description="用户原选模型标识")
    bill_type: str = Field(description="计费类型(chat;tool_llm;kb_inject;asr;tts)")
    input_tokens: int = Field(description="输入Token数")
    cached_input_tokens: int = Field(description="缓存命中的输入Token数")
    output_tokens: int = Field(description="输出Token数")
    credits: int = Field(description="消耗积分数")
    credits_saved: int = Field(description="缓存命中节省积分数")
    tool_credits: int | None = Field(default=None, description="工具调用额外积分")
    quota_consumed: int = Field(description="实际扣减配额")
    pre_deduct: int = Field(description="预扣积分数")
    create_time: datetime | None = Field(default=None, description="创建时间")


class BillingRecordQuery(OrmResult):
    conversation_id: int | None = Field(default=None, description="会话ID")
    bill_type: str | None = Field(default=None, description="计费类型")
    model_id: str | None = Field(default=None, description="模型标识")
    date_start: datetime | None = Field(default=None, description="开始时间")
    date_end: datetime | None = Field(default=None, description="结束时间")
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=20, ge=1, le=100, description="每页数量")


# ── 余额流水 ──────────────────────────────────────────


class CreditLogResult(OrmResult):
    id: int = Field(description="主键")
    user_id: int = Field(description="用户ID")
    source: str = Field(description="变动来源")
    amount: Decimal = Field(description="变动金额(正数增加;负数扣减)")
    balance_after: Decimal = Field(description="变动后账户余额")
    related_id: int | None = Field(default=None, description="关联业务记录ID")
    reason: str | None = Field(default=None, description="变动原因")
    operator_id: int | None = Field(default=None, description="操作人ID")
    create_time: datetime | None = Field(default=None, description="创建时间")


class CreditLogQuery(OrmResult):
    source: str | None = Field(default=None, description="变动来源")
    date_start: datetime | None = Field(default=None, description="开始时间")
    date_end: datetime | None = Field(default=None, description="结束时间")
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=20, ge=1, le=100, description="每页数量")


# ── 余额账户 ──────────────────────────────────────────


class BalanceResult(OrmResult):
    user_id: int = Field(description="用户ID")
    credits_balance: Decimal = Field(description="积分余额")
    arrears_status: bool = Field(description="是否欠费")
    daily_used: int = Field(description="今日已用")
    daily_limit: int = Field(description="日限额")
    monthly_used: int = Field(description="本月已用")
    monthly_limit: int = Field(description="月限额")


# ── 统计 ──────────────────────────────────────────────


class BillingStatQuery(OrmResult):
    model_id: str | None = Field(default=None, description="模型标识")
    bill_type: str | None = Field(default=None, description="计费类型")
    date_start: datetime | None = Field(default=None, description="开始时间")
    date_end: datetime | None = Field(default=None, description="结束时间")
    group_by: str = Field(default="model", description="统计维度(user/model/bill_type/day)")


class BillingStatResult(OrmResult):
    dimension: str = Field(description="统计维度值")
    total_credits: int = Field(description="积分总消耗")
    total_input_tokens: int = Field(description="输入Token总数")
    total_output_tokens: int = Field(description="输出Token总数")
    cache_hit_rate: float = Field(description="缓存命中率")
    credits_saved: int = Field(description="缓存节省积分")
    degradation_count: int = Field(description="降级次数")


# ── 账单 ──────────────────────────────────────────────


class BillResult(OrmResult):
    user_id: int = Field(description="用户ID")
    month: str = Field(description="账期月份(YYYY-MM)")
    total_consume: int = Field(description="总消耗积分")
    total_recharge: int = Field(description="总充值积分")
    total_refund: int = Field(description="总退款积分")
    balance_start: Decimal = Field(description="期初余额")
    balance_end: Decimal = Field(description="期末余额")
    item_summary: dict = Field(description="按bill_type维度的明细汇总")


# ── 退款 ──────────────────────────────────────────────


class RefundCreateRequest(OrmResult):
    billing_id: int = Field(description="原计费记录ID")
    amount: int = Field(description="退款积分数")
    reason: str = Field(..., min_length=1, description="退款原因")


class RefundAuditRequest(OrmResult):
    approved: bool = Field(description="是否通过(true通过;false驳回)")
    audit_remark: str | None = Field(default=None, description="审核意见")


class RefundResult(OrmResult):
    id: int = Field(description="主键")
    user_id: int = Field(description="用户ID")
    billing_id: int = Field(description="原计费记录ID")
    amount: int = Field(description="退款积分数")
    reason: str = Field(description="退款原因")
    status: int = Field(description="退款状态(1:待审核;2:已通过;3:已驳回)")
    auditor_id: int | None = Field(default=None, description="审核人ID")
    audit_remark: str | None = Field(default=None, description="审核意见")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


# ── 调整 ──────────────────────────────────────────────


class AdjustRequest(OrmResult):
    user_id: int = Field(description="用户ID")
    amount: int = Field(description="调整积分(正数增加;负数扣减)")
    reason: str = Field(..., min_length=1, description="调整原因")
