from decimal import Decimal

from sqlalchemy import BigInteger, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiBilling(AppendOnlyModel):
    """AI计费记录表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_billing"
    __table_args__ = {"comment": "AI计费记录表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="用户ID(关联sys_user.id)"
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="会话ID(关联sys_ai_conversation.id)"
    )
    message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="消息ID(关联sys_ai_message.id)"
    )
    request_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="请求唯一ID(支撑对账与异常追溯)"
    )
    provider_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="实际供应商ID(关联sys_ai_provider.id)"
    )
    model: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="实际使用模型标识(降级场景为降级模型)"
    )
    actual_model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="用户原选模型标识(NULL表示未降级)"
    )
    error_code: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="调用失败错误码(如429/5xx,成功为NULL)"
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="调用耗时(毫秒)")
    bill_type: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="计费类型(chat;tool_llm;kb_inject;asr;tts)"
    )
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输入Token数(含缓存命中部分)"
    )
    cached_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="其中缓存命中的输入Token数"
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输出Token数"
    )
    credits: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="消耗积分数(按实际模型计费比例换算)"
    )
    credits_saved: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="缓存命中节省积分数"
    )
    tool_credits: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="工具调用额外LLM Token积分(tool_llm类型记录)"
    )
    quota_consumed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="实际扣减配额(credits-预扣退还差额)"
    )
    pre_deduct: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="预扣积分数"
    )
    cost: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 4), nullable=True, comment="本次调用估算成本(成本线异步回填,元;未配置成本价为0)"
    )
