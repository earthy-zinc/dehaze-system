from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiTrace(AppendOnlyModel):
    """AI对话过程链汇总记录表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_trace"
    __table_args__ = {"comment": "AI对话过程链汇总记录表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    trace_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="过程链ID(复用日志链路trace_id，全链路串联)"
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="所属会话ID(关联sys_ai_conversation.id)"
    )
    message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="关联助手回复消息ID(关联sys_ai_message.id)"
    )
    agent_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="执行智能体编码"
    )
    trace_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="conversation",
        comment=(
            "过程链类型(conversation主对话; summary会话摘要压缩; memory_extraction记忆提取; "
            "suggestion类似问题推荐; step_summary步骤摘要)"
        ),
    )
    model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="实际使用模型标识"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
        comment="执行状态(1:成功;2:失败;3:中断;4:超时)",
    )
    error_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="失败类型(工具失败/模型异常/超时/配额拒绝)"
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="整条回复总耗时(毫秒)"
    )
    first_token_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="首Token延迟(毫秒，首个输出token到达耗时)"
    )
    llm_call_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本次回复的LLM调用次数"
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="总Token消耗(与计费口径一致)"
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输入Token消耗"
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输出Token消耗"
    )
    cached_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="缓存命中Token数(与计费口径一致)"
    )
    step_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="推理步数(防循环观测，超阈值显著标注)"
    )
    context_snapshot: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="上下文构成快照JSON(系统提示/历史/记忆/检索/工具清单及各占比、压缩/截断事件)",
    )
    error_detail: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="异常详情(消息+堆栈截断,失败/中断时填充)",
    )
