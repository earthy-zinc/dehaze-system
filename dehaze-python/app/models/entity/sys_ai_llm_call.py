from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiLlmCall(AppendOnlyModel):
    """AI对话每次LLM调用明细表(span级，只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_llm_call"
    __table_args__ = {"comment": "AI对话每次LLM调用明细表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    trace_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="关联过程链ID(关联sys_ai_trace.trace_id)"
    )
    seq: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="调用序号(1起递增，贯穿推理步骤链路)"
    )
    step_position: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="关联推理步骤序号(关联sys_ai_agent_thought.position，可为空)",
    )
    model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="本次调用模型(多步推理中可能切换模型)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="调用状态(1:成功;2:失败;3:超时)"
    )
    error_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="失败类型"
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="本次调用总耗时(毫秒)"
    )
    first_token_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="本次调用首Token延迟(毫秒，首个输出token到达耗时)"
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输入Token消耗"
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输出Token消耗"
    )
    cached_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="缓存命中Token数(未提供缓存统计的模型置0)"
    )
    tool_call: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="工具调用信息JSON(has_tool_call/tool_name/args_summary)"
    )
    input_snapshot: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="本次调用输入构成JSON(system/消息按角色计数/tools/用户信息)"
    )
    output_snapshot: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="本次调用输出摘要JSON(文本截断 + tool_calls参数)"
    )
    attempts: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="物理调用尝试明细JSON(逐Key/逐路由: provider_id/key_id/model/status/error_code/latency_ms)",
    )
