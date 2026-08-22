from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiAgentThought(AppendOnlyModel):
    """AI推理过程表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_agent_thought"
    __table_args__ = {"comment": "AI推理过程表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    message_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联消息ID(触发推理的assistant消息)"
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="会话ID(冗余，便于按会话查询推理链路)"
    )
    position: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="步骤序号(从1开始，同一消息内排序)"
    )
    thought: Mapped[str | None] = mapped_column(Text, nullable=True, comment="LLM思考内容")
    tool: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="工具名称(MCP tool标识)"
    )
    tool_input: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="工具输入参数(JSON)"
    )
    observation: Mapped[str | None] = mapped_column(Text, nullable=True, comment="工具返回摘要")
    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="步骤一句话摘要(LLM生成,两级展示一级:步骤摘要)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="步骤状态(1:成功;2:失败;3:跳过)"
    )
    latency_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="工具调用耗时(毫秒)"
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="失败原因(status=2时填充)"
    )
