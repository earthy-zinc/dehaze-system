from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AppendOnlyModel


class SysAiMcpCall(AppendOnlyModel):
    """外部MCP工具调用审计表(只追加，不使用逻辑删除)"""

    __tablename__ = "sys_ai_mcp_call"
    __table_args__ = {"comment": "外部MCP工具调用审计表(只追加)"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="调用用户ID(关联sys_user.id,NULL表示系统调用)"
    )
    server_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联Server ID(关联sys_ai_mcp_server.id)"
    )
    server_name: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="Server名称(冗余快照)"
    )
    tool_name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="被调用的工具名"
    )
    request: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="调用载荷(JSON)"
    )
    response: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="响应结果(JSON文本)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="调用状态(0:失败;1:成功)"
    )
    result: Mapped[str] = mapped_column(
        String(16), nullable=False, default="success", comment="调用结果(success;failure)"
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="调用耗时(毫秒)"
    )
