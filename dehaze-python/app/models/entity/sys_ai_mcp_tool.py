from typing import Any

from sqlalchemy import BigInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiMcpTool(BaseModel):
    __tablename__ = "sys_ai_mcp_tool"
    __table_args__ = {"comment": "外部MCP Server工具清单表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    server_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联Server ID(关联sys_ai_mcp_server.id)"
    )
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="工具名(Server内唯一)"
    )
    description: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="工具描述"
    )
    input_schema: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="参数schema概要(JSON)"
    )
