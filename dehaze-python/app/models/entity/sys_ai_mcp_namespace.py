from typing import Any

from sqlalchemy import BigInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiMcpNamespace(BaseModel):
    __tablename__ = "sys_ai_mcp_namespace"
    __table_args__ = {"comment": "外部MCP Server命名空间配置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    server_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联Server ID(关联sys_ai_mcp_server.id)"
    )
    namespace: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="命名空间标识(工具分组,对齐McpNamespaceVO.name)"
    )
    tool_names: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="分组内工具名数组(JSON)"
    )
