from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiMcpServer(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_mcp_server"
    __table_args__ = {"comment": "外部MCP Server注册表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, comment="Server名称(唯一)"
    )
    description: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="描述"
    )
    protocol_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="streamable-http",
        comment="传输协议(stdio;streamable-http;sse)",
    )
    endpoint: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="端点URL(stdio可为空)"
    )
    auth_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="鉴权方式(none;api_key;oauth2等)"
    )
    credentials: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="凭据密文(JSON,AES加密后base64,仅录入/更新不回显明文)",
    )
    health: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="健康状态(online;offline)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
    tool_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="工具数量(冗余,注册/拉取时更新)"
    )
