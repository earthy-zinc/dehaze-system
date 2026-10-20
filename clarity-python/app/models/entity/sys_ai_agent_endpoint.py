from typing import Any

from sqlalchemy import BigInteger, Index, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiAgentEndpoint(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_agent_endpoint"
    __table_args__ = (
        Index("uk_base_url", "base_url", unique=True),
        {"comment": "AI外部A2A端点注册表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="端点名称")
    agent_card_url: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Agent Card地址(发现端点,如 https://host/.well-known/agent.json)",
    )
    base_url: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="A2A端点地址(如 https://host/a2a)"
    )
    auth_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="http",
        comment=(
            "认证方式(apiKey;http;oauth2;openIdConnect;mutualTLS,"
            "遵循Agent Card securitySchemes声明)"
        ),
    )
    credential: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="凭证密文(AES加密后base64编码,运行时解密按声明方案注入请求头)",
    )
    agent_card: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="缓存的Agent Card JSON(注册时拉取,作为发现依据)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
