from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiProviderKey(BaseModel):
    __tablename__ = "sys_ai_provider_key"
    __table_args__ = (
        Index("idx_provider", "provider_id", "status"),
        {"comment": "AI供应商API密钥表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    provider_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联供应商ID(关联sys_ai_provider.id)"
    )
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="Key名称(备注,如OpenAI主账号;备用账号)"
    )
    key_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, comment="密钥哈希(SHA256 hex,固定64字符,用于查重)"
    )
    key_prefix: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="密钥前缀(展示用,如sk-proj-aB...)"
    )
    key_cipher: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="密钥密文(AES-256-CBC加密后base64编码,运行时解密)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="优先级(数字越小越优先)"
    )
    weight: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="权重(同优先级按权重加权随机选取)"
    )
    daily_quota: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="日调用上限(供应商侧限额,可选)"
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="过期时间")
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最后使用时间"
    )
    last_used_by: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="最后使用的用户ID"
    )
