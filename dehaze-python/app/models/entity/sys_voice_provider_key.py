from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysVoiceProviderKey(BaseModel):
    __tablename__ = "sys_voice_provider_key"
    __table_args__ = (
        Index("idx_provider", "provider_id", "status"),
        {"comment": "语音引擎API密钥表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    provider_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="关联引擎ID(关联sys_voice_provider.id)"
    )
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="Key名称(备注,如阿里云ASR主账号;备用账号)"
    )
    key_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, comment="密钥哈希(SHA256 hex,固定64字符,用于查重)"
    )
    key_prefix: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="密钥前缀(展示用)"
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
        Integer, nullable=True, comment="日调用上限(引擎侧限额,可选)"
    )
    rpm_limit: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="分钟调用频率上限(可选,NULL表示不限制)"
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="过期时间")
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最后使用时间"
    )
    last_used_by: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="最后使用的用户ID"
    )
