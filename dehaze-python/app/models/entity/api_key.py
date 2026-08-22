from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysApiKey(BaseModel):
    __tablename__ = "sys_api_key"
    __table_args__ = {"comment": "API密钥表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    name: Mapped[str | None] = mapped_column(String(128), comment="密钥名称")
    key_prefix: Mapped[str | None] = mapped_column(String(16), comment="密钥前缀")
    key_hash: Mapped[str | None] = mapped_column(String(64), unique=True, comment="密钥哈希")
    status: Mapped[int] = mapped_column(SmallInteger, default=1, comment="状态(1:正常;0:禁用)")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="过期时间")
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最后使用时间"
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="吊销时间")
    daily_quota: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="日调用配额(NULL或0表示不限制)"
    )
    monthly_quota: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="月调用配额(NULL或0表示不限制)"
    )
    rpm_limit: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="每分钟请求数上限RPM(NULL或0表示不限制)"
    )
    model_whitelist: Mapped[list | None] = mapped_column(
        JSON, nullable=True, comment="模型白名单(NULL或空数组表示继承用户可见模型)"
    )
