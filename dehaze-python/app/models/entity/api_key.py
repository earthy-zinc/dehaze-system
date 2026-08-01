from datetime import datetime
from typing import Optional

from app.models.base import BaseModel
from sqlalchemy import BigInteger, DateTime, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column


class SysApiKey(BaseModel):
    __tablename__ = 'sys_api_key'
    __table_args__ = {'comment': 'API密钥表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment='用户ID')
    name: Mapped[Optional[str]] = mapped_column(String(128), comment='密钥名称')
    key_prefix: Mapped[Optional[str]] = mapped_column(
        String(16), comment='密钥前缀')
    key_hash: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, comment='密钥哈希')
    status: Mapped[int] = mapped_column(
        SmallInteger, default=1, comment='状态(1:正常;0:禁用)')
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment='过期时间')
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment='最后使用时间')
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment='吊销时间')
