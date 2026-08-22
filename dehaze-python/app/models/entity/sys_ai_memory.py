from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin

# 软删记忆物理清理窗口（天）：软删后保留 30 天供用户恢复，超期由定时任务物理删除
MEMORY_RECOVERY_WINDOW_DAYS = 30


class SysAiMemory(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_memory"
    __table_args__ = {"comment": "AI长期记忆表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    memory_type: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="记忆类型(episodic/semantic/procedural)"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="记忆内容")
    metadata_: Mapped[Any | None] = mapped_column(
        "metadata", JSON, nullable=True, comment="结构化属性"
    )
    importance: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=50, comment="重要性评分(0-100)"
    )
    access_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="检索命中次数"
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最后访问时间"
    )
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="conversation", comment="来源"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
    archived: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否归档"
    )
    delete_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="软删时间(30天恢复窗口判定，超期由定时任务物理清理)"
    )
