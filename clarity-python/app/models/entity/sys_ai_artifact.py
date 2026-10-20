from typing import Any

from sqlalchemy import BigInteger, SmallInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysAiArtifact(BaseModel):
    __tablename__ = "sys_ai_artifact"
    __table_args__ = {"comment": "AI中间产物表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="会话ID")
    message_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="关联消息ID")
    type: Mapped[str] = mapped_column(String(32), nullable=False, comment="产物类型")
    ref_type: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="引用业务表")
    ref_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="引用业务表ID")
    summary: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="业务摘要元数据(绝不存URL)"
    )
    is_invalid: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="引用对象是否已失效"
    )
