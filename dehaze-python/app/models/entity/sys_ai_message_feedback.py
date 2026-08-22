from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, SmallInteger, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiMessageFeedback(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_message_feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uk_message_user"),
        {"comment": "AI消息反馈表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    message_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="消息ID(仅assistant消息可反馈)"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="消息所属会话ID(冗余,支撑按会话维度统计)"
    )
    model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="生成该消息的模型标识(按模型统计满意度与归因)"
    )
    source: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="internal",
        comment="反馈来源(internal:内部API;compat:第三方兼容API)",
    )
    rating: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, comment="评分(1:点赞;-1:点踩)"
    )
    tags: Mapped[Any | None] = mapped_column(
        JSON,
        nullable=True,
        comment="预设标签(JSON数组,点赞:accurate/detailed/concise/creative;点踩:incorrect/irrelevant/incomplete/too_long/bad_citation/harmful)",
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True, comment="反馈内容(可选)")
    processed: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=0,
        comment="闭环处理状态(0:待处理;1:已处理,由XXL-Job定时扫描)",
    )
    process_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="闭环处理完成时间(支撑闭环时效统计)"
    )
