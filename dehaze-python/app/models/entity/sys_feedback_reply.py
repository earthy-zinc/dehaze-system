from typing import Optional

from sqlalchemy import BigInteger, String, JSON, SmallInteger
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysFeedbackReply(BaseModel):
    __tablename__ = 'sys_feedback_reply'
    __table_args__ = {'comment': '反馈回复表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    feedback_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='反馈ID')
    replier_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='回复人ID')
    replier_type: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, comment='回复人类型(1:用户;2:管理员)')
    content: Mapped[str] = mapped_column(String(2000), nullable=False, comment='回复内容')
    reply_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, comment='回复类型')
    attachments: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment='附件URL（JSON数组）')
    deleted: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
