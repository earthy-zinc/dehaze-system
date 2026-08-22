from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysRating(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_rating"
    __table_args__ = {"comment": "处理结果评分评价表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="评价用户ID")
    pred_log_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="关联处理日志ID")
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="算法ID")
    rating: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, comment="评分(1-5星)")
    comment: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="评价文字")
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True, comment="评价标签（JSON数组）")
    image_urls: Mapped[list | None] = mapped_column(
        JSON, nullable=True, comment="截图URL（JSON数组）"
    )
    is_anonymous: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="是否匿名(0:否;1:是)"
    )
    is_hidden: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="是否隐藏(0:否;1:是)"
    )
    admin_reply: Mapped[str | None] = mapped_column(
        String(2000), nullable=True, comment="管理员回复内容"
    )
    reply_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="管理员回复时间"
    )
