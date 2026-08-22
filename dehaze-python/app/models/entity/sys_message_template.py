from typing import Any

from sqlalchemy import JSON, BigInteger, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysMessageTemplate(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_message_template"
    __table_args__ = {"comment": "消息模板表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, comment="模板编码")
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="模板名称")
    type: Mapped[str] = mapped_column(String(32), nullable=False, comment="消息类型")
    title_template: Mapped[str] = mapped_column(
        String(255), nullable=False, default="", comment="标题模板"
    )
    content_template: Mapped[str] = mapped_column(Text, nullable=False, comment="正文模板")
    priority: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=2, comment="默认优先级"
    )
    channels: Mapped[Any | None] = mapped_column(JSON, nullable=True, comment="默认推送渠道")
    variables: Mapped[Any | None] = mapped_column(JSON, nullable=True, comment="变量定义")
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="状态(1:启用;0:禁用)"
    )
