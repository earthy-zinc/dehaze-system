"""
文件相关实体模型
"""

from typing import Optional

from app.database import Base
from app.models.base import BaseModel
from sqlalchemy import (CHAR, BigInteger, Index, Integer,
                        String, Text)
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysFile(BaseModel):
    __tablename__ = 'sys_file'
    __table_args__ = (
        Index('md5_key', 'md5', unique=True),
        {'comment': '文件表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='文件id')
    type: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment='文件类型')
    url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment='文件url')
    name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment='文件原始名')
    object_name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment='文件存储名')
    size: Mapped[str] = mapped_column(
        String(100), nullable=False, default='0', comment='文件大小(格式化)')
    size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=0, comment='文件大小(字节)')
    path: Mapped[str] = mapped_column(
        String(255), nullable=False, comment='文件路径')
    md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, unique=True,
                                     comment='文件的MD5值，用于比对文件是否相同')
    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment='逻辑删除标识(0:未删除;1:已删除)')
