"""
文件相关实体模型
"""

from typing import Optional

from app.database import Base
from app.models.base import BaseModel
from sqlalchemy import (CHAR, VARCHAR, BigInteger, Column, Index, Integer,
                        String, Text)
from sqlalchemy.orm import Mapped, mapped_column


class SysFile(BaseModel):
    __tablename__ = 'sys_file'
    __table_args__ = (
        Index('md5_key', 'md5', unique=True),
        {'comment': '文件表'}
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment='文件id')
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
    path: Mapped[str] = mapped_column(
        String(255), nullable=False, comment='文件路径')
    md5: Mapped[str] = mapped_column(CHAR(32), nullable=False, unique=True,
                                     comment='文件的MD5值，用于比对文件是否相同')


class SysWpxFile(Base):
    __tablename__ = 'sys_wpx_file'
    __table_args__ = {'comment': 'WPX文件表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='id')
    origin_file_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, comment='旧文件id')
    origin_md5: Mapped[str] = mapped_column(CHAR(32), unique=True,
                                            nullable=False, comment='旧文件的MD5值')
    origin_path: Mapped[str] = mapped_column(
        VARCHAR(255), nullable=False, comment='旧文件路径')
    new_file_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, comment='新文件id')
    new_path: Mapped[str] = mapped_column(
        VARCHAR(255), nullable=False, comment='新文件路径')
    new_md5: Mapped[str] = mapped_column(
        CHAR(32), unique=True, nullable=False, comment='新文件的MD5值')
