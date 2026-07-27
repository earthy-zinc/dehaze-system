"""
WPX 文件映射实体（对齐 Java SysWpxFile）
"""

from typing import Optional

from sqlalchemy import BigInteger, CHAR, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysWpxFile(BaseModel):
    __tablename__ = 'sys_wpx_file'
    __table_args__ = (
        Index('idx_origin_md5', 'origin_md5'),
        {'comment': 'WPX 文件映射表'},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='id')
    origin_file_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment='旧文件id')
    origin_md5: Mapped[str] = mapped_column(
        CHAR(32), nullable=False, unique=True, comment='旧文件的MD5值')
    origin_path: Mapped[str] = mapped_column(
        String(255), nullable=False, comment='旧文件路径')
    new_file_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment='新文件id')
    new_path: Mapped[str] = mapped_column(
        String(255), nullable=False, comment='新文件路径')
    new_md5: Mapped[str] = mapped_column(
        CHAR(32), nullable=False, unique=True, comment='新文件的MD5值')
