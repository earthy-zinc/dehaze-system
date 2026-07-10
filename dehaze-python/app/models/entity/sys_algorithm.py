"""
算法模型实体
"""

from typing import Optional

from app.models.base import BaseModel
from sqlalchemy import BigInteger, Column, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysAlgorithm(BaseModel):
    __tablename__ = 'sys_algorithm'
    __table_args__ = {'comment': '算法模型表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                    autoincrement=True, comment='模型id')
    parent_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, default=0, comment='模型的父id')
    type: Mapped[str] = mapped_column(String(100), default='', comment='模型类型')
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, comment='模型名称')
    img: Mapped[Optional[str]] = mapped_column(Text, comment='模型图片')
    path: Mapped[str] = mapped_column(
        String(255), default='', comment='模型存储路径')
    size: Mapped[Optional[str]] = mapped_column(String(100), comment='模型大小')
    params: Mapped[Optional[str]] = mapped_column(String(255), comment='模型参数')
    flops: Mapped[Optional[str]] = mapped_column(
        String(255), comment='模型浮点运算次数')
    import_path: Mapped[Optional[str]] = mapped_column(
        String(255), comment='模型代码导入路径')
    description: Mapped[Optional[str]] = mapped_column(
        String(2048), comment='针对该模型的详细描述')
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=1, comment='状态(1:启用；0:禁用)')
