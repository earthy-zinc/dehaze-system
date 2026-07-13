"""
图像输入历史记录实体
对齐 dehaze-java SysInputHistory.java + BaseEntity
对齐 config/sql/schema.sql 中 sys_input_history 表定义
"""

from typing import Optional

from sqlalchemy import BigInteger, Boolean, Integer, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysInputHistory(BaseModel):
    """图像输入历史记录表 (对齐 Java SysInputHistory)"""
    __tablename__ = 'sys_input_history'
    __table_args__ = {'comment': '图像输入历史记录表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='用户ID')
    original_image_url: Mapped[Optional[str]] = mapped_column(String(500), comment='原始图片URL')
    original_thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), comment='原始缩略图URL')
    result_image_url: Mapped[Optional[str]] = mapped_column(String(500), comment='处理结果图片URL')
    result_thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), comment='结果缩略图URL')
    algorithm_id: Mapped[Optional[int]] = mapped_column(BigInteger, comment='算法ID')
    algorithm_name: Mapped[Optional[str]] = mapped_column(String(100), comment='算法名称（冗余）')
    algorithm_params: Mapped[Optional[str]] = mapped_column(Text, comment='算法参数（JSON）')
    processing_time: Mapped[Optional[int]] = mapped_column(Integer, comment='处理耗时（毫秒）')
    # 处理状态（1=成功，2=失败，3=处理中）
    status: Mapped[Optional[int]] = mapped_column(
        mysql_types.TINYINT, default=3, comment='处理状态（1=成功，2=失败，3=处理中）')
    input_source: Mapped[Optional[str]] = mapped_column(String(20), comment='图片来源（upload/camera/sample）')
    is_favorite: Mapped[Optional[bool]] = mapped_column(
        Boolean, default=False, comment='是否收藏')
    sync_status: Mapped[Optional[int]] = mapped_column(
        mysql_types.TINYINT, default=0, comment='同步状态（0=未同步，1=已同步）')
