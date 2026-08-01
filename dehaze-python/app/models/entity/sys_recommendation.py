"""
推荐记录实体
对齐 dehaze-java SysRecommendation.java + BaseEntity
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, JSON
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysRecommendation(BaseModel):
    __tablename__ = "sys_recommendation"
    __table_args__ = {"comment": "推荐记录表"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="主键")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    image_md5: Mapped[str] = mapped_column(String(32), nullable=False, comment="图像MD5（关联特征分析缓存）")
    target_type: Mapped[str] = mapped_column(String(32), nullable=False, default="algorithm", comment="推荐对象类型")
    top_algorithms: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, comment="推荐算法列表（JSON数组）")
    analysis_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="图像特征分析结果（JSON）")
    feedback: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment="推荐反馈(0:未反馈;1:有用;2:无用)")
    adopted_algorithm_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, comment="用户采纳的算法ID")
