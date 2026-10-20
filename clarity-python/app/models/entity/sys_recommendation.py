"""
推荐记录实体
对齐 clarity-java SysRecommendation.java + BaseEntity
"""

from sqlalchemy import JSON, BigInteger, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysRecommendation(BaseModel):
    __tablename__ = "sys_recommendation"
    __table_args__ = {"comment": "推荐记录表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    image_md5: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="图像MD5（关联特征分析缓存）"
    )
    target_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="model", comment="推荐对象类型"
    )
    top_models: Mapped[list | None] = mapped_column(
        JSON, nullable=True, comment="推荐图像模型列表（JSON数组，统一模型规范 slug）"
    )
    analysis_result: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="图像特征分析结果（JSON）"
    )
    feedback: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=0, comment="推荐反馈(0:未反馈;1:有用;2:无用)"
    )
    adopted_model_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="用户采纳的图像模型标识（统一模型规范 slug）"
    )
