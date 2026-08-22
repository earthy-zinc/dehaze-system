"""
推荐规则配置实体
对齐 dehaze-java SysRecommendationRule.java + BaseEntity
"""

from sqlalchemy import JSON, BigInteger, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysRecommendationRule(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_recommendation_rule"
    __table_args__ = {"comment": "推荐规则配置表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    rule_name: Mapped[str] = mapped_column(String(64), nullable=False, comment="规则名称")
    scene_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="场景类型")
    algorithm_ids: Mapped[list] = mapped_column(
        JSON, nullable=False, comment="候选算法ID列表（JSON数组）"
    )
    weight: Mapped[int] = mapped_column(
        mysql_types.INTEGER, nullable=False, default=0, comment="规则权重（数值越大越优先）"
    )
    enabled: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment="是否启用(0:禁用;1:启用)"
    )
